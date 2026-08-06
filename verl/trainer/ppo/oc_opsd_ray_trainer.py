"""
OC-OPSD (Observation-Calibrated GRPO) Ray Trainer.

Extends RayPPOTrainer with two teacher forward passes (Full + Ablated) that
use short-horizon future observations as privileged context.  The log-prob
residual between the two teachers is used to calibrate the GRPO advantage at
token level — no auxiliary loss is added.

Training objective (unchanged GRPO clipped loss, advantage replaced):
  A^OC_{i,k,t} = A_i · (1 + β · m_oc_{i,k,t} · sign(A_seq_i) · q_{i,k,t})
  where q = tanh((log π_F - log π_A) / 2)  ∈ [-1, 1]
        m_oc = response_mask · m_future · m_adv

See verl/trainer/ppo/oc_opsd_utils.py for the core math.
See change_achieve_log/change_goal.md for the full design spec.
"""

import collections
import math
import os
import warnings
from pprint import pprint

import numpy as np

import ray
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    _timer,
    apply_invalid_action_penalty,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.oc_opsd_utils import (
    ASSISTANT_MARKERS,                       # Issue-1: module-level import (was in for-loop)
    build_action_schema,
    build_k1_future_block,
    build_k2_future_block,
    insert_future_evidence_before_response,
    normalize_advantages_shape,
    compute_oc_advantage_modulation,
    compute_oc_residual_opsd_loss,           # Baseline C
)
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

from agent_system.multi_turn_rollout import adjust_batch


# ---------------------------------------------------------------------------
# Teacher batch construction helper (module-level, pure function)
# ---------------------------------------------------------------------------

def _build_oc_teacher_batch(
    batch: DataProto,
    full: bool,
    env_name: str,
    tokenizer,
    max_prompt_length: int,
) -> tuple:  # (DataProto, float) — teacher batch + truncated_ratio
    """
    Build a Full or Ablated Teacher batch for OC-OPSD.

    For each sample i, the teacher prompt is constructed as:
        teacher_prompt = insert_future_evidence_before_response(
            original_prompt_text,
            full/ablated_future_block,
        )

    For samples where available_future_len == 0, no future block is added,
    so teacher_prompt == student_prompt.  In that case Full == Ablated,
    e = 0, q = 0, m_oc = 0, and A_oc = A exactly (vanilla GRPO fallback).

    Follows the same pattern as build_teacher_batch() in rlsd_ray_trainer.py:
      - batch.batch["input_ids"]  shape: [bs, max_prompt_len + response_len]
      - batch.batch["responses"]  shape: [bs, response_len]
      - Decode valid prompt tokens via attention_mask (left-padded).
      - Tokenize modified teacher prompt (left-pad to max_prompt_length).
      - Concatenate teacher_prompt_ids + response_ids → teacher_input_ids.

    Args:
        batch:            The current training batch (post-rollout, post-adjust_batch).
        full:             True → Full Teacher (real future obs).
                          False → Ablated Teacher (obs replaced by "not provided").
        env_name:         Environment name for action schema dispatch.
        tokenizer:        Model tokenizer.
        max_prompt_length: Maximum prompt token length from config.

    Returns:
        DataProto with keys: input_ids, attention_mask, position_ids, responses.
        Shape is [bs, max_prompt_length + response_length].
    """
    bs = batch.batch["input_ids"].size(0)
    response_length = batch.batch["responses"].size(1)

    teacher_input_ids_list = []
    teacher_attention_mask_list = []
    teacher_position_ids_list = []

    # Pull future-evidence fields from non_tensor_batch (numpy arrays of shape [bs])
    future_obs_1   = batch.non_tensor_batch.get("future_obs_text_1",   np.array([""] * bs))
    future_obs_2   = batch.non_tensor_batch.get("future_obs_text_2",   np.array([""] * bs))
    future_act_1   = batch.non_tensor_batch.get("future_action_text_1", np.array([""] * bs))
    avail_future   = batch.non_tensor_batch.get("available_future_len",  np.zeros(bs, dtype=int))

    # OC-OPSD debug: print 2 sample prompts for human alignment verification.
    _oc_debug = os.environ.get("OC_OPSD_DEBUG_TEACHER", "0") == "1"
    _oc_debug_count = 0
    _oc_debug_max = 2
    _oc_debug_label = "Full" if full else "Ablated"

    _truncated_list: list[bool] = []   # Issue-5: track truncation per sample

    for i in range(bs):
        # ---- 1. Decode the original student prompt (strip left-padding) ----
        # batch["input_ids"][i] = [padded_prompt | response] (from generate_sequences)
        original_input_ids    = batch.batch["input_ids"][i]         # [max_prompt + response_len]
        original_attention_mask = batch.batch["attention_mask"][i]  # same shape

        prompt_length    = original_input_ids.size(0) - response_length
        prompt_ids       = original_input_ids[:prompt_length]
        prompt_mask      = original_attention_mask[:prompt_length]

        # Find first non-padding token (left-padded prompts)
        nonzero_positions = prompt_mask.nonzero(as_tuple=True)[0]
        valid_start = int(nonzero_positions[0].item()) if len(nonzero_positions) > 0 else 0
        valid_prompt_ids = prompt_ids[valid_start:]
        prompt_text = tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)

        # ---- 2. Build future evidence block ----
        afl = int(avail_future[i])   # 0, 1, or 2

        if afl == 0:
            # No future obs available: teacher prompt == student prompt.
            # Full == Ablated → e = 0 → q = 0 → m_oc = 0 → A_oc = A.
            teacher_prompt_text = prompt_text

        elif afl == 1:
            obs_k1  = str(future_obs_1[i]) if future_obs_1[i] else ""
            block = build_k1_future_block(obs_k1=obs_k1, full=full)
            teacher_prompt_text = insert_future_evidence_before_response(prompt_text, block)

        else:  # afl >= 2
            obs_k1     = str(future_obs_1[i]) if future_obs_1[i] else ""
            act_raw_1  = str(future_act_1[i]) if future_act_1[i] else ""
            obs_k2     = str(future_obs_2[i]) if future_obs_2[i] else ""
            schema_k1  = build_action_schema(act_raw_1, env_name)
            block = build_k2_future_block(
                obs_k1=obs_k1,
                schema_k1=schema_k1,
                obs_k2=obs_k2,
                full=full,
            )
            teacher_prompt_text = insert_future_evidence_before_response(prompt_text, block)

        # ---- 4. Tokenize teacher prompt ----
        teacher_prompt_ids = tokenizer.encode(teacher_prompt_text, add_special_tokens=False)
        prompt_len_before_trunc = len(teacher_prompt_ids)

        # Left-truncate if too long (keep end of prompt: current obs + future block).
        truncated = prompt_len_before_trunc > max_prompt_length
        if truncated:
            teacher_prompt_ids = teacher_prompt_ids[-max_prompt_length:]
            prompt_len_after_trunc = len(teacher_prompt_ids)

            # Issue-5 / P2: per-sample length info gated behind OC_OPSD_DEBUG_TEACHER
            # to avoid flooding stdout in production (many samples may truncate).
            if _oc_debug:
                print(
                    f"[OC-OPSD {_oc_debug_label}] sample {i} prompt truncated: "
                    f"{prompt_len_before_trunc} → {prompt_len_after_trunc} tokens "
                    f"(dropped {prompt_len_before_trunc - prompt_len_after_trunc} from head)."
                )
            # Issue-6: warn if assistant marker was lost during truncation.
            # ASSISTANT_MARKERS and warnings are module-level (Issue-1/P5 fix).
            tail_text = tokenizer.decode(teacher_prompt_ids, skip_special_tokens=False)
            if not any(m in tail_text for m in ASSISTANT_MARKERS):
                warnings.warn(
                    f"[OC-OPSD] sample {i}: assistant marker lost after left-truncation "
                    f"(teacher prompt too long). Future evidence insertion may be malformed.",
                    stacklevel=2,
                )
        _truncated_list.append(truncated)

        # ---- 3. Debug: print before tokenizing (Issue-3: expanded fields) ----
        # step_index, is_empty bools, response_valid_len, response text[:100] added
        # per spec section 15 requirements.
        if _oc_debug and _oc_debug_count < _oc_debug_max:
            # Determine response_valid_len for the debug print
            _resp_mask_dbg = original_attention_mask[-response_length:]
            _valid_len_dbg = int(_resp_mask_dbg.sum().item())
            _resp_ids_dbg  = batch.batch["responses"][i][:_valid_len_dbg]
            _resp_text_dbg = tokenizer.decode(_resp_ids_dbg, skip_special_tokens=True)
            _step_idx_dbg  = batch.non_tensor_batch.get("step_index",
                                                          [None] * bs)[i]
            print(
                f"\n=== OC-OPSD {_oc_debug_label} Teacher Debug [sample {i}] ===\n"
                f"  step_index           : {_step_idx_dbg}\n"
                f"  available_future_len : {afl}\n"
                f"  future_obs_1 empty   : {not str(future_obs_1[i])}\n"
                f"  future_obs_2 empty   : {not str(future_obs_2[i])}\n"
                f"  future_obs_1 (120ch) : {str(future_obs_1[i])[:120]!r}\n"
                f"  future_action_raw_1  : {str(future_act_1[i])[:80]!r}\n"
                f"  action_schema        : {(build_action_schema(str(future_act_1[i]), env_name) if afl >= 2 else 'N/A')!r}\n"
                f"  future_obs_2 (120ch) : {str(future_obs_2[i])[:120]!r}\n"
                f"  response_valid_len   : {_valid_len_dbg}/{response_length}\n"
                f"  response_text[:100]  : {_resp_text_dbg[:100]!r}\n"
                f"  prompt_truncated     : {truncated} ({prompt_len_before_trunc}→"
                f"{len(teacher_prompt_ids)} tokens)\n"
                f"  prompt tail (300ch)  :\n"
                f"  {teacher_prompt_text[-300:]!r}"
            )
            _oc_debug_count += 1

        teacher_prompt_ids = torch.tensor(teacher_prompt_ids, dtype=torch.long)
        actual_prompt_len  = len(teacher_prompt_ids)

        # ---- 5. Left-pad to max_prompt_length ----
        pad_length = max_prompt_length - actual_prompt_len
        if pad_length > 0:
            pad_ids = torch.full((pad_length,), tokenizer.pad_token_id, dtype=torch.long)
            teacher_prompt_ids = torch.cat([pad_ids, teacher_prompt_ids])
            t_prompt_mask = torch.cat([
                torch.zeros(pad_length,       dtype=torch.long),
                torch.ones(actual_prompt_len, dtype=torch.long),
            ])
        else:
            t_prompt_mask = torch.ones(actual_prompt_len, dtype=torch.long)

        # ---- 6. Concatenate with VALID response ids only (Bug-2 fix) ----
        # spec section 7: teacher input must not contain trailing padding tokens.
        # Only valid (non-padding) response tokens are included; explicit pad tokens
        # are appended at the end to maintain a uniform [max_prompt_len + response_len]
        # shape for stacking.  Attention mask zeros out the padding positions so the
        # model never attends to them.
        response_ids_i = batch.batch["responses"][i]               # [response_len]

        # Bug-1 fix: use the authoritative response_mask when available.
        # In sglang + multi_turn mode, attention_mask contains interleaved observation
        # tokens, so [-response_length:] would over-count and shift the valid_len.
        # batch.batch["response_mask"] (set by compute_response_mask in fit()) is
        # the canonical mask for model-generated response tokens only.
        if "response_mask" in batch.batch:
            response_mask_raw = batch.batch["response_mask"][i]        # [response_len]
        else:
            response_mask_raw = original_attention_mask[-response_length:]  # fallback
        valid_len = int(response_mask_raw.sum().item())

        valid_response_ids = response_ids_i[:valid_len]               # [valid_len]
        resp_pad_len       = response_length - valid_len
        if resp_pad_len > 0:
            resp_pad = torch.full((resp_pad_len,), tokenizer.pad_token_id, dtype=torch.long)
            full_response_ids  = torch.cat([valid_response_ids, resp_pad])
            full_response_mask = torch.cat([
                torch.ones(valid_len,     dtype=torch.long),
                torch.zeros(resp_pad_len, dtype=torch.long),
            ])
        else:
            full_response_ids  = valid_response_ids
            full_response_mask = response_mask_raw  # already all-ones

        teacher_full_ids  = torch.cat([teacher_prompt_ids, full_response_ids])
        teacher_full_mask = torch.cat([t_prompt_mask,      full_response_mask])
        teacher_pos_ids   = compute_position_id_with_mask(teacher_full_mask.unsqueeze(0))[0]

        teacher_input_ids_list.append(teacher_full_ids)
        teacher_attention_mask_list.append(teacher_full_mask)
        teacher_position_ids_list.append(teacher_pos_ids)

    # Compute truncated_ratio for Issue-2: must be logged as wandb metric.
    n_truncated    = sum(_truncated_list)
    truncated_ratio = float(n_truncated) / max(bs, 1)
    if n_truncated > 0:
        print(
            f"[OC-OPSD {_oc_debug_label}] {n_truncated}/{bs} teacher prompts truncated "
            f"(ratio={truncated_ratio:.3f}).  Future evidence may have been silently dropped."
        )

    teacher_batch = DataProto.from_dict(
        tensors={
            "input_ids":      torch.stack(teacher_input_ids_list),
            "attention_mask": torch.stack(teacher_attention_mask_list),
            "position_ids":   torch.stack(teacher_position_ids_list),
            "responses":      batch.batch["responses"],  # unchanged
        },
    )
    # Issue-2: return truncated_ratio so callers can log it as a wandb metric.
    return teacher_batch, truncated_ratio


# ---------------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------------

class OCOPSDRayTrainer(RayPPOTrainer):
    """
    Observation-Calibrated GRPO Trainer.

    Implements OC-OPSD: for each agent step that has future observations
    (available_future_len > 0) and a non-trivial GRPO advantage (|A| > eps_A),
    the trainer:
      1. Builds a Full Teacher prompt (student context + future evidence).
      2. Builds an Ablated Teacher prompt (same structure, obs replaced by
         "not provided").
      3. Runs a single no_grad forward pass over both (concatenated batch).
      4. Computes q = tanh((log π_F - log π_A) / 2) per token.
      5. Replaces batch["advantages"] with A_oc = A · (1 + β · m_oc · sign(A_seq) · q).

    No auxiliary loss is added; actor update uses pure GRPO clipped loss.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        oc_cfg = self.config.algorithm.get("oc_opsd", {})
        self.oc_beta     = float(oc_cfg.get("beta",     0.5))
        self.oc_eps_A    = float(oc_cfg.get("eps_A",    1e-6))
        # Default to "alfworld" so build_action_schema uses the correct template
        # even if the config key is missing.  An empty string silently falls
        # through to the "a future action" fallback for every K=2 sample.
        self.oc_env_name = str(oc_cfg.get("env_name",   "alfworld"))
        self.oc_max_prompt_length = self.config.data.max_prompt_length
        # Placeholder reads — not yet used by the first-version implementation,
        # but kept here for config validation and future extension.
        self.oc_K_future    = int(oc_cfg.get("K_future",    2))   # future window size
        self.oc_H_history   = int(oc_cfg.get("H_history",  10))   # history window (reserved)
        # AOC variant: only reinforce when OC signal agrees with GRPO direction
        self.oc_aligned_only = bool(oc_cfg.get("aligned_only", False))

        # Per-trajectory top-K step selection for teacher forward.
        # 1.0 (default) = all eligible steps (current behaviour, no overhead).
        # < 1.0 = select only the top-ratio fraction of steps per trajectory,
        # ranked by mean NLL of the student's response tokens (highest NLL first).
        # Example: 0.5 → within each trajectory, score only the 50% of steps
        # where the student was most uncertain about its own generated response.
        self.oc_top_step_ratio = float(oc_cfg.get("top_step_ratio", 1.0))

        # ------------------------------------------------------------------
        # Baseline B — Full Teacher OPSD (π_F as SDAR teacher)
        #   L_B = agg( σ(β·(log π_F − log π_S)) · (log π_F − log π_S) )
        #   Reuses existing actor use_sdar_loss mechanism; no new actor code.
        #   Default OFF to leave main OC-OPSD experiment unchanged.
        # ------------------------------------------------------------------
        self.oc_use_full_teacher_opsd = bool(oc_cfg.get("use_full_teacher_opsd", False))
        if self.oc_use_full_teacher_opsd:
            from omegaconf import open_dict
            with open_dict(self.config):
                self.config.actor_rollout_ref.actor.use_sdar_loss    = True
                self.config.actor_rollout_ref.actor.sdar_loss_coef   = float(
                    oc_cfg.get("full_teacher_opsd_coef", 0.01))
                self.config.actor_rollout_ref.actor.sdar_gate_beta   = float(
                    oc_cfg.get("full_teacher_opsd_gate_beta", 5.0))

        # ------------------------------------------------------------------
        # Baseline C — OC-Residual OPSD (log π_F − log π_A as weight)
        #   L_C = agg( (log π_F − log π_A) · (log π_F − log π_S) )
        #   Requires a new actor loss branch; handled in dp_actor.py.
        #   Default OFF to leave main OC-OPSD experiment unchanged.
        # ------------------------------------------------------------------
        self.oc_use_residual_opsd = bool(oc_cfg.get("use_oc_residual_opsd", False))
        if self.oc_use_residual_opsd:
            from omegaconf import open_dict
            with open_dict(self.config):
                self.config.actor_rollout_ref.actor.use_oc_residual_opsd_loss = True
                self.config.actor_rollout_ref.actor.oc_residual_opsd_coef     = float(
                    oc_cfg.get("oc_residual_opsd_coef", 0.01))

    # ------------------------------------------------------------------
    # Teacher log-prob computation
    # ------------------------------------------------------------------

    def _compute_nll_top_step_mask(
        self,
        batch: DataProto,
        response_mask: torch.Tensor,
        avail_future_arr: np.ndarray,
        A_seq_np: np.ndarray,
    ) -> np.ndarray:
        """
        Compute per-trajectory NLL-based step selection mask.

        Only steps that are *eligible* (available_future_len > 0 AND |A_seq| > eps_A)
        participate in the ranking.  Within each trajectory, the top `top_step_ratio`
        fraction of eligible steps (by mean response NLL, descending) are selected.

        When top_step_ratio >= 1.0, returns the eligible mask directly (all eligible
        steps selected, no NLL computation needed).

        Args:
            batch:             Current training batch.
            response_mask:     [bs, response_len] binary mask for response tokens.
            avail_future_arr:  [bs] int — available_future_len per flat sample.
            A_seq_np:          [bs] float — |A_seq| per flat sample.

        Returns:
            step_select_mask: [bs] bool numpy — True for steps to teacher-score.
        """
        bs = response_mask.shape[0]

        # Eligible steps: must have future obs AND non-trivial advantage
        eligible_mask = (
            (avail_future_arr > 0) & (np.abs(A_seq_np) > self.oc_eps_A)
        )

        if self.oc_top_step_ratio >= 1.0:
            # All eligible steps; no NLL ranking needed
            return eligible_mask

        # Step-level NLL from old_log_probs (already in batch — zero extra compute).
        # step_nll[i] = mean neg log-prob of flat sample i's valid response tokens.
        old_lp      = batch.batch["old_log_probs"]   # [bs, response_len]
        resp_m      = response_mask.float()           # [bs, response_len]
        step_nll    = (
            -(old_lp * resp_m).sum(-1) / resp_m.sum(-1).clamp(min=1)
        )
        step_nll_np = step_nll.detach().cpu().numpy()   # [bs]

        # Per-trajectory top-K selection (only within eligible steps)
        traj_uids   = batch.non_tensor_batch.get("traj_uid", np.arange(bs, dtype=object))
        select_mask = np.zeros(bs, dtype=bool)

        for uid in np.unique(traj_uids):
            # Eligible steps of this trajectory only
            idx = np.where((traj_uids == uid) & eligible_mask)[0]
            if len(idx) == 0:
                continue
            k = max(1, math.ceil(len(idx) * self.oc_top_step_ratio))
            # Descending NLL → most uncertain steps first
            top_idx = idx[np.argsort(step_nll_np[idx])[::-1][:k]]
            select_mask[top_idx] = True

        # Final mask: eligible AND in top-K NLL within trajectory
        return eligible_mask & select_mask

    def _compute_full_ablated_log_probs(
        self,
        batch: DataProto,
        step_select_mask: np.ndarray | None = None,
    ) -> tuple:
        """
        Build Full + Ablated teacher batches and return per-token log-probs.

        Two paths controlled by step_select_mask:

        Full-batch path (step_select_mask is None):
          - Operate on ALL bs samples (same as SDAR / RLSD pattern).
          - adjust_batch() guarantees bs % world_size == 0 → 2*bs also divisible.
          - Ineligible samples get Full == Ablated (e=0, q=0, m_oc=0, A_oc=A).
          - No pad/unpad needed.

        Sub-batch path (step_select_mask is provided, top_step_ratio < 1.0):
          - Only selected steps go through teacher forward.
          - n_selected = step_select_mask.sum() is NOT guaranteed % world_size == 0.
          - Requires pad_dataproto_to_divisor + unpad_dataproto.
          - Non-selected steps have log_probs = 0 → exact GRPO fallback.
          - Saves ~(1-ratio) × teacher forward compute.

        Args:
            batch:             Current training batch.
            step_select_mask:  [bs] bool numpy — steps to teacher-score.
                               None → full-batch path (backward compatible).

        Returns:
            full_log_probs:    [bs, response_len]  (detached, 0 for non-selected)
            ablated_log_probs: [bs, response_len]  (detached, 0 for non-selected)
            truncated_ratio:   float
        """
        bs           = batch.batch["input_ids"].size(0)
        response_len = batch.batch["responses"].size(1)
        device       = batch.batch["advantages"].device

        # ---- Shared response mask (for C-2: zero out padding positions) ----
        if "response_mask" in batch.batch:
            response_mask_float = batch.batch["response_mask"].float()
        else:
            response_mask_float = batch.batch["attention_mask"][:, -response_len:].float()

        def _teacher_forward(src_batch: DataProto, n: int) -> tuple:
            """Build Full+Ablated, concat, forward, return (combined_lp, trunc_ratio)."""
            full_tb,    full_tr = _build_oc_teacher_batch(
                batch=src_batch, full=True,
                env_name=self.oc_env_name, tokenizer=self.tokenizer,
                max_prompt_length=self.oc_max_prompt_length,
            )
            ablated_tb, _       = _build_oc_teacher_batch(
                batch=src_batch, full=False,
                env_name=self.oc_env_name, tokenizer=self.tokenizer,
                max_prompt_length=self.oc_max_prompt_length,
            )
            combined = DataProto.from_dict(tensors={
                key: torch.cat([full_tb.batch[key], ablated_tb.batch[key]], dim=0)
                for key in ["input_ids", "attention_mask", "position_ids", "responses"]
            })
            combined.meta_info = src_batch.meta_info.copy()

            # Pad to world_size multiple (required by chunk() inside compute_log_prob)
            padded, pad_size = pad_dataproto_to_divisor(
                combined, self.actor_rollout_wg.world_size
            )
            out_padded = self.actor_rollout_wg.compute_log_prob(padded)
            out        = unpad_dataproto(out_padded, pad_size)

            lp = out.batch["old_log_probs"].detach().to(device)  # [2*n, response_len]
            return lp, full_tr

        # ==== Full-batch path ====
        if step_select_mask is None:
            combined_lp, truncated_ratio = _teacher_forward(batch, bs)
            full_log_probs    = combined_lp[:bs] * response_mask_float
            ablated_log_probs = combined_lp[bs:] * response_mask_float

        # ==== Sub-batch path (NLL step selection) ====
        else:
            full_log_probs    = torch.zeros(bs, response_len, device=device)
            ablated_log_probs = torch.zeros(bs, response_len, device=device)
            n_selected        = int(step_select_mask.sum())
            truncated_ratio   = 0.0

            if n_selected > 0:
                mask_idx_np = np.where(step_select_mask)[0]
                mask_idx_t  = torch.tensor(mask_idx_np, dtype=torch.long, device=device)

                # Construct sub-batch DataProto
                sub_tensors = {}
                for k, v in batch.batch.items():
                    idx = torch.tensor(mask_idx_np, dtype=torch.long, device=v.device)
                    sub_tensors[k] = v[idx]

                sub_non_tensor = {}
                for k, v in batch.non_tensor_batch.items():
                    try:
                        if hasattr(v, "__len__") and len(v) == bs:
                            sub_non_tensor[k] = v[mask_idx_np]
                        else:
                            sub_non_tensor[k] = v   # global field, keep as-is
                    except Exception:
                        sub_non_tensor[k] = v

                sub_batch = DataProto.from_dict(tensors=sub_tensors)
                sub_batch.non_tensor_batch = sub_non_tensor
                sub_batch.meta_info        = batch.meta_info.copy()

                # Teacher forward on sub-batch (2*n_selected samples, padded)
                combined_lp, truncated_ratio = _teacher_forward(sub_batch, n_selected)
                # combined_lp: [2*n_selected, response_len]

                # C-2: zero out padding positions using sub-batch response mask
                if "response_mask" in sub_tensors:
                    sub_resp_mask = sub_tensors["response_mask"].float().to(device)
                else:
                    sub_resp_mask = (sub_tensors["attention_mask"]
                                    [:, -response_len:].float().to(device))

                full_lp_sub    = combined_lp[:n_selected] * sub_resp_mask
                ablated_lp_sub = combined_lp[n_selected:] * sub_resp_mask

                assert full_lp_sub.shape == (n_selected, response_len), \
                    f"full_lp_sub {full_lp_sub.shape} != ({n_selected}, {response_len})"

                # Scatter back to [bs, response_len]; non-selected remain 0
                full_log_probs[mask_idx_t]    = full_lp_sub
                ablated_log_probs[mask_idx_t] = ablated_lp_sub

        assert full_log_probs.shape    == (bs, response_len), \
            f"full_log_probs {full_log_probs.shape} != ({bs}, {response_len})"
        assert ablated_log_probs.shape == full_log_probs.shape

        return full_log_probs, ablated_log_probs, truncated_ratio

    # ------------------------------------------------------------------
    # Diagnostic data collection
    # ------------------------------------------------------------------

    def _save_oc_diagnostic(
        self,
        batch: DataProto,
        full_log_probs: torch.Tensor,
        ablated_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        avail_future_arr: np.ndarray,
        step_select_mask,          # np.ndarray[bool, shape [bs]] or None
    ) -> None:
        """
        Collect per-step diagnostic statistics and save to parquet.

        Called after _compute_full_ablated_log_probs() returns, before
        compute_oc_advantage_modulation() runs.  All three log-prob tensors
        are live at this point.

        Output: {diagnostic_dir}/step_{global_steps:06d}.parquet
        Each row = one sample in the batch (one env-step in one trajectory).
        """
        try:
            import pandas as pd   # lazy import – zero cost when disabled
            import threading

            # ---- 1. Resolve config ----
            _diag_cfg = self.config.algorithm.get("oc_opsd", {})   # safe: no crash if key absent
            diag_dir = _diag_cfg.get("diagnostic_dir", None)
            if diag_dir is None:
                diag_dir = os.path.join(
                    self.config.trainer.default_local_dir, "oc_diagnostics"
                )
            os.makedirs(diag_dir, exist_ok=True)
            out_path = os.path.join(diag_dir, f"step_{self.global_steps:06d}.parquet")

            # ---- 2. Move tensors to CPU float32 numpy ----
            # All three are [bs, response_len]; response_mask is int64 [bs, response_len]
            old_lp  = batch.batch["old_log_probs"].detach().float().cpu()   # [bs, R]
            full_lp = full_log_probs.detach().float().cpu()                  # [bs, R]
            abl_lp  = ablated_log_probs.detach().float().cpu()               # [bs, R]
            adv     = advantages.detach().float().cpu()                       # [bs, R]
            mask    = response_mask.detach().float().cpu()                    # [bs, R]

            bs = old_lp.shape[0]

            # ---- 2b. Truncate to max_samples to bound file size ----
            max_n = int(_diag_cfg.get("diagnostic_max_samples", 512))
            n = min(bs, max_n)
            if n < bs:
                old_lp  = old_lp[:n];  full_lp = full_lp[:n];  abl_lp = abl_lp[:n]
                adv = adv[:n];  mask = mask[:n]
                avail_future_arr = avail_future_arr[:n]
                if step_select_mask is not None:
                    step_select_mask = step_select_mask[:n]
            bs = n

            # ---- 3. Compute per-sample aggregates (masked mean over response tokens) ----
            mask_sum_raw = mask.sum(dim=1)               # true token count, may be 0
            mask_sum     = mask_sum_raw.clamp(min=1.0)   # safe divisor only

            def masked_mean(x):
                # x: [bs, R]; returns numpy float32 array of shape [bs]
                return ((x * mask).sum(dim=1) / mask_sum).numpy()

            delta_F    = full_lp - old_lp           # log π_F - log π_S  [bs, R]
            delta_A    = abl_lp  - old_lp           # log π_A - log π_S  [bs, R]
            residual_e = full_lp - abl_lp           # log π_F - log π_A  [bs, R]
            q_val      = torch.tanh(residual_e / 2) # q ∈ [-1, 1]        [bs, R]

            # Scalar advantage per sample: mean over valid response tokens
            adv_scalar = masked_mean(adv)

            records = {
                "global_step":          [self.global_steps] * bs,
                "batch_idx":            list(range(bs)),
                # --- identity fields from non_tensor_batch ---
                # NOTE: [:bs] slice is required — when n < original_bs (truncation fired),
                # batch.non_tensor_batch still has length original_bs; we must cap to n.
                "traj_uid":             list(batch.non_tensor_batch.get(
                                            "traj_uid", np.array([""] * bs)))[:bs],
                "uid":                  list(batch.non_tensor_batch.get(
                                            "uid",      np.array([""] * bs)))[:bs],
                "step_index":           list(batch.non_tensor_batch.get(
                                            "step_index", np.zeros(bs, dtype=int)))[:bs],
                "available_future_len": list(avail_future_arr),
                # --- text field for labeling / inspection ---
                "future_obs_text_1":    list(batch.non_tensor_batch.get(
                                            "future_obs_text_1", np.array([""] * bs)))[:bs],
                # --- per-sample token count (true count, may be 0) ---
                "num_tokens":           mask_sum_raw.numpy().astype(int).tolist(),
                # --- step_selected: unified meaning = has future obs (avail_future_len > 0) ---
                "step_selected":        list(avail_future_arr > 0),
                # --- is_nll_selected: True/False only in sub-batch path; None in full-batch ---
                "is_nll_selected":      (
                    list(step_select_mask.astype(bool))
                    if step_select_mask is not None
                    else [None] * bs   # NaN in DataFrame = full-batch path, not applicable
                ),
                # --- advantage (pre-OC) ---
                "advantage":            adv_scalar.tolist(),
                # --- log-prob delta aggregates ---
                "step_delta_F":         masked_mean(delta_F).tolist(),
                "step_delta_A":         masked_mean(delta_A).tolist(),
                "step_residual_e":      masked_mean(residual_e).tolist(),
                "step_q":               masked_mean(q_val).tolist(),
                # --- absolute value aggregates ---
                "step_abs_delta_F":     masked_mean(delta_F.abs()).tolist(),
                "step_abs_delta_A":     masked_mean(delta_A.abs()).tolist(),
                "step_abs_residual_e":  masked_mean(residual_e.abs()).tolist(),
                "step_abs_q":           masked_mean(q_val.abs()).tolist(),
                # --- raw log-prob means (useful for sanity checks) ---
                "mean_old_log_prob":    masked_mean(old_lp).tolist(),
                "mean_full_log_prob":   masked_mean(full_lp).tolist(),
                "mean_abl_log_prob":    masked_mean(abl_lp).tolist(),
            }

            # ---- 4. Write parquet (async, daemon thread – does not block training) ----
            df = pd.DataFrame(records)
            def _write(df=df, path=out_path):
                df.to_parquet(path, index=False, compression="snappy")
            threading.Thread(target=_write, daemon=True).start()

        except Exception as _diag_exc:
            # Diagnostic errors must never interrupt training.
            print(
                f"[OC-OPSD DIAG WARNING] step={self.global_steps}: "
                f"diagnostic save failed: {_diag_exc}"
            )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def fit(self):
        """
        OC-OPSD training loop.

        Differs from RLSDRayTrainer.fit() in the advantage section:
          - No skill provider / teacher-before-advantage pattern.
          - GRPO advantages computed first (need A_seq for sign_A and m_adv).
          - Full + Ablated teacher forward runs after advantage computation.
          - Advantages replaced by A_oc via compute_oc_advantage_modulation.
          - No auxiliary loss injected into the actor.
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="OC-OPSD Training",
        )
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics    = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # Pop keys consumed by the gen-batch (not needed for actor update)
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                for k in ["multi_modal_data", "raw_prompt", "tools_kwargs", "env_kwargs"]:
                    if k in batch.non_tensor_batch:
                        non_tensor_batch_keys_to_pop.append(k)
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):

                    # ---- 1. Rollout ----
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                            gen_batch=gen_batch,
                            actor_rollout_wg=self.actor_rollout_wg,
                            envs=self.envs,
                            is_train=True,
                        )

                    del batch
                    batch = gen_batch_output

                    batch = adjust_batch(self.config, batch)
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # G3-B1: assert OC modulation mask == policy loss mask in multi-turn.
                    # dp_actor uses loss_mask[:, -response_len:] for multi_turn;
                    # compute_response_mask must produce the identical tensor, otherwise
                    # A_oc is modulated with a different mask than the one applying the loss.
                    # Current path (vllm + multi_turn.enable=False) is safe; this guard
                    # catches future regressions when switching to sglang + multi_turn.
                    if self.config.actor_rollout_ref.rollout.multi_turn.enable:
                        if "loss_mask" in batch.batch:
                            _response_len = batch.batch["responses"].size(1)
                            _loss_mask_slice = batch.batch["loss_mask"][:, -_response_len:]
                            assert torch.equal(
                                batch.batch["response_mask"], _loss_mask_slice
                            ), (
                                "[OC-OPSD] response_mask != loss_mask[:, -response_len:] "
                                "in multi-turn mode. OC modulation would apply to wrong tokens."
                            )

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = (
                        torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    )

                    # ---- 1b. Group-level reward statistics (before reward shaping) ----
                    # Compute from episode_rewards in non_tensor_batch (set by rollout).
                    # uid   = GRPO group ID (shared by all rollout.n trajectories from same prompt).
                    # traj_uid = unique per trajectory; used to deduplicate step-level rows.
                    # episode_rewards is stored once per step → each trajectory appears T times.
                    # Dedup by traj_uid first so each trajectory contributes exactly one reward
                    # value to the group statistics (fixes group_reward_var being compressed to 0).
                    _uid_arr  = batch.non_tensor_batch.get("uid",           None)
                    _traj_uid = batch.non_tensor_batch.get("traj_uid",      None)
                    _ep_rw    = batch.non_tensor_batch.get("episode_rewards", None)
                    if _uid_arr is not None and _ep_rw is not None and len(_uid_arr) > 0:
                        # --- dedup: one entry per trajectory ---
                        if _traj_uid is not None:
                            _seen = set()
                            _dedup_uid, _dedup_rw = [], []
                            for _tu, _u, _r in zip(_traj_uid, _uid_arr, _ep_rw):
                                if _tu not in _seen:
                                    _seen.add(_tu)
                                    _dedup_uid.append(_u)
                                    _dedup_rw.append(float(_r))
                        else:
                            # traj_uid unavailable — fallback: all rows (less accurate)
                            _dedup_uid = list(_uid_arr)
                            _dedup_rw  = [float(r) for r in _ep_rw]
                        # --- group by uid ---
                        _uid_to_rw = collections.defaultdict(list)
                        for _u, _r in zip(_dedup_uid, _dedup_rw):
                            _uid_to_rw[_u].append(_r)
                        if len(_uid_to_rw) > 0:
                            _group_sr = [
                                sum(r > 0 for r in rws) / len(rws)
                                for rws in _uid_to_rw.values()
                            ]
                            metrics["episode/group_any_success"] = float(
                                sum(s > 0.0 for s in _group_sr) / len(_group_sr)
                            )
                            metrics["episode/group_all_success"] = float(
                                sum(s >= 1.0 for s in _group_sr) / len(_group_sr)
                            )
                            metrics["episode/group_sr_std"] = float(
                                np.std(_group_sr) if len(_group_sr) > 1 else 0.0
                            )
                            metrics["episode/group_reward_var"] = float(np.mean([
                                np.var(rws) if len(rws) > 1 else 0.0
                                for rws in _uid_to_rw.values()
                            ]))

                    # ---- 2. Rewards ----
                    with _timer("reward", timing_raw):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                batch, self.config, self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(
                                batch, self.reward_fn
                            )

                    # ---- 3. Student log probs ----
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(
                            loss_mat=entropys,
                            loss_mask=response_masks,
                            loss_agg_mode=loss_agg_mode,
                        )
                        metrics["actor/entropy_loss"] = entropy_loss.detach().item()
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    # ---- 4. Reference policy log probs (optional, for KL) ----
                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    # ---- 5. GRPO advantages ----
                    with _timer("adv", timing_raw):
                        reward_extra_infos_dict: dict
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        if self.config.actor_rollout_ref.actor.get(
                            "use_invalid_action_penalty", True
                        ):
                            batch, invalid_metrics = apply_invalid_action_penalty(
                                batch,
                                invalid_action_penalty_coef=(
                                    self.config.actor_rollout_ref.actor.invalid_action_penalty_coef
                                ),
                            )
                            metrics.update(invalid_metrics)

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = (
                                batch.batch["token_level_scores"]
                            )

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                            step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                            gigpo_mode=self.config.algorithm.gigpo.mode,
                            gigpo_enable_similarity=self.config.algorithm.gigpo.enable_similarity,
                            gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                        )
                        # batch["advantages"] now has shape [bs, response_len]

                    # ---- 6. OC-OPSD: Full + Ablated teacher forward ----
                    # Skip entirely when beta=0: A_oc = A trivially (pure GRPO),
                    # running two teacher forwards would waste ~2x compute for nothing.
                    if self.oc_beta != 0.0:
                        # ---- 6. OC-OPSD: Full + Ablated teacher forward ----
                        # Teacher forward runs on the FULL batch (same pattern as RLSD).
                        # adjust_batch() already ensures bs % world_size == 0, so the
                        # concatenated [2*bs] batch is also divisible — no pad/unpad needed.
                        # Ineligible samples (afl=0) get Full==Ablated prompts → e=0, q=0,
                        # m_oc=0, A_oc=A.  Their teacher forward is ~5% of bs (terminal steps)
                        # and cheaper than the DataProto sub-batch indexing that replaced it.
                        with _timer("oc_teacher_forward", timing_raw):
                            advantages    = batch.batch["advantages"]    # [bs, response_len]
                            response_mask = batch.batch["response_mask"] # [bs, response_len]

                            # available_future_len: needed here for avail_future_arr (step 7)
                            avail_future_np = batch.non_tensor_batch.get(
                                "available_future_len",
                                np.zeros(advantages.shape[0], dtype=int),
                            )
                            # Issue-2: guard dtype conversion with a diagnostic error
                            try:
                                avail_future_arr = np.array(avail_future_np, dtype=np.int64)
                            except (ValueError, TypeError) as _e:
                                raise ValueError(
                                    f"[OC-OPSD] available_future_len cannot be cast to int64. "
                                    f"dtype={getattr(avail_future_np, 'dtype', type(avail_future_np))}, "
                                    f"first 3 values={avail_future_np[:3]}. "
                                    f"Original error: {_e}"
                                ) from _e

                            # Critical fix: enforce K_future config as an upper bound.
                            # Without this, K=1 ablation runs identically to K=2 because
                            # rollout data stores the actual available_future_len (up to 2).
                            # np.minimum is a no-op for default K_future=2 (0/1/2 unchanged).
                            avail_future_arr = np.minimum(avail_future_arr, self.oc_K_future)
                            # Write back so _build_oc_teacher_batch reads the clipped value.
                            # Without this line, teacher prompt construction still uses the
                            # original afl=2 from non_tensor_batch and builds K=2 prompts
                            # even when K_future=1 is configured.
                            batch.non_tensor_batch["available_future_len"] = avail_future_arr

                            # ---- NLL-based step mask (computed BEFORE teacher forward) ----
                            # Needs A_seq_np to determine eligible steps (|A_seq| > eps_A).
                            # Note: normalize_advantages_shape is called again inside
                            # compute_oc_advantage_modulation (P3 redundant call, acceptable).
                            _, _A_seq = normalize_advantages_shape(advantages, response_mask)
                            _A_seq_np = _A_seq.detach().abs().cpu().numpy()

                            # step_select_mask: eligible & top-NLL-ratio within each traj.
                            # None when top_step_ratio >= 1.0 (full-batch path, no overhead).
                            if self.oc_top_step_ratio < 1.0:
                                step_select_mask = self._compute_nll_top_step_mask(
                                    batch, response_mask, avail_future_arr, _A_seq_np
                                )
                            else:
                                step_select_mask = None   # full-batch path

                            # ---- Teacher forward (only for selected steps) ----
                            full_log_probs, ablated_log_probs, trunc_ratio = (
                                self._compute_full_ablated_log_probs(
                                    batch, step_select_mask=step_select_mask
                                )
                            )

                            # ---- 6c. Diagnostic data collection (optional) ----
                            # Config access wrapped in try/except: OmegaConf structured
                            # config may raise ConfigAttributeError if key absent.
                            # Minor known issues (non-blocking):
                            #   m1: reward field absent (can cross-analyse with advantage)
                            #   m2: response_decoded text absent (qualitative check only)
                            #   m3: diagnostic call inside oc_teacher_forward timer (~5-20ms bias)
                            #   m4: num_tokens=0 yields 0.0 mean; analysis must filter df[num_tokens>0]
                            try:
                                _diag_cfg     = self.config.algorithm.get("oc_opsd", {})
                                _diag_enabled = bool(_diag_cfg.get("save_diagnostic_data", False))
                                _diag_freq    = max(1, int(_diag_cfg.get("diagnostic_save_freq", 10)))
                            except Exception:
                                _diag_enabled = False
                                _diag_freq    = 10
                            if _diag_enabled and (self.global_steps % _diag_freq == 0):
                                self._save_oc_diagnostic(
                                    batch=batch,
                                    full_log_probs=full_log_probs,
                                    ablated_log_probs=ablated_log_probs,
                                    advantages=advantages,
                                    response_mask=response_mask,
                                    avail_future_arr=avail_future_arr,
                                    step_select_mask=step_select_mask,
                                )

                            metrics["oc_opsd/future_block_truncated_ratio"] = trunc_ratio

                            # Selection metrics (only when sub-batch path is active)
                            if step_select_mask is not None:
                                # eligible = afl>0 AND |A_seq|>eps_A (matches _compute_nll_top_step_mask)
                                _eligible = (
                                    (avail_future_arr > 0) & (np.abs(_A_seq_np) > self.oc_eps_A)
                                )
                                metrics["oc_opsd/nll_eligible_step_ratio"] = float(_eligible.mean())
                                metrics["oc_opsd/nll_selected_step_ratio"] = float(
                                    step_select_mask.mean()
                                )

                        # ---- 6b. Baseline injections (OFF by default) ----
                        # Baseline B: Full Teacher OPSD
                        #   Put full_log_probs as teacher → actor uses use_sdar_loss branch.
                        # Baseline C: OC-Residual OPSD
                        #   Put full + ablated log_probs → actor uses use_oc_residual_opsd_loss branch.
                        # Both flags default to False; main OC-OPSD experiment is unaffected.
                        _dev = advantages.device
                        if self.oc_use_full_teacher_opsd or self.oc_use_residual_opsd:
                            # Baseline B and C both need full_log_probs as teacher
                            batch.batch["teacher_log_probs"] = full_log_probs.to(_dev)
                        if self.oc_use_residual_opsd:
                            # Baseline C additionally needs ablated_log_probs for OC gate
                            batch.batch["oc_ref_log_probs"] = ablated_log_probs.to(_dev)

                        # ---- 7. OC advantage modulation ----
                        with _timer("oc_modulation", timing_raw):
                            # available_future_len tensor for compute_oc_advantage_modulation
                            available_future_len = torch.tensor(
                                avail_future_arr,
                                dtype=torch.long,
                                device=advantages.device,
                            )  # [bs]

                            A_oc, oc_metrics = compute_oc_advantage_modulation(
                                advantages=advantages,
                                full_log_probs=full_log_probs.to(advantages.device),
                                ablated_log_probs=ablated_log_probs.to(advantages.device),
                                response_mask=response_mask,
                                available_future_len=available_future_len,
                                beta=self.oc_beta,
                                eps_A=self.oc_eps_A,
                                aligned_only=self.oc_aligned_only,
                            )

                            # Sanity check: sign_flip_count should be 0 when beta <= 1
                            if oc_metrics.get("oc_opsd/sign_flip_count", 0) > 0:
                                print(
                                    f"[OC-OPSD WARNING] step={self.global_steps}: "
                                    f"sign_flip_count={oc_metrics['oc_opsd/sign_flip_count']} "
                                    f"(beta={self.oc_beta:.3f}). This indicates a bug."
                                )

                            batch.batch["advantages"] = A_oc
                            metrics.update(oc_metrics)

                    # ---- 8. Update actor (pure GRPO, no aux loss) ----
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(
                            critic_output.meta_info["metrics"]
                        )
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = (
                                self.config.actor_rollout_ref.rollout.multi_turn.enable
                            )
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(
                            actor_output.meta_info["metrics"]
                        )
                        metrics.update(actor_output_metrics)

                    # ---- 9. Dump rollout generations (optional) ----
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            inputs  = self.tokenizer.batch_decode(
                                batch.batch["prompts"], skip_special_tokens=True
                            )
                            outputs = self.tokenizer.batch_decode(
                                batch.batch["responses"], skip_special_tokens=True
                            )
                            scores  = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # ---- 10. Validation & checkpoint ----
                    test_start_step = self.config.trainer.get("test_start_step", 0)
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (
                            is_last_step
                            or (
                                self.global_steps >= test_start_step
                                and self.global_steps % self.config.trainer.test_freq == 0
                            )
                        )
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # ---- Logging ----
                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch":        epoch,
                })
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    compute_throughout_metrics(
                        batch=batch, timing_raw=timing_raw, n_gpus=n_gpus
                    )
                )

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
