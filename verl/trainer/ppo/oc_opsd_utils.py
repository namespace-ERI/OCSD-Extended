"""
OC-OPSD utility functions  (verl/trainer/ppo/oc_opsd_utils.py)

Provides the core algorithmic helpers for Observation-Calibrated GRPO (OC-OPSD).
This module is a pure utility layer — it depends only on `re` and `torch` and
can be imported safely from any part of the training stack without risk of
circular imports with trainer / actor / rollout / ray modules.

Public API
----------
extract_action_text          – strip <think> wrappers, return executable action
build_action_schema          – naturalize action (mask entity/query/answer args)
insert_future_evidence_before_response
                             – splice future evidence block before assistant marker
build_k1_future_block        – K=1 Full / Ablated future evidence text
build_k2_future_block        – K=2 Full / Ablated future evidence text
normalize_advantages_shape   – normalize advantages to [bs, resp_len] + A_seq [bs]
compute_oc_advantage_modulation
                             – compute q, A_oc, and diagnostic metrics
"""

import re
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Known assistant-turn start markers across common chat templates.
# insert_future_evidence_before_response uses rfind over this list.
ASSISTANT_MARKERS = [
    "<|im_start|>assistant",                          # Qwen / ChatML
    "<|start_header_id|>assistant<|end_header_id|>",  # Llama-3
    "<|assistant|>",                                  # Phi-3 / some Mistral templates
    "[ASSISTANT]",                                    # some custom templates (NC-3 fix)
    "### Assistant:",                                 # Alpaca-style
    "Assistant:",                                     # plain text fallback
]


# ===========================================================================
# 1. extract_action_text
# ===========================================================================

def extract_action_text(response_str: str) -> str:
    """
    Extract the executable action from a raw model response string.

    Different environments use different action tag conventions:
      • ALFWorld / WebShop : <action>...</action>
      • Search             : <search>...</search>  or  <answer>...</answer>
        (returned WITH outer tags so build_action_schema can detect type)

    Priority order:
      1. <action>...</action>   → inner content (stripped)
      2. <search>...</search>   → full tag (preserved for schema dispatch)
      3. <answer>...</answer>   → full tag (preserved for schema dispatch)
      4. 'Action:' line prefix  → text after 'Action:' on same line
      5. Fallback               → stripped response_str, or "a future action"

    Args:
        response_str: Raw decoded model response (may contain <think> blocks,
                      special tokens, or trailing pad garbage).

    Returns:
        Executable action string, never None or empty.
    """
    # Priority 1: <action>...</action>  (ALFWorld, WebShop)
    m = re.search(r'<action>(.*?)</action>', response_str, re.DOTALL | re.IGNORECASE)
    if m:
        extracted = m.group(1).strip()
        if extracted:
            return extracted

    # Priority 2: <search>...</search>  (Search env search call)
    # Return WITH outer tags so schema builder can detect "search" type.
    m = re.search(r'(<search>.*?</search>)', response_str, re.DOTALL | re.IGNORECASE)
    if m:
        extracted = m.group(1).strip()
        if extracted:
            return extracted

    # Priority 3: <answer>...</answer>  (Search env final answer)
    m = re.search(r'(<answer>.*?</answer>)', response_str, re.DOTALL | re.IGNORECASE)
    if m:
        extracted = m.group(1).strip()
        if extracted:
            return extracted

    # Priority 4: 'Action:' prefix (generic fallback format)
    m = re.search(r'(?i)^action:\s*(.+)', response_str, re.MULTILINE)
    if m:
        extracted = m.group(1).strip()
        if extracted:
            return extracted

    # Priority 5: last resort — return stripped raw response
    stripped = response_str.strip()
    return stripped if stripped else "a future action"


# ===========================================================================
# 2. build_action_schema
# ===========================================================================

def build_action_schema(action: str, env_name: str) -> str:
    """
    Convert a raw executable action to a naturalized action schema.

    Core principle: preserve action intent (verb / tool name), replace
    entity / query / answer / path arguments with generic placeholders.

    Dispatches to environment-specific helpers based on env_name.
    Unrecognised environments fall back to "a future action".

    Args:
        action:   Executable action text (output of extract_action_text).
        env_name: Environment name string from config, e.g.
                  "alfworld/AlfredTWEnv", "webshop", "search".

    Returns:
        Naturalized schema string, e.g. "take an item from a receptacle".
        Always returns a non-empty string.
    """
    env = (env_name or "").lower()

    if "alfworld" in env:
        return _alfworld_schema(action)
    elif "minihack" in env:
        # Preserve movement direction, the entire executable action in this task.
        direction = re.sub(r'</?answer>', '', action.lower()).strip()
        allowed = {'north', 'east', 'south', 'west', 'northeast', 'southeast', 'southwest', 'northwest'}
        return f'move {direction}' if direction in allowed else 'a movement action'
    elif "scienceworld" in env:
        return _scienceworld_schema(action)
    elif "webshop" in env:
        return _webshop_schema(action)
    elif "search" in env:
        return _search_schema(action)
    else:
        # Unknown environment — safe fallback, no information leakage.
        return "a future action"


def _scienceworld_schema(action: str) -> str:
    """Mask object names/answers while retaining the scientific operation."""
    value = action.strip().lower()
    if value in {'look around', 'inventory', 'task', 'wait', 'wait1'}:
        return value
    for prefix, schema in [
        ('focus on ', 'focus on a target'), ('go to ', 'go to a location'),
        ('look at ', 'look at an object'), ('move ', 'move an object to a container'),
        ('pour ', 'pour a substance into a container'), ('use ', 'use a tool on a target'),
        ('connect ', 'connect electrical components'), ('disconnect ', 'disconnect components'),
        ('activate ', 'activate a device'), ('deactivate ', 'deactivate a device'),
        ('pick up ', 'pick up an object'), ('open ', 'open a container'),
        ('close ', 'close a container'), ('mix ', 'mix substances'),
        ('eat ', 'eat an object'), ('read ', 'read an object'),
        ('examine ', 'examine a target'), ('drop ', 'drop an object'),
        ('dunk ', 'dunk an object in a substance'), ('flush ', 'flush a container')]:
        if value.startswith(prefix):
            return schema
    return 'a scientific interaction'


def _alfworld_schema(action: str) -> str:
    """
    ALFWorld action schema mapping.

    ALFWorld's admissible command templates are fixed and generated in
    agent_system/environments/env_package/alfworld/alfworld/agents/utils/misc.py.
    The full template set is:

        go to {recep}
        open {recep}
        close {recep}
        take {obj} from {recep}
        move {obj} to {recep}       ← place/put action (NOT "put X in/on Y")
        use {lamp}                  ← look_at_obj_in_light task
        heat {obj} with {microwave}
        cool {obj} with {fridge}
        clean {obj} with {cleaner}
        slice {obj} with {knife}
        inventory / look / help     ← state-query; no entity; fall through to fallback
        examine {obj}
        examine {recep}

    Two-argument patterns are checked before single-argument ones to prevent
    prefix false-positives (e.g. "move" before "open").
    """
    a = action.strip().lower()

    # --- two-argument patterns ---
    if re.match(r"^take .+ from .+", a):
        return "take an item from a receptacle"
    if re.match(r"^move .+ to .+", a):
        # "move {obj} to {recep}" is the actual place/put action in ALFWorld.
        return "move an item to a location"
    if re.match(r"^heat .+ with .+", a):
        return "heat an item with an appliance"
    if re.match(r"^cool .+ with .+", a):
        return "cool an item with an appliance"
    if re.match(r"^clean .+ with .+", a):
        return "clean an item with a fixture"
    if re.match(r"^slice .+ with .+", a):
        return "slice an item with a tool"

    # --- single-argument patterns ---
    if re.match(r"^go to .+", a):
        return "go to a location"
    if re.match(r"^open .+", a):
        return "open a receptacle"
    if re.match(r"^close .+", a):
        return "close a receptacle"
    if re.match(r"^use .+", a):
        # "use {lamp}" — specific to look_at_obj_in_light tasks.
        return "use a lamp"
    if re.match(r"^examine .+", a):
        # Both "examine {obj}" and "examine {recep}" collapse to the same schema.
        return "examine a target"

    # "inventory", "look", "help" produce no entity argument; fall through.
    return "a future action"


def _search_schema(action: str) -> str:
    """
    Search environment action schema.

    Search env uses XML-style tags (see prompts/search.py):
      <search>query text</search>   →  <search>a query</search>
      <answer>final answer</answer>  →  <answer>an answer</answer>

    The tag is preserved so the Ablated Teacher knows whether the next step
    is a search call or a final answer, without revealing the content.
    """
    a = action.strip()
    if re.match(r'<search>', a, re.IGNORECASE):
        return "<search>a query</search>"
    if re.match(r'<answer>', a, re.IGNORECASE):
        return "<answer>an answer</answer>"
    return "a future action"


def _webshop_schema(action: str) -> str:
    """
    WebShop action schema — Intent-Aware version (v2).

    WebShop uses lowercase tool[arg] format delivered inside <action> tags.
    The projection function (webshop/projection.py) lowercases and strips the
    action before writing it to future_action_text_1, so `action` is always
    lowercase here.

    v2 (Intent-Aware) schema mapping
    ---------------------------------
    search[...]                         →  search[a query]
    click[buy now]                      →  click[purchase button]
    click[back to search / next/prev]   →  click[navigation or pagination]
    click[short value, no apostrophe]   →  click[a product attribute]
      (sizes: "10", "xl", "large"; colors: "red", "black"; etc.)
    click[anything else]                →  click[a specific product]
      (product link text, typically longer and/or contains apostrophe)

    Rationale: collapsing all click actions to "click[an item]" (v1) removes
    all intent information from the K=2 Next-action scaffold, making the Full/
    Ablated teacher comparison nearly uninformative.  Distinguishing purchase,
    navigation, attribute selection, and product clicks gives the teacher strong
    directional signal — especially for the "click[purchase button]" case, which
    provides decisive evidence about whether the preceding search/browse steps
    were on track.

    Heuristic for attribute vs product:
      WebShop attribute values (radio button inputs) are short single-token or
      two-token values without apostrophes: "red", "10", "xl", "dark blue".
      Product link text tends to be longer or contains "'" (Men's, Women's).
      Threshold: len(clicked) <= 20 and "'" not in clicked → attribute.

    # -------------------------------------------------------------------------
    # v1 (original, kept for reference / easy rollback):
    #
    # def _webshop_schema(action: str) -> str:
    #     a = action.strip()
    #     m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\[.*\]$', a, re.DOTALL)
    #     if m:
    #         tool = m.group(1).lower()
    #         if tool == "search":
    #             return "search[a query]"
    #         elif tool == "click":
    #             return "click[an item]"
    #         else:
    #             return f"{m.group(1)}[an argument]"
    #     return "a future action"
    # -------------------------------------------------------------------------
    """
    a = action.strip().lower()

    # ---- search ----
    if re.match(r'^search\[', a):
        return "search[a query]"

    # ---- parse tool[arg] ----
    m = re.match(r'^([a-z_][a-z0-9_]*)\[(.*)\]$', a, re.DOTALL)
    if not m:
        return "a future action"

    tool    = m.group(1)
    clicked = m.group(2).strip()

    if tool != "click":
        # Non-click, non-search tool (rare in WebShop) — preserve tool name.
        return f"{tool}[an argument]"

    # ---- click intent classification ----

    # 1. Purchase button — highest priority, exact match.
    #    WebShop's END_BUTTON constant is "buy now" (lowercased).
    if clicked == "buy now":
        return "click[purchase button]"

    # 2. Navigation / Pagination — exact match on fixed button labels.
    if clicked in {"back to search", "next page", "prev page", "previous page"}:
        return "click[navigation or pagination]"

    # 3. Product attribute — short values without apostrophes.
    #    Radio-button option values (colors, sizes, materials, styles) are
    #    typically ≤ 20 chars and contain no possessive apostrophe.
    #    Examples: "red", "10", "xl", "dark blue", "one size", "cotton".
    if len(clicked) <= 20 and "'" not in clicked:
        return "click[a product attribute]"

    # 4. Product link — anything else (longer text, brand/model names, often
    #    contains apostrophe from "Men's" / "Women's" etc.).
    return "click[a specific product]"


# ===========================================================================
# 3. insert_future_evidence_before_response
# ===========================================================================

def insert_future_evidence_before_response(
    prompt_text: str,
    future_block: str,
) -> str:
    """
    Insert a future evidence block into a chat prompt just before the
    assistant response marker.

    The future evidence must reside in the context (user / system) side of
    the prompt.  Placing it AFTER the assistant marker causes the model to
    score the future evidence as part of its own previously generated
    response, which is semantically wrong and will corrupt the logprobs.

    Strategy:
      1. Search all ASSISTANT_MARKERS with rfind (latest occurrence).
      2. Insert future_block immediately before the latest marker found,
         separated by double newlines.
      3. If no marker is found, append future_block to the end of prompt_text.

    Args:
        prompt_text:  Full chat prompt string for the current step.
        future_block: Future evidence text (from build_k1/k2_future_block).

    Returns:
        Modified prompt string with future_block inserted at the correct position.
    """
    insert_pos = -1
    for marker in ASSISTANT_MARKERS:
        pos = prompt_text.rfind(marker)
        if pos > insert_pos:
            insert_pos = pos

    block = "\n\n" + future_block.strip() + "\n\n"

    if insert_pos >= 0:
        before = prompt_text[:insert_pos].rstrip()
        after  = prompt_text[insert_pos:].lstrip()
        return before + block + after
    else:
        # No marker found — append; future evidence still precedes response ids
        # which are concatenated externally by the trainer.
        return prompt_text.rstrip() + block


# ===========================================================================
# 4. build_k1_future_block
# ===========================================================================

def build_k1_future_block(obs_k1: str, full: bool) -> str:
    """
    Build the future evidence block for the K=1 case (only o_{k+1} available).

    K=1 contains ONLY the immediate observation after the current action.
    It must NOT contain 'Next action' or 'After next action' fields — those
    belong exclusively to K=2.

    Args:
        obs_k1: Text of o_{k+1}.  Ignored (replaced with "not provided") when
                full=False.
        full:   True  → Full Teacher block (real observation).
                False → Ablated Teacher block ("not provided").

    Returns:
        Multi-line future evidence block string.
    """
    obs_line = obs_k1.strip() if (full and obs_k1) else "not provided"
    return (
        "Future evidence:\n"
        "After current action:\n"
        f"Observation: {obs_line}"
    )


# ===========================================================================
# 5. build_k2_future_block
# ===========================================================================

def build_k2_future_block(
    obs_k1: str,
    schema_k1: str,
    obs_k2: str,
    full: bool,
) -> str:
    """
    Build the future evidence block for the K=2 case (o_{k+1}, g(a_{k+1}), o_{k+2}).

    Full Teacher and Ablated Teacher share the SAME 'Next action' line (the
    naturalized schema g(a_{k+1})).  The ONLY difference is whether the
    'Observation' lines show real text or "not provided".

    This strict structural alignment is critical: the residual
        e = log π_F - log π_A
    should reflect only the incremental effect of seeing future observations,
    not any difference in prompt structure.

    Args:
        obs_k1:    Text of o_{k+1}.  Ignored for ablated.
        schema_k1: Naturalized action schema g(a_{k+1}).  Shared by both.
                   Falls back to "a future action" if empty.
        obs_k2:    Text of o_{k+2}.  Ignored for ablated.
        full:      True  → Full Teacher; False → Ablated Teacher.

    Returns:
        Multi-line future evidence block string.
    """
    obs1_line    = obs_k1.strip() if (full and obs_k1) else "not provided"
    obs2_line    = obs_k2.strip() if (full and obs_k2) else "not provided"
    action_schema = schema_k1.strip() if (schema_k1 and schema_k1.strip()) else "a future action"

    return (
        "Future evidence:\n"
        "After current action:\n"
        f"Observation: {obs1_line}\n"
        "\n"
        "Next action:\n"
        f"{action_schema}\n"
        "\n"
        "After next action:\n"
        f"Observation: {obs2_line}"
    )


# ===========================================================================
# 6. normalize_advantages_shape
# ===========================================================================

def normalize_advantages_shape(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple:
    """
    Normalize advantages to token-level shape [bs, response_len] and extract
    the sample-level scalar A_seq [bs].

    OC-OPSD needs two granularities of advantages:
      • A      [bs, response_len] — for token-level modulation multiplication
      • A_seq  [bs]               — for sign_A and m_adv (must be sample-level)

    Using sign(A) (token-level) instead of sign(A_seq) (sample-level) would
    give sign=0 for any masked-to-zero token, silently neutralising the
    modulation for those tokens.

    Supported input shapes:
      [bs]              → expand to [bs, response_len]
      [bs, 1]           → expand to [bs, response_len]
      [bs, response_len]→ use as-is; A_seq = masked-mean per sample

    Args:
        advantages:    Advantage tensor, one of the supported shapes.
        response_mask: Binary mask [bs, response_len], 1 for valid response tokens.

    Returns:
        (A, A_seq):
          A:     torch.Tensor  [bs, response_len]
          A_seq: torch.Tensor  [bs]
    """
    bs, resp_len = response_mask.shape

    if advantages.dim() == 1:
        # Input: [bs]
        A_seq = advantages
        A = advantages.view(bs, 1).expand(bs, resp_len).clone()

    elif advantages.dim() == 2 and advantages.shape[1] == 1:
        # Input: [bs, 1]
        A_seq = advantages.view(bs)
        A = advantages.expand(bs, resp_len).clone()

    elif advantages.dim() == 2 and advantages.shape == response_mask.shape:
        # Input: [bs, response_len]  (current GRPO already broadcasts here)
        A = advantages
        denom = response_mask.sum(dim=-1).clamp(min=1)
        A_seq = (advantages * response_mask).sum(dim=-1) / denom

    else:
        raise ValueError(
            f"Unsupported advantages shape {tuple(advantages.shape)}. "
            f"Expected [bs={bs}], [bs,1], or [bs,response_len={resp_len}]."
        )

    return A, A_seq


# ===========================================================================
# 7. compute_oc_advantage_modulation
# ===========================================================================

def compute_oc_advantage_modulation(
    advantages: torch.Tensor,
    full_log_probs: torch.Tensor,
    ablated_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    available_future_len: torch.Tensor,
    beta: float = 0.5,
    eps_A: float = 1e-6,
    aligned_only: bool = False,
) -> tuple:
    """
    Compute the OC-OPSD token-level advantage modulation.

    Core formula
    ------------
      e              = log π_F(y_t) − log π_A(y_t)       observation residual
      q              = tanh(e / 2)                         ∈ [−1, 1]
      sign_A         = sign(A_seq)                         sample-level sign
      m_future       = (available_future_len > 0)          [bs, 1]
      m_adv          = (|A_seq| > eps_A)                   [bs, 1]
      m_oc           = response_mask · m_future · m_adv    [bs, response_len]
      modulation     = 1 + β · m_oc · sign_A · q
      A_oc           = A · modulation

    Invariants
    ----------
      m_oc == 0   →   A_oc == A          (exact vanilla-GRPO fallback)
      β ≤ 1       →   sign(A_oc) == sign(A)   (sign-preserving)

    Inputs
    ------
      advantages:           [bs] | [bs,1] | [bs, response_len]
      full_log_probs:       [bs, response_len]  (0.0 where teacher not scored)
      ablated_log_probs:    [bs, response_len]  (0.0 where teacher not scored)
      response_mask:        [bs, response_len]  binary, 1 for valid response tokens
      available_future_len: [bs]  int tensor (values 0 / 1 / 2)
      beta:                 float ≤ 1, modulation strength (default 0.5)
      eps_A:                float, near-zero advantage threshold (default 1e-6)

    Returns
    -------
      (A_oc, metrics):
        A_oc:    torch.Tensor [bs, response_len]
        metrics: dict  {str: float}  — all keys prefixed with 'oc_opsd/'
    """
    bs, resp_len = response_mask.shape

    # ---- 1. Shape normalization ----
    A, A_seq = normalize_advantages_shape(advantages, response_mask)

    # ---- 2. sign_A: sample-level, broadcast to [bs, response_len] ----
    sign_A = torch.sign(A_seq).view(bs, 1)          # [bs, 1]

    # ---- 3. Masks ----
    m_future = (available_future_len > 0).float().view(bs, 1)   # [bs, 1]
    m_adv    = (A_seq.abs() > eps_A).float().view(bs, 1)        # [bs, 1]
    m_oc     = response_mask.float() * m_future * m_adv          # [bs, response_len]

    # ---- 4. Observation residual ----
    # e = 0 where teacher was not scored (full == ablated == 0), so q = 0
    # and modulation_factor = 1, meaning A_oc = A exactly for those tokens.
    e = full_log_probs - ablated_log_probs    # [bs, response_len]
    q = torch.tanh(e / 2.0)                  # [bs, response_len], ∈ [−1, 1]

    # ---- 5. Modulation ----
    if aligned_only:
        # AOC (Aligned-only OC): clamp the signed signal to [0, +∞)
        # → only reinforce when OC agrees with GRPO, never suppress
        # → modulation_factor ∈ [1, 1+β]  (no downward adjustment)
        signed_q = torch.clamp(sign_A * q, min=0.0)
    else:
        signed_q = sign_A * q                              # standard OC-OPSD
    modulation_factor = 1.0 + beta * m_oc * signed_q      # [bs, response_len]
    A_oc = A * modulation_factor                           # [bs, response_len]
    # When m_oc == 0: modulation_factor = 1.0, so A_oc = A * 1 = A (exact).

    # ---- Issue-4: m_oc=0 exact degeneracy assertion (spec section 14 check-3) ----
    # Run always (cheap: no_grad + boolean mask); catches modulation_factor bugs early.
    with torch.no_grad():
        no_oc_mask = (~m_oc.bool()) & response_mask.bool()
        if no_oc_mask.any():
            max_diff = (A_oc[no_oc_mask] - A[no_oc_mask]).abs().max().item()
            assert max_diff < 1e-6, (
                f"m_oc=0 tokens deviate from vanilla GRPO: max_diff={max_diff:.2e} "
                f"(expected 0). Check modulation_factor computation."
            )

    # ---- 6. Metrics (all under torch.no_grad()) ----
    with torch.no_grad():
        oc_mask   = m_oc.bool()
        resp_mask = response_mask.bool()
        has_oc    = oc_mask.any()
        n_oc      = oc_mask.sum().clamp(min=1)
        n_resp    = resp_mask.sum().clamp(min=1)

        # modulation_factor min / max — only over m_oc=1 tokens; empty-safe
        if has_oc:
            mod_min = modulation_factor[oc_mask].min()
            mod_max = modulation_factor[oc_mask].max()
        else:
            mod_min = torch.tensor(1.0, device=A_oc.device)
            mod_max = torch.tensor(1.0, device=A_oc.device)

        # q / e metrics: restrict to m_oc=1 tokens (n_oc as denominator)
        # Tokens where teacher was not scored have e=q=0; including them in
        # the average would dilute the signal and misrepresent OC behaviour.
        q_oc = q[oc_mask]      if has_oc else q.new_zeros(1)
        e_oc = e[oc_mask]      if has_oc else e.new_zeros(1)

        # align_ratio: fraction of OC tokens where sign(A_seq) · q > 0
        # (OC residual agrees with GRPO direction → reinforcing modulation)
        sign_A_bc = sign_A.expand_as(q)
        align_num = ((sign_A_bc * q > 0).float() * m_oc).sum()
        align_ratio = (align_num / n_oc).item()

        metrics = {
            # --- residual / q (over m_oc=1 tokens) ---
            "oc_opsd/q_mean":                    q_oc.mean().item(),
            "oc_opsd/q_abs_mean":                q_oc.abs().mean().item(),
            "oc_opsd/q_pos_ratio":               (q_oc > 0).float().mean().item(),
            "oc_opsd/q_neg_ratio":               (q_oc < 0).float().mean().item(),
            "oc_opsd/residual_mean":             e_oc.mean().item(),
            "oc_opsd/residual_abs_mean":         e_oc.abs().mean().item(),
            # --- masks ---
            "oc_opsd/m_oc_ratio":                (m_oc.sum() / n_resp).item(),
            "oc_opsd/m_future_ratio":            m_future.mean().item(),
            "oc_opsd/adv_nonzero_ratio":         m_adv.mean().item(),
            "oc_opsd/available_future_len_mean": available_future_len.float().mean().item(),
            # --- A_oc (over all response tokens) ---
            "oc_opsd/A_oc_mean":                 ((A_oc * response_mask).sum() / n_resp).item(),
            "oc_opsd/A_oc_abs_mean":             ((A_oc.abs() * response_mask).sum() / n_resp).item(),
            # --- modulation factor (over m_oc=1 tokens) ---
            "oc_opsd/modulation_factor_mean":    modulation_factor[oc_mask].mean().item() if has_oc else 1.0,
            "oc_opsd/modulation_factor_min":     mod_min.item(),
            "oc_opsd/modulation_factor_max":     mod_max.item(),
            # --- sanity checks ---
            # sign_flip_count: should be 0 when beta <= 1; non-zero indicates a bug
            "oc_opsd/sign_flip_count":           ((A_oc * A < 0) & resp_mask).sum().item(),
            # align_ratio: fraction of OC tokens where OC residual reinforces GRPO
            "oc_opsd/align_ratio":               align_ratio,
            # adv_delta_abs_mean: mean absolute change in advantage from OC modulation
            "oc_opsd/adv_delta_abs_mean":        ((A_oc - A).abs() * response_mask).sum().div(n_resp).item(),
        }

        # ---- Advantage quality: token-level discriminability before vs after OC ----
        # adv_std_ratio > 1.0 means OC increases token-level advantage spread (more discriminative).
        # Computed over ALL valid response tokens (not just m_oc=1 tokens) so it reflects
        # the full impact of OC modulation on the batch.
        valid_A     = A[resp_mask]        # response tokens, pre-OC   [N_valid]
        valid_A_oc  = A_oc[resp_mask]     # response tokens, post-OC  [N_valid]
        if valid_A.numel() > 1:
            _adv_std_before = valid_A.std().item()
            _adv_std_after  = valid_A_oc.std().item()
            metrics["oc_opsd/adv_std_before"] = _adv_std_before
            metrics["oc_opsd/adv_std_after"]  = _adv_std_after
            metrics["oc_opsd/adv_std_ratio"]  = _adv_std_after / (_adv_std_before + 1e-8)
        else:
            metrics["oc_opsd/adv_std_before"] = 0.0
            metrics["oc_opsd/adv_std_after"]  = 0.0
            metrics["oc_opsd/adv_std_ratio"]  = 1.0

    return A_oc, metrics


# ===========================================================================
# Baseline losses (for ablation / comparison experiments)
# ===========================================================================

def compute_oc_residual_opsd_loss(
    student_log_probs: torch.Tensor,
    full_log_probs: torch.Tensor,
    ablated_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple:
    """
    Baseline C — OC-Residual OPSD loss.

    Uses the OC residual e = log π_F − log π_A as a token-level importance
    weight for the distillation signal from π_F to the current student policy:

        L_C = mean_mask( e · (log π_F − log π_S_current) )
        where e = log π_F − log π_A  (detached)

    Interpretation:
      - e > 0: π_F assigns higher prob than π_A → future obs supports this token
               → strengthen distillation (positive weight)
      - e < 0: π_A assigns higher prob → future obs hurts this token
               → reversed signal (mild anti-distillation)

    Gradient flows only through student_log_probs (π_S_current from actor
    forward); full_log_probs and ablated_log_probs are detached.

    Args:
        student_log_probs:  [bs, response_len]  current policy log probs (grad)
        full_log_probs:     [bs, response_len]  π_F  (detached, no grad)
        ablated_log_probs:  [bs, response_len]  π_A  (detached, no grad)
        response_mask:      [bs, response_len]  binary

    Returns:
        (loss, metrics):
          loss:    scalar
          metrics: dict with "oc_residual_opsd/*" keys
    """
    full_log_probs    = full_log_probs.detach()
    ablated_log_probs = ablated_log_probs.detach()

    # OC residual as token-level weight (detached)
    e = full_log_probs - ablated_log_probs          # [bs, response_len]

    # Distillation signal from Full Teacher to current student (has gradient)
    signal = full_log_probs - student_log_probs      # [bs, response_len]

    # OC-weighted distillation: tokens where future obs helps get stronger signal
    weighted = e * signal                            # [bs, response_len]

    mask_sum = response_mask.sum().clamp(min=1)
    loss = (weighted * response_mask).sum() / mask_sum

    with torch.no_grad():
        metrics = {
            "oc_residual_opsd/e_mean":     (e * response_mask).sum().div(mask_sum).item(),
            "oc_residual_opsd/e_abs_mean": (e.abs() * response_mask).sum().div(mask_sum).item(),
            "oc_residual_opsd/signal_mean":(signal * response_mask).sum().div(mask_sum).item(),
            "oc_residual_opsd/loss":        loss.detach().item(),
        }

    return loss, metrics
