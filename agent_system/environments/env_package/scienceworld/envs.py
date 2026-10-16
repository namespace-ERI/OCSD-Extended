







from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import ray
except ImportError:
    ray = None


Variation = Tuple[int, int]


def _normalize_variations(values: Sequence[Sequence[int]]) -> List[Variation]:
    variations = []
    for index, value in enumerate(values):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"Variation entry {index} must be [task_index, variation_id].")
        task_index, variation_id = int(value[0]), int(value[1])
        if task_index < 0 or variation_id < 0:
            raise ValueError(f"Variation entry {index} contains a negative index: {value!r}.")
        variations.append((task_index, variation_id))
    if not variations:
        raise ValueError("ScienceWorld variation pool is empty.")
    if len(set(variations)) != len(variations):
        raise ValueError("ScienceWorld variation pool contains duplicate task/variation pairs.")
    return variations


def load_variation_manifest(path: str, split: str) -> List[Variation]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"ScienceWorld manifest not found: {manifest_path}. Run "
            "`python examples/data_preprocess/scienceworld.py` first."
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if split not in payload:
        raise KeyError(f"Manifest {manifest_path} does not contain split {split!r}.")
    return _normalize_variations(payload[split])


def _truncate_list(items: Sequence[str], limit: int) -> Tuple[List[str], int]:
    values = [str(item) for item in items]
    if limit <= 0 or len(values) <= limit:
        return values, 0
    return values[:limit], len(values) - limit


class ScienceWorldWorker:
    pass

    def __init__(self, variations: Sequence[Sequence[int]], env_kwargs: Dict[str, Any]):
        try:
            from scienceworld import ScienceWorldEnv
        except ImportError as exc:
            raise ImportError(
                "ScienceWorld is required. Install Java 8+ and run "
                "`python -m pip install scienceworld==1.2.3`."
            ) from exc

        kwargs = dict(env_kwargs or {})
        self.variations = _normalize_variations(variations)
        self.simplification = str(kwargs.get("simplification", "easy") or "")
        self.reward_mode = str(kwargs.get("reward_mode", "score_delta")).lower()
        if self.reward_mode not in {"binary", "score_delta", "milestone"}:
            raise ValueError(
                "scienceworld.reward_mode must be binary, score_delta, or milestone."
            )
        self.success_reward = float(kwargs.get("success_reward", 10.0))
        self.success_score = float(kwargs.get("success_score", 100.0))
        self.max_action_templates = int(kwargs.get("max_action_templates", 40))
        self.max_visible_objects = int(kwargs.get("max_visible_objects", 80))
        self.max_valid_actions = int(kwargs.get("max_valid_actions", 200))
        self.max_action_context_chars = int(kwargs.get("max_action_context_chars", 6000))

        self.env = ScienceWorldEnv(
            taskName=None,
            serverPath=kwargs.get("jar_path") or None,
            envStepLimit=int(kwargs.get("env_step_limit", 30)),
        )
        self.task_names = list(self.env.get_task_names())
        self.pool_index: Optional[int] = None
        self.task_index: Optional[int] = None
        self.variation_id: Optional[int] = None
        self.task_description = ""
        self.last_observation = ""
        self.last_rendered_observation = ""
        self.last_info: Dict[str, Any] = {}
        self.previous_score = 0.0
        self.episode_done = False

    @staticmethod
    def _deduplicate_commands(commands: Sequence[Any]) -> List[str]:
        values = []
        seen = set()
        for command in commands:
            value = " ".join(str(command).split()).strip()
            if not value or value in seen:
                continue
            seen.add(value)
            values.append(value)
        return values

    def _valid_commands(self, raw_info: Optional[Dict[str, Any]]) -> List[str]:
        info_commands = (raw_info or {}).get("valid")
        if isinstance(info_commands, (list, tuple)):
            return self._deduplicate_commands(info_commands)
        try:
            return self._deduplicate_commands(
                self.env.get_valid_action_object_combinations()
            )
        except (AttributeError, NotImplementedError):


            return []

    def _format_valid_commands(self, commands: Sequence[str]) -> str:
        visible, omitted = _truncate_list(commands, self.max_valid_actions)
        header = "Simulator-parseable candidates:"
        lines = [header]
        included = 0
        for command in visible:
            candidate = f"- {command}"
            suffix = "\n" + candidate
            if (
                self.max_action_context_chars > 0
                and len("\n".join(lines)) + len(suffix) > self.max_action_context_chars
            ):
                break
            lines.append(candidate)
            included += 1
        omitted += len(visible) - included
        if omitted:
            note = f"[{omitted} additional valid commands omitted from the prompt]"
            if self.max_action_context_chars <= 0 or len("\n".join(lines + [note])) <= self.max_action_context_chars:
                lines.append(note)
        return "\n".join(lines)

    def _available_actions(
        self, raw_info: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, List[str]]:
        valid_commands = self._valid_commands(raw_info)
        if valid_commands:
            return self._format_valid_commands(valid_commands), valid_commands




        actions, omitted_actions = _truncate_list(
            self.env.get_possible_actions(), self.max_action_templates
        )
        objects, omitted_objects = _truncate_list(
            self.env.get_possible_objects(), self.max_visible_objects
        )
        action_note = f" (+{omitted_actions} omitted)" if omitted_actions else ""
        object_note = f" (+{omitted_objects} omitted)" if omitted_objects else ""
        context = (
            f"Action templates{action_note}: {actions}\n"
            f"Visible object referents{object_note}: {objects}\n"
            "Replace placeholders such as OBJ with one visible referent and emit one exact command."
        )
        if self.max_action_context_chars > 0 and len(context) > self.max_action_context_chars:
            context = context[: self.max_action_context_chars].rstrip() + "\n[Action context truncated]"
        return context, []

    def _render_observation(self, observation: str, available_actions: str) -> str:
        return (
            f"Task goal:\n{self.task_description}\n\n"
            f"Current observation:\n{observation}\n\n"
            f"Currently available command vocabulary:\n{available_actions}"
        )

    def _enrich_info(self, raw_info: Optional[Dict[str, Any]], observation: str) -> Dict[str, Any]:
        info = dict(raw_info or {})
        score = float(info.get("score", self.previous_score))
        available_actions, admissible_commands = self._available_actions(info)
        info.update(
            {
                "task_id": f"{self.task_index}:{self.variation_id}",
                "task_index": int(self.task_index),
                "task_name": self.task_names[int(self.task_index)],
                "task_type": self.task_names[int(self.task_index)],
                "variation_id": int(self.variation_id),
                "variation_index": int(self.pool_index),
                "task_description": self.task_description,
                "available_actions": available_actions,
                "admissible_commands": admissible_commands,
                "observation_text": observation,
                "score": score,
                "task_score": score,
                "won": score == self.success_score,
            }
        )
        return info

    def reset(self, pool_index: int):
        index = int(pool_index)
        if index < 0 or index >= len(self.variations):
            raise IndexError(
                f"ScienceWorld variation index {index} is outside [0, {len(self.variations)})."
            )
        self.pool_index = index
        self.task_index, self.variation_id = self.variations[index]
        if self.task_index >= len(self.task_names):
            raise IndexError(
                f"Task index {self.task_index} is unavailable in installed ScienceWorld "
                f"({len(self.task_names)} tasks)."
            )

        task_name = self.task_names[self.task_index]
        self.env.load(task_name, self.variation_id, self.simplification)
        observation, raw_info = self.env.reset()
        self.task_description = str(self.env.get_task_description()).strip()
        self.previous_score = float((raw_info or {}).get("score", 0.0))
        self.episode_done = False
        self.last_observation = str(observation)
        info = self._enrich_info(raw_info, self.last_observation)
        info.update(
            {
                "won": False,
                "raw_reward": 0.0,
                "score_delta": 0.0,
                "subgoal_completed": False,
                "action_is_effective": True,
            }
        )
        self.last_info = dict(info)
        self.last_rendered_observation = self._render_observation(
            self.last_observation, info["available_actions"]
        )
        return self.last_rendered_observation, info

    def step(self, action: str):
        if self.episode_done:
            info = dict(self.last_info)
            info.update({"terminal_step_replayed": True, "action_is_effective": False})
            return self.last_rendered_observation, 0.0, True, info

        command = str(action).strip()
        if not command:
            info = dict(self.last_info)
            info.update(
                {
                    "won": False,
                    "raw_reward": 0.0,
                    "score_delta": 0.0,
                    "subgoal_completed": False,
                    "action_is_effective": False,
                    "invalid_format_noop": True,
                }
            )
            return self.last_rendered_observation, 0.0, False, info

        observation, raw_reward, done, raw_info = self.env.step(command)
        info = self._enrich_info(raw_info, str(observation))
        score = float(info["score"])
        score_delta = score - self.previous_score
        won = bool(score == self.success_score)
        subgoal_completed = score_delta > 0.0

        if self.reward_mode == "binary":
            reward = self.success_reward if won else 0.0
        elif self.reward_mode == "score_delta":
            reward = score_delta / max(self.success_score, 1.0)
        else:
            reward = (1.0 if subgoal_completed else 0.0) + (
                self.success_reward if won else 0.0
            )

        info.update(
            {
                "won": won,
                "raw_reward": float(raw_reward),
                "score_delta": score_delta,
                "subgoal_completed": subgoal_completed,
                "action_is_effective": str(observation) != self.last_observation,
            }
        )
        self.previous_score = score
        self.last_observation = str(observation)
        self.last_info = dict(info)
        self.last_rendered_observation = self._render_observation(
            self.last_observation, info["available_actions"]
        )
        self.episode_done = bool(done or won)
        return self.last_rendered_observation, float(reward), self.episode_done, info

    def render(self):
        return self.last_rendered_observation

    def close(self):
        env = getattr(self, "env", None)
        if env is not None:
            self.env = None
            env.close()


class ScienceWorldMultiProcessEnv:
    pass

    mode = "text"

    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: Optional[Dict[str, float]] = None,
        is_train: bool = True,
        env_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if ray is None:
            raise ImportError("ScienceWorld training requires Ray.")
        if not ray.is_initialized():
            ray.init()
        if not is_train and int(group_n) != 1:
            raise ValueError("ScienceWorld validation requires group_n=1.")

        kwargs = dict(env_kwargs or {})
        self.is_train = bool(is_train)
        self.group_n = int(group_n)
        self.env_num = int(env_num)
        self.num_processes = self.env_num * self.group_n
        self.active_processes = self.num_processes
        self.rng = np.random.default_rng(seed)
        self.current_indices: Optional[List[int]] = None
        split = "train" if self.is_train else str(kwargs.pop("eval_split", "test"))
        direct_variations = kwargs.pop("variations", None)
        if direct_variations is None:
            manifest_path = kwargs.pop("variation_manifest", None)
            if not manifest_path:
                raise ValueError("scienceworld.variation_manifest must be configured.")
            self.variations = load_variation_manifest(str(manifest_path), split)
        else:
            self.variations = _normalize_variations(direct_variations)
        self.variation_to_index = {
            variation: index for index, variation in enumerate(self.variations)
        }

        worker_options = resources_per_worker or {"num_cpus": 0.1}
        remote_worker = ray.remote(**worker_options)(ScienceWorldWorker)
        self.workers = [
            remote_worker.remote(self.variations, kwargs)
            for _ in range(self.num_processes)
        ]

    def _resolve_variation_index(self, value) -> int:
        if isinstance(value, dict):
            if value.get("task_index") is None or value.get("variation_id") is None:
                raise ValueError(
                    "ScienceWorld task identity requires task_index and variation_id."
                )
            value = (value["task_index"], value["variation_id"])
        if isinstance(value, (tuple, list)):
            if len(value) != 2:
                raise ValueError("ScienceWorld task identity must contain exactly two values.")
            variation = (int(value[0]), int(value[1]))
            mapping = getattr(self, "variation_to_index", None)
            if mapping is None:
                mapping = {
                    item: index for index, item in enumerate(self.variations)
                }
            if variation not in mapping:
                raise KeyError(
                    f"ScienceWorld task identity {variation} is not in the selected split."
                )
            return int(mapping[variation])
        return int(value)

    def _normalize_indices(self, indices: Optional[Iterable[int]]) -> List[int]:
        if indices is None:
            if self.env_num > len(self.variations):
                raise ValueError(
                    f"Requested {self.env_num} unique ScienceWorld tasks from a pool of "
                    f"{len(self.variations)}."
                )
            unique = self.rng.choice(len(self.variations), size=self.env_num, replace=False)
            return np.repeat(unique, self.group_n).astype(np.int64).tolist()

        normalized = [self._resolve_variation_index(index) for index in indices]
        if len(normalized) == self.env_num:
            normalized = np.repeat(normalized, self.group_n).tolist()
        valid_partial_eval = (
            not self.is_train
            and self.group_n == 1
            and 0 < len(normalized) <= self.num_processes
        )
        if len(normalized) != self.num_processes and not valid_partial_eval:
            raise ValueError(
                f"Expected {self.env_num} variation indices or {self.num_processes} "
                f"per-worker indices, got {len(normalized)}."
            )
        if any(index < 0 or index >= len(self.variations) for index in normalized):
            raise IndexError("At least one ScienceWorld variation index is outside the split pool.")
        return normalized

    def _collect_reset(self, indices: Sequence[int]):
        results = ray.get(
            [worker.reset.remote(index) for worker, index in zip(self.workers, indices)]
        )
        observations, infos = zip(*results)
        return list(observations), list(infos)

    def reset(self, seeds=None, variation_indices=None):


        if seeds is not None and variation_indices is not None:
            raise ValueError("Pass either seeds or variation_indices, not both.")
        indices = self._normalize_indices(
            variation_indices if variation_indices is not None else seeds
        )
        self.current_indices = list(indices)
        self.active_processes = len(indices)
        return self._collect_reset(indices)

    def reset_to_current(self):
        if self.current_indices is None:
            raise RuntimeError("reset_to_current() called before reset().")
        return self._collect_reset(self.current_indices)

    def step(self, actions: Sequence[str]):
        if len(actions) != self.active_processes:
            raise ValueError(
                f"Expected {self.active_processes} actions for active workers, got {len(actions)}."
            )
        results = ray.get(
            [
                worker.step.remote(action)
                for worker, action in zip(
                    self.workers[: self.active_processes], actions
                )
            ]
        )
        observations, rewards, dones, infos = zip(*results)
        return list(observations), list(rewards), list(dones), list(infos)

    def render(self, mode=None, env_idx=None):
        if env_idx is not None:
            return ray.get(self.workers[env_idx].render.remote())
        return ray.get(
            [worker.render.remote() for worker in self.workers[: self.active_processes]]
        )

    def close(self):
        if ray is None:
            return
        workers = getattr(self, "workers", [])
        for worker in workers:
            try:
                ray.get(worker.close.remote())
            except Exception:


                pass
            finally:
                ray.kill(worker)
        self.workers = []

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def build_scienceworld_envs(
    seed=0,
    env_num=1,
    group_n=1,
    resources_per_worker=None,
    is_train=True,
    env_kwargs=None,
):
    return ScienceWorldMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )
