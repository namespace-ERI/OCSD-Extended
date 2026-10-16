
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np

try:
    import ray
except ImportError:
    ray = None


def _decode_message(value: Any) -> str:
    if value is None:
        return ""
    array = np.asarray(value, dtype=np.uint8)
    return array.tobytes().split(b"\x00", 1)[0].decode("utf-8", errors="replace").strip()


def _decode_screen_description(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace").strip("\x00 ")
    array = np.asarray(value)
    if array.dtype == np.uint8:
        return array.tobytes().split(b"\x00", 1)[0].decode(
            "utf-8", errors="replace"
        ).strip()
    return str(value).strip()


def _render_grid(grid: Any) -> str:
    rows = []
    for row in np.asarray(grid, dtype=np.uint8):
        text = row.tobytes().decode("latin-1", errors="replace").replace("\x00", "").rstrip()
        rows.append(text)
    while rows and not rows[-1].strip():
        rows.pop()
    return "\n".join(rows)


def _symbol_legend(observation: Dict[str, Any]) -> Dict[str, list[str]]:
    descriptions = observation.get("screen_descriptions")
    if descriptions is None:
        return {}
    descriptions = np.asarray(descriptions)
    if descriptions.ndim < 2:
        return {}

    grid = None
    for key in ("chars_crop", "chars"):
        candidate = observation.get(key)
        if candidate is not None and np.asarray(candidate).shape == descriptions.shape[:2]:
            grid = np.asarray(candidate)
            break
    if grid is None:
        return {}

    legend: Dict[str, set[str]] = {}
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            value = int(grid[row, column])
            if value in (0, 32):
                continue
            description = _decode_screen_description(descriptions[row, column])
            if description:
                legend.setdefault(chr(value), set()).add(description)
    return {symbol: sorted(values) for symbol, values in legend.items()}


def observation_to_text(observation: Dict[str, Any]) -> str:
    pass
    grid = observation.get("chars_crop")
    if grid is None:
        grid = observation.get("chars")
    if grid is None:
        raise KeyError("MiniHack observation has neither chars_crop nor chars")

    parts = []
    message = _decode_message(observation.get("message"))
    if message:
        parts.append(f"Message: {message}")
    parts.extend(("Dungeon view:", _render_grid(grid)))

    legend = _symbol_legend(observation)
    if legend:
        parts.append(
            "The meaning of common symbols in the dungeon view is "
            "(the same symbol may have different descriptions):"
        )
        parts.extend(
            f"{symbol}: {', '.join(descriptions)}"
            for symbol, descriptions in sorted(legend.items())
        )
    return "\n".join(parts)


def build_minihack_retrieval_key(observation_text: str) -> str:
    pass







    text = str(observation_text or "")
    marker = "Dungeon view:"
    if marker in text:
        grid_text = text.split(marker, 1)[1]
    else:
        grid_text = text
    legend_marker = "The meaning of common symbols in the dungeon view is"
    if legend_marker in grid_text:
        grid_text = grid_text.split(legend_marker, 1)[0]

    rows = [line.rstrip() for line in grid_text.splitlines()]
    while rows and not rows[0].strip():
        rows.pop(0)
    while rows and not rows[-1].strip():
        rows.pop()
    if not rows:
        return "MiniHack room layout; terrain=unknown"

    occupied_columns = [
        column
        for row in rows
        for column, symbol in enumerate(row)
        if not symbol.isspace()
    ]
    if not occupied_columns:
        return "MiniHack room layout; terrain=unknown"
    left = min(occupied_columns)
    right = max(occupied_columns)
    cropped = [row[left:right + 1].ljust(right - left + 1) for row in rows]

    player = None
    goals = []
    traps = []
    monsters = []
    terrain_rows = []
    for row_index, row in enumerate(cropped):
        encoded = []
        for column_index, symbol in enumerate(row):
            position = (row_index, column_index)
            if symbol == "@":
                player = position
                encoded.append("P")
            elif symbol == ">":
                goals.append(position)
                encoded.append("G")
            elif symbol == "^":
                traps.append(position)
                encoded.append("T")
            elif symbol.isalpha():
                monsters.append(position)
                encoded.append("M")
            elif symbol in "|-":
                encoded.append("#")
            elif symbol == " ":
                encoded.append("_")
            elif symbol.isprintable():
                encoded.append(symbol)
            else:
                encoded.append("?")
        terrain_rows.append("".join(encoded))

    def relative_feature(position):
        if player is None:
            return "relative=unknown"
        row_delta = position[0] - player[0]
        column_delta = position[1] - player[1]
        vertical = "north" if row_delta < 0 else "south" if row_delta > 0 else ""
        horizontal = "west" if column_delta < 0 else "east" if column_delta > 0 else ""
        direction = vertical + horizontal if vertical or horizontal else "same"
        return (
            f"direction={direction},manhattan={abs(row_delta) + abs(column_delta)},"
            f"chebyshev={max(abs(row_delta), abs(column_delta))},"
            f"offset={row_delta}:{column_delta}"
        )

    goal_features = sorted(relative_feature(position) for position in goals)
    trap_features = sorted(relative_feature(position) for position in traps)
    monster_features = sorted(relative_feature(position) for position in monsters)
    return "; ".join(
        (
            "MiniHack room layout",
            "goals=" + ("|".join(goal_features) if goal_features else "not_visible"),
            "traps=" + ("|".join(trap_features) if trap_features else "none_visible"),
            "monsters=" + ("|".join(monster_features) if monster_features else "none_visible"),
            "terrain=" + "/".join(terrain_rows),
        )
    )


class MiniHackRoomWorker:
    pass

    def __init__(self, env_id: str, env_kwargs: Dict[str, Any]):
        try:
            import gymnasium as gym
            import minihack
        except ImportError as exc:
            raise ImportError(
                "MiniHack Room requires `pip install --no-deps nle==1.3.0 "
                "minihack==1.0.2` "
                "inside a Linux environment."
            ) from exc

        self.env_id = env_id
        kwargs = dict(env_kwargs)
        kwargs.setdefault(
            "observation_keys",
            (
                "glyphs",
                "message",
                "chars_crop",
                "glyphs_crop",
                "screen_descriptions",
                "chars",
            ),
        )


        kwargs.setdefault("fix_moon_phase", True)
        self.env = gym.make(env_id, **kwargs)
        self.episode_seed = None
        self.last_observation = None
        self.last_text = ""
        self.last_position = None
        self.episode_done = False
        self.terminal_info = None

    @staticmethod
    def _position(observation):
        blstats = np.asarray(observation.get("blstats", []))
        if blstats.size < 2:
            return None
        return int(blstats[0]), int(blstats[1])

    def reset(self, seed_for_reset: int):
        self.episode_seed = int(seed_for_reset)
        self.episode_done = False
        self.terminal_info = None





        unwrapped = self.env.unwrapped
        if not hasattr(unwrapped, "seed"):
            raise RuntimeError(
                f"{self.env_id} does not expose NLE's deterministic seed API."
            )
        try:
            unwrapped.seed(
                self.episode_seed,
                self.episode_seed,
                reseed=False,
                lgen=self.episode_seed,
            )
        except TypeError:


            unwrapped.seed(self.episode_seed, self.episode_seed, reseed=False)

        try:
            result = self.env.reset(seed=self.episode_seed)
        except TypeError:
            result = self.env.reset()

        if isinstance(result, tuple) and len(result) == 2:
            observation, info = result
        else:
            observation, info = result, {}
        self.last_observation = observation
        self.last_position = self._position(observation)
        self.last_text = observation_to_text(observation)
        info = dict(info or {})
        info.update(
            {
                "won": False,
                "episode_seed": self.episode_seed,
                "task_id": f"{self.env_id}:{self.episode_seed}",
            }
        )
        return self.last_text, info

    def step(self, action: int):
        if self.episode_done:
            info = dict(self.terminal_info or {})
            info["action_is_effective"] = False
            info["terminal_step_replayed"] = True
            return self.last_text, 0.0, True, info

        if int(action) < 0:

            return self.last_text, 0.0, False, {
                "won": False,
                "action_is_effective": False,
                "episode_seed": self.episode_seed,
                "task_id": f"{self.env_id}:{self.episode_seed}",
            }

        result = self.env.step(int(action))
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            observation, reward, done, info = result
            terminated, truncated = bool(done), False

        position = self._position(observation)
        won = bool(float(reward) > 0.0)
        self.last_observation = observation
        self.last_text = observation_to_text(observation)
        info = dict(info or {})
        info.update(
            {
                "won": won,


                "action_is_effective": True,
                "episode_seed": self.episode_seed,
                "task_id": f"{self.env_id}:{self.episode_seed}",
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        self.last_position = position
        if done:
            self.episode_done = True
            self.terminal_info = dict(info)


        binary_reward = 1.0 if won else 0.0
        return self.last_text, binary_reward, bool(done), info

    def render(self):
        return self.last_text

    def close(self):
        self.env.close()


class MiniHackRoomMultiProcessEnv:
    pass

    mode = "text"

    def __init__(
        self,
        env_id: str,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        resources_per_worker: Optional[Dict[str, float]] = None,
        is_train: bool = True,
        env_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if ray is None:
            raise ImportError("MiniHack Room requires Ray. Install this repository's dependencies first.")
        if not ray.is_initialized():
            ray.init()

        self.env_id = env_id
        self.is_train = is_train
        self.group_n = int(group_n)
        self.env_num = int(env_num)
        self.num_processes = self.env_num * self.group_n
        self.active_num_processes = self.num_processes
        self.rng = np.random.default_rng(seed)
        self.current_seeds = None

        worker_options = resources_per_worker or {"num_cpus": 0.1}
        remote_worker = ray.remote(**worker_options)(MiniHackRoomWorker)
        worker_kwargs = dict(env_kwargs or {})
        self.workers = [
            remote_worker.remote(self.env_id, worker_kwargs)
            for _ in range(self.num_processes)
        ]

    def _normalize_seeds(self, seeds: Optional[Iterable[int]]):
        if seeds is None:
            low, high = (0, 2**30) if self.is_train else (2**30, 2**31 - 1)
            unique = self.rng.integers(low, high, size=self.env_num)
            return np.repeat(unique, self.group_n).astype(np.int64).tolist()

        normalized = [int(seed) for seed in seeds]
        if len(normalized) == self.env_num and self.group_n > 1:
            normalized = np.repeat(normalized, self.group_n).tolist()
        if (
            not normalized
            or len(normalized) > self.num_processes
            or len(normalized) % self.group_n != 0
        ):
            raise ValueError(
                f"Expected 1..{self.num_processes} per-worker seeds divisible by "
                f"group_n={self.group_n}, got {len(normalized)}."
            )
        return normalized

    def reset(self, seeds=None):
        seeds = self._normalize_seeds(seeds)
        self.active_num_processes = len(seeds)
        self.current_seeds = list(seeds)
        return self._collect_reset(seeds)

    def _collect_reset(self, seeds):
        active_count = getattr(self, "active_num_processes", len(seeds))
        results = ray.get(
            [
                worker.reset.remote(seed)
                for worker, seed in zip(self.workers[:active_count], seeds)
            ]
        )
        observations, infos = zip(*results)
        return list(observations), list(infos)

    def reset_to_current(self):
        if self.current_seeds is None:
            raise RuntimeError("reset_to_current() called before reset().")
        return self._collect_reset(self.current_seeds)

    def step(self, actions):
        if len(actions) != self.active_num_processes:
            raise ValueError(f"Expected {self.active_num_processes} actions, got {len(actions)}.")
        results = ray.get(
            [
                worker.step.remote(action)
                for worker, action in zip(self.workers[:self.active_num_processes], actions)
            ]
        )
        observations, rewards, dones, infos = zip(*results)
        return list(observations), list(rewards), list(dones), list(infos)

    def render(self, mode=None, env_idx=None):
        if env_idx is not None:
            return ray.get(self.workers[env_idx].render.remote())
        return ray.get(
            [worker.render.remote() for worker in self.workers[:self.active_num_processes]]
        )

    def close(self):
        if ray is None:
            return
        for worker in getattr(self, "workers", []):
            try:
                ray.get(worker.close.remote())
            finally:
                ray.kill(worker)
        self.workers = []

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def build_minihack_room_envs(
    env_id="MiniHack-Room-Ultimate-5x5-v0",
    seed=0,
    env_num=1,
    group_n=1,
    resources_per_worker=None,
    is_train=True,
    env_kwargs=None,
):
    return MiniHackRoomMultiProcessEnv(
        env_id=env_id,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )
