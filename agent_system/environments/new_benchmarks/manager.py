"""Native environments; grouped resets and explicit task metadata for every trainer."""
from collections import defaultdict
import numpy as np

from agent_system.environments.base import EnvironmentManagerBase
from agent_system.memory import SimpleMemory


class NewBenchmarkManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config, kind):
        super().__init__(envs, projection_f, config)
        self.kind = kind
        self.memory = SimpleMemory()

    def reset(self, kwargs=None):
        identities = None
        if kwargs is not None:
            identities = list(kwargs)
            identities = [dict(x) for x in identities]
        if identities:
            if self.kind == 'minihack_room':
                obs, infos = self.envs.reset(seeds=[int(x['seed']) for x in identities])
            else:
                pairs = [(int(x['task_index']), int(x['variation_id'])) for x in identities]
                obs, infos = self.envs.reset(variation_indices=pairs)
        else:
            raise ValueError('New benchmarks require explicit env_kwargs from prepared data')
        self.memory.reset(batch_size=len(obs))
        self.last_obs = list(obs)
        self.task_types = ['room_navigation' if self.kind == 'minihack_room' else i['task_name'] for i in infos]
        self.goals = ['Move the player (@) onto the downstairs tile (>).' if self.kind == 'minihack_room'
                      else i['task_description'] for i in infos]
        self.last_infos = list(infos)
        return self._observations(obs), self._infos(infos, [True] * len(obs))

    def _infos(self, infos, valid):
        return [{**info, 'task_type': self.task_types[i], 'benchmark': self.kind,
                 'is_action_valid': bool(valid[i])} for i, info in enumerate(infos)]

    def _observations(self, obs):
        texts = []
        history_length = int(self.config.env.history_length)
        for i, current in enumerate(obs):
            history = self.memory[i][-history_length:] if history_length else []
            rendered = '\n'.join(f"Observation: {h['text_obs']}\nAction: {h['action']}" for h in history)
            if self.kind == 'minihack_room':
                instructions = ('Map: @ player, > downstairs goal, . floor, | and - walls, ^ visible trap, '
                                'letters usually monsters. Moving into a monster attacks it. '
                                'Use only north, east, south, west, northeast, southeast, southwest, northwest. '
                                'Return concise <reason>...</reason> and exactly one <answer>direction</answer>.')
            else:
                instructions = ('Complete the scientific task in causal order. A parseable command may still '
                                'be ineffective or harmful. Copy exactly one simulator command. '
                                'Return concise <reason>...</reason> and exactly one <action>command</action>.')
            texts.append(f'{instructions}\nTask: {self.goals[i]}\n'
                         f'Steps taken: {len(self.memory[i])}\nRecent history:\n{rendered}\n'
                         f'Current observation:\n{current}')
        # Task type is public environment metadata, never an answer or variation outcome.
        return {'text': texts, 'image': None, 'anchor': list(obs),
                'task_type': list(self.task_types), 'benchmark': [self.kind] * len(obs)}

    def step(self, text_actions):
        actions, valid = self.projection_f(text_actions)
        if self.kind == 'scienceworld':
            # Validate against the FULL candidate set, not the truncated prompt list.
            for i, action in enumerate(actions):
                candidates = self.last_infos[i].get('admissible_commands') or []
                lookup = {' '.join(c.split()).casefold(): c for c in candidates}
                canonical = lookup.get(' '.join(action.split()).casefold()) if candidates else action
                valid[i] = bool(valid[i] and canonical)
                actions[i] = canonical if valid[i] else ''
        obs, rewards, dones, infos = self.envs.step(actions)
        history_actions = actions
        if self.kind == 'minihack_room':
            names = ['north', 'east', 'south', 'west', 'northeast', 'southeast', 'southwest', 'northwest']
            history_actions = [names[a] if 0 <= a < len(names) else 'invalid (no-op)' for a in actions]
        self.memory.store({'text_obs': self.last_obs, 'action': history_actions})
        self.last_obs, self.last_infos = list(obs), list(infos)
        return self._observations(obs), np.asarray(rewards, dtype=np.float32), np.asarray(dones, dtype=bool), self._infos(infos, valid)

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for t in reversed(range(len(total_batch_list[batch_idx]))):
            if total_batch_list[batch_idx][t]['active_masks']:
                info = total_infos[batch_idx][t]
                success['success_rate'].append(float(info['won']))
                if self.kind == 'scienceworld':
                    success['native_score'].append(float(info['score']))
                return


def make_new_benchmark_envs(config):
    from omegaconf import OmegaConf
    kind = 'minihack_room' if 'minihack' in config.env.env_name.lower() else 'scienceworld'
    resources = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)
    groups = max(1, int(config.env.rollout.n))
    managers = []
    for train, batch, n in [(True, config.data.train_batch_size, groups),
                           (False, config.data.val_batch_size, 1)]:
        if kind == 'minihack_room':
            from agent_system.environments.env_package.minihack_room import build_minihack_room_envs, minihack_room_projection
            envs = build_minihack_room_envs(env_id='MiniHack-Room-Ultimate-5x5-v0', seed=config.env.seed,
                       env_num=batch, group_n=n, is_train=train, resources_per_worker=resources,
                       env_kwargs={'max_episode_steps': int(config.env.max_steps)})
            projection = minihack_room_projection
        else:
            from agent_system.environments.env_package.scienceworld import build_scienceworld_envs, scienceworld_projection
            settings = OmegaConf.to_container(config.env.scienceworld, resolve=True)
            settings.update(env_step_limit=int(config.env.max_steps))
            envs = build_scienceworld_envs(seed=config.env.seed, env_num=batch, group_n=n,
                       is_train=train, resources_per_worker=resources, env_kwargs=settings)
            projection = scienceworld_projection
        managers.append(NewBenchmarkManager(envs, projection, config, kind))
    return tuple(managers)
