from functools import partial
import gym
from gym.spaces import Box
from gym.wrappers import TimeLimit
import numpy as np
from .multiagentenv import MultiAgentEnv
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray, to_list


class SecuritiesMulti(MultiAgentEnv):
    def __init__(self, batch_size=None, **kwargs):
        super().__init__(batch_size, **kwargs)
        self.n_agents = _n_agents = 4
        self.n_actions = _n_actions = 3

        self._observation_space = _observation_space = gym.spaces.Dict(
            {
                'agent_state': gym.spaces.Box(
                    low=-1, high=1, shape=(self.n_agents, 198), dtype=np.float32
                ),
                'global_state': gym.spaces.Box(
                    low=-1, high=1, shape=(self.n_agents, 424), dtype=np.float32
                )
            }
        )
        self._action_space = _action_space = gym.spaces.Dict(
            {f'agent_{x}': gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32) for x in range(self.n_agents)}
        )
        self._reward_space = _reward_space = gym.spaces.Box(-float('inf'), +float('inf'), shape=(1,), dtype=np.float32)
        self._init_flag = False
        self._final_eval_reward = 0.
        return

    def get_agent_obs(self):
        _ = self
        obs = np.random.uniform(low=-1, high=1, size=(4, 198))
        return obs

    def get_global_obs(self):
        _ = self
        obs_one = np.random.uniform(-1, 1, (424,))
        obs = np.array([obs_one for _ in range(self.n_agents)])
        return obs

    def reset(self):
        self._final_eval_reward = 0.
        if not self._init_flag:
            self._init_flag = True

        obs = {'agent_state': self.get_agent_obs(), 'global_state': self.get_global_obs()}
        return obs

    def step(self, action) -> BaseEnvTimestep:
        _ = action

        obs = {'agent_state': self.get_agent_obs(), 'global_state': self.get_global_obs()}
        obs = to_ndarray(obs)

        rew = 1.
        rew = to_ndarray([rew])  # Transformed to an array with shape (1, )
        self._final_eval_reward += rew
        done = False
        info = {}
        if done:
            info['final_eval_reward'] = self._final_eval_reward

        return BaseEnvTimestep(obs, rew, done, info)
