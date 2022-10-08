import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os
import sys
from easydict import EasyDict
import platform
import warnings

import gym
from gym.spaces import Box
from gym.wrappers import TimeLimit

from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray, to_list


if True:
    # from myclasses.objs import LocalAccount
    # import myclasses.factor.myfactor as f
    # from myclasses.common.Securities import Securities
    # from myclasses.qqmail.qqemail import SendEmail
    #
    # from myclasses.Securities_trading.CFFEX_trading.IOTradingEnv2 import IOTradingEnv2 as Env
    from .multiagentenv import MultiAgentEnv


class OptionsMulti(MultiAgentEnv, ):
    now: list = None
    now_str: list = None
    name = 'OptionsMultiEnv'

    def __init__(self, batch_size=None, **kwargs):
        super().__init__(batch_size, **kwargs)
        self._cfg = kwargs['env_args']
        if 'share_memory' not in self._cfg.__class__.__dict__.keys():
            self._cfg.__class__.share_memory = {}
        self.share_memory = self._cfg.share_memory
        self.start_time = _start_time = self._cfg.start_time
        self.debug_flag = _debug_flag = self._cfg.debug_flag
        self.add_agent_id = _add_agent_id = kwargs["env_args"]["add_agent_id"]
        self.n_agents = _n_agents = 4
        self.n_actions = _n_actions = 3
        self.agent_obs_shape = _agent_obs_shape = self._cfg.agent_obs_shape
        self.global_obs_shape = _global_obs_shape = self._cfg.global_obs_shape
        self._observation_space = _observation_space = gym.spaces.Dict(
            {
                'agent_state': gym.spaces.Box(
                    low=-1, high=1, shape=(self.n_agents, self.agent_obs_shape), dtype=np.float32
                ),
                'global_state': gym.spaces.Box(
                    low=-1, high=1, shape=(self.n_agents, self.global_obs_shape), dtype=np.float32
                )
            }
        )
        self._action_space = _action_space = gym.spaces.Dict(
            {f'agent_{x}': gym.spaces.Box(low=-1, high=1, shape=(self.n_actions,), dtype=np.float32) for x in
             range(self.n_agents)}
        )
        self._reward_space = _reward_space = gym.spaces.Box(-float('inf'), +float('inf'), shape=(1,), dtype=np.float32)
        self._init_flag = False
        self._final_eval_reward = 0.
        return

    def get_agent_obs(self):
        obs = np.random.uniform(-1, 1, (self.n_agents, self.agent_obs_shape)).astype(np.float32)
        return obs

    def get_global_obs(self):
        obs_one = np.random.uniform(-1, 1, (self.global_obs_shape,)).astype(np.float32)
        obs = np.array([obs_one for _ in range(self.n_agents)], dtype=np.float32)
        return obs

    def reset(self):
        self._final_eval_reward = 0.

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

    def close(self):
        pass
