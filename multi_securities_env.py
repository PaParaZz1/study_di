from typing import Any, Union, List
import gym
import numpy as np
import pandas as pd
from easydict import EasyDict

from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray, to_list
from ding.utils import ENV_REGISTRY
from .multi_options_train_env import MultiOptionsTrainingEnv
from multi_env.securities_multi import OptionsMulti


@ENV_REGISTRY.register('multi_securities')
class MultiSecurities(BaseEnv):

    pass

    def __init__(self, cfg: EasyDict = None):
        self._cfg = cfg
        self._final_eval_reward = 0
        self._init_flag = False
        self.n_agent = 4
        self._observation_space = _observation_space = gym.spaces.Dict(
            {
                'agent_state': gym.spaces.Box(
                    low=-1, high=1, shape=(self.n_agent, cfg.agent_obs_shape), dtype=np.float32
                ),
                'global_state': gym.spaces.Box(
                    low=-1, high=1, shape=(self.n_agent, cfg.global_obs_shape), dtype=np.float32
                )
            }
        )
        self._action_space = _action_space = gym.spaces.Dict(
            {f'agent_{x}': gym.spaces.Box(low=-1, high=1, shape=(cfg.action_shape,), dtype=np.float32) for x in
             range(self.n_agent)}
        )
        self._reward_space = _reward_space = gym.spaces.Box(-float('inf'), +float('inf'), shape=(1,), dtype=np.float32)
        return

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        pass

    # def get_agent_obs(self):
    #     _ = self
    #     obs = np.random.uniform(low=-1, high=1, size=(4, 198)).astype(np.float32)
    #     return obs

    # def get_global_obs(self):
    #     _ = self
    #     obs_one = np.random.uniform(-1, 1, (424,)).astype(np.float32)
    #     obs = np.array([obs_one for _ in range(self.n_agent)])
    #     return obs

    def reset(self):
        if not self._init_flag:
            self._env = OptionsMulti(env_args=self._cfg)
            self._init_flag = True
        obs = self._env.reset()
        return obs

    def step(self, action) -> BaseEnvTimestep:
        _ = action
        action = to_ndarray(action)
        obs, rew, done, info = self._env.step(action)
        rew = to_ndarray([rew])
        if done:
            info['final_eval_reward'] = self._final_eval_reward
        return BaseEnvTimestep(obs, rew, done, info)

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def random_action(self) -> np.ndarray:
        random_action = self.action_space.sample()
        random_action = to_ndarray([random_action], dtype=np.float32)
        return random_action

    @property
    def num_agents(self) -> Any:
        return self._num_agents

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    def __repr__(self) -> str:
        return "DI-engine Multi Securities Env({})".format(self._cfg.env_id)


if __name__ == '__main__':
    env = MultiSecurities()
    action_shape = env.action_space.shape
    obs_shape = env.observation_space.shape
    _state = env.reset()
    _ = _state
    _action = env.action_space.sample()
    _state, _reward, _done, _info = env.step(_action)
    _ = _state
