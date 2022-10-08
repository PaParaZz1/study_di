import gym
import numpy as np
import pandas as pd
from easydict import EasyDict

from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray, to_list
from ding.utils import ENV_REGISTRY


@ENV_REGISTRY.register('multi_options')
class MultiOptionsTrainingEnv(BaseEnv):
    pass

    def __init__(self, cfg: EasyDict = None):
        self._cfg = cfg
        self._final_eval_reward = 0
        self._init_flag = False
        self.n_agent = 4
        self.observation_space = _observation_space = gym.spaces.Dict(
            {
                'agent_state': gym.spaces.Box(
                    low=-1, high=1, shape=(self.n_agent, 198), dtype=np.float32
                ),
                'global_state': gym.spaces.Box(
                    low=-1, high=1, shape=(self.n_agent, 424), dtype=np.float32
                )
            }
        )
        self.action_space = _action_space = gym.spaces.Dict(
            {f'agent_{x}': gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32) for x in range(self.n_agent)}
        )
        self.reward_space = _reward_space = gym.spaces.Box(-float('inf'), +float('inf'), shape=(1,), dtype=np.float32)
        return

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        pass

    def get_agent_obs(self):
        _ = self
        obs = np.random.uniform(low=-1, high=1, size=(4, 198))
        return obs

    def get_global_obs(self):
        _ = self
        obs_one = np.random.uniform(-1, 1, (424,))
        obs = np.array([obs_one for _ in range(self.n_agent)])
        return obs

    def reset(self):
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

    # @property
    # def observation_space(self) -> gym.spaces.Space:
    #     return self._observation_space
    #
    # @property
    # def action_space(self) -> gym.spaces.Space:
    #     return self._action_space
    #
    # @property
    # def reward_space(self) -> gym.spaces.Space:
    #     return self._reward_space


if __name__ == '__main__':
    env = MultiOptionsTrainingEnv()
    action_shape = env.action_space.shape
    obs_shape = env.observation_space.shape
    _state = env.reset()
    _ = _state
    _action = env.action_space.sample()
    _state, _reward, _done, _info = env.step(_action)
    _ = _state
