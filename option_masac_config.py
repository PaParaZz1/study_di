from torch import nn
from easydict import EasyDict

collector_env_num = 1
evaluator_env_num = 1
n_agent = 4

default_config = dict(
    exp_name='save/multi_securities',
    env=dict(
        agent_obsk=4,
        add_agent_id=False,
        episode_limit=1000,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
    ),
    policy=dict(
        cuda=True,
        random_collect_size=1000,
        multi_agent=True,
        model=dict(
            agent_obs_shape=198,
            global_obs_shape=424,
            action_shape=4,
            twin_critic=True,
            action_space='reparameterization',
            actor_head_hidden_size=512,
            actor_head_layer_num=3,
            critic_head_hidden_size=512,
            critic_head_layer_num=3,
            activation=nn.Mish(),
        ),
        learn=dict(
            update_per_collect=1,
            batch_size=256,
            learning_rate_q=1e-4,
            learning_rate_policy=1e-4,
            learning_rate_alpha=3e-5,
            ignore_done=False,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=1,
            reparameterization=True,
            auto_alpha=True,
            log_space=True,
        ),
        collect=dict(
            n_sample=20,
            unroll_len=1,
        ),
        command=dict(),
        eval=dict(evaluator=dict(eval_freq=100, )),
        other=dict(replay_buffer=dict(replay_buffer_size=1_000_000, ), ),
    ),
)

main_config = EasyDict(default_config)

create_config = dict(
    env=dict(
        type='multi_securities',
        import_names=['MultiEnv.multi_securities_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='sac',
        import_names=['ding.policy.sac'],
    ),
    replay_buffer=dict(type='naive', ),
)
create_config = EasyDict(create_config)

if __name__ == '__main__':
    # or you can enter `ding -m serial -c ant_masac_config.py -s 0`
    from ding.entry.serial_entry import serial_pipeline

    serial_pipeline((main_config, create_config), seed=0)
