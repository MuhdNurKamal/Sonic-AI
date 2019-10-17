import os
from sonic_util import make_env
from stable_baselines.deepq.policies import CnnPolicy
from stable_baselines import DQN
import numpy as np
import tensorflow as tf

best_mean_reward, n_steps = -np.inf, 0
tensorboard_log = './tb_log'
total_timesteps = 5000


def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global best_mean_reward

    self_ = _locals['self']
    # Print stats every 1000 calls
    if self_.num_timesteps % 1000 == 0:
        print("n steps: ", self_.num_timesteps)

    # Log scalar values
    if 'info' in _locals.keys():
        for key, value in _locals['info'].items():
            summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
            _locals['writer'].add_summary(summary, self_.num_timesteps)

    # Returning False will stop training early
    return True


def main():
    env = make_env()

    model = DQN(CnnPolicy, env, verbose=1, tensorboard_log=tensorboard_log)

    saved_model_filename = "sonic_stable_dqn"

    # Learn from previous run
    if os.path.isfile(saved_model_filename):
        model.load(saved_model_filename)

    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save("sonic_stable_dqn")

    obs = env.reset()


if __name__ == '__main__':
    main()
