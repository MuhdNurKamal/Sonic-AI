import os
from sonic_util import make_env
from stable_baselines.deepq.policies import CnnPolicy
from stable_baselines import DQN
import numpy as np

best_mean_reward, n_steps = -np.inf, 0
tensorboard_log = './tb_log'
total_timesteps = 5000


def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if n_steps % 1000 == 0:
        print("n steps: ", n_steps)

    n_steps += 1

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
