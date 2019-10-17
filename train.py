import os
from sonic_util import make_env
from stable_baselines.deepq.policies import CnnPolicy
from stable_baselines import DQN
import numpy as np

best_mean_reward, n_steps = -np.inf, 0


def callback(_locals, _globals):
    global best_mean_reward, n_steps
    n_steps += 1
    if n_steps % 500 == 0:
        print("Total timesteps passed: ", n_steps)
    return True


def main():
    env = make_env(log_dir="./log/")

    model = DQN(CnnPolicy, env, verbose=1)
    saved_model_filename = "sonic_stable_dqn"

    # Learn from previous run
    if os.path.isfile(saved_model_filename):
        model.load(saved_model_filename)

    model.learn(total_timesteps=25000, callback=callback)
    model.save("sonic_stable_dqn")

    obs = env.reset()


if __name__ == '__main__':
    main()
