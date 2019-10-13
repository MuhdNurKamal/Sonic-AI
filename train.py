import gym
import retro
import os
from baselines import deepq
from sonic_util import make_env
from stable_baselines.deepq.policies import CnnPolicy
from stable_baselines import DQN


def main():
    env = make_env()

    model = DQN(CnnPolicy, env, verbose=1)
    saved_model_filename = "sonic_stable_dqn"

    # Learn from previous run
    if os.path.isfile(saved_model_filename):
        model.load(saved_model_filename)

    model.learn(total_timesteps=25000)
    model.save("sonic_stable_dqn")

    obs = env.reset()


if __name__ == '__main__':
    main()
