import gym
import retro

from baselines import deepq
from sonic_util import make_env
from stable_baselines import DQN
from stable_baselines.deepq.policies import CnnPolicy


def main():
    env = make_env(stack=True, scale_rew=True)
    saved_model_filename = "sonic_stable_dqn"

    model = DQN(CnnPolicy, env, verbose=1)
    model.load(saved_model_filename)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()


if __name__ == '__main__':
    main()
