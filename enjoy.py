import gym
import retro

from baselines import deepq
from sonic_util import make_env

def main():
    env = make_env(stack=True, scale_rew=True)
    act = deepq.learn(env, network='mlp', total_timesteps=0, load_path="dqn_sonic.pkl")

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()