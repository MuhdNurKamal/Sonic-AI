import gym
import retro
import os
from baselines import deepq
from sonic_util import make_env


def callback(lcl, _glb):
    # stop training if reward exceeds 10000
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 10000
    return is_solved


def main():
    env = make_env(stack=True, scale_rew=True)
    load_path = "dqn_sonic.pkl" if os.path.isfile("./dqn_sonic.pkl") else None
    if load_path:
        print("Loading from " + "dqn_sonic.pkl")
    act = deepq.learn(
        env,
        network='mlp',
        lr=1e-3,
        total_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback,
        load_path=load_path
    )
    print("Saving model to dqn_sonic.pkl")
    act.save("dqn_sonic.pkl")


if __name__ == '__main__':
    main()
