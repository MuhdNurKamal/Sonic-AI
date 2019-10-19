import gym
import retro

from baselines import deepq
from sonic_util import make_env
from stable_baselines import DQN
from stable_baselines.deepq.policies import CnnPolicy
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def main():
    env = make_env()
    saved_model_filename = "sonic_stable_dqn"
    max_screen_x = 0

    model = DQN.load(saved_model_filename)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

        if info['screen_x'] > max_screen_x:
            max_screen_x = info['screen_x']
            logger.info("Max screen x: " + str(max_screen_x))
        if done:
            env.reset()
        else:
            env.render()


if __name__ == '__main__':
    main()
