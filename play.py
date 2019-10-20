import sys
from os.path import isfile
from sonic_util import make_env
from stable_baselines import DQN
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def main():
    env = make_env()
    max_screen_x = 0

    model = DQN.load(saved_model_file_path)

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
    if len(sys.argv) != 2:
        logger.warning("Usage: python train.py saved_model_file_path")
        exit()

    saved_model_file_path = sys.argv[1]

    if isfile(saved_model_file_path):
        logger.info("Loading from " + saved_model_file_path)
    else:
        logger.warning("No such file " + saved_model_file_path)
        exit()

    main()
