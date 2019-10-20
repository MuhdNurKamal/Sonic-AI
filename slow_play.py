import sys
from os.path import isfile
from sonic_util import make_env
from stable_baselines import DQN
import logging
import time

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

    fps = 60
    frames_per_timestep = 4
    speed_up_factor = 1.5
    wait_time = frames_per_timestep / fps / speed_up_factor
    while True:
        t1 = time.time()

        action, _states = model.predict(obs)

        t2 = time.time()
        
        t3 = wait_time - (t2 - t1)

        if t3 > 0:
            time.sleep(t3)
        
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
