import logging
import os
import random
import time

import numpy as np
import tensorflow as tf
from stable_baselines import DQN
from stable_baselines.deepq.policies import CnnPolicy

from sonic_util import make_env

# Setup logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

random.seed(time.time())

best_mean_reward = -np.inf
TENSORBOARD_LOG_DIR = './tb_log'
TOTAL_TIMESTEPS = 10000
STEP_LOGGING_FREQ = 1000
SAVING_FREQ = 1000
REPLAY_BUFFER_SIZE = 1e6

saved_model_filename = "sonic_stable_dqn.zip"
env = make_env(act=random.randint(1, 3))
model = DQN(CnnPolicy, env, verbose=1, tensorboard_log=TENSORBOARD_LOG_DIR, buffer_size=REPLAY_BUFFER_SIZE)


def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global best_mean_reward, env

    self_ = _locals['self']

    # Log every step_logging_freq
    if self_.num_timesteps % STEP_LOGGING_FREQ == 0:
        logging.info("At n steps: " + str(self_.num_timesteps))

    # Save every step_logging_freq
    if self_.num_timesteps % SAVING_FREQ == 0:
        logging.info("Saving model at n steps: " + str(self_.num_timesteps))
        model.save(saved_model_filename)

    # Log scalar values
    if 'info' in _locals.keys():
        for key, value in _locals['info'].items():
            summary = tf.Summary(value=[tf.Summary.Value(tag="info/" + key, simple_value=value)])
            _locals['writer'].add_summary(summary, self_.num_timesteps)

    if 'done' in _locals.keys() and _locals['done']:
        env.load_state("GreenHillZone.Act" + str(random.randint(1, 3)))

    # Returning False will stop training early
    return True


def main():
    # Learn from previous run
    if os.path.isfile(saved_model_filename):
        logging.info("Loading model from file: " + saved_model_filename)
        model.load(saved_model_filename, env=env)
    else:
        logging.info("Creating model from scratch...")

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    model.save(saved_model_filename)

    obs = env.reset()


if __name__ == '__main__':
    main()
