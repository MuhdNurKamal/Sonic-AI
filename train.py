from os.path import isfile

from sonic_util import make_env
from stable_baselines.deepq.policies import CnnPolicy
from stable_baselines import DQN
import numpy as np
import tensorflow as tf
import logging

TENSORBOARD_LOG_DIR = './tb_log'
TOTAL_TIMESTEPS = 5000
STEP_LOGGING_FREQ = 1000
SAVING_FREQ = 1000
REPLAY_BUFFER_SIZE = 1e6
SAVED_MODEL_FILENAME = "sonic_stable_dqn.zip"


def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    self_ = _locals['self']

    # Log every step_logging_freq
    if self_.num_timesteps % STEP_LOGGING_FREQ == 0:
        logging.info("At n steps: " + str(self_.num_timesteps))

    # Save every step_logging_freq
    if self_.num_timesteps % SAVING_FREQ == 0:
        logging.info("Saving model at n steps: " + str(self_.num_timesteps))
        model.save(SAVED_MODEL_FILENAME)

    # Log scalar values
    if 'info' in _locals.keys():
        for key, value in _locals['info'].items():
            summary = tf.Summary(value=[tf.Summary.Value(tag="info/" + key, simple_value=value)])
            _locals['writer'].add_summary(summary, self_.num_timesteps)

    # Returning False will stop training early
    return True


def main():
    global model
    # Learn from previous run
    if isfile(SAVED_MODEL_FILENAME):
        logging.info("Loading model from file: " + SAVED_MODEL_FILENAME)
        model = DQN.load(SAVED_MODEL_FILENAME,
                         env=env,
                         verbose=0,
                         tensorboard_log=TENSORBOARD_LOG_DIR,
                         buffer_size=REPLAY_BUFFER_SIZE)
    else:
        logging.info("Creating model from scratch...")
        model = DQN(CnnPolicy,
                    env,
                    verbose=0,
                    tensorboard_log=TENSORBOARD_LOG_DIR,
                    buffer_size=REPLAY_BUFFER_SIZE)

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    model.save(SAVED_MODEL_FILENAME)

    obs = env.reset()


if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)

    env = make_env()
    model = None
    main()
