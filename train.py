from os.path import isfile

from sonic_util import make_env
from stable_baselines.deepq.policies import CnnPolicy
from stable_baselines import DQN
import numpy as np
import tensorflow as tf
import logging
from sys import argv

TENSORBOARD_LOG_DIR = './tb_log'
TOTAL_TIMESTEPS = 5000
STEP_LOGGING_FREQ = 1000
SAVING_FREQ = 1000
REPLAY_BUFFER_SIZE = 1e6
DEFAULT_SAVED_MODEL_FILENAME = "sonic_stable_dqn.zip"
DECORATOR = ("*" * 100 + "\n") * 5
saved_model_name = DEFAULT_SAVED_MODEL_FILENAME


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
        model.save(saved_model_name)

    # Log scalar values
    if 'info' in _locals.keys():
        for key, value in _locals['info'].items():
            summary = tf.Summary(value=[tf.Summary.Value(tag="info/" + key, simple_value=value)])
            _locals['writer'].add_summary(summary, self_.num_timesteps)

    # Returning False will stop training early
    return True


def main():
    global model
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    model.save(saved_model_name)

    obs = env.reset()


if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)

    env = make_env()
    model = None

    print(DECORATOR)
    if len(argv) == 1:
        if isfile(saved_model_name):
            logging.info("Loading model from file: " + saved_model_name)
            model = DQN.load(saved_model_name,
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
    elif len(argv) == 2:
        saved_model_name = argv[1]
        if isfile(saved_model_name) and saved_model_name.endswith(".zip"):
            logging.info("Loading model from file: " + saved_model_name)
            model = DQN.load(saved_model_name,
                             env=env,
                             verbose=0,
                             tensorboard_log=TENSORBOARD_LOG_DIR,
                             buffer_size=REPLAY_BUFFER_SIZE)
        else:
            logger.warning("Usage: \npython train.py \nOR\npython train.py model_to_load_from.zip")
            exit()
    else:
        logger.warning("Usage: \npython train.py \nOR\npython train.py model_to_load_from.zip")
        exit()

    main()
