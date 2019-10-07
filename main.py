import math
import random
from itertools import count

import matplotlib
import matplotlib.pyplot as plt
import retro
import torch

from screen_utils import get_screen
from dqn_agent import DQNAgent

env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1').unwrapped
# set up matplotlib
matplotlib.use('TkAgg')
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using CUDA: ", torch.cuda.is_available())

episode_durations = []


def main():
    # Change agent here
    agent = DQNAgent(env, device)
    num_episodes = 50
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset()
        last_screen = get_screen(env, device)
        current_screen = get_screen(env, device)
        state = current_screen - last_screen
        for t in count():
            # Select and perform an action
            action = agent.select_action(state)
            _, reward, done, _ = env.step(action)

            env.render()

            reward = torch.tensor([reward], device=device)

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen(env, device)
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            agent.post_action_update(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            agent.optimize_model(state)
            if done:
                episode_durations.append(t + 1)
                # plot_durations()
                break

        agent.post_episode_update(i_episode=i_episode)

    print('Complete')
    env.close()
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
