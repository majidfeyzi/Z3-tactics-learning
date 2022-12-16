import datetime

import numpy as np


class RandomAgent:

    # Print results every 100 step
    PRINT_FREQUENCY = 100

    @staticmethod
    def random_search(env, episodes):
        """ Random search strategy implementation """

        total_rewards = list()
        mean_rewards = list()
        for episode in range(1, episodes + 1):
            step_counter = 0
            state = env.reset()
            done = False
            total_reward = 0
            while not done:

                # Get and keep env steps count for print
                step_counter = env.step_counter

                # Sample random actions
                action = env.action_space.sample()
                # Take action and extract results
                next_state, reward, done, _, _ = env.step(action)
                # Update reward
                total_reward += reward
                if done:
                    break

            # Add total_reward to the total_rewards and compute last 50 episode mean
            total_rewards.append(total_reward)
            mean_score = np.mean(total_rewards[-100:])  # Get average of last 100 rewards
            mean_rewards.append(mean_score)

            if episode % RandomAgent.PRINT_FREQUENCY == 0:
                print("{}: Random Agent: Episode {}-{}: Mean score = {} in {} episodes"
                      .format(datetime.datetime.now(),
                              abs(episode - RandomAgent.PRINT_FREQUENCY) + 1, episode, mean_score,
                              RandomAgent.PRINT_FREQUENCY))

        return mean_rewards
