import datetime

from env.env import Env


class BruteForceAgent:

    # Print results every 500 step
    PRINT_FREQUENCY = 500

    # This const variable specify max length of combinations
    COMBINATION_MIN_LENGTH = 3
    COMBINATION_MAX_LENGTH = 23

    # Maximum number of permutations that we can generate randomly for large combinations
    MAX_COUNT_OF_RANDOM_PERMUTATIONS = 50000

    @staticmethod
    def run_permutation(env_params, chunk, print_interval, results):
        """ Run permutation """

        # Create env
        smt_file_name = env_params[0]
        env_actions = env_params[1]
        number_of_episodes = env_params[2]
        verbose = env_params[3]
        env = Env(smt_file_name, env_actions, number_of_episodes, verbose=verbose)

        counter = 1
        permutations_size = len(chunk)
        total_rewards = list()
        for permutation in chunk:
            env.reset()
            total_reward = 0
            for action in permutation:

                # Call in try..catch to prevent process stop problem in HPC
                try:

                    # Find index of action in env actions
                    index = env_actions.index(action)

                    # Take action and extract results
                    next_state, reward, done, _, _ = env.step(index)

                    # Update reward
                    total_reward += reward

                except Exception as e:
                    print(str(e))

            total_rewards.append(total_reward)

            counter += 1

            if counter % print_interval == 0 or counter == len(chunk):
                print(
                    "{}: Brute Force Permutation {}-{}, max reward is {} in prem size {}, total checked perms is {}/{}"
                        .format(datetime.datetime.now(),
                                abs(counter - print_interval) + 1, counter,
                                max(total_rewards[-print_interval:]),
                                len(permutation), counter,
                                permutations_size))

        results[smt_file_name]["max_rewards"].append(total_rewards)
