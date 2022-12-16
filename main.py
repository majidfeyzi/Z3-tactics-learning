import json
import multiprocessing
import os
import shutil

import matplotlib.pyplot as plt
from IPython.display import clear_output

from brute_force_agent import BruteForceAgent
from dqn_agent import Agent
from env.env import Env
from env.utils.action_reader import ActionReader
from env.utils.files_handler import FilesHandler
from random_agent import RandomAgent


def get_learned_files_path(logic, dataset_name):
    """
    Return path of file that keep leaned files list for given logic and dataset
    :param logic logic name that is using in file name
    :param dataset_name data set name that is using in file name
    """
    return os.path.join(os.path.join(os.getcwd(), "shared"), "learned_files_" + logic + "_" + dataset_name + ".txt")


def read_learned_files_list(logic, dataset_name):
    """
    Return list of files that its learning done for given logic and dataset
    :param logic logic name that is using in file name
    :param dataset_name data set name that is using in file name
    """
    files = list()
    path = get_learned_files_path(logic, dataset_name)
    if os.path.isfile(path) and os.access(path, os.R_OK):
        with open(path, 'r') as file:
            for line in file:
                f_path = line[:-1]  # Remove linebreak which is the last character of the string
                files.append(f_path)  # Add item to the list
    return files


def write_learned_files_list(learned_files_list, logic, dataset_name):
    """
    Write and keep learned files list for given logic and dataset
    :param logic logic name that is using in file name
    :param dataset_name data set name that is using in file name
    """
    path = get_learned_files_path(logic, dataset_name)
    with open(path, 'w') as file:
        for item in learned_files_list:
            file.write('%s\n' % item)


def plot_result(dqn_agent_values, mean_steps, random_agent_values=[], max_reward=0, plot_title="", save_path=""):
    """ Plot the reward curve and histogram of results over time """

    # Remove first items because avg of them not computed
    if len(dqn_agent_values) > 100:
        dqn_agent_values = dqn_agent_values[-(len(dqn_agent_values) - 100):]
    if len(mean_steps) > 100:
        mean_steps = mean_steps[-(len(mean_steps) - 50):]
    if len(random_agent_values) > 100:
        random_agent_values = random_agent_values[-(len(random_agent_values) - 100):]

    # Update the window after each episode
    clear_output(wait=True)

    # Define the figure
    figure, axis = plt.subplots(nrows=2, ncols=1, figsize=(12, 5))
    figure.suptitle(plot_title)
    axis[0].set_ylabel("Rewards")
    axis[1].set_xlabel("Episodes")
    axis[1].set_ylabel("Steps")
    axis[0].plot(dqn_agent_values, c="blue", label="DQN agent rewards", linestyle="-")
    axis[1].plot(mean_steps, c="dodgerblue", label="DQN agent steps", linestyle="-.")
    if len(random_agent_values) > 0:
        axis[0].plot(random_agent_values, c="red", label="Random strategy rewards", linestyle=":")
    if max_reward > 0:
        axis[0].axhline(max_reward, c="green", label="Max reward", linestyle="--")

    # Save plot as file in given path
    if len(save_path) > 0:
        abs_path = os.path.abspath(save_path)
        figure.savefig(abs_path, format="svg")
    else:
        plt.show()

    # See https://stackoverflow.com/a/21884375
    plt.close(figure)


def plot_rlimits_and_probes(mean_rlimits, mean_probes, plot_title="", save_path=""):
    """ Plot the rlimits and probes that is using in reward computation. """

    # Update the window after each episode
    clear_output(wait=True)

    # Define the figure
    nrows = len(mean_probes[0]) + 1
    figure, axis = plt.subplots(nrows=nrows, ncols=1, figsize=(12, nrows * 5))
    figure.suptitle(plot_title)
    if nrows > 1:
        axis[0].plot(mean_rlimits, c="blue", label="RLimits", linestyle="-")
        axis[nrows - 1].set_xlabel("Episodes")
        axis[0].set_ylabel("RLimits")
        for i in range(0, nrows - 1):
            probes_column = [column[i] for column in mean_probes]
            axis[i + 1].plot(probes_column, c="dodgerblue", label="Probe[{}]".format(i + 1), linestyle="-")
    else:
        axis.plot(mean_rlimits, c="blue", label="RLimits", linestyle="-")
        axis.set_xlabel("Episodes")
        axis.set_ylabel("RLimits")

    # Save plot as file in given path
    if len(save_path) > 0:
        abs_path = os.path.abspath(save_path)
        figure.savefig(abs_path)
    else:
        plt.show()


def clear_dir(dir_path):
    """ Remove content of directory """
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def run_dqn_agent(smt_file_name, policy_file_name, actions, number_of_episodes, results):
    """ Run DQN agent and produce results """
    env = Env(smt_file_name, actions, number_of_episodes, verbose=False)
    state_size = env.observation_space.n
    action_size = env.action_space.n
    agent = Agent(state_size, action_size, policy_file_name)
    if os.path.exists("shared" + os.sep + policy_file_name):
        agent.load_weights()
    (mean_rewards, mean_steps, mean_rlimits, mean_probes) = agent.start_training(env, number_of_episodes, 0)
    results[smt_file_name]["mean_rewards"] = mean_rewards
    results[smt_file_name]["mean_steps"] = mean_steps
    results[smt_file_name]["mean_rlimits"] = mean_rlimits
    results[smt_file_name]["mean_probes"] = mean_probes
    results[smt_file_name]["agent"] = str(agent)


def get_chunks(ls, n):
    """ Yield successive n-sized chunks from lst """
    for i in range(0, len(ls), n):
        yield ls[i:i + n]


def run_brute_force_agent(smt_file_name, actions, number_of_episodes, results, process_manager):
    """
    Run brute force agent and max possible reward.
    Here we apply all actions permutations on env and get rewards.
    """

    results[smt_file_name]["max_rewards"] = process_manager.list()
    permutations = ActionReader().get_actions_permutations(actions,
                                                           BruteForceAgent.COMBINATION_MIN_LENGTH,
                                                           BruteForceAgent.COMBINATION_MAX_LENGTH,
                                                           BruteForceAgent.MAX_COUNT_OF_RANDOM_PERMUTATIONS)

    # Divide permutations into smaller pieces based on the number of processes
    chunk_size = int(len(permutations) / (number_of_processes - 1))
    chunks = list(get_chunks(permutations, chunk_size))

    # Create processes (one for each processor)
    processes = list()
    env_params = (smt_file_name, actions, number_of_episodes, False)
    for chunk in chunks:
        process = multiprocessing.Process(target=BruteForceAgent.run_permutation,
                                          args=(env_params, chunk, BruteForceAgent.PRINT_FREQUENCY, results))
        processes.append(process)

    return processes


def run_random_agent(smt_file_name, actions, number_of_episodes, results):
    """ Run random agent and produce results """
    env_random_agent = Env(smt_file_name, actions, number_of_episodes, verbose=False)
    results[smt_file_name]["random_rewards"] = RandomAgent().random_search(env_random_agent, number_of_episodes)


if __name__ == '__main__':

    number_of_processes = 4
    number_of_episodes = 5000
    logic = "qf_bv"
    dataset_name = "bruttomesso"
    smt_files_dir = "train" + os.sep + logic + os.sep + dataset_name
    policy_file_name = "policy_" + logic + "_" + dataset_name + ".h5"

    # Get type of running as input
    run_other_agents_input = input("Do you want just learning?\n"
                                   "If you want to run program just for learning so type 'y' and then press enter,\n"
                                   "else if you want to run program for learning and also running brute force agent"
                                   "and pure chance agent too, type 'n' and then press enter.\n")
    if run_other_agents_input.lower() != 'n' and run_other_agents_input.lower() != 'y':
        print("Invalid input!")
        exit(0)
    run_other_agents = True if run_other_agents_input.lower() == 'n' else False

    # First clear plots directory content to prevent conflict when creating new plots for results
    plots_dir_path = "shared" + os.sep + "plots" + os.sep + logic + os.sep + dataset_name
    if not os.path.exists(plots_dir_path):
        os.makedirs(plots_dir_path)
    elif len(read_learned_files_list(logic, dataset_name)) == 0:
        clear_dir(plots_dir_path)

    # Create one process manager and use it in all processes
    process_manager = multiprocessing.Manager()

    # Shared variable to keep results for each env (smt file)
    results = process_manager.dict()

    # Train agent for all smt files in train folder
    actions = ActionReader().get_random_actions()
    for (smt_file_name, smt_file_size) in FilesHandler.get_files_by_file_size(smt_files_dir):

        # Print name of env to specify which env is running at this moment
        print("------------------------------------------------------------------------------------")
        print(smt_file_name)

        # Ignore files that learned before
        learned_files = read_learned_files_list(logic, dataset_name)
        learned_before = False
        for f in learned_files:
            if smt_file_name in f:
                learned_before = True
                break
        if learned_before:
            continue

        # Create new record in results dictionary to add results to this new record
        results[smt_file_name] = process_manager.dict()

        # Create brute force processes
        ps_brute_force_agent = run_brute_force_agent(smt_file_name, actions, number_of_episodes, results, process_manager)

        # Create random search (pure chance) process
        p_random_agent = multiprocessing.Process(target=run_random_agent, args=(smt_file_name, actions, number_of_episodes, results))

        # Only when we want to have plots, we need max reward and random agent results
        if run_other_agents:

            # Run brute force processes
            counter = 1
            for p in ps_brute_force_agent:
                print("Brute force process " + str(counter) + " started")
                print("Brute force process count " + str(len(ps_brute_force_agent)))
                p.start()
                counter += 1
            print("Brute force processes started")

            # Run random search (pure chance) process
            p_random_agent.start()
            print("Random agent process started")

        # Create and run DQN agent process
        run_dqn_agent(smt_file_name, policy_file_name, actions, number_of_episodes, results)

        # Wait until processes are finished
        if run_other_agents:
            for p in ps_brute_force_agent:
                p.join()
            print("Brute force processes finished")
            p_random_agent.join()
            print("Random agent process finished")

        # Get DQN agent results
        mean_rewards = results[smt_file_name]["mean_rewards"]
        mean_steps = results[smt_file_name]["mean_steps"]
        mean_rlimits = results[smt_file_name]["mean_rlimits"]
        mean_probes = results[smt_file_name]["mean_probes"]
        agent_json = json.loads(results[smt_file_name]["agent"])  # Deserialize agent
        agent = Agent(agent_json["state_size"], agent_json["action_size"], policy_file_name)  # Create agent using deserialized data

        # Generate plot title and saving path
        title = "learning_rate: " + str(agent.learning_rate) + \
                ", epsilon: " + str(agent.epsilon) + \
                ", epsilon decay: " + str(agent.epsilon_decay) + \
                ", gamma: " + str(agent.gamma) + \
                ", memory_size: " + str(agent.memory_size) + \
                ", target_update_interval: " + str(agent.target_update_interval) + \
                ",\ntraining_interval: " + str(agent.training_interval) + \
                ", learning_start: " + str(agent.learning_start) + \
                ", batch_size: " + str(agent.batch_size) + \
                ", action_size: " + str(agent.action_size) + \
                ", smt_file: " + smt_file_name
        splitted_path = smt_file_name.split(os.sep)
        plot_file_name = plots_dir_path + os.sep + splitted_path[len(splitted_path) - 1].replace(".smt2", ".png")

        # All processes finished, so we can plot the results
        if run_other_agents:
            # Get brute force result
            random_rewards = results[smt_file_name]["random_rewards"]

            # Get random search (pure chance) results
            max_rewards = list()
            for rewards in results[smt_file_name]["max_rewards"]:
                max_rewards.append(max(rewards))
            max_reward = max(max_rewards)

            plot_result(dqn_agent_values=mean_rewards, mean_steps=mean_steps, random_agent_values=random_rewards,
                        max_reward=max_reward, plot_title=title, save_path=plot_file_name)
        else:
            plot_result(dqn_agent_values=mean_rewards, mean_steps=mean_steps,
                        plot_title=title, save_path=plot_file_name)

        rlimits_and_probes_plot_file_name = plots_dir_path + os.sep + splitted_path[len(splitted_path) - 1].replace(".smt2", "") + "_rlimits_and_probes.png"
        plot_rlimits_and_probes(mean_rlimits=mean_rlimits, mean_probes=mean_probes, plot_title=title,
                                save_path=rlimits_and_probes_plot_file_name)

        # Update learned files list
        learned_files.append(str(smt_file_name))
        write_learned_files_list(learned_files, logic, dataset_name)
