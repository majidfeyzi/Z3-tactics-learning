import os
import random
import time
import numpy as np

from IPython.core.display import clear_output
from matplotlib import pyplot as plt

from env.utils.action_selector import ActionSelector
from test_learning_dqn_agent import AgentLearningTest
from env.env_learning_test import EnvLearningTest
from env.utils.action_reader import ActionReader
from env.utils.files_handler import FilesHandler
from env.utils.probes_reader import ProbesReader
from env.utils.smt_solver import SMTSolver
from env.utils.solver_provider import SolverProvider
from env.utils.tactic_handler import TacticHandler


def plot_result(default_speeds, learned_speeds, random_speeds, default_rlimits, learned_rlimits, random_rlimits,
                smt_files_sizes, plot_title="", save_path=""):
    """ Plot the speeds and rlimits reduction """

    # Update the window after each episode
    clear_output(wait=True)

    # Define the figure
    figure, axis = plt.subplots(nrows=2, ncols=1, figsize=(len(smt_files_sizes), 15))
    figure.suptitle(plot_title)
    axis[0].set_ylabel("Average Speeds")
    axis[0].plot(learned_speeds, c="blue", label="Learned tactics", linestyle="-")
    axis[0].plot(default_speeds, c="green", label="Without tactics", linestyle="--")
    axis[0].plot(random_speeds, c="red", label="Random tactics", linestyle="-.")
    axis[0].set_xticks([i for i in range(0, len(default_speeds))])
    axis[0].set_xticklabels(smt_files_sizes)
    axis[0].legend(loc="best")
    axis[1].set_ylabel("RLimit")
    axis[1].set_xlabel("Files (Ordered by size)")
    axis[1].plot(learned_rlimits, c="blue", label="Learned tactics", linestyle="-")
    axis[1].plot(default_rlimits, c="green", label="Without tactics", linestyle="--")
    axis[1].plot(random_rlimits, c="red", label="Random tactics", linestyle="-.")
    axis[1].set_xticks([i for i in range(0, len(default_rlimits))])
    axis[1].set_xticklabels(smt_files_sizes)
    axis[1].legend(loc="best")

    # Save plot as file in given path
    if len(save_path) > 0:
        abs_path = os.path.abspath(save_path)
        figure.savefig(abs_path, format="svg")
    else:
        plt.show()

    # See https://stackoverflow.com/a/21884375
    plt.close(figure)


def run_test(smt_files, title, save_path):
    solver_provider = SolverProvider((len(smt_files) * (iteration_for_each_smt * 3)) + number_of_tactics_to_apply)

    default_speeds = list()
    learned_speeds = list()
    random_speeds = list()
    default_rlimits = list()
    learned_rlimits = list()
    random_rlimits = list()
    smt_files_sizes = list()

    print(
        "                                               smt file name | "
        "time without any tactic | "
        "time with learned tactics | "
        "time random tactics | "
        "solver time without tactics | "
        "solver time with learned tactics | "
        "solver time with random tactics | "
        "rlimit without any tactic | "
        "rlimit with learned tactics | "
        "rlimit with random tactics")

    for (smt_file_name, smt_file_size) in smt_files:

        print("---------------------------------------------------------------------------------------------"
              "---------------------------------------------------------------------------------------------"
              "---------------------------------------------------------------------------------------------"
              "----------------------------------------------")

        default_speed = list()
        learned_speed = list()
        random_speed = list()
        default_rlimit = list()
        learned_rlimit = list()
        random_rlimit = list()

        for _ in range(0, iteration_for_each_smt):

            # Use learned policy to select best tactics for solving inequalities
            # Actions count must be same with policy file shape
            env = EnvLearningTest(smt_file_name, actions, verbose=False)

            # Running codes in try...except, prevent app failing in run for inputs that cause exception
            try:

                # First run without any tactic and measure time
                start_time_for_no_tactic = time.process_time()

                # Convert selected actions to z3 tactics and check formula (inequalities) satisfiability
                # by applying selected tactics and print the satisfiability result
                solver = SMTSolver(solver_provider, env.generate_formula_goal(), None, None, 300000, False)
                solver.solve()
                _, _, _, rlimit_1, time_1, _ = solver.get_result()  # Get result
                del solver

                end_time_for_no_tactic = time.process_time()

                # Second run with learned tactics and measure time
                start_time_for_learned_tactics = time.process_time()

                # Select trained actions
                selected_actions = ActionSelector.select_tactics_based_on_learning(env, agent, actions, number_of_tactics_to_apply)

                # Convert selected actions to z3 tactics and check formula (inequalities) satisfiability
                # by applying selected tactics and print the satisfiability result
                tactics_in_z3_format = list()
                for action in selected_actions:
                    tactics_in_z3_format.append(tactic_handler.convert_tactic_tuple_str_to_z3_format(action))
                z3_tactic_str_1 = tactic_handler.combine_tactics(tactics_in_z3_format)
                z3_tactic = tactic_handler.convert_string_to_z3_tactic(z3_tactic_str_1)
                solver = SMTSolver(solver_provider, env.generate_formula_goal(), z3_tactic_str_1, z3_tactic, 300000, False)
                solver.solve()
                _, _, _, rlimit_2, time_2, _ = solver.get_result()  # Get result
                del solver

                end_time_for_learned_tactics = time.process_time()

                # Third run with random tactics and measure time
                start_time_for_random_tactics = time.process_time()

                # Convert selected actions to z3 tactics and check formula (inequalities) satisfiability
                # by applying selected tactics and print the satisfiability result
                tactics_in_z3_format = list()
                selected_actions_3 = list()
                for _ in range(number_of_tactics_to_apply):
                    selected_actions_3.append(random.choice(actions))
                for action in selected_actions_3:
                    tactics_in_z3_format.append(tactic_handler.convert_tactic_tuple_str_to_z3_format(action))
                z3_tactic_str_3 = tactic_handler.combine_tactics(tactics_in_z3_format)
                z3_tactic = tactic_handler.convert_string_to_z3_tactic(z3_tactic_str_3)
                solver = SMTSolver(solver_provider, env.generate_formula_goal(), z3_tactic_str_3, z3_tactic, 300000, False)
                solver.solve()
                _, _, _, rlimit_3, time_3, _ = solver.get_result()  # Get result
                del solver

                end_time_for_random_tactics = time.process_time()

                time_for_no_tactic = end_time_for_no_tactic - start_time_for_no_tactic  # Execution time without any tactic
                time_for_learned_tactics = end_time_for_learned_tactics - start_time_for_learned_tactics  # Execution time with learned tactics
                time_for_random_tactics = end_time_for_random_tactics - start_time_for_random_tactics  # Execution time with random tactics

                rlimit_1_final = rlimit_1 if rlimit_2 == -1 else rlimit_2  # Cancel -1 rlimit affect in results
                rlimit_3_final = rlimit_1 if rlimit_3 == -1 else rlimit_3  # Cancel -1 rlimit affect in results

                learned_speed.append(time_for_learned_tactics)
                default_speed.append(time_for_no_tactic)
                random_speed.append(time_for_random_tactics)
                default_rlimit.append(rlimit_1)
                learned_rlimit.append(rlimit_1_final)
                random_rlimit.append(rlimit_3_final)

                # print("learning selected tactics = {}".format(z3_tactic_str_1))
                # print("randomly selected tactics = {}".format(z3_tactic_str_3))

                print("{: >60} | {: >23} | {: >25} | {: >19} | {: >27} | {: >32} | {: >31} | {: >25} | {: >27} | {: >26}"
                      .format(smt_file_name,
                              time_for_no_tactic,
                              time_for_learned_tactics,
                              time_for_random_tactics,
                              time_1,
                              time_2,
                              time_3,
                              rlimit_1,
                              rlimit_1_final if rlimit_1_final == rlimit_2 else str(rlimit_1_final) + "(" + str(rlimit_2) + ")",
                              rlimit_3_final if rlimit_3_final == rlimit_3 else str(rlimit_3_final) + "(" + str(rlimit_3) + ")"))

            except Exception as e:
                print("exception: " + str(e))

            del env  # To reset z3

        learned_speeds.append(np.mean(learned_speed))
        default_speeds.append(np.mean(default_speed))
        random_speeds.append(np.mean(random_speed))
        learned_rlimits.append(np.mean(learned_rlimit))
        default_rlimits.append(np.mean(default_rlimit))
        random_rlimits.append(np.mean(random_rlimit))
        smt_files_sizes.append(smt_file_size)

        print("learning speedup:                {:.3f}x".format(default_speeds[-1]/learned_speeds[-1]))
        print("random speedup:                  {:.3f}x".format(default_speeds[-1]/random_speeds[-1]))
        print("learning rlimit reduction:       {:.3f}x".format(default_rlimits[-1]/learned_rlimits[-1]))
        print("random rlimit reduction:         {:.3f}x".format(default_rlimits[-1]/random_rlimits[-1]))

    plot_result(default_speeds, learned_speeds, random_speeds, default_rlimits, learned_rlimits, random_rlimits,
                smt_files_sizes, title, save_path)


if __name__ == '__main__':

    np.random.seed(0)
    random.seed(0)

    number_of_tactics_to_apply = 1  # Max can be number of all actions
    iteration_for_each_smt = 10
    logic = "qf_nia"
    dataset_name = "AProVE"
    difficulty = "hard"
    smt_files_dir = "test" + os.sep + logic + os.sep + dataset_name
    policy_file_name = "policy_" + logic + "_" + dataset_name + ".h5"

    plots_dir_path = "shared" + os.sep + "results" + os.sep + logic + os.sep + dataset_name
    if not os.path.exists(plots_dir_path):
        os.makedirs(plots_dir_path)

    actions = ActionReader().get_random_actions()
    state_size = ProbesReader.PROBES_SIZE
    action_size = len(actions)
    agent = AgentLearningTest(state_size, action_size, policy_file_name)
    agent.load_weights()
    tactic_handler = TacticHandler()

    smt_files = FilesHandler.get_files_by_file_size(smt_files_dir + os.sep + difficulty)
    plot_file_name = plots_dir_path + os.sep + difficulty + ".svg"
    run_test(smt_files, "Learning test result (" + difficulty + " - " + str(number_of_tactics_to_apply) + " tactic)", plot_file_name)


