import copy

import numpy as np
import z3

from env.utils.probes_reader import ProbesReader
from env.utils.smt_solver import SMTSolver
from env.utils.solver_provider import SolverProvider
from env.utils.tactic_handler import TacticHandler


class Env:
    """
    Env for z3 tactics to learn to choose best tactic.
    Each state in this env is made from formula probes that achieved even.
    Initial state is a state without any tactic.
    Final state is state in which formula satisfiability is trivially decidable or we achieve max step count or timeout.
    Actions are one tactic with one of the its params (or maybe without param).
    """

    def __init__(self, smt_file_name, actions, number_of_episodes, verbose=False):
        """
        Create env instance (Constructor)
        :param smt_file_name: name of smt2 file that contain formula
        :param actions: actions of environment
        :param number_of_episodes: count of episodes that agent act in this env
        :param verbose: specify prints done or not
        """

        # To get same results in multiple and different runs on same env
        np.random.seed(0)

        # Read formula from file and remove check-sat or check-sat-using from it and then pass it to z3
        self.smt_file_name = smt_file_name
        self.formula = self.__read_formula(smt_file_name)

        # Get available actions
        self.action_space = ActionBox(actions)

        # We need create solvers at the beginning because of z3 bug, also
        # +2 is for that we need to get initial rlimit in env creating and resting
        self.solver_provider = SolverProvider((number_of_episodes + 2) * self.action_space.n)

        # Create required objects and env config
        self.tactic_handler = TacticHandler()
        self.step_counter = 0
        self.max_step = self.action_space.n  # Max step count in each episode
        self.verbose = verbose

        # Generate start state
        self.formula_goal = self.__generate_formula_goal()
        self.start_state = ProbesReader.get_probes(self.formula_goal)
        self.state = copy.deepcopy(self.start_state)

        # This is required to measure goodness of action
        self.last_rlimit = self.__get_initial_rlimit()

        # Create observation state
        self.observation_space = StateBox(self.state)

        # Cache of z3 results as dictionary
        self.cache = {}

    def step(self, action):
        """
        Run action on env and get the result.
        :param action: index of action in actions to do in env (one tactic and one of the its params)
        :return: (nex_state, reward, done) tuple
        """

        # Get action by it index
        action = self.action_space.get(action)

        # Append new tactic (action) to exist tactics
        tactic_in_z3_format = self.tactic_handler.convert_tactic_tuple_str_to_z3_format(action)

        if self.verbose:
            print("step " + str(self.step_counter))

        # Tactics change in each step, so we can use it as key in cache
        z3_tactic_str = self.tactic_handler.combine_tactics([tactic_in_z3_format])
        key = str(self.state) + "_" + z3_tactic_str

        # Load results from cache if exist to speed up
        if key in self.cache:
            satisfiability, self.formula_goal, tactic, rlimit, time, probes = self.cache[key]  # Get result from cache
        else:

            # Check satisfiability with new tactics
            z3_tactic = self.tactic_handler.convert_string_to_z3_tactic(z3_tactic_str) if z3_tactic_str != "" else None
            solver = SMTSolver(self.solver_provider, self.formula_goal, z3_tactic_str, z3_tactic, 5000, self.verbose)
            solver.solve()
            satisfiability, self.formula_goal, tactic, rlimit, time, probes = solver.get_result()  # Get result
            del solver

            # Cache results
            value = (satisfiability, self.formula_goal, tactic, rlimit, time, probes)
            self.cache[key] = value

        # This must be call before __check_done and env reset
        self.step_counter += 1

        # Get reward and create next state
        done, is_trivially_satisfiable = self.__check_done()
        reward = self.__get_reward(rlimit, is_trivially_satisfiable)
        next_state = probes

        # Update last rlimit must be done after computing reward
        self.last_rlimit = rlimit

        if done:
            self.reset()

        if self.verbose:
            print("{ rlimit: " + str(rlimit) +
                  ", satisfiability: " + satisfiability +
                  ", formula_goal_len: " + str(len(self.formula_goal)) +
                  ", time: " + str(time) +
                  ", tactic: " + str(tactic_in_z3_format) +
                  ", done: " + str(done) +
                  ", reward: " + "{:.10f}".format(reward) + " }")

        self.state = copy.deepcopy(next_state)
        return next_state, reward, done, rlimit, probes

    def reset(self):
        """
        Reset env and change env state to initial state.
        :return: return current/initial state
        """
        self.formula_goal = self.__generate_formula_goal()
        self.state = copy.deepcopy(self.start_state)
        self.last_rlimit = self.__get_initial_rlimit()
        self.step_counter = 0
        return self.state

    def __get_reward(self, rlimit, is_trivially_satisfiable):
        """
        Compute reward based on rlimit.
        :param rlimit: rlimit that achieved from solver after check formula satisfiability with applied tactic
        :param is_trivially_satisfiable: boolean value that specify formula goal is trivially satisfiable or not
        :return: reward per action (immediate reward)
        """

        MAX_REWARD = 5
        STEP_REWARD = -2
        STEP_REWARD_FOR_RLIIMIT_REDUCTION = -1
        PENALTY_REWARD = -MAX_REWARD

        if rlimit == 0 or is_trivially_satisfiable:
            return MAX_REWARD
        if rlimit == -1:
            return PENALTY_REWARD

        rlimit_reduction = self.last_rlimit - rlimit
        if rlimit_reduction > 0:
            return STEP_REWARD_FOR_RLIIMIT_REDUCTION

        return STEP_REWARD

    def __check_done(self):
        """
        This method specify that episode finished or not.
        :return: a boolean that say us episode finished or not
        """
        is_max_steps_reached = self.step_counter > self.max_step
        is_trivially_satisfiable = self.formula_goal.size() == 0 or (self.formula_goal.size() == 1 and len(self.formula_goal[0].children()) == 0)
        return is_max_steps_reached or is_trivially_satisfiable, is_trivially_satisfiable

    def __generate_formula_goal(self):
        """
        Just generate formula goal using formula
        :return: generated formula goal
        """
        formula_goal = z3.Goal()
        formula_goal.add(self.formula)
        return formula_goal

    def __get_initial_rlimit(self):
        """
        Get rlimit of initial formula goal
        :return: rlimit of initial formula goal
        """
        solver = SMTSolver(self.solver_provider, self.__generate_formula_goal(), None, None, 5000, self.verbose)
        solver.solve()
        satisfiability, self.formula_goal, tactic, rlimit, time, probes = solver.get_result()  # Get result
        del solver
        return rlimit

    @staticmethod
    def __read_formula(smt_file_name):
        """
        Read formula from smt2 file.
        :param smt_file_name: path of smt2 file
        :return: z3 formula
        """
        return z3.parse_smt2_file(smt_file_name)


class ActionBox:
    def __init__(self, actions):
        self.actions = actions
        self.n = len(actions)

    def sample(self):
        """
        Generate one random action and return it index in actions list.
        :return: index of randomly generated action
        """
        actions_array = np.array(self.actions)
        index = np.random.choice(len(actions_array), 1)
        return index[0]

    def get(self, index):
        """
        Get action by index
        :param index index of action
        :return: action
        """
        return self.actions[index]


class StateBox:
    def __init__(self, state):
        self.state = state
        self.n = len(state)
