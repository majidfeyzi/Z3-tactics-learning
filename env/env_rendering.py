import copy

import numpy as np
import z3

from env.utils.probes_reader import ProbesReader
from env.utils.tactic_handler import TacticHandler


class EnvRendering:
    """
    Env for z3 tactics to render learned policy.
    Each state in this env is made from formula probes that achieved even.
    Initial state is a state without any tactic.
    Final state is state in which formula satisfiability is trivially decidable or we achieve max step count or timeout.
    Actions are one tactic with one of the its params (or maybe without param).
    This Environment is created to prevent Disk I/O overhead, because this env using smt string to work and not
    file with .smt2 format.
    """

    def __init__(self, smt_string, actions, verbose=False):
        """
        Create env instance (Constructor)
        :param smt_string: smt as string
        :param actions: actions of environment
        :param verbose: specify prints done or not
        """

        # Read formula from file and remove check-sat or check-sat-using from it and then pass it to z3
        self.formula = self.__read_formula(smt_string)

        # Get available actions
        self.action_space = ActionBox(actions)

        # Create required objects and env config
        self.tactic_handler = TacticHandler()
        self.max_step = self.action_space.n  # Max step count in each episode
        self.verbose = verbose

        # Generate start state
        # Start state have a random tactic
        self.formula_goal = z3.Goal()
        self.formula_goal.add(self.formula)
        self.start_state = ProbesReader.get_probes(self.formula_goal)
        self.state = copy.deepcopy(self.start_state)

        # Create observation state
        self.observation_space = StateBox(self.state)

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

        # Apply new tactics to goal
        z3_tactic_str = self.tactic_handler.combine_tactics([tactic_in_z3_format])
        z3_tactic = self.tactic_handler.convert_string_to_z3_tactic(z3_tactic_str) if z3_tactic_str != "" else None
        if z3_tactic:
            goal = z3_tactic(self.formula_goal)[0]
        else:
            goal = self.formula_goal
        probes = ProbesReader.get_probes(goal)

        # Just create next state
        next_state = probes

        if self.verbose:
            print("{ new_state: " + str(probes) + ", tactic: " + str(tactic_in_z3_format) + " }")

        self.state = copy.deepcopy(next_state)
        return next_state

    def reset(self):
        """
        Reset env and change env state to initial state.
        :return: return current/initial state
        """
        self.state = copy.deepcopy(self.start_state)
        return self.state

    @staticmethod
    def __read_formula(smt_string):
        """
        Read formula from smt2 file.
        :param smt_string: assertion as string
        :return: z3 formula
        """
        temp_smt_string = ""
        for line in smt_string.split("\n"):
            new_line = line
            if 'check-sat' not in line:
                temp_smt_string += new_line + "\n"
        formula = z3.parse_smt2_string(temp_smt_string)
        return formula


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
