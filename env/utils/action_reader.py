import itertools
import json
import os
import pathlib
import random

import numpy as np

from env.utils.tactic_handler import TacticHandler


class ActionReader:
    """
    Class to read actions from json file.
    """

    # This constant variable specify number of actions that is using in environment
    ACTIONS_COUNT = 23

    def __init__(self):
        self.tactic_handler = TacticHandler()

    def get_all_actions(self):
        """
        Separate tactic params and build actions.
        Actions are tuples as (tactic, param, param value)
        :return: list of tuples as (tactic, param, param value)
        """
        tactics = self.tactic_handler.get_tactics_with_params()
        actions = list()
        for (tactic, params) in tactics:
            actions.append((tactic, None, None))  # tactic without param itself can be an action
            if params is not None:
                for param in params:
                    actions.append((tactic, param, True))
        return actions

    def get_random_actions(self):
        """
        Separate tactic params and build actions.
        Actions are tuples as (tactic, param, param value)
        Actions are selecting as random.
        Randomly generated action is caching as json to use them when rendering.
        :return: list of tuples as (tactic, param, param value)
        """
        path = os.path.join(os.path.join(
            pathlib.Path(__file__).parent.parent.parent.resolve(), "shared"),
            "randomly_selected_actions.json"
        )
        
        if os.path.isfile(path) and os.access(path, os.R_OK):
            with open(path, 'r') as randomly_selected_actions_file:
                actions_json = json.loads(randomly_selected_actions_file.read())
                random_actions = list()
                for action_json in actions_json:
                    random_actions.append((action_json[0], action_json[1], action_json[2]))
                return random_actions
        else:
            actions = self.get_all_actions()
            random_actions = list()
            while len(random_actions) < self.ACTIONS_COUNT:
                random_action = random.choice(actions)
                if random_action not in random_actions:
                    random_actions.append(random_action)

            # Cache randomly selected actions as json in json file
            with open(path, 'w') as randomly_selected_actions_file:
                json.dump(random_actions, randomly_selected_actions_file)
            return random_actions

    @staticmethod
    def get_actions_all_permutations(actions, min_length, max_length):
        """
        Get all permutations of all actions
        :param actions: actions to compute its permutations
        :param max_length: max length of permutations
        :param min_length: min length of permutations
        :return: list of actions permutations tuples as (tactic, param, param value)
        """
        permutations = list()
        for i in range(min_length, max_length):
            combinations = list(itertools.combinations(actions, i))
            for c in combinations:
                perms = itertools.permutations(c)
                permutations.extend(perms)
        return list(set(permutations))

    def get_actions_permutations(self, actions, min_length, max_length, random_max_count):
        """
        Get permutations of all actions.
        For combinations/permutations with length > 4 we use random selection, because computing all
        combinations or permutations take so much time.
        Randomly generated permutations for large combinations can be lower than specified random max count, but
        this difference is so little and can be ignored.
        :param actions: actions to compute its permutations
        :param min_length: min length of permutations
        :param max_length: max length of permutations
        :param random_max_count: maximum number of randomly generated permutations for large combinations
        :return: list of actions permutations tuples as (tactic, param, param value)
        """

        threshold = 4

        if min_length >= max_length:
            raise Exception("Min length must be smaller than max length")

        permutations = set()
        if max_length <= threshold:
            permutations.update(self.get_actions_all_permutations(actions, min_length, max_length))
        else:
            permutations.update(self.get_actions_all_permutations(actions, min_length, threshold))
            chunk_size = int(random_max_count / (max_length - threshold))
            for i in range(threshold, max_length):
                combinations = list(itertools.combinations(actions, i))
                counter = 1
                while counter <= chunk_size:
                    combination = random.choice(combinations)
                    permutation = self.__convert_permutation_to_tuple(np.random.permutation(combination))
                    if permutation not in permutations:
                        permutations.add(permutation)
                        counter += 1

        return list(permutations)

    @staticmethod
    def __convert_permutation_to_tuple(permutation):
        """ Don't use tuple(map(tuple, permutation)) to convert permutation to tuple """
        t = []
        for i in permutation:
            t.append((i[0], i[1], i[2]))
        return tuple(t)
