import json
import os
import random

from z3 import z3


class TacticHandler:
    """
    Class to working with z3 tactics, convert z3 tactics to string and vise versa and etc.
    """

    @staticmethod
    def __read_tactics_json():
        """
        Return content of tactics json file as json.
        :return: json of tactics
        """
        json_file = open(os.path.dirname(__file__) + "/../tactics.json")
        tactics_json = json.load(json_file)
        json_file.close()
        return tactics_json

    def get_tactics_with_params(self):
        """
        Separate tactics json and specified each tactic params.
        :return: list of tuples as (tactic, tactic params as list)
        """
        tactics_json = self.__read_tactics_json()
        tactics_config = tactics_json["tactics_config"]
        all_tactics = tactics_config["all_tactics"]
        allowed_params = tactics_config["allowed_params"]
        all_tactics_with_params = list()
        for tactic in all_tactics:
            params = None
            if tactic in allowed_params:
                params = allowed_params[tactic]["boolean"]
            all_tactics_with_params.append((tactic, params))
        return all_tactics_with_params

    def __generate_random_tactic_str(self, count):
        """
        Generate random tactic as string using exist tactics.
        :param count: number of random tactics that must be generated
        :return: list of randomly generated tactics as string
        """
        tactics = self.get_tactics_with_params()
        selected_tactics = list()
        while len(selected_tactics) < count:
            (random_tactic, params) = random.choice(tactics)
            random_param = params
            random_param_value = None
            if params is not None:
                random_param = random.choice(params)
                random_param_value = random.choice([True, False])
            selected_tactics.append((random_tactic, random_param, random_param_value))
            selected_tactics = list(set(selected_tactics))
        return selected_tactics

    @staticmethod
    def convert_tactic_tuple_str_to_z3_format(tactic_tuple):
        """
        Convert tactic string to z3 tactic
        :param tactic_tuple: tuple of tactic as (tactic, param, param value)
        :return: z3 tactic with it param (if param exist)
        """
        (tactic, param, param_value) = tactic_tuple
        if param is not None and param_value is not None:
            return "With({};{})".format(tactic, str(param) + "=" + str(param_value))
        else:
            return "Tactic({})".format(tactic)

    @staticmethod
    def combine_tactics(tactics_str_list):
        """
        Method that combine given tactics using AnThen combinator
        :param tactics_str_list: list of tactics string
        :return: combined tactics as string
        """
        if len(tactics_str_list) >= 2:
            result = "AndThen("
            for tactic_str in tactics_str_list:
                result += tactic_str + ","
            result = result[:-1]
            result += ")"
            return result
        elif len(tactics_str_list) == 1:
            return tactics_str_list[0]
        else:
            return ""

    def generate_random_tactics(self, count):
        """
        Generate random z3 tactic using exist tactics.
        :param count: number of random tactics that must be generated
        :return: list of randomly generated tactics
        """
        random_tactics_str = self.__generate_random_tactic_str(count)
        tactics_in_z3_format = list()
        for random_tactic_str in random_tactics_str:
            tactic_in_z3_format = self.convert_tactic_tuple_str_to_z3_format(random_tactic_str)
            tactics_in_z3_format.append(tactic_in_z3_format)
        return tactics_in_z3_format

    def convert_string_to_z3_tactic(self, s):
        """
        Convert string to z3 tactic.
        :param s: string of tactic
        :return: z3 tactic
        """
        if s[:7] == 'Tactic(' and s[-1] == ')':
            return z3.Tactic(s[7:-1])
        elif s[:8] == 'AndThen(' and s[-1] == ')':
            tokens = s[8:-1].split(',')
            tactics = list(map(self.convert_string_to_z3_tactic, tokens))
            return z3.AndThen(*tactics)
        elif s[:5] == 'With(' and s[-1] == ')':
            tokens = s[5:-1].split(';')
            tactic = tokens[0]

            params = []
            for token in tokens[1:]:
                x, val = token.split('=')
                if val == 'True':
                    params.append(x)
                    params.append(True)
                elif val == 'False':
                    params.append(x)
                    params.append(False)
                elif val.isdigit():
                    params.append(x)
                    params.append(int(val))
                else:
                    assert False, 'param {} = {} invalid'.format(x, val)

            # z3.With doesn't work correct
            # return z3.With(tactic, params)
            t = self.__to_tactic(tactic)
            p = z3.args2params(params, {}, t.ctx)
            return z3.Tactic(z3.Z3_tactic_using_params(t.ctx.ref(), t.tactic, p.params), t.ctx)

        else:
            return z3.Tactic(s)

    @staticmethod
    def __to_tactic(t, ctx=None):
        """
        This private method is z3 method and is using because of that z3.With doesn't work correct
        :param t: tactic
        :param ctx: context
        :return: z3 tactic
        """
        if isinstance(t, z3.Tactic):
            return t
        else:
            return z3.Tactic(t, ctx)
