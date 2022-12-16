import z3


class ProbesReader:

    # This const variable specify number of probes that is using for environment states
    PROBES_SIZE = 27

    # Only non boolean probes can affect in computing rewards. This range specify that which probes can be used in
    # computing rewards, or in other word which probes are non boolean probes.
    REWARD_PROBES_RANGE = range(0, 0)

    @staticmethod
    def get_probes(formula_goal):
        """ Get formula measures (Probes) """
        formula_goal_str = str(formula_goal)
        probes = list()

        # This probes can be changed in environment by applying tactics. Reducing value number of this probes
        # leading us to more rewards.
        probes.append(z3.Probe('size')(formula_goal))  # Number of assertions in the given goal
        probes.append(z3.Probe('num-exprs')(formula_goal))  # Number of expressions/terms in the given goal
        probes.append(z3.Probe('num-consts')(formula_goal))  # Number of non Boolean constants in the given goal
        probes.append(z3.Probe('num-bool-consts')(formula_goal))  # Number of Boolean constants in the given goal
        probes.append(z3.Probe('num-arith-consts')(formula_goal))  # Number of arithmetic constants in the given goal
        probes.append(z3.Probe('num-bv-consts')(formula_goal))  # Number of bit-vector constants in the given goal
        probes.append(formula_goal_str.count('+'))
        probes.append(formula_goal_str.count('-'))
        probes.append(formula_goal_str.count('/'))
        probes.append(formula_goal_str.count('*'))
        probes.append(formula_goal_str.count('=='))
        probes.append(formula_goal_str.count('!='))
        probes.append(formula_goal_str.count('>='))
        probes.append(formula_goal_str.replace('>=', '').count('>'))
        probes.append(formula_goal_str.count('<='))
        probes.append(formula_goal_str.replace('<=', '').count('<'))
        probes.append(formula_goal_str.replace('bvand', '').count('and'))
        probes.append(formula_goal_str.replace('bvor', '').count('or'))
        probes.append(formula_goal_str.count('bvand'))
        probes.append(formula_goal_str.count('bvor'))
        probes.append(formula_goal_str.count('bvxor'))
        probes.append(formula_goal_str.count('bvashr'))
        probes.append(formula_goal_str.count('bvshl'))

        # True if the goal contains integer/real constants that do not have lower/upper bounds
        probes.append(z3.Probe('is-unbounded')(formula_goal))

        # This probes are fix in one smt2 file but can be different in different smt2 files
        # In other word, this probes are fix in one environment but can be different in different environments
        probes.append(z3.Probe('is-qfbv')(formula_goal))
        probes.append(z3.Probe('is-qfnia')(formula_goal))
        probes.append(z3.Probe('is-qfnra')(formula_goal))

        return probes
