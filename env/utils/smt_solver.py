import time

from z3 import z3

from env.utils.probes_reader import ProbesReader


class SMTSolver:
    """
    Class that solve formula with given tactic and generate result.
    Smt solver is running in separate thread, because of that after time we stop the operation (thread).
    If solver can't apply tactic and check formula satisfiability in specified timeout, so the result will be
    unknown result tuple.
    """

    def __init__(self, solver_provider, input_goal, tactic_str, tactic, timeout, verbose=False):
        """
        Create solver instance (Constructor)
        :param input_goal: z3 formula that we want to check it satisfiability with given tactic.
        :param tactic_str: string of given tactic. just to print.
        :param tactic: given z3 tactic to apply to formula.
        :param verbose: specify prints done or not.
        """
        super().__init__()
        self.solver_provider = solver_provider
        self.input_goal = input_goal
        self.tactic_str = tactic_str
        self.result = None
        self.output_goal = None
        self.timeout = timeout
        self.verbose = verbose
        self.probes = ProbesReader.get_probes(input_goal)

        # Set timeout for tactic (applicable when applying tactic)
        if tactic:
            self.tactic = z3.Tactic(z3.Z3_tactic_try_for(tactic.ctx.ref(), tactic.tactic, self.timeout))
        else:
            self.tactic = None

        # Set z3 config
        z3.set_option("timeout", self.timeout)  # Globally set timeout for z3
        z3.Z3_global_param_set("timeout", str(self.timeout))

        # To get same results in multiple and different runs
        # z3.set_option("parallel.enable", False)  # Because we guess paralleling can cause non-deterministic results
        # z3.set_option("parallel.threads.max", 1)  # Because we guess paralleling can cause non-deterministic results
        # z3.set_option("fixedpoint.spacer.random_seed", 0)
        # z3.set_option("sat.random_seed", 0)
        # z3.set_option("nlsat.seed", 0)
        # z3.set_option("fp.spacer.random_seed", 0)
        # z3.set_option("smt.random_seed", 0)
        # z3.set_option("sls.random_seed", 0)

    def solve(self):
        """ Apply given tactic to given formula and then check formula satisfiability"""

        # Receive solver from solver provider and use it
        solver = self.solver_provider.pop()
        solver.set("timeout", self.timeout)  # Set solver timeout

        try:

            if self.verbose:
                print("strategy: " + str(self.tactic_str))

            # Apply tactic to goal
            if self.tactic:
                self.output_goal = self.tactic(self.input_goal)[0]
            else:
                self.output_goal = self.input_goal

            if self.verbose:
                print("sub goals count: " + str(len(self.output_goal)))

            # Get initial rlimit and time
            time_start = time.process_time()
            rlimit_before = self.get_rlimit(solver)

            # Check satisfiability of formula with applied tactic
            solver.add(self.output_goal.as_expr())
            satisfiability = solver.check()  # Check satisfiability

            # Get rlimit and time after checking satisfiability
            rlimit_after = self.get_rlimit(solver)
            time_end = time.process_time()

            # Compute final rlimit and time
            final_rlimit = rlimit_after - rlimit_before
            final_time = time_end - time_start

            # Update and print formulas measures (Probes)
            self.probes = ProbesReader.get_probes(self.output_goal)
            if self.verbose:
                print(str(self.probes))

            # Generate result. result is getting using get_result method
            self.result = str(satisfiability), self.output_goal, self.tactic, final_rlimit, final_time, self.probes

            if self.verbose:
                print("statistics: " + str(solver.statistics()))
        except Exception as e:
            if self.verbose:
                print("exception occurred")

        # We must remove solver provided by solver provide after it job finished to save memory space
        del solver

    @staticmethod
    def get_rlimit(solver):
        """
        Get rlimit of solver. If couldn't get rlimit so return 0
        (See this comment: https://github.com/Z3Prover/z3/issues/4662#issuecomment-679959426).
        :param solver: z3 solver that used to solve formula (checking satisfiability)
        :return: rlimit amount of z3 solver
        """
        statistics = solver.statistics()
        for i in range(len(statistics)):
            if statistics[i][0] == 'rlimit count':
                return statistics[i][1]
        return 0

    def get_result(self):
        """
        Get result of solving formula using given tactic.
        This method will be call after solver generate result.
        Result can be None if z3 can't check satisfiability in specified timeout.
        So we return known values as result.
        return value is tuple as (satisfiability, formula, tactic, rlimit, time, probes).
        If result be None, satisfiability is unknown, formula is input formula, both rlimit and time are -1,
        probes are input formula probes.
        :return: achieved result after apply tactic.
        """

        if self.result is None:
            self.result = 'unknown', self.input_goal, self.tactic, -1, -1, self.probes
        return self.result
