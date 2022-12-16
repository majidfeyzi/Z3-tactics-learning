import z3


class SolverProvider:
    """
    This class is created to provide solvers for solving formulas.
    Using this class cause deterministic statistics in z3. See https://github.com/Z3Prover/z3/issues/5969 for more
    information about why we do this.
    """

    __solvers = list()
    __index = -1  # Keep index of solver in list

    def __init__(self, count):
        """
        Create solver instance (Constructor)
        :param count: number of solvers that we need to create.
        """
        for i in range(0, count):
            # Create z3 solver to solver formula (See https://github.com/Z3Prover/z3/issues/5969)
            self.__solvers.append(z3.SimpleSolver())
        self.__index = len(self.__solvers) - 1

    def pop(self):
        """
        Get one of the solvers.
        Using stack (pop method) also cause non deterministic statistics.
        Deleting popped element also cause non deterministic statistics.
        """
        solver = self.__solvers[self.__index]
        self.__index -= 1
        return solver

