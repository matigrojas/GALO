"""
.. module:: crossover
   :platform: Unix, Windows
   :synopsis: Module implementing crossover operators.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""

class FloatSolution():
    """ Class representing float solutions """

    def __init__(self, lower_bound: [float], upper_bound: [float], number_of_objectives: int,
                 number_of_constraints: int = 0):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.number_of_variables = len(lower_bound)
        self.number_of_objectives = number_of_objectives
        self.number_of_constraints = number_of_constraints
        self.variables = [[] for _ in range(self.number_of_variables)]
        self.objectives = [0.0 for _ in range(self.number_of_objectives)]
        self.constraints = [0.0 for _ in range(self.number_of_constraints)]
        self.attributes = {}

    def __copy__(self):
        new_solution = FloatSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_objectives,
            self.number_of_constraints)
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]
        new_solution.constraints = self.constraints[:]

        new_solution.attributes = self.attributes.copy()

        return new_solution