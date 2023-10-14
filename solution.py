"""
.. module:: crossover
   :platform: Unix, Windows
   :synopsis: Module implementing crossover operators.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""

class FloatSolution():
    """ Class representing float solutions """

    def __init__(self, lower_bound: [float], upper_bound: [float]):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.number_of_variables = len(lower_bound)
        self.variables = [[] for _ in range(self.number_of_variables)]
        self.objectives = [0.0]
        self.attributes = {}

    def __copy__(self):
        new_solution = FloatSolution(
            self.lower_bound,
            self.upper_bound,)
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]

        new_solution.attributes = self.attributes.copy()

        return new_solution