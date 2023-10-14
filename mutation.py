import random
from solution import FloatSolution

"""
.. module:: mutation
   :platform: Unix, Windows
   :synopsis: Module implementing mutation operators.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""
    
class UniformMutation():

    def __init__(self, probability: float, perturbation: float = 0.5):
        self.perturbation = perturbation
        self.probability = probability

    def execute(self, solution: FloatSolution) -> FloatSolution:

        for i in range(solution.number_of_variables):
            rand = random.random()
            
            if rand <= self.probability:
                tmp = (random.random() - 0.5) * self.perturbation
                tmp += solution.variables[i]

                if tmp < solution.lower_bound[i]:
                    tmp = solution.lower_bound[i]
                elif tmp > solution.upper_bound[i]:
                    tmp = solution.upper_bound[i]

                solution.variables[i] = tmp

        return solution

    def get_name(self):
        return 'Uniform mutation (UM)'
    
