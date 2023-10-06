import random
from solution import FloatSolution

"""
.. module:: mutation
   :platform: Unix, Windows
   :synopsis: Module implementing mutation operators.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""

class SimpleRandomMutation():

    def __init__(self, probability: float):
        self.probability = probability

    def execute(self, solution: FloatSolution) -> FloatSolution:

        for i in range(solution.number_of_variables):
            rand = random.random()
            if rand <= self.probability:
                solution.variables[i] = solution.lower_bound[i] + \
                                        (solution.upper_bound[i] - solution.lower_bound[i]) * random.random()
        return solution

    def get_name(self):
        return 'Simple random_search mutation (RM)'
    
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
    
class NonUniformMutation():

    def __init__(self, probability: float, perturbation: float = 0.5, max_iterations: float = 0.5):
        self.perturbation = perturbation
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.probability = probability


    def execute(self, solution: FloatSolution) -> FloatSolution:

        for i in range(solution.number_of_variables):
            if random.random() <= self.probability:
                rand = random.random()

                if rand <= 0.5:
                    tmp = self.__delta(solution.upper_bound[i] - solution.variables[i], self.perturbation)
                else:
                    tmp = self.__delta(solution.lower_bound[i] - solution.variables[i], self.perturbation)

                tmp += solution.variables[i]

                if tmp < solution.lower_bound[i]:
                    tmp = solution.lower_bound[i]
                elif tmp > solution.upper_bound[i]:
                    tmp = solution.upper_bound[i]

                solution.variables[i] = tmp

        return solution

    def set_current_iteration(self, current_iteration: int):
        self.current_iteration = current_iteration

    def __delta(self, y: float, b_mutation_parameter: float):
        return (y * (1.0 - pow(random.random(),
                               pow((1.0 - 1.0 * self.current_iteration / self.max_iterations), b_mutation_parameter))))

    def get_name(self):
        return 'Non uniform mutation (NUM)'
    
class PolynomialMutation():

    def __init__(self, probability: float, distribution_index: float = 0.20):
        self.distribution_index = distribution_index
        self.probability=probability

    def execute(self, solution: FloatSolution) -> FloatSolution:
        for i in range(solution.number_of_variables):
            rand = random.random()

            if rand <= self.probability:
                y = solution.variables[i]
                yl, yu = solution.lower_bound[i], solution.upper_bound[i]

                if yl == yu:
                    y = yl
                else:
                    delta1 = (y - yl) / (yu - yl)
                    delta2 = (yu - y) / (yu - yl)
                    rnd = random.random()
                    mut_pow = 1.0 / (self.distribution_index + 1.0)
                    if rnd <= 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (pow(xy, self.distribution_index + 1.0))
                        deltaq = pow(abs(val), mut_pow) - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (pow(xy, self.distribution_index + 1.0))
                        deltaq = 1.0 - pow(abs(val), mut_pow)

                    y += deltaq * (yu - yl)
                    if y < solution.lower_bound[i]:
                        y = solution.lower_bound[i]
                    if y > solution.upper_bound[i]:
                        y = solution.upper_bound[i]

                solution.variables[i] = y

        return solution

    def get_name(self):
        return 'Polynomial mutation (PM)'
    
