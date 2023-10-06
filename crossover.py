import random
import copy
import numpy as np
from solution import FloatSolution

"""
.. module:: Galo Crossover
   :platform: Unix, Windows
   :synopsis: Module implementing the CKX Operator.

.. moduleauthor:: Matias G. Rojas, Ana Carolina Olivera, Pablo Javier Vidal
"""

class GaloCrossoverConfigurable():

    def __init__(self, probability: float = 1.0,
                 max_evaluations: int = 10000,
                 ratio_start: float = 0.99,
                 ratio_var: float = 0.5,
                 p_elite_start: float = 0.7,
                 p_elite_var: float = 0.3):
        self.probability=probability
        self.ratio_start = ratio_start
        self.ratio_var = ratio_var
        self.p_elite_start = p_elite_start
        self.p_elite_var = p_elite_var

        self.max_evaluations = max_evaluations
        self.evaluations = 0

    def execute(self,parents: [FloatSolution])->[FloatSolution]:
        child_1 = copy.deepcopy(parents[0])
        if random.random() <= self.probability:
            ratio = self.ratio_start+self.ratio_var*(self.evaluations/self.max_evaluations)
            p_elite = self.p_elite_start + self.p_elite_var * (self.evaluations / self.max_evaluations)
            div = 100

            increment = np.random.uniform(child_1.lower_bound/div, child_1.upper_bound/div)
            child_1.variables = (p_elite * np.array(parents[2].variables) +
                                    (1-p_elite) * np.array(parents[1].variables) + increment) * ratio
            child_1.variables += np.array(parents[0].variables) * (1-ratio)
            child_1.variables = np.clip(child_1.variables, child_1.lower_bound, child_1.upper_bound)

        return [child_1]

    def set_current_evaluation(self,evaluation: int):
        self.evaluations = evaluation

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 1

    def get_name(self) -> str:
        return 'Galo Crossover'
    
