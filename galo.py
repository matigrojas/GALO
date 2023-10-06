import numpy as np
import time
import math
import copy
import random

class GeneticAntLionOptimizer():

    def __init__(self,
                 problem,
                 termination_criterion,
                 crossover,
                 mutation,
                 selection,
                 population_size: int = 50,
                 number_of_antlions:int = 50,
                 ):
        super(GeneticAntLionOptimizer,self).__init__()

        self.problem = problem
        self.population_size = population_size
        self.number_of_antlions = number_of_antlions

        self.mutation_operator = mutation
        self.crossover_operator = crossover
        self.random_walk_selection_operator = selection

        self.termination_criterion = termination_criterion

        self.antlions_population = []
        self.ants_population = []

        self.elite_antlion = None
        self.observable.register(termination_criterion)

    def run(self):
        """ Execute the algorithm. """
        self.start_computing_time = time.time()

        self.create_initial_solutions()
        self.antlions_population = self.evaluate(self.antlions_population)
        self.ants_population = self.evaluate(self.ants_population)

        print(f"Progress: {self.evaluations}/{self.max_evaluations}, Fitness: {self.get_result().objectives[0]}")
        self.init_progress()

        while not self.stopping_condition_is_met():
            self.step()
            self.update_progress()

        self.total_computing_time = time.time() - self.start_computing_time

    def evaluate(self, solution_list):
        return self.evaluator.evaluate(solution_list,self.problem)

    def create_initial_solutions(self):
        self.antlions_population = [self.problem.create_solution() for _ in range(round(self.number_of_antlions))]
        self.ants_population = [self.problem.create_solution() for _ in range(round(self.population_size))]

    def set_elite_antlion(self):
        self.antlions_population.sort(key=lambda s: s.objectives[0])

        if (self.elite_antlion is None) or (self.antlions_population[0].objectives[0] < self.elite_antlion.objectives[0]):
            self.elite_antlion = copy.deepcopy(self.antlions_population[0])

    def init_progress(self) -> None:
        self.set_elite_antlion()
        self.evaluations = self.population_size

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def step(self):
        for i in range(len(self.ants_population)):

            self.antlions_population.sort(key=lambda s:s.objectives[0])

            antlion = self.random_walk_selection_operator.execute(self.antlions_population)

            antlion_pos = 0
            for i in range(len(self.antlions_population)):
                if self.antlions_population[i] is antlion:
                    antlion_pos = i
                    break

            if hasattr(self.crossover_operator, 'set_current_evaluation'):
                self.crossover_operator.set_current_evaluation(self.evaluations)

            self.ants_population[i] = random.choice(self.crossover_operator.execute([self.ants_population[i],
                                                                      self.antlions_population[antlion_pos],
                                                                       self.elite_antlion]))

            self.ants_population[i] = self.mutation_operator.execute(self.ants_population[i])

            self.ants_population[i] = self.evaluator.evaluate([self.ants_population[i]],self.problem)[0]
            self.update_progress()

            #Verify if the new ant is better than the selected antlion
            if self.ants_population[i].objectives[0] < self.antlions_population[antlion_pos].objectives[0]:
                aux = copy.deepcopy(self.antlions_population[antlion_pos])
                self.antlions_population[antlion_pos] = copy.deepcopy(self.ants_population[i])
                self.ants_population[i] = aux
                self.set_elite_antlion()

    def update_progress(self) -> None:
        self.evaluations += 1

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def get_result(self):
        return self.elite_antlion

    def get_name(self) -> str:
        return "Float Antlion Optimization"

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met