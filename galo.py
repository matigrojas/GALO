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
            #while antlion is None:
            #    antlion,antlion_pos = self.random_walk_selection_operator.execute(self.antlions_population)

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

        #self.combine()

    def combine(self):
        combination = copy.deepcopy(self.ants_population)
        combination.extend(copy.deepcopy(self.antlions_population))
        combination.sort(key=lambda s: s.objectives[0])
        self.antlions_population = combination[0:self.number_of_antlions]
        self.ants_population = combination[self.number_of_antlions:]

    def update_progress(self) -> None:
        self.evaluations += 1

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def get_result(self) -> R:
        return self.elite_antlion

    def get_name(self) -> str:
        return "Float Antlion Optimization"

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met


class MicroGAFFNN():

    def __init__(self,
                 problem,
                 mutation,
                 crossover,
                 selection,
                 offspring_population_size: int = 6,
                 population_size: int = 5,
                 max_evaluations: int = 1000,
                 freq: int = 100):
        self.problem=problem
        
        self.population_size=population_size
        self.offspring_population_size=offspring_population_size
        
        self.mutation_operator=mutation
        self.crossover_operator=crossover
        self.selection_operator=selection
        
        self.best_solution = None
        self.evaluations = 0
        self.max_evaluations = max_evaluations

        self.mating_pool_size = \
            self.offspring_population_size * \
            self.crossover_operator.get_number_of_parents() // self.crossover_operator.get_number_of_children()
        
        self.start_computing_time = 0
        self.total_computing_time = 0

        self.imp = 0
        self.freq = freq

        self.history = []

    def create_initial_solutions(self):
        population = [self.problem.create_solution()
            for _ in range(self.population_size)]
        return population

    def init_progress(self) -> None:
        self.evaluations = self.population_size

        n = self.problem.get_input_size()
        weights_l1 = (n**2)*2+n
        weights_l2 = n*2+1
        biases = n*2+2
        self.crossover_operator.set_network_architecture(weights_l1,weights_l2,biases)

    def set_best_solution(self,best_solution):
        self.best_solution = best_solution

    def run(self):
        """ Execute the algorithm. """
        self.start_computing_time = time.time()

        self.solutions = self.create_initial_solutions()
        self.solutions = self.evaluate(self.solutions)

        self.init_progress()

        print(f"Progress: {self.evaluations}/{self.max_evaluations}, Fitness: {self.get_result().objectives[0]}")
        self.history.append(self.get_result().objectives[0])

        while not self.termination_criterion_is_met():
            self.step()
            if math.floor(self.evaluations / self.freq) == self.imp:
                fitness = self.get_result().objectives[0]#np.mean([x.objectives[0] for x in self.solutions])
                print(f"Progress: {self.evaluations}/{self.max_evaluations}, Fitness: {fitness}")
                self.history.append(fitness)
                self.imp += 1
            

        self.total_computing_time = time.time() - self.start_computing_time    

    def step(self):
        mating_population = self.selection(self.solutions)
        offspring_population = self.reproduction(mating_population)
        offspring_population = self.evaluate(offspring_population)
        self.evaluations += self.offspring_population_size

        self.solutions = self.replacement(self.solutions, offspring_population)

        if not self.termination_criterion_is_met():
            if self.is_nominal_convergence(self.solutions):
                self.solutions = self.reset_population(self.solutions)

    def selection(self, population):
        mating_population = []

        for _ in range(self.mating_pool_size):
            solution = self.selection_operator.execute(population)
            mating_population.append(solution)

        return mating_population

    def reproduction(self, mating_population):
        number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()

        #offspring_population = []
        for i in range(0, self.offspring_population_size, 2):
            parents = []
            for j in range(2):
                parents.append(mating_population[i + j])
            if len(parents) < number_of_parents_to_combine:
                parents.append(self.solutions[0])

            offspring = self.crossover_operator.execute(parents)

            if self.mutation_operator is not None:
                for solution in offspring:
                    self.mutation_operator.execute(solution)
                    #offspring_population.append(solution)
                    #if len(offspring_population) >= self.offspring_population_size:
                    #    break
            
        return offspring
    
    def replacement(self, population, offspring_population):
        population.extend(offspring_population)

        population.sort(key=lambda s: s.objectives[0])

        return population[:self.population_size]

    def is_nominal_convergence(self, solutions) -> bool:
        acum_mean = 0.0
        for sol in solutions[1:]:
            acum_mean += np.mean(np.abs(np.array(solutions[0].variables) - np.array(sol.variables)))
        return True if acum_mean <= 0.001 else False

    def reset_population(self, solutions):
        solutions = [solutions[0]]
        if not self.termination_criterion_is_met():
            for _ in range(1, self.population_size):
                if not self.termination_criterion_is_met():
                    solutions.append(self.evaluate([self.problem.create_solution()])[0])
                    self.evaluations += 1
                else:
                    break
        return solutions
    
    def evaluate(self, population: [] = None):
        for i in range(len(population)):
            population[i] = self.problem.evaluate(population[i])
        return population
    
    def termination_criterion_is_met(self):
        return self.evaluations >= self.max_evaluations
    
    def get_name(self) -> str:
        return 'Micro Genetic algorithm'
    
    def get_result(self):
        return self.solutions[0]
    
    def get_history(self):
        return self.history