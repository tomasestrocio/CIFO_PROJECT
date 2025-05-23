# -------------------------------------------
# Importing necessary libraries
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon

# Class for the Genetic Algorithm
# -------------------------------------------
class GeneticAlgorithm:
    inverses = {1:2, 2:1, 3:4, 4:3, 5:6, 6:5, 7:8, 8:7,
                9:10, 10:9, 11:12, 12:11, 13:14, 14:13,
                15:16, 16:15, 17:18, 18:17}

    def __init__(self, scramble_cube, population_size=100, chromosome_length=30, 
                 crossover_rate=0.8, mutation_rate=0.2, elitism_rate=0.10, 
                 max_generations=1500, 
                 mutation_type='random', crossover_type='one_point', selection_type='roulette'):
        self.scramble_cube = scramble_cube
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.crossover_rate = crossover_rate
        self.mutation_rate_base = mutation_rate
        self.elitism_rate = elitism_rate
        self.max_generations = max_generations
        self.mutation_type = mutation_type    # 'simple', 'swap', 'segment', 'random'
        self.crossover_type = crossover_type  # 'one_point' or 'two_point'
        self.selection_type = selection_type  # 'roulette' or 'tournament'

    def is_inverse(self, move1, move2):                 #Checks if two moves are inverses of each other
        return self.inverses.get(move1) == move2

    def choose_random_move(self, previous_move=None):   #Chooses a random move, avoiding the inverse of the previous move
        move = random.randint(1, 18)
        while previous_move is not None and self.is_inverse(previous_move, move):
            move = random.randint(1, 18)
        return move

    def generate_random_chromosome(self):               #Generates a random chromosome of moves
        chrom = []
        for i in range(self.chromosome_length):
            prev = chrom[i-1] if i > 0 else None
            chrom.append(self.choose_random_move(prev))
        return chrom

    def clean_chromosome(chromosome):                   #Cleans the chromosome by removing inverse moves
        if not chromosome:
            return []
        new_chrom = [chromosome[0]]
        inverses = GeneticAlgorithm.inverses
        for move in chromosome[1:]:
            if new_chrom and inverses.get(new_chrom[-1]) == move:
                new_chrom.pop()
            else:
                new_chrom.append(move)
        return new_chrom

    def initial_population(self):                       #Generates the initial population of chromosomes
        return [self.generate_random_chromosome() for _ in range(self.population_size)]

    def evaluate_population(self, population):          #Evaluates the fitness of each chromosome in the population
        fitnesses = []
        for chromosome in population:
            cube_copy = self.scramble_cube.copy()
            cube_copy.apply_sequence(chromosome)
            fitnesses.append(cube_copy.fitness())
        return fitnesses

    def roulette_wheel_selection(self, population, fitnesses):  #Selects a parent using roulette wheel selection
        epsilon = 1e-6
        max_fit = max(fitnesses) + epsilon
        scores = [max_fit - f for f in fitnesses]
        total = sum(scores)
        pick = random.uniform(0, total)
        current = 0
        for chrom, score in zip(population, scores):
            current += score
            if current >= pick:
                return chrom
        return population[-1]

    def tournament_selection(self, population, fitnesses, k=3): #Selects a parent using tournament selection
        selected = random.sample(list(zip(population, fitnesses)), k)
        selected.sort(key=lambda x: x[1])
        return selected[0][0]

    def select_parent(self, population, fitnesses):             #Selects a parent based on the configured selection type
        if self.selection_type == 'roulette':
            return self.roulette_wheel_selection(population, fitnesses)
        elif self.selection_type == 'tournament':
            return self.tournament_selection(population, fitnesses)

    def one_point_crossover(self, parent1, parent2):            # Perform one-point crossover between two parents
        if len(parent1) < 2 or len(parent2) < 2:                # If parents are too short, return them unchanged
            return parent1[:], parent2[:]
        point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

    def two_point_crossover(self, parent1, parent2):            # Perform two-point crossover between two parents
        if len(parent1) < 3 or len(parent2) < 3:
            return parent1[:], parent2[:]
        p1, p2 = sorted(random.sample(range(1, min(len(parent1), len(parent2))), 2))    # Two random points
        child1 = parent1[:p1] + parent2[p1:p2] + parent1[p2:]
        child2 = parent2[:p1] + parent1[p1:p2] + parent2[p2:]
        return child1, child2

    def crossover(self, parent1, parent2):                      # Select and perform the specified crossover type
        if self.crossover_type == 'one_point':
            return self.one_point_crossover(parent1, parent2)
        elif self.crossover_type == 'two_point':
            return self.two_point_crossover(parent1, parent2)

    def mutate(self, chromosome, mutation_rate):                # Apply mutation to a chromosome based on the mutation type
        mtype = self.mutation_type
        if mtype == 'random':           # Randomly choose a mutation type
            mtype = random.choice(['simple', 'swap', 'segment'])

        if mtype == 'simple':           # Simple mutation: replace moves with random ones
            for i in range(len(chromosome)):
                if random.random() < mutation_rate:
                    prev = chromosome[i-1] if i > 0 else None
                    chromosome[i] = self.choose_random_move(prev)

        elif mtype == 'swap':           # Swap mutation: swap two random moves
            if len(chromosome) > 1:
                i, j = random.sample(range(len(chromosome)), 2)
                chromosome[i], chromosome[j] = chromosome[j], chromosome[i]

        elif mtype == 'segment':        # Segment mutation: reverse a random segment
            if len(chromosome) > 2:
                i, j = sorted(random.sample(range(len(chromosome)), 2))
                chromosome[i:j+1] = reversed(chromosome[i:j+1])

        return chromosome

    def run(self):
        # Main loop for the genetic algorithm
        population = self.initial_population()
        best_solution = None
        best_fit = float('inf')        # Initialize best fitness
        generation = 0
        fitness_progress = []
        mutation_rate = self.mutation_rate_base
        no_improve_counter = 0

        while generation < self.max_generations:
            fitnesses = self.evaluate_population(population)        # Evaluate fitness of population
            current_best_fit = min(fitnesses)                       # Find best fitness in current generation
            current_best_index = fitnesses.index(current_best_fit)

            if current_best_fit < best_fit:                         # Update best solution if improvement
                best_fit = current_best_fit
                best_solution = population[current_best_index][:]
                no_improve_counter = 0
            else:
                no_improve_counter += 1

            fitness_progress.append(best_fit)                       # Track progress

            if best_fit == 0:                                       # Stop if perfect solution is found
                print(f"Solution found in generation {generation}")
                return best_solution, generation, fitness_progress

            if no_improve_counter >= 50:                            # Increase mutation rate if no improvement
                mutation_rate = min(mutation_rate * 1.2, 1.0)
                no_improve_counter = 0
            else:
                mutation_rate = self.mutation_rate_base

            new_population = []
            num_elite = int(self.elitism_rate * self.population_size)
            sorted_population = [chrom for _, chrom in sorted(zip(fitnesses, population), key=lambda x: x[0])]
            new_population.extend(sorted_population[:num_elite])

            while len(new_population) < self.population_size:
                parent1 = self.select_parent(population, fitnesses) # Select first parent
                parent2 = self.select_parent(population, fitnesses) # Select second parent
                if random.random() < self.crossover_rate:           # Perform crossover
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1[:], parent2[:]
                child1 = self.mutate(child1, mutation_rate)         # Mutate first child
                child2 = self.mutate(child2, mutation_rate)         # Mutate second child
                child1 = GeneticAlgorithm.clean_chromosome(child1)
                child2 = GeneticAlgorithm.clean_chromosome(child2)

                new_population.append(child1)                       # Add first child to new population
                if len(new_population) < self.population_size:      # Add second child if space remains
                    new_population.append(child2)
            population = new_population                              # Update population
            generation += 1                                          # Increment generation

        return best_solution, generation, fitness_progress