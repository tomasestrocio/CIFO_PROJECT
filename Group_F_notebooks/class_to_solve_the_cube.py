# Importing necessary libraries
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon



# -------------------------------------------
# Class to Solve the Cube
# -------------------------------------------
class RubiksCubeSolver:
    def __init__(self, scramble_moves=10):
        #Initializes the solver with a scrambled cube
        self.scramble_cube, self.scramble_moves = self.scramble(scramble_moves)

    def scramble(self, moves_count):
        #"Scrambles the cube with a given number of random moves
        cube = Cube()
        moves = [random.randint(1, 18) for _ in range(moves_count)]      # Generate random moves
        for move in moves:
            cube.apply_move(move)                                        # Apply each move to the cube
        return cube, moves

    def solve(self):
        #Solves the cube using GA + Local Search + Greedy
        print("Scramble moves", self.scramble_moves)
        self.scramble_cube.print_cube()

        # Initialize the genetic algorithm
        ga = GeneticAlgorithm(
            scramble_cube=self.scramble_cube,
            mutation_type='swap',
            crossover_type='two_point',
            selection_type='tournament'
                )
        solution, generations, fitness_progress = ga.run()                 # Run the genetic algorithm

        print("\nGA Solution::", solution)
        cube_after_ga = self.scramble_cube.copy()
        cube_after_ga.apply_sequence(solution)                             # Apply the GA solution to the cube

        # Perform local search improvement
        improved_solution = self.local_improvement(solution, iterations=50)
        print("\nAfter local search:", improved_solution)

        cube_after_local = self.scramble_cube.copy()
        cube_after_local.apply_sequence(improved_solution)                 # Apply the improved solution

        # Perform greedy improvement
        greedy_seq, final_fit = self.greedy_improvement(cube_after_local)
        print("\nAfter greedy search", greedy_seq)

        # Combine all solutions and apply them to the cube
        final_sequence = improved_solution + greedy_seq
        final_cube = self.scramble_cube.copy()
        final_cube.apply_sequence(final_sequence)

        print("\nFinal cube state:")
        final_cube.print_cube()
        print("Final fitness:", final_cube.fitness())

         # Plot the fitness evolution
        self.plot_fitness(fitness_progress)

    def local_improvement(self, chromosome, iterations=50):
        #Local search improvement: modifies 1, 2, or 3 genes at a time
        best_chrom = chromosome[:]
        cube_copy = self.scramble_cube.copy()
        cube_copy.apply_sequence(best_chrom)                  # Apply the initial chromosome
        best_fit = cube_copy.fitness()
        chrom_length = len(best_chrom)

        if chrom_length < 3:                                 # If the chromosome is too short, return it unchanged
            return best_chrom

        for _ in range(iterations):
            improved = False
            for i in range(chrom_length):
                if i >= len(best_chrom):
                    continue

                # Test modification of 1 gene
                candidate = best_chrom[:]
                candidate[i] = random.randint(1, 18)        # Replace one move
                candidate = GeneticAlgorithm.clean_chromosome(candidate)    # Clean invalid moves
                if len(candidate) < 1:
                    continue
                cube_candidate = self.scramble_cube.copy()
                cube_candidate.apply_sequence(candidate)
                if cube_candidate.fitness() < best_fit:     # Update if fitness improves
                    best_chrom = candidate
                    best_fit = cube_candidate.fitness()
                    improved = True

                # Test modification of 2 genes
                for j in range(i+1, chrom_length):
                    if j >= len(best_chrom):
                        continue
                    candidate2 = best_chrom[:]
                    candidate2[i] = random.randint(1, 18)
                    candidate2[j] = random.randint(1, 18)
                    candidate2 = GeneticAlgorithm.clean_chromosome(candidate2)
                    if len(candidate2) < 2:
                        continue
                    cube_candidate2 = self.scramble_cube.copy()
                    cube_candidate2.apply_sequence(candidate2)
                    if cube_candidate2.fitness() < best_fit:
                        best_chrom = candidate2
                        best_fit = cube_candidate2.fitness()
                        improved = True

                     # Test modification of 3 genes
                    for k in range(j+1, chrom_length):
                        if k >= len(best_chrom):
                            continue
                        candidate3 = best_chrom[:]
                        candidate3[i] = random.randint(1, 18)
                        candidate3[j] = random.randint(1, 18)
                        candidate3[k] = random.randint(1, 18)
                        candidate3 = GeneticAlgorithm.clean_chromosome(candidate3)
                        if len(candidate3) < 3:
                            continue
                        cube_candidate3 = self.scramble_cube.copy()
                        cube_candidate3.apply_sequence(candidate3)
                        if cube_candidate3.fitness() < best_fit:
                            best_chrom = candidate3
                            best_fit = cube_candidate3.fitness()
                            improved = True

            if not improved:
                break  # Stop early if no improvement
        return best_chrom

    def greedy_improvement(self, cube, max_iters=100):
        # Greedy search (1, 2, or 3 steps)
        sequence = []
        fit = cube.fitness()
        for _ in range(max_iters):
            best_sequence = []
            best_fit = fit

            # Test all 1-move sequences
            for move1 in range(1, 19):
                candidate1 = cube.copy()
                candidate1.apply_move(move1)
                fit1 = candidate1.fitness()
                if fit1 < best_fit:             # Update if fitness improves
                    best_sequence = [move1]
                    best_fit = fit1

                 # Test all 2-move sequences
                for move2 in range(1, 19):
                    candidate2 = candidate1.copy()
                    candidate2.apply_move(move2)
                    fit2 = candidate2.fitness()
                    if fit2 < best_fit:
                        best_sequence = [move1, move2]
                        best_fit = fit2

                    # Test all 3-move sequences
                    for move3 in range(1, 19):
                        candidate3 = candidate2.copy()
                        candidate3.apply_move(move3)
                        fit3 = candidate3.fitness()
                        if fit3 < best_fit:
                            best_sequence = [move1, move2, move3]
                            best_fit = fit3

            if best_sequence:
                for move in best_sequence:
                    cube.apply_move(move)       # Apply the best sequence
                    sequence.append(move)
                fit = best_fit
            else:
                break           # Stop if no improvement    
        return sequence, fit

    def plot_fitness(self, fitness_progress):
        # Plots the fitness evolution over generations
        plt.figure(figsize=(8,5))
        plt.plot(range(len(fitness_progress)), fitness_progress, marker='o')
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.title("Fitness Evolution (GA)")
        plt.grid(True)
        plt.show()
