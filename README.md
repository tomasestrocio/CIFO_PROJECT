# CIFO_PROJECT
 This study presents a comprehensive genetic-algorithm approach to solving the 3×3 Rubik’s
 Cube, whose approximately 4.30 × 1019 reachable configurations pose an immense search challenge.
 We extend prior work by developing a flexible cube representation that supports all eighteen face
 and slice moves and by implementing interchangeable crossover, mutation and selection operators
 with automatic elimination of redundant moves.
 After the evolutionary phase, candidate solutions undergo a multigene local search followed by a
 greedy look-ahead refinement. We systematically evaluate every combination of mutation, crossover
 and selection strategies via grid search, then employ random search to tune hyperparameters includ
ing population size, chromosome length, crossover and mutation probabilities, elitism proportion and
 generation limit.
 Experiments comprising thirty independent runs for each of twelve operator schemes identify an
 optimal configuration—200 individuals, 30-move chromosomes, 60 % crossover probability, 30 % mu
tation probability, 1 % elitism, segment mutation, two-point crossover and tournament selection over
 1500 generations. Wilcoxon signed-rank tests confirm that both refinement phases yield significant
 performance gains. Although only two trials achieved perfect solutions for ten-move scrambles, the
 algorithm consistently reduces scramble disorder, underscoring the promise of genetic algorithms as
 domain-agnostic solvers for complex combinatorial challenges.
