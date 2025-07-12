
# 🧬 Solving the 3×3 Rubik’s Cube with Genetic Algorithms

## 📘 Overview

This study presents a **comprehensive genetic algorithm approach** to solving the 3×3 Rubik’s Cube, whose approximately **4.30 × 10¹⁹** reachable configurations pose an immense search challenge.

We extend prior work by:

- Developing a flexible cube representation that supports **all eighteen face and slice moves**
- Implementing **interchangeable crossover, mutation, and selection operators**
- Automatically **eliminating redundant moves**

---

## 🧠 Solution Strategy

After the evolutionary phase, candidate solutions go through two refinement steps:

1. **Multigene Local Search**
2. **Greedy Look-Ahead Refinement**

We conduct a **systematic evaluation** of every combination of mutation, crossover, and selection strategies using **grid search**, followed by **random search** to tune hyperparameters including:

- Population size  
- Chromosome length  
- Crossover and mutation probabilities  
- Elitism proportion  
- Generation limit  

---

## 🔬 Experimental Setup

- **30 independent runs** for each of 12 operator schemes  
- Grid and random search for operator and hyperparameter tuning

---

## ✅ Optimal Configuration

The best-performing setup was:

- **Population Size**: 200  
- **Chromosome Length**: 30 moves  
- **Crossover Probability**: 60%  
- **Mutation Probability**: 30%  
- **Elitism**: 1%  
- **Mutation Operator**: Segment Mutation  
- **Crossover Operator**: Two-Point Crossover  
- **Selection Method**: Tournament Selection  
- **Generations**: 1500  

---

## 📈 Results

- **Wilcoxon signed-rank tests** confirmed that both refinement phases yielded significant performance improvements
- Only **two trials** achieved perfect solutions for ten-move scrambles
- However, the algorithm **consistently reduced scramble disorder**

---

## 🧩 Conclusion

While perfect solves were rare, the consistent improvement in scramble state **demonstrates the promise of genetic algorithms** as powerful, domain-agnostic solvers for complex **combinatorial challenges** like the Rubik’s Cube.

