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
# Class that represents the Cube
# -------------------------------------------
class Cube:
    move_functions = {}

    def __init__(self):
        #Initializes the solved cube
        self.state = {
            "U": np.full((3, 3), 1),     #Upper face
            "R": np.full((3, 3), 2),     #Right face
            "F": np.full((3, 3), 3),     #Front face
            "D": np.full((3, 3), 4),     #Down face
            "L": np.full((3, 3), 5),     #Left face
            "B": np.full((3, 3), 6)      #Back face
        }

    def copy(self):
        #Returns a copy of the cube
        new_cube = Cube()
        new_cube.state = {face: self.state[face].copy() for face in self.state}
        return new_cube

    def rotate_face(self, face, clockwise=True):
        #Rotates the face clockwise or counterclockwise
        self.state[face] = np.rot90(self.state[face], -1 if clockwise else 1)

    def apply_move(self, move):
        #Executes the corresponding move function
        Cube.move_functions[move](self)

    def apply_sequence(self, sequence):
         # Iterates over the sequence and applies each move
        for move in sequence:
            self.apply_move(move)

    def fitness(self):
       #Calculates the fitness of the cube 
        penalty = 0
        for face in ["U", "R", "F", "D", "L", "B"]:
            center = self.state[face][1, 1]      # Center of the face
            # Counts the facelets out of place
            for i in range(3):
                for j in range(3):
                    if (i, j) != (1, 1) and self.state[face][i, j] != center:
                        penalty += 1
        return penalty

    def print_cube(self):
        for face in ["U", "R", "F", "D", "L", "B"]:
            print(f"Face {face}:\n{self.state[face]}\n")

    @staticmethod
    def init_moves():
        #Rotates a given face of the cube and returns the rotated face
        def rotate_face(face, clockwise=True):
            return np.rot90(face, -1 if clockwise else 1)

        def move_U(cube):
            cube.rotate_face('U', True)
            temp = cube.state["F"][0, :].copy()
            cube.state["F"][0, :] = cube.state["R"][0, :].copy()
            cube.state["R"][0, :] = cube.state["B"][0, :].copy()
            cube.state["B"][0, :] = cube.state["L"][0, :].copy()
            cube.state["L"][0, :] = temp

        def move_Ui(cube):
            cube.rotate_face('U', False)
            temp = cube.state["F"][0, :].copy()
            cube.state["F"][0, :] = cube.state["L"][0, :].copy()
            cube.state["L"][0, :] = cube.state["B"][0, :].copy()
            cube.state["B"][0, :] = cube.state["R"][0, :].copy()
            cube.state["R"][0, :] = temp

        def move_D(cube):
            cube.rotate_face('D', True)
            temp = cube.state["F"][2, :].copy()
            cube.state["F"][2, :] = cube.state["L"][2, :].copy()
            cube.state["L"][2, :] = cube.state["B"][2, :].copy()
            cube.state["B"][2, :] = cube.state["R"][2, :].copy()
            cube.state["R"][2, :] = temp

        def move_Di(cube):
            cube.rotate_face('D', False)
            temp = cube.state["F"][2, :].copy()
            cube.state["F"][2, :] = cube.state["R"][2, :].copy()
            cube.state["R"][2, :] = cube.state["B"][2, :].copy()
            cube.state["B"][2, :] = cube.state["L"][2, :].copy()
            cube.state["L"][2, :] = temp

        def move_F(cube):
            cube.rotate_face('F', True)
            temp = cube.state["U"][2, :].copy()
            cube.state["U"][2, :] = np.flip(cube.state["L"][:, 2].copy())
            cube.state["L"][:, 2] = cube.state["D"][0, :].copy()
            cube.state["D"][0, :] = np.flip(cube.state["R"][:, 0].copy())
            cube.state["R"][:, 0] = temp

        def move_Fi(cube):
            cube.rotate_face('F', False)
            temp = cube.state["U"][2, :].copy()
            cube.state["U"][2, :] = cube.state["R"][:, 0].copy()
            cube.state["R"][:, 0] = np.flip(cube.state["D"][0, :].copy())
            cube.state["D"][0, :] = cube.state["L"][:, 2].copy()
            cube.state["L"][:, 2] = np.flip(temp)

        def move_B(cube):
            cube.rotate_face('B', True)
            temp = cube.state["U"][0, :].copy()
            cube.state["U"][0, :] = cube.state["R"][:, 2].copy()[::-1]
            cube.state["R"][:, 2] = cube.state["D"][2, :].copy()
            cube.state["D"][2, :] = cube.state["L"][:, 0].copy()[::-1]
            cube.state["L"][:, 0] = temp

        def move_Bi(cube):
            cube.rotate_face('B', False)
            temp = cube.state["U"][0, :].copy()
            cube.state["U"][0, :] = cube.state["L"][:, 0].copy()
            cube.state["L"][:, 0] = np.flip(cube.state["D"][2, :].copy())
            cube.state["D"][2, :] = cube.state["R"][:, 2].copy()
            cube.state["R"][:, 2] = np.flip(temp)

        def move_L(cube):
            cube.rotate_face('L', True)
            temp = cube.state["U"][:, 0].copy()
            cube.state["U"][:, 0] = cube.state["B"][:, 2].copy()[::-1]
            cube.state["B"][:, 2] = np.flip(cube.state["D"][:, 0].copy())
            cube.state["D"][:, 0] = cube.state["F"][:, 0].copy()
            cube.state["F"][:, 0] = temp

        def move_Li(cube):
            cube.rotate_face('L', False)
            temp = cube.state["U"][:, 0].copy()
            cube.state["U"][:, 0] = cube.state["F"][:, 0].copy()
            cube.state["F"][:, 0] = cube.state["D"][:, 0].copy()
            cube.state["D"][:, 0] = np.flip(cube.state["B"][:, 2].copy())
            cube.state["B"][:, 2] = np.flip(temp)

        def move_R(cube):
            cube.rotate_face('R', True)
            temp = cube.state["U"][:, 2].copy()
            cube.state["U"][:, 2] = cube.state["F"][:, 2].copy()
            cube.state["F"][:, 2] = cube.state["D"][:, 2].copy()
            cube.state["D"][:, 2] = np.flip(cube.state["B"][:, 0].copy())
            cube.state["B"][:, 0] = np.flip(temp)

        def move_Ri(cube):
            cube.rotate_face('R', False)
            temp = cube.state["U"][:, 2].copy()
            cube.state["U"][:, 2] = np.flip(cube.state["B"][:, 0].copy())
            cube.state["B"][:, 0] = np.flip(cube.state["D"][:, 2].copy())
            cube.state["D"][:, 2] = cube.state["F"][:, 2].copy()
            cube.state["F"][:, 2] = temp

        def move_M(cube):
            temp = cube.state["U"][:, 1].copy()
            cube.state["U"][:, 1] = cube.state["F"][:, 1].copy()
            cube.state["F"][:, 1] = cube.state["D"][:, 1].copy()
            cube.state["D"][:, 1] = np.flip(cube.state["B"][:, 1].copy())
            cube.state["B"][:, 1] = np.flip(temp)

        def move_Mi(cube):
            temp = cube.state["U"][:, 1].copy()
            cube.state["U"][:, 1] = np.flip(cube.state["B"][:, 1].copy())
            cube.state["B"][:, 1] = np.flip(cube.state["D"][:, 1].copy())
            cube.state["D"][:, 1] = cube.state["F"][:, 1].copy()
            cube.state["F"][:, 1] = temp

        def move_E(cube):
            temp = cube.state["F"][1, :].copy()
            cube.state["F"][1, :] = cube.state["R"][1, :].copy()
            cube.state["R"][1, :] = cube.state["B"][1, :].copy()
            cube.state["B"][1, :] = cube.state["L"][1, :].copy()
            cube.state["L"][1, :] = temp

        def move_Ei(cube):
            temp = cube.state["F"][1, :].copy()
            cube.state["F"][1, :] = cube.state["L"][1, :].copy()
            cube.state["L"][1, :] = cube.state["B"][1, :].copy()
            cube.state["B"][1, :] = cube.state["R"][1, :].copy()
            cube.state["R"][1, :] = temp

        def move_S(cube):
            temp = cube.state["U"][1, :].copy()
            cube.state["U"][1, :] = np.flip(cube.state["L"][:, 1].copy())
            cube.state["L"][:, 1] = cube.state["D"][1, :].copy()
            cube.state["D"][1, :] = np.flip(cube.state["R"][:, 1].copy())
            cube.state["R"][:, 1] = temp

        def move_Si(cube):
            temp = cube.state["U"][1, :].copy()
            cube.state["U"][1, :] = cube.state["R"][:, 1].copy()
            cube.state["R"][:, 1] = np.flip(cube.state["D"][1, :].copy())
            cube.state["D"][1, :] = cube.state["L"][:, 1].copy()
            cube.state["L"][1, :] = np.flip(temp)

        # Dictionary that maps the move number to the corresponding function 
        Cube.move_functions = {
            1: move_U, 2: move_Ui, 3: move_D, 4: move_Di,
            5: move_F, 6: move_Fi, 7: move_B, 8: move_Bi,
            9: move_L, 10: move_Li, 11: move_R, 12: move_Ri,
            13: move_M, 14: move_Mi, 15: move_E, 16: move_Ei,
            17: move_S, 18: move_Si
        }

