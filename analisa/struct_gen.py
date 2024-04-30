# Structure generator
#
# @author: Fredy Gabriel Ramírez Villanueva
# Starting this code at: 2024 / 04 / 29

import numpy as np
# from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LatticeBeam:

    a: float = 10  # left dimension
    b: float = 10  # right dimension
    nL: int = 2  # number of sections to the left
    nr: int = 2  # number of sections to the right
    dL: float = 1  # left height of the lattice beam
    dr: float = 1  # right height of the lattice beam
    hL: float = 0  # height of the left support
    hm: float = 2  # height of the top
    hr: float = 0  # height of the right support

    def lslope(self):
        '''Left slope of the floor.
        Can be positive or negative.
        '''
        return (self.hm - self.hL) / self.a

    def rslope(self):
        '''Right slope of the floor.
        Can be positive or negative.
        '''
        return (self.hr - self.hm) / self.b

    def lversor(self):
        '''Left versor'''
        x = self.a
        y = self.hm - self.hL
        v = np.array([x, y])
        return v / np.linalg.norm(v)

    def rversor(self):
        '''Right versor'''
        x = self.a
        y = self.hr - self.hm
        v = np.array([x, y])
        return v / np.linalg.norm(v)

    def left_roof(self):
        '''Roof left side lenght'''
        x = self.a
        y = self.hm - self.hL
        v = np.array([x, y])
        return np.linalg.norm(v)

    def right_roof(self):
        '''Roof right side lenght'''
        x = self.a
        y = self.hr - self.hm
        v = np.array([x, y])
        return np.linalg.norm(v)

    def left_step(self):
        '''Distance between nodes in long. bar at the left side of the roof'''
        return self.left_roof() / self.nL

    def right_step(self):
        '''Distance between nodes in long. bar at the right side of the roof'''
        return self.right_roof() / self.nr

    def mix_lists(list1, list2):
        final_list = []  # List to save the result
        for i in range(len(list2)):  # We go to the end of the shortest list
            final_list.append(list1[i])  # Add the element of list1
            final_list.append(list2[i])  # Add element of list2
        final_list.append(list1[-1])  # Add the last element of list1
        return final_list

    def builder(self):
        '''Returns nodes coordinates and the connectivity table'''

        pL = self.left_step()
        pr = self.right_step()
        rot = np.array([  # -90 degrees rotation matrix
            [0, 1],
            [-1, 0]
        ])

        A1 = np.array([0, self.hL])  # leftest node of the lattice beam
        B1 = A1 + pL/2 * self.lversor() + self.dL * (rot @ self.lversor())

        A2 = np.array([self.a + self.b + self.hr])  # rightest node
        B2 = A2 - pr/2 * self.rversor() + self.dr * (rot @ self.lversor())

        # Left roof
        upper_left = [A1 + i*pL*self.lversor() for i in range(self.nL)]
        lower_left = [B1 + i*pL*self.lversor() for i in range(self.nL - 1)]
        left = self.mix_lists(upper_left, lower_left)

        # Right roof
        upper_right = [A2 - i*pr*self.rversor() for i in range(self.nr)]
        lower_right = [B2 - i*pL*self.rversor() for i in range(self.nr - 1)]
        right = self.mix_lists(upper_right, lower_right)
        right = right[::-1][1:]

        # Combine
        nodes_coords = left + right

        return nodes_coords


if __name__ == "__main__":

    cab = LatticeBeam()
    nodes = cab.builder()
    print(nodes)
