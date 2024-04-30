# Structure generator
#
# @author: Fredy Gabriel Ramírez Villanueva
# Starting this code at: 2024 / 04 / 29

import numpy as np
# from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class LatticeBeam:

    # For node coordinates
    a: float = 10  # left dimension
    b: float = 10  # right dimension
    nL: int = 2  # number of sections to the left
    nr: int = 3  # number of sections to the right
    dL: float = 1  # left height of the lattice beam
    dr: float = 1  # right height of the lattice beam
    hL: float = 0  # height of the left support
    hm: float = 2  # height of the top
    hr: float = 0  # height of the right support

    # For conectivity table
    long_mat: int = 1  # longitudinal rod material number
    long_sec: int = 1  # longitudinal rod section number
    diag_mat: int = 1  # diagonal rod material number
    diag_sec: int = 1  # diagonal rod section number

    def lslope(self) -> float:
        '''Left slope of the floor.
        Can be positive or negative.
        '''
        return (self.hm - self.hL) / self.a

    def rslope(self) -> float:
        '''Right slope of the floor.
        Can be positive or negative.
        '''
        return (self.hr - self.hm) / self.b

    def lversor(self) -> np.ndarray:
        '''Left versor'''
        x = self.a
        y = self.hm - self.hL
        v = np.array([x, y])
        return v / np.linalg.norm(v)

    def rversor(self) -> np.ndarray:
        '''Right versor'''
        x = self.b
        y = self.hr - self.hm
        v = np.array([x, y])
        return v / np.linalg.norm(v)

    def left_roof(self) -> float:
        '''Roof left side lenght'''
        x = self.a
        y = self.hm - self.hL
        v = np.array([x, y])
        return np.linalg.norm(v)

    def right_roof(self) -> float:
        '''Roof right side lenght'''
        x = self.b
        y = self.hr - self.hm
        v = np.array([x, y])
        return np.linalg.norm(v)

    def left_step(self) -> float:
        '''Distance between nodes in long. bar at the left side of the roof'''
        return self.left_roof() / self.nL

    def right_step(self) -> float:
        '''Distance between nodes in long. bar at the right side of the roof'''
        return self.right_roof() / self.nr

    def mix_lists(self, list1, list2) -> list:
        final_list = []  # List to save the result
        for i in range(len(list2)):  # We go to the end of the shortest list
            final_list.append(list1[i])  # Add the element of list1
            final_list.append(list2[i])  # Add element of list2
        final_list.append(list1[-1])  # Add the last element of list1
        return final_list

    def coords(self) -> List[Tuple]:
        '''Returns nodes coordinates and the connectivity table'''

        pL = self.left_step()
        pr = self.right_step()
        lv = self.lversor()
        rv = self.rversor()

        rot = np.array([  # -90 degrees rotation matrix
            [0, 1],
            [-1, 0]
        ])

        lvp = rot @ lv
        rvp = rot @ rv

        A1 = np.array([0, self.hL])  # leftest node of the lattice beam
        B1 = A1 + pL/2 * lv + self.dL * lvp

        A2 = np.array([self.a + self.b, self.hr])  # rightest node
        B2 = A2 - pr/2 * rv + self.dr * rvp

        # Left roof
        upper_left = [A1 + i*pL*lv for i in range(self.nL + 1)]
        lower_left = [B1 + i*pL*lv for i in range(self.nL)]
        left = self.mix_lists(upper_left, lower_left)

        # Right roof
        upper_right = [A2 - i*pr*rv for i in range(self.nr + 1)]
        lower_right = [B2 - i*pr*rv for i in range(self.nr)]
        right = self.mix_lists(upper_right, lower_right)
        right = right[::-1][1:]

        # Combine
        coords = left + right

        return list(map(tuple, coords))

    def builder(self) -> List[Tuple]:
        '''Returns'''
        coords = self.coords()
        nnodes = len(coords)

        diags = [(i, i+1, self.diag_mat, self.diag_sec)
                 for i in range(nnodes - 1)]
        long_sup = [(2*k, 2*k+2, self.long_mat, self.long_sec)
                    for k in range(self.nL + self.nr)]
        long_inf = [(2*k+1, 2*k+3, self.long_mat, self.long_sec)
                    for k in range(self.nL + self.nr - 1)]

        conect = long_sup + long_inf + diags

        return coords, conect


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    cab = LatticeBeam(a=17.66, b=17.66, nL=30, nr=30, dL=1, dr=1, hL=7,
                      hm=9.65, hr=7)
    cab.builder()
    nodes, conect = cab.builder()
    X, Y = zip(*nodes)
    plt.scatter(X, Y)
    for b in conect:
        x = (X[b[0]], X[b[1]])
        y = (Y[b[0]], Y[b[1]])
        plt.plot(x, y, 'b')
    plt.axis('scaled')
    plt.grid()
    plt.show()
