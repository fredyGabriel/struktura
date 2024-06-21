# Structure generator
#
# @author: Fredy Gabriel Ramírez Villanueva
# Starting this code at: 2024 / 04 / 29

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class LatticeBeam:
    """Represents a lattice beam structure.

    Nodes a bars numbering starts on 1.

    Args:
        a: Left dimension of the beam.
        b: Right dimension of the beam.
        nL: Number of sections to the left.
        nr: Number of sections to the right.
        dL: Left height of the lattice beam
        dr: Right height of the lattice beam
        hL: Height of the left support
        hm: Height of the top
        hr: Height of the right support
        long_mat: Longitudinal rod material number
        long_sec: Longitudinal rod section number
        diag_mat: Diagonal rod material number
        diag_sec: Diagonal rod section number
        sup_vert_load: Surface vertical load. Positiva sign upwards.
        trus_sep: Distance between trusses
        nod_dist_purlins: Distance between purlins in nodes

    Example of use:
        To obtain the node coordinates an the conectivity table.
            > cab = LatticeBeam()
            > nodes, conect = cab.builder()
    """

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
    # Every count starts at zero
    long_mat: int = 0  # longitudinal rod material number
    long_sec: int = 0  # longitudinal rod section number
    diag_mat: int = 0  # diagonal rod material number
    diag_sec: int = 0  # diagonal rod section number

    # For loads
    sur_vert_load: float = 1.0  # Surface vertical load
    trus_sep: float = 5.0  # Distance between trusses
    nod_dist_purlins: int = 1  # Distance between purlins in nodes

    # Luego del builder
    node_coords: Optional[List[Tuple]] = None
    conectivity: Optional[List[Tuple]] = None
    vloads: Optional[dict] = None

    # For ploting
    title: str = 'Lattice Beam'

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

        A1 = np.array([0, self.hL])  # leftest node coords of the lattice beam
        B1 = A1 + pL/2 * lv + self.dL * lvp  # First node coords. at bottom

        A2 = np.array([self.a + self.b, self.hr])  # rightest coords. node
        B2 = A2 - pr/2 * rv + self.dr * rvp  # last node coords. at bottom

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
        self.node_coords = list(map(tuple, coords))

    def nknots(self):
        '''Number of nodes'''
        if self.node_coords is None:
            raise TypeError("First run coords()")
        else:
            return len(self.node_coords)

    def nbars(self):
        '''Number of bars'''
        if self.conectivity is None:
            raise TypeError("First run builder()")
        else:
            return len(self.conectivity)

    def top_nodes(self) -> list:
        '''List of top node numbers'''
        return [k for k in range(self.nknots()) if k % 2 == 0]

    def bottom_nodes(self) -> list:
        '''List of bottom node numbers'''
        return [k for k in range(self.nknots()) if k % 2 != 0]

    def nodal_vloads(self) -> dict:
        '''Vertical nodal loads at the top nodes.

        Args:
            q: surface load
            sep: separation of trusses
            each: every how many nodes a purlin is located

        Returns:
            Dictionary of type: {number_of_node: vertical_load, ...}
        '''
        nL = self.nL
        nr = self.nr
        q = self.sur_vert_load
        sep = self.trus_sep
        each = self.nod_dist_purlins

        pL = q*sep * self.a / nL * each
        pr = q*sep * self.b / nr * each

        # Nodes with loads
        top_left = self.top_nodes()[:nL]
        top_right = self.top_nodes()[nL:]
        nleft = [n for i, n in enumerate(top_left) if i % each == 0]
        vleft = [pL] * len(nleft)
        vleft[0] = pL / 2

        if nL % each != 0:
            nleft.append(self.top_nodes()[self.nL])

            # Load at ridge because left distribution
            vleft.append(pL/each * (nL % each))

        nright = [n for i, n in enumerate(top_right[::-1])
                  if i % each == 0][::-1]
        vright = [pr] * len(nright)

        # Adding load at ridge because right distribution
        if nr % each != 0:
            vleft[-1] += pr/each * (nr % each)
        else:
            vleft[-1] += pr
            nright = nright[1:]
            vright = vright[1:]

        keys = nleft + nright
        verticals = vleft + vright
        verticals[-1] = verticals[-1] / 2

        values = [(0, v) for v in verticals]

        self.vloads = dict(zip(keys, values))

    def builder(self) -> List[Tuple]:
        '''Build the truss with vertical loads'''
        # print("Getting nodal coordinates...")
        self.coords()
        nnodes = self.nknots()

        diags = [(i, i+1, self.diag_mat, self.diag_sec)
                 for i in range(nnodes - 1)]
        long_sup = [(2*k, 2*k+2, self.long_mat, self.long_sec)
                    for k in range(self.nL + self.nr)]
        long_inf = [(2*k+1, 2*k+3, self.long_mat, self.long_sec)
                    for k in range(self.nL + self.nr - 1)]

        # print('Building the connectivity table...')
        self.conectivity = long_sup + long_inf + diags

        # print('Computing nodal loads...')
        self.nodal_vloads()

        # print('Process completed successfully.')

    def plot(self, node_numbering: bool = False):
        nodes = self.node_coords
        conect = self.conectivity
        nn = self.nknots()

        plt.figure(figsize=(self.a + self.b, self.hm))
        X, Y = zip(*nodes)
        plt.scatter(X, Y)

        for b in conect:
            x = (X[b[0]], X[b[1]])
            y = (Y[b[0]], Y[b[1]])
            plt.plot(x, y, 'b')

        if node_numbering:
            for i in range(nn):
                plt.annotate(i, (X[i], Y[i]))

        plt.axis('scaled')
        plt.title(self.title)
        plt.grid()
        plt.show()


if __name__ == "__main__":

    cab = LatticeBeam(a=17.66, b=17.66, nL=30, nr=30, dL=1, dr=1, hL=7,
                      hm=9.65, hr=7)
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
