"""
Hamiltonian for the Monté Carlo simulation
"""
from itertools import combinations
import numpy as np
import numpy.linalg as la
from LiveChromatin.utils import pdistmap

class Hamiltonian():
    """
    Use keyword arguments
    """
    def __init__(self, structure, **kwargs):
        self.structure = structure
        self.n = len(self.structure)
        self.kwargs = kwargs
        self.keywords = {}
        self.update_keywords()

    def value(self):
        H = 0.0
        for functionName in self.kwargs['function_list']:
            H += getattr(self, functionName)()
        return H

    def set_kwargs(self, structure, **kwargs):
        self.structure = structure
        self.n = len(self.structure)
        if len(kwargs) > 0: self.kwargs = kwargs
        self.update_keywords()

    def update_keywords(self):
        if not 'function_list'.lower() in tuple(self.kwargs.keys()):
            self.kwargs['function_list'] = ['harmonic_bond', 'cosine_angle', 'spherical_confinement']

        self.keywords['harmonic_bond'] = ['bonding_profile', 'kcis', 'ktrans', 'r0']
        self.keywords['cosine_angle'] = ['bonding_profile', 'kbend']
        self.keywords['spherical_confinement'] = ['density', 'kconf']

    def harmonic_bond(self):
        bondmap = self.kwargs['bonding_profile'].toarray() * self.kwargs['ktrans']
        for i,j in zip(range(self.n-1), range(1, self.n)):
            if bondmap[i,j]:
                bondmap[i,j] = self.kwargs['kcis']
        return np.sum(bondmap * np.square(pdistmap(self.structure) - self.kwargs['r0']))

    def cosine_angle(self):
        coo = self.kwargs['bonding_profile'].tocoo()
        H = 0.0
        bending_energy = lambda k, vi, vj: k * (1 - (vi*vj).sum() / (la.norm(vi)*la.norm(vj)))
        for row in np.arange(self.n):
            target = coo.row == row
            target_T = coo.col == row

            for i,j in combinations(np.unique(np.concatenate((coo.row[target_T],
                                                              coo.col[target]))), 2):
                H += bending_energy(self.kwargs['kbend'], self.structure[row] - self.structure[i],
                                      self.structure[j] - self.structure[row])
        return H

    def spherical_confinement(self):
        R = (3 * self.n / (4 * 3.141592 * self.kwargs['density'])) ** (1 / 3.0)
        r_ = self.structure - self.structure.mean(0)
        r = (np.square(r_).sum(1) + (1e-2)**2)
        H = np.zeros_like(r)
        H = self.kwargs['kconf'] * r - 1
        H[r <= (R - 1/self.kwargs['kconf'])] = 0.0
        return H.sum()
