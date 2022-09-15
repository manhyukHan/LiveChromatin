import sys, os
import numpy as np
import numpy.linalg as la

from scipy.sparse import dok_matrix
from sympy.ntheory import factorint

from pickle import dump, load
from itertools import combinations

import copy
from LiveChromatin.energy import Hamiltonian
import LiveChromatin.utils as utils

global lattice_vector_set__
global nn_dist__
global params__
global epigenetic_codes__

lattice_vector_set__ = np.array([[0.5,0.5,0],[-0.5,-0.5,0],[0.5,-0.5,0],[-0.5,0.5,0],
                                    [0,0.5,0.5],[0,-0.5,-0.5],[0,0.5,-0.5],[0,-0.5,0.5],
                                    [0.5,0,0.5],[-0.5,0,-0.5],[0.5,0,-0.5],[-0.5,0,0.5]])

nn_dist__ = la.norm(lattice_vector_set__, axis=1).mean()

epigenetic_codes__ = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103,
                               107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227,
                               229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293])

params__ = {
    'kb' : 1e-3,
    'ku' : 1e-4,

    'ec1' : 0.001,
    'et1' : 0.001,
    'ec2' : 0.001,
    'et2' : 0.001,

    'turnover' : (1e-4,1e-4),
    'e01' : 0,
    'e02' : 0,

    'cisbond' : 1.5,
    'transbond' : 1.5,
    'bending' : 1.0,
    'states': np.array([-1,0,1]), # {I, U, A} = {-1, 0, 1}

    'nndist' : nn_dist__,
    
    'r0' : nn_dist__,
    'density' : 0.3,
    'kconf': 10,
}


class Simulation_Setup():
    """
    setup of simulation
    """
    def __init__(self) -> None:
        global params__
        global lattice_vector_set__
        global nn_dist__
        global epigenetic_codes__

        self.lattice_vector_set__ = lattice_vector_set__

        self.params__ = params__

        self.epi_states__ = params__['states']

        self.nn_dist__ = nn_dist__

        self.epigenetic_codes__ = epigenetic_codes__

class RandomWalker(Simulation_Setup):
    """
    3D random walker class
     - lattice scheme

    """
    def __init__(self, seed = None, dimension = 3, stride = 1, exclusive_volume = 0):
        super().__init__()
        self.dimension = dimension
        self.RNG = np.random.default_rng(seed)
        self.traj = np.zeros((1, self.dimension))
        self.moveset = self.lattice_vector_set__
        self.stride = stride
        self.exclusive_volume = exclusive_volume

    def mover(self, p, mask):
        effective_set = self.stride * self.moveset[~mask]
        if len(effective_set) >0:
            return effective_set[int(p * len(effective_set))]
        else:
            return np.zeros((1, self.dimension))

    def initialize_trajectory(self):
        self.traj = np.zeros((1, self.dimension))

    def selfavoiding(self, target):
        mask = np.zeros(len(self.moveset), dtype=bool)
        for i,p in enumerate(target + self.stride * self.moveset):
            if (np.sqrt(np.sum((self.traj - p)**2, 1)) <= self.exclusive_volume).any():
                mask[i] = True
        return mask

        """
            if (self.traj == p).prod(1).any():
                mask[i] = True
        return mask
        """

    def elongate(self, n):
        for t in range(n):
            last = self.traj[-1]
            mask = self.selfavoiding(last)

            if mask.all():
                sys.stderr.write(f'Walker stops at {t+1} steps because of self-avoidity')
                break
            move = self.mover(self.RNG.uniform(0,1), mask)
            self.traj = np.concatenate([self.traj, (last + move)[None,:]], axis=0)

    def diffuse(self):
        n = np.random.randint(len(self.traj))
        target = self.traj[n].copy()
        mask = self.selfavoiding(target)
        move = self.mover(self.RNG.uniform(0,1), mask)
        self.traj[n] = target + move

def simul_annealing_temperature_generator(templow, temphigh, tlow, thigh):
    cnt = 0
    status = 'low'
    while True:
        cnt += 1
        if status == 'low':
            if cnt > tlow:
                cnt = 0
                status = 'high'
            yield templow
        else:
            if cnt > thigh:
                cnt = 0
                status = 'low'
            yield temphigh

def constant_temp_generator(temp):
    while True:
        yield temp

def ChrMover_MC_generator(hamiltonian, randomwalker, temperature):
    while True:
        T = temperature.__next__()
        x0 = randomwalker.traj.copy()
        hamiltonian.set_kwargs(x0)
        E0 = hamiltonian.value()
        randomwalker.diffuse()
        hamiltonian.set_kwargs(randomwalker.traj)

        dE = hamiltonian.value() - E0
        if - dE / T > 0:
            yield randomwalker.traj.copy()
        if np.random.uniform(0,1) < np.exp(-dE / T, dtype=np.float128):
            yield randomwalker.traj.copy()
        else:
            hamiltonian.set_kwargs(x0)
            randomwalker.traj = x0
            yield x0

def calc_state_transition_propensity(kwargs):
    return (kwargs['turnover'] +
            kwargs['trans'] * kwargs['n_trans'] + kwargs['cis'] * kwargs['n_cis'])

class Gillespie(Simulation_Setup):
    """
    Gillespie algorithm simulator
     - propensity/ reaction matrix generation
     - simulation
    """
    def __init__(self, x0, init_state, init_bond, params=None, criterion=None,
                 reaction_matrix_controller=None) -> None:
        super().__init__()
        if params is None:
            self.params = self.params__
        else:
            self.params = params
        self.x0 = x0
        self.init_state = init_state
        self.init_bond = init_bond
        self.n = len(self.x0)
        if criterion is None:
            self.criterion = self.nn_dist__
        else:
            self.criterion = criterion

        self.struc_traj = [self.x0]
        self.time_traj = [0]
        self.state_traj = [self.init_state]
        self.bond_traj = [self.init_bond]
        self.mu_traj = []
        self.RNG = np.random.default_rng()

        if reaction_matrix_controller is None:
            self.reaction_matrix_controller = ReactionMatrixController()
        else:
            self.reaction_matrix_controller = reaction_matrix_controller

    def _encode_state(self, state):
        encoded_state = np.zeros(state.shape)
        for i, s in enumerate(self.params['states']):
            encoded_state[state == s] = self.epigenetic_codes__[i]
        return encoded_state

    def _create_bond_propensity(self, criteria=None):
        if criteria is None:
            criteria = self.nn_dist__
        encoded_state = self._encode_state(self.state_traj[-1])

        return self.reaction_matrix_controller('bond')(self.struc_traj[-1], self.bond_traj[-1], criteria, encoded_state, self.params)

    def _create_state_propensity(self):
        return self.reaction_matrix_controller('state')(self.state_traj[-1], self.struc_traj[-1], self.criterion, self.params)

    def step(self, bond_criteria = None, callback = None, callbackargs = None):

        next_total_bond_reaction_matrix, next_total_bond_propensity = self._create_bond_propensity(bond_criteria)
        next_state_reaction_matrix, next_state_propensity = self._create_state_propensity()

        if (callback is not None) and (callable(callback)):
            next_total_bond_reaction_matrix, next_total_bond_propensity, next_state_reaction_matrix, next_state_propensity = callback(self, callbackargs,
                                                                                                                                      bond_reaction_matrix = next_total_bond_reaction_matrix,
                                                                                                                                      bond_propensity = next_total_bond_propensity,
                                                                                                                                      state_reaction_matrix = next_state_reaction_matrix,
                                                                                                                                      state_propensity = next_state_propensity)
        state_profile = self.state_traj[-1].copy()
        bond_profile = self.bond_traj[-1].copy()

        total_propensity = np.concatenate([next_total_bond_propensity[:,None], next_state_propensity[:,None]], axis=0)
        total_abs_propensity = np.abs(total_propensity)

        a0 = total_abs_propensity.sum()
        if a0 == 0:
            self.state_traj.append(state_profile)
            self.bond_traj.append(bond_profile)
            self.time_traj.append(self.tmax)
            return

        tau = np.divide(-np.log(self.RNG.uniform()), a0)
        mu = 0
        s = total_abs_propensity[0]
        r0 = self.RNG.uniform() * a0

        while s < r0:
            mu += 1
            s += total_abs_propensity[mu]

        bond_state_boundary = len(next_total_bond_reaction_matrix)
        self.mu_traj.append((mu, bond_state_boundary))
        if mu < bond_state_boundary:
            # change bond profile
            if (total_propensity[mu] < 0).any():
                i,j = list(next_total_bond_reaction_matrix.keys())[mu]
                bond_profile[i,j] = 0
            else:
                i,j = list(next_total_bond_reaction_matrix.keys())[mu]
                bond_profile[min(i,j), max(i,j)] = 1
        else:
            # change state profile
            mu -= bond_state_boundary
            state_profile += next_state_reaction_matrix[mu]

        self.state_traj.append(state_profile)
        self.bond_traj.append(bond_profile)
        self.time_traj.append(self.time_traj[-1] + tau)


    def run(self, tmax, bond_criteria = None, callback = None, callbackargs = None):
        self.tmax = tmax
        while self.time_traj[-1] < self.tmax:
            self.step(bond_criteria, callback, callbackargs)

class LiveChromatin(Simulation_Setup):
    """
    Live Chromatin model
     - Integrate Monté Carlo based structure optimization

    params
     - n : int : number of segment of the chromatin polymer
     - init_bond : scipy.sparse.dok_matrix : initial bonding profile (if not declared, cis-only)
     - init_state : ndarray (n x 1) : initial state profile
     -

    """
    def __init__(self, n, init_bond, init_state, temperature,
                 seed=None, randomwalker=None, params=None, criterion=None,
                 reaction_matrix_controller=None) -> None:
        super().__init__()
        self.n = n
        self.init_bond = init_bond
        self.init_state = init_state
        self.temperature = temperature
        self.criterion = criterion
        if reaction_matrix_controller is None:
            self.reaction_matrix_controller = ReactionMatrixController()
        else:
            self.reaction_matrix_controller = reaction_matrix_controller

        self.montécarlo_track_in_step = []

        if randomwalker is None:
            self.randomwalker = RandomWalker(seed, 3)
            self.randomwalker.elongate(n-1)
        else:
            self.randomwalker = copy.deepcopy(randomwalker)

        if params is not None:
            self.params__ = copy.deepcopy(params)

        self.x0 = self.randomwalker.traj.copy()

        self.liv_structure_traj = [self.x0]
        self.liv_bond_traj = [self.init_bond]
        self.liv_state_traj = [self.init_state]
        self.liv_energy_traj = []

        self.Gillespie = Gillespie(self.x0, self.init_state, self.init_bond, self.params__, self.criterion,
                                   reaction_matrix_controller=self.reaction_matrix_controller)
        self.hamiltonian = Hamiltonian(self.Gillespie.struc_traj[-1],
                                       bonding_profile=self.Gillespie.bond_traj[-1],
                                       kcis=self.params__['cisbond'], ktrans=self.params__['transbond'], r0=self.params__['r0'],
                                       kbend=self.params__['bending'],
                                       density=self.params__['density'], kconf=self.params__['kconf'])
        self.liv_energy_traj.append(self.hamiltonian.value())

    def step(self, epi_tmax, mc_steps, bond_criteria = None, callback = None, callbackargs = None, save_traj=True):
        self.Gillespie = Gillespie(self.liv_structure_traj[-1], self.liv_state_traj[-1], self.liv_bond_traj[-1], self.params__, self.criterion,
                                   reaction_matrix_controller=self.reaction_matrix_controller)
        self.Gillespie.run(epi_tmax, bond_criteria, callback, callbackargs)
        self.hamiltonian.set_kwargs(self.Gillespie.struc_traj[-1],
                                       bonding_profile=self.Gillespie.bond_traj[-1],
                                       kcis=self.params__['cisbond'], ktrans=self.params__['transbond'], r0=self.params__['r0'],
                                       kbend=self.params__['bending'],
                                       density=self.params__['density'], kconf=self.params__['kconf'])
        MontéCarlo = ChrMover_MC_generator(self.hamiltonian, self.randomwalker, self.temperature)
        self.liv_energy_traj.append(self.hamiltonian.value())

        if save_traj:
            for i in range(mc_steps):
                MontéCarlo.__next__()
                self.montécarlo_track_in_step.append(self.randomwalker.traj.copy())
        else:
            for i in range(mc_steps):
                MontéCarlo.__next__()

        self.liv_structure_traj.append(self.randomwalker.traj.copy())
        self.liv_bond_traj.append(self.Gillespie.bond_traj[-1])
        self.liv_state_traj.append(self.Gillespie.state_traj[-1])

    def run(self, steps,
            epi_tmax, mc_steps, bond_criteria = None,
            callback = None, callbackargs = None, save_traj=True):
        for s in range(steps):
            self.step(epi_tmax, mc_steps, bond_criteria = bond_criteria, callback = callback, callbackargs = callbackargs, save_traj=save_traj)

    @staticmethod
    def callback_loading_state(instances, callbackargs, **kwargs):
        i = callbackargs['index']
        s = callbackargs['state']
        instances.state_traj[-1][i] = s
        bond_reaction_matrix = kwargs['bond_reaction_matrix']
        bond_propensity = kwargs['bond_propensity']
        state_reaction_matrix = kwargs['state_reaction_matrix']
        state_propensity = kwargs['state_propensity']

        targets = state_reaction_matrix[:,i] == 0
        new_state_reaction_matrix = state_reaction_matrix[targets]
        new_state_propensity = state_propensity[targets]

        return (bond_reaction_matrix, bond_propensity,
                new_state_reaction_matrix, new_state_propensity)

    def save(self, path):
        parameters = {'n': self.n, 'init_bond': self.init_bond, 'init_state': self.init_state, #'temperature': self.temperature,
                      'criterion': self.criterion, 'reaction_matrix_controller': self.reaction_matrix_controller,
                      'randomwalker': self.randomwalker, 'params': self.params__}

        if not os.path.isdir(path):
            os.mkdir(path)
        if not os.path.isdir(path+'/data'):
            os.mkdir(path+'/data')

        with open(path + '/data/parameters.bin', 'wb') as f:
            dump(parameters, f)

        np.save(path+'/data/state_trajectory', self.liv_state_traj, allow_pickle=True)
        np.save(path+'/data/structure_trajectory', self.liv_structure_traj, allow_pickle=True)
        np.save(path+'/data/bond_trajectory', self.liv_bond_traj, allow_pickle=True)
        np.save(path+'/data/energy_trajectory', self.liv_energy_traj, allow_pickle=True)

    @staticmethod
    def load(path, temperature):
        datapath = path + '/data/'
        with open(datapath+'parameters.bin', 'rb') as f:
            parameters = load(f)
        state_traj = np.load(datapath+'state_trajectory.npy', allow_pickle=True)
        structure_traj = np.load(datapath+'structure_trajectory.npy', allow_pickle=True)
        bond_traj = np.load(datapath+'bond_trajectory.npy', allow_pickle=True)
        energy_traj = np.load(datapath+'energy_trajectory.npy', allow_pickle=True)

        self = LiveChromatin(parameters['n'], parameters['init_bond'], parameters['init_state'], temperature=temperature, #parameters['temperature'],
                             randomwalker=parameters['randomwalker'], params=parameters['params'], criterion=parameters['criterion'],
                             reaction_matrix_controller=parameters['reaction_matrix_controller'])
        self.liv_state_traj = list(state_traj)
        self.liv_structure_traj = list(structure_traj)
        self.liv_bond_traj = list(bond_traj)
        self.liv_energy_traj = list(energy_traj)

        return self

class ReactionMatrixController():
    """
    Reaction matrix/propensity controller class
    Basic input
        - Bond reaction matrix : (structure, bond, criterion, encoded state, and params)
        - State reaction matrix : (state, structure, criterion, and parameters)
    output
        - reaction matrix, propensity

    New reaction matrix generator is a method that uses the class variables, and gets input variables posted above.
    """
    def __init__(self, bond_function_name = 'default', state_function_name='default',
                 **kwargs) -> None:
        self.kwargs = kwargs
        self.bond_function_name = bond_function_name
        self.state_function_name = state_function_name

    def __call__(self, reaction_type='bond'):
        if reaction_type.lower() == 'bond':
            if self.bond_function_name.lower() == 'default':
                function_name = 'homotypic_universal_bond'
            else:
                function_name = self.bond_function_name
        elif reaction_type.lower() == 'state':
            if self.state_function_name.lower() == 'default':
                function_name = 'three_state_reaction_matrix'
            else:
                function_name = self.state_function_name

        return getattr(self, function_name)

    def homotypic_universal_bond(self, structure, bond_profile, criterion, encoded_state, params):
        """
        homotypic and universal bond creation profile maker.

        monomer in the same state, closer than the certian threshold (criteria) can be connected.
        Bond-connecting ratio (kb) is universal regardless of the epigenetic state.

        this is the default bond reaction matrix generator.
        """
        n = len(structure)
        next_remove_reaction = bond_profile.copy()
        next_remove_reaction.setdiag(0,1)

        encoded_matrix = encoded_state[:,None]@encoded_state[None,:]
        mask = np.ones((n,n))

        for e in np.unique(encoded_state):
            mask[encoded_matrix == e**2] = 0
        mask += bond_profile.toarray()
        mask = mask.astype(bool)

        dmap = utils.pdistmap(structure)
        dmap[mask] = 2 * criterion
        next_create_reaction = dok_matrix((np.triu((dmap <= criterion).astype(int), 2)))

        next_total_reaction_matrix = (params['kb'] * next_create_reaction -
                                      params['ku'] * next_remove_reaction).todok()
        next_total_propensity = np.array(list(next_total_reaction_matrix.values()))

        return next_total_reaction_matrix, next_total_propensity

    def general_bond(self, structure, bond_profile, criterion, encoded_state, params):
        """
        general bond reaction matrix with different reaction rate.

        keyward_arguments:
            - bond_creation_rate_matrix: (mxm ndarray) m: np.unique(encoded_state).size
            - unique_epi_state
        """
        if not 'bond_creation_rate_matrix'.lower() in self.kwargs.keys():
            raise ValueError('bond_creation_rate_matrix keyward argument is needed')

        n = len(structure)
        states = self.kwargs['unique_epi_state']

        next_remove_reaction = bond_profile.copy()
        next_remove_reaction.setdiag(0, 1)

        encoded_matrix = encoded_state[:,None]@encoded_state[None,:]
        dmap = utils.pdistmap(structure)
        kb_map = np.zeros(encoded_matrix.shape)

        for ep in np.unique(encoded_matrix):
            try:
                p,q = tuple(factorint(int(ep)).keys())
                i,j = int(np.where(states == p)[0]), int(np.where(states == q)[0])
            except ValueError:
                p = tuple(factorint(int(ep)).keys())[0]
                i = int(np.where(states == p)[0])
                j = i
            kb_map[encoded_matrix == ep] = self.kwargs['bond_creation_rate_matrix'][i,j]

        next_total_reaction_matrix = (dok_matrix(kb_map * params['kb'] * np.triu((dmap <= criterion).astype(int), 2)) -
                                params['ku'] * next_remove_reaction).todok()
        next_total_propensity = np.array(list(next_total_reaction_matrix.values()))

        return next_total_reaction_matrix, next_total_propensity

    def three_state_reaction_matrix(self, state_profile, structure, criterion, params):
        """
        three state (inactive, unmodified, active)

        this is the default state reaction matrix generator.

        """
        next_state_propensity = []
        next_state_reaction_matrix = []
        kwargs = {'turnover': 0,
                  'n_trans' : 0,
                  'n_cis' : 0,
                  'trans' : 0,
                  'cis' : 0}

        for i,s in enumerate(state_profile):
            if i == 0:
                n_cis_active = int(state_profile[i+1] == 1)
                n_cis_inactive = int(state_profile[i+1] == -1)

            elif i == len(state_profile) - 1:
                n_cis_active = int(state_profile[i-1] == 1)
                n_cis_inactive = int(state_profile[i-1] == -1)

            else:
                n_cis_active = int(state_profile[i-1] == 1) + int(state_profile[i+1] == 1)
                n_cis_inactive = int(state_profile[i-1] == -1) + int(state_profile[i+1] == -1)

            n_trans_active = 0
            n_trans_inactive = 0

            targets = np.ones(len(structure), dtype=bool)
            targets[i] = 0
            deviations = np.sqrt(np.sum((structure - structure[i])**2, 1))

            if i == 0:
                targets[1] = False
                nn_dev = deviations[1]
            elif i == len(state_profile)-1:
                targets[-2] = False
                nn_dev = deviations[-2]
            else:
                targets[i-1] = False
                targets[i+1] = False
                nn_dev = 1/2 * (deviations[i-1] + deviations[i+1])

            targets[deviations > criterion] = 0

            neighbors = np.arange(len(structure), dtype=int)[targets]
            for n in neighbors:
                d = deviations[n] / nn_dev
                if state_profile[n] == -1: n_trans_inactive += 1 / d
                elif state_profile[n] == 1: n_trans_active += 1 / d

            if s == -1:
                # I -> U
                kwargs['turnover'] = params['turnover'][0]
                kwargs['n_trans'] = n_trans_active
                kwargs['n_cis'] = n_cis_active
                kwargs['cis'] = params['ec1']
                kwargs['trans'] = params['et1']

                next_state_propensity.append(calc_state_transition_propensity(kwargs))
                reaction = np.zeros(state_profile.shape)
                reaction[i] = 1
                next_state_reaction_matrix.append(reaction)

            if s == 0:
                # U -> A
                kwargs['turnover'] = params['e01']
                kwargs['n_trans'] = n_trans_active
                kwargs['n_cis'] = n_cis_active
                kwargs['cis'] = params['ec1']
                kwargs['trans'] = params['et1']

                next_state_propensity.append(calc_state_transition_propensity(kwargs))
                reaction = np.zeros(state_profile.shape)
                reaction[i] = 1
                next_state_reaction_matrix.append(reaction)

                # U -> I
                kwargs['turnover'] = params['e02']
                kwargs['n_trans'] = n_trans_inactive
                kwargs['n_cis'] = n_cis_inactive
                kwargs['cis'] = params['ec2']
                kwargs['trans'] = params['et2']

                next_state_propensity.append(calc_state_transition_propensity(kwargs))
                reaction = np.zeros(state_profile.shape)
                reaction[i] = -1
                next_state_reaction_matrix.append(reaction)

            if s == 1:
                # A -> U
                kwargs['turnover'] = params['turnover'][1]
                kwargs['n_trans'] = n_trans_inactive
                kwargs['n_cis'] = n_cis_inactive
                kwargs['cis'] = params['ec2']
                kwargs['trans'] = params['et2']

                next_state_propensity.append(calc_state_transition_propensity(kwargs))
                reaction = np.zeros(state_profile.shape)
                reaction[i] = -1
                next_state_reaction_matrix.append(reaction)

        next_state_propensity = np.array(next_state_propensity)
        next_state_reaction_matrix = np.array(next_state_reaction_matrix)

        return next_state_reaction_matrix, next_state_propensity
