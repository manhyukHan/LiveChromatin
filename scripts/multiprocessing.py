import os
import numpy as np

from pickle import load, dump

from concurrent.future import ProcessPoolExecutor, as_completed

import LiveChromatin.simulation as ls
import LiveChromatin.utils as lu

global __from_scratch # bool
global __n_cpu # int
global __scratchpath # path, str type
global __resultpath # path, str type

### MULTIPROCESSING ###
def pool_initializer(**kwargs):
    for key, value in kwargs.items():
        global()['__' + key] = value

def run_single_simulator_from_index(subset_index):
    for i in subset_index:
        if __from_scratch:
            with open(__scratchpath + f'{i}/simulator.bin', 'rb') as f:
                livchromatin = load(f)
            dmaps_init[i] = lu.pdistmap(livchromatin.liv_structure_traj[0])
            livchromatins_in_pool.append(livchromatin)
        else:
            try:
                with open(__resultpath + f'index_{i}/simulator.bin', 'rb') as f:
                    livchromatin = load(f)
            except:
                param = __param
                randomwalker = ls.RandomWalker(stride = param['stride'], exclusive_volume = param['exclusive_volume'])
                randomwalker.elongate(param['n']-1)
                randomwalker.stride = 1.1
                livchromatin = ls.LiveChromatin(param['n'], param['init_bond'], param['temperature'],
                                                randomwalker=randomwalker, params = param, criterion = 8,
                                                 reaction_matrix_controller = __reaction_matrix_controller)
            dmaps_init[i] = lu.pdistmap(livchromatin.liv_structure_traj[0])
            livchromatin.save(__scratchpath + f'{i})
            livchromatins_in_pool.append(livchromatin)
