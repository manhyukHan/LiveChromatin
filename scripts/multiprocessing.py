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

if __name__ == '__main__':
    ### PARAMETERS ###
    n = 80
    init_state = np.zeros(n)
    #init_state[np.arange(38,42,1, dtype=int)] = 1
    init_bond = dok_matrix(np.zeros((n,n)))
    init_bond.setdiag(1,1)
    nn_dist = ls.Simulation_Setup().nn_dist__
    param = ls.params__
    stride = 3

    param['kb'] = 1e-3
    param['ku'] = 1e-4
    param['ec1'] = 0#1e-4
    param['ec2'] = 0#1e-4
    param['et1'] = 0#1e-5
    param['et2'] = 0#1e-5
    param['e01'] = 0
    param['e02'] = 0
    param['turnover'] = (1e-4, 1e-4)
    param['cisbond'] = 1.5
    param['transbond'] = 1.5
    param['bending'] = 1
    param['stride'] = stride
    param['r0'] = param['stride'] * nn_dist
    param['states'] = np.array([-1,0,1])
    param['density'] = 5.0

    bond_creation_rate_matrix = np.triu(np.ones((3,3)))
    bond_creation_rate_matrix[1,1] = 1
    bond_creation_rate_matrix[1,2] = 0

    ### Simulation Setup ###
    numcells = 50
    dmaps_init = np.zeros((numcells,n,n))
    dmaps_after = np.zeros_like(dmaps_init)

    callback = None
    temperature = ls.Temperature(templow=1,temphigh=90,tlow=8,thigh=2)

    reaction_matrix_controller = ls.ReactionMatrixController(bond_function_name='general_bond',
                                                            bond_creation_rate_matrix=bond_creation_rate_matrix,
                                                            unique_epi_state=ls.epigenetic_codes__[:3])

    if not os.path.isdir(__scratchpath):
        os.mkdir(__scratchpath)

    livchromatins_of_dmap = []
    pool = ProcessPoolExecutor(n_cpus, initializer=pool_initializer, initargs=(livchromatins_of_dmap, __scratchpath, 3 * nn_dist,
                                                                            callback, {'index':int(n/2),'state':1}))
    p_list = []
    subset_index_list = [np.arange(len(livchromatins_of_dmap))[i::n_cpus] for i in range(n_cpus)]

    for subset_index in subset_index_list:
        p_list.append(pool.submit(run_single_thread, subset_index))

    for p in as_completed(p_list):
        status = p.result()
        print(status)
    pool.shutdown()

    if not os.path.isdir(resultpath):
        os.mkdir(resultpath)

    for i in range(numcells):
        with open(__scratchpath + f'{i}/simulator.bin', 'rb') as f:
            livchromatin = load(f)
            livchromatin.save(resultpath + f'index_{i}')
            dmaps_after[i] = lu.pdistmap(livchromatin.liv_structure_traj[-1])

    np.save(resultpath+'dmaps_init', dmaps_init)
    np.save(resultpath+'dmaps_after', dmaps_after)

    os.system(f'rm -rf {__scratchpath}*')
