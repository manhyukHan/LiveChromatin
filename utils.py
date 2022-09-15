import numpy as np
from scipy.interpolate import interp1d

rmsd = lambda x, ref: la.norm(x - ref)
Rg = lambda msd: np.sqrt(1 / (2*msd.shape[0]) * msd.sum())

def mv_avg(x,w):
    x = x[:,None]
    return np.nanmean(np.concatenate([x[i:(i+1 - w)] for i in range(w-1)] + [x[w-1:]], axis=1), 1)

def kappa(gram) -> np.ndarray:
    """
    kappa mapping from gram matrix on manifold to euclidean distance matrix

    gram : np.ndarray (n x n) : gram matrix with rank r

    Return
    ------
    euclidean_distances : np.ndarray (n x n) : corresponding EDM
    """
    n = gram.shape[0]
    e = np.ones((n,1))

    def __diag(matrix):
        return np.diag(matrix).reshape(matrix.shape[0],1)

    return (__diag(gram)@e.T) + (e@__diag(gram).T) - 2*gram

def pdistmap(traj):
    corr = traj@traj.T
    return np.sqrt(corr.diagonal()[:,None] + corr.diagonal()[None,:] - 2*corr)

def plot(ax, pol, m, c_list=None, size=300, cmap=((0.0,0.0,0.0), (1.0,0.0,0.0), (0.0,0.0,1.0)), linecolor='rainbow', linesize=4, linedensity=100):
    mask = np.isnan(pol)
    n = np.arange(m)[~mask[:,0]].max()
    f1 = interp1d(np.arange(m)[~mask[:,0]], pol[:,0][~mask[:,0]], kind='quadratic')
    f2 = interp1d(np.arange(m)[~mask[:,1]], pol[:,1][~mask[:,1]], kind='quadratic')
    f3 = interp1d(np.arange(m)[~mask[:,2]], pol[:,2][~mask[:,2]], kind='quadratic')
    cint = interp1d(np.arange(m)[~mask[:,0]], np.array(c_list)[~mask[:,0]], kind='linear')
    domain = np.linspace(np.arange(m)[~mask[:,0]].min(), n, n*linedensity)
    ax.computed_zorder = False
    if linecolor == 'rainbow': linecolor = [cmap[int(cc)] for cc in cint(domain)]
    ax.scatter(f1(domain), f2(domain), f3(domain),alpha=0.08,s=linesize, zorder=0,
           c=linecolor)
    l = ax.scatter(pol[:,0], pol[:,1], pol[:,2], s=size, alpha=1, zorder=1,
           c=[cmap[int(s)] for s in c_list])
    return l
