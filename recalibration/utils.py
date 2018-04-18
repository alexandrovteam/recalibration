import numpy as np
def moving_average(x,y,window):
    x,y = list(map(np.asarray,[x,y]))
    edges = np.arange(np.min(x), np.max(x)+window, window)
    bins = np.digitize(x, edges)
    ma = np.zeros((np.max(bins),))
    for ii in np.arange(np.max(bins)):
        ma[ii] = np.median(y[bins==ii])
    return edges[0:-1] + window/2., ma

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def find_nearest(v, t):
    """
    v: vector of values
    t: target value
    returns
    index of nearest value, nearest value
    """
    v,t = map(np.asarray, [v, t])
    if v.shape == (): #number passed - only one possible result
        return 0, v
    if t.shape ==(): # number passed, force to array
        t = np.asarray([t,])
    ix = np.searchsorted(v,t)
    ix[ix==0]+=1
    ix[ix==v.shape[0]] -= 1
    ix = ix-1+np.argmin(np.abs(np.vstack([v[ix-1], v[ix]]) - t[::]), axis=0)
    return ix, v[ix]

def get_deltas_mix(target, reference, ppm=True):
    """
    matches most likely pairs of peaks between two vectors and returns a list of value differences
    :param target: vector one
    :param reference: vector two
    :return: list of pairwise differences, sorted in ascending value order. length == len(v2), nan where matching value not found
    """
    target, reference = map(lambda x: np.asarray(x).T, [target, reference])
    dmz = np.abs(target[:, 0][:, np.newaxis] - reference[:,0])
    ix0 = np.arange(dmz.shape[0])
    ix1 = np.argmin(dmz, axis=1)
    assert len(ix0) == len(ix1)
    ix2 = np.arange(dmz.shape[1])
    ix3 = np.argmin(dmz, axis=0)
    assert len(ix2) == len(ix3)
    matchix = set((x,y) for x,y in zip(ix2,ix3)) & set((x,y) for x,y in zip(ix1, ix0))
    m1 = np.asarray([target[m[1],0] for m in matchix])
    m2 = np.asarray([reference[m[0],0] for m in matchix])
    _deltas = m1 - m2
    if ppm:
        _deltas = 1e6*_deltas/np.mean(np.vstack([m2, m2]), axis=0)
    deltas = np.nan * np.zeros(reference.shape[0])
    deltas[[m[0] for m in matchix]] = _deltas
    return deltas


def get_deltas_intensity(target, reference, ppm=True):
    """
    matches most likely pairs of peaks between two vectors and returns a list of value differences
    :param target: vector one
    :param reference: vector two
    :return: list of pairwise differences, sorted in ascending value order. length == len(v2), nan where matching value not found
    """
    target, reference = map(lambda x: np.asarray(x).T, [target, reference])
    dmz = np.abs(target[:, 0][:, np.newaxis] - reference[:,0])
    dmz = np.sqrt(np.square(dmz) + np.square( (target[:, 1][:, np.newaxis] - reference[:,1]) / (target[:, 1][:, np.newaxis] + reference[:,1])/2 ))
    ix0 = np.arange(dmz.shape[1])
    ix1 = np.argmin(dmz, axis=1)
    ix2 = np.arange(dmz.shape[0])
    ix3 = np.argmin(dmz, axis=0)
    matchix = set((x,y) for x,y in zip(ix2,ix3)) & set((x,y) for x,y in zip(ix1, ix0))
    m1 = np.asarray([target[m[0],0] for m in matchix])
    m2 = np.asarray([reference[m[1],0] for m in matchix])
    _deltas = m2 - m1
    if ppm:
        _deltas = 1e6*_deltas/np.mean(np.vstack([m2, m2]), axis=0)
    deltas = np.nan * np.zeros(dmz.shape[0])
    deltas[[m[1] for m in matchix]] = _deltas
    return deltas

def get_deltas(x1, x2, ppm=True):
    """
    Find the ppm difference between the nearest value from x2 in x1
    :param x1: array of value(s) to search in
    :param x2: array of value(s) to search for
    :return: ppms, array same size as x2
    """
    x1, x2 = map(np.asarray, [x1, x2])
    ixs, mzs= find_nearest(x1, x2)
    deltas = x2 - mzs
    if ppm:
        ppms = 1e6 * (deltas)/x2
        return ppms
    return deltas

def get_deltas_values(v1, v2, ppm=True):
    v = np.sort(np.concatenate([v1,v2]))
    diff = np.diff(v)
    if ppm:
        return 1e6 * diff / v[1:]
    return diff

def get_top_peaks(x, y, n=500):
    if len(x)>n:
        ix = np.sort(np.argsort(y)[-n:])
    else:
        ix = np.argsort(x)
    return x[ix], y[ix]

def select_peaks(mzs, intensities, max_per_chunk=75, n_chunks=10, bins=None):
    if bins is None:
        bins = np.linspace(mzs[0], mzs[-1], n_chunks)
    dig = np.digitize(mzs, bins)
    ixs = []
    for ii in np.arange(len(bins)):
        _ix = dig == ii
        _mzs, _ints = mzs[_ix], intensities[_ix]
        if len(_mzs) <= max_per_chunk:
            ixs.append(
                np.where(_ix)[0]
            )
        else:
            ixs.append(
                np.where(_ix)[0][np.argsort(_ints)[-max_per_chunk:]]
            )
    ixs = np.hstack(ixs)
    return mzs[ixs], intensities[ixs]


def estimate_linear_shift(s1, s2, n=None, ppm=True, n_chunks=1):
    bins = np.linspace(np.min([s1[0][0], s2[0][0]]), np.max([s1[0][-1], s2[0][-1]]), n_chunks+1)
    #v1, v2 = [select_peaks(*s, bins=bins, max_per_chunk=n) for s in [s1, s2]]
    v1, v2 = [get_top_peaks(*s, n) for s in [s1, s2]]
    deltas = get_deltas_mix(v2, v1, ppm=ppm)
    deltas = np.sort(deltas[~np.isnan(deltas)])
    if len(deltas) == 0:
        return 0
    return -np.percentile(deltas, [25, 75])[np.argmax([0, np.median(deltas)])]
