from pyImagingMSpec.dataset import InMemoryDataset, ImzmlDataset
from joblib import Parallel, delayed
from recalibration.utils import estimate_linear_shift
import numpy as np
import networkx as nx



def get_matching_peaks(deltas, n_est=100):
    p = np.polyfit(np.arange(n_est), deltas[0:n_est], 1)
    fit_error = np.abs(deltas - np.polyval(p, np.arange(0,len(deltas))))
    t = 1.05 * np.max(np.abs(fit_error[0:n_est])) # max error in fitted points plus 5% tolerance
    return deltas[0 : np.argmax(fit_error > t)] # find first value above error tolerance, and stop

def process_node(ref_spec, comp_spec, n_peaks=500):
    ii, s0 = ref_spec
    edges = []
    for jj, s1 in comp_spec:
        v = estimate_linear_shift(s0, s1, n_peaks)
        if v > 0:
            edges.append((str(ii), str(jj), {'weight': np.abs(v)}))
        else:
            edges.append((str(jj), str(ii), {'weight': np.abs(v)}))
    return edges


def get_adjacent_spectra(ii, imzml):
    """
    best not to have data access in parallel loop to avoid race problems
    """
    ref_spec = ii, imzml.get_spectrum(ii)
    comp_spec = []
    c = imzml.coordinates[ii]
    for i, j in zip([c[0], c[0]+1], [c[1]+1, c[1]]):
        try:
            jj = np.where(np.sum(imzml.coordinates == [i,j,1], axis=1) == 3)[0][0]
        except IndexError as e:
            #([i,j,1], e)
            continue
        spec = imzml.get_spectrum(jj)
        comp_spec.append((jj, spec))
    return ref_spec, comp_spec


def format_graph(imzml, edges):
    G = nx.DiGraph()
    for ii, c in enumerate(imzml.coordinates):
        G.add_node(str(ii), x=float(c[0]), y=float(c[1]))
    for edge in edges:
        G.add_weighted_edges_from([(e[0], e[1], float(e[2]['weight'])) for e in edge])
    return G


def write_graphml(fname, G):
    nx.write_graphml(G, fname)


def create_mass_shift_graph(imzml_fn, num_cores=2):
    # load imzml
    imzml = ImzmlDataset(imzml_fn)
    # calculate mass shift per spectrum
    edges = Parallel(n_jobs=num_cores)(
        delayed(process_node)(*get_adjacent_spectra(ii, imzml)) for ii in np.arange(len(imzml.coordinates)))
    # format as a graph
    graph = format_graph(imzml, edges)
    return graph