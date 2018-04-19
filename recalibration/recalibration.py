import numpy as np
from recalibration import utils
from scipy.optimize import least_squares
from recalibration import shift_graph
from recalibration.metaspace import get_reference_mzs
import pandas as pd

from pyimzml import ImzMLParser, ImzMLWriter
from cpyImagingMSpec import ImzbReader
import subprocess


def power(x, a, b, c):
    return a * np.power(x, -b) + c

def exp(x, a, b, c):
    return a * np.exp(-b * x) + c

def poly(x, t):
    return np.polyval(x, t)

def log(x, a, b, c):
    return a*np.log(-b*x)+c

def lsfunc(x, t, y):
    e = poly(x, t) - y
    return e

def fit_spectrum(mzs, intensities, ref_mzs, ref_pcts, max_delta_ppm, mz_min, mz_max, x0=(1, 1, 1),
                 weight_by_occurance=True, stabilise=True, intensity_threshold=0, ppm_flag=True):
    mzs, intensities = map(lambda x: x[intensities>intensity_threshold], [mzs, intensities])
    intensities, mzs = map(lambda x: x[mzs > mz_min], [intensities, mzs])
    intensities, mzs  = map(lambda x: x[mzs < mz_max], [intensities, mzs])
    ref_pcts, ref_mzs  = map(lambda x: x[ref_mzs > mz_min], [ref_pcts, ref_mzs])
    ref_pcts, ref_mzs = map(lambda x: x[ref_mzs < mz_max], [ref_pcts, ref_mzs])

    delta = -utils.get_deltas_mix((mzs, intensities), (ref_mzs, ref_pcts),  ppm=ppm_flag)
    delta[np.abs(delta) > max_delta_ppm] = np.nan

    ref_mzs, ref_pcts, delta = map(lambda x:x[~np.isnan(delta)], [ref_mzs, ref_pcts, delta])
    _x = []
    _y = []
    if stabilise is not None:
        for s in range(stabilise):
            _x.extend([mz_min,mz_max])
            _y.extend([0, 0])

    for ref_mz, ref_pct, dt in zip(ref_mzs, ref_pcts.astype('int'), delta):
        if not weight_by_occurance:
            ref_pct = 1
        for ii in np.arange(ref_pct):
            _x.append(ref_mz)
            _y.append(dt)
    _x, _y = map(np.asarray, (_x, _y))
    _r = least_squares(lsfunc, x0, loss='soft_l1', f_scale=0.5, args=(_x, _y))
    return _r, (_x, _y)

def get_shift_model(mzs, ints, ref_mzs, max_delta_ppm=30, x0=(0,0,0), stabilise=1, ppm_flag=True, plot=False):
    r, data = fit_spectrum(
        *utils.select_peaks(mzs, ints, max_per_chunk=1, bins=np.arange(mzs[0], mzs[-1], 0.1)),
        #mzs, ints,
        ref_mzs['mz'].values, ref_mzs['count'],
        mz_min=mzs.min(), mz_max=mzs.max(),
        max_delta_ppm=max_delta_ppm, x0=x0, stabilise=stabilise, ppm_flag=ppm_flag
    )
    if plot:
        import matplotlib.pyplot as plt
        shifts = poly(r.x, mzs)
        plt.figure()
        _x = utils.select_peaks(mzs, ints, max_per_chunk=1, bins=np.arange(mzs[0], mzs[-1], 0.1))[0]
        plt.scatter(_x, np.zeros(len(_x)), label='mean spectrum', alpha=0.8)
        plt.scatter(ref_mzs['mz'].values, np.zeros(len(ref_mzs['mz'].values)), label='reference', alpha=0.6)
        plt.scatter(data[0], data[1], label='matched')
        plt.plot(mzs, shifts, label='fitted error')
        plt.legend()
        plt.show()
    if ppm_flag:
        return lambda x: poly(r.x, x)*1e-6*x, data, r.x
    else:
        return lambda x: poly(r.x, x), data, r.x

def mean_spectrum_aligned(imzml):
    mzs, _ = map(np.asarray, imzml.getspectrum(0))
    intensities = np.zeros(mzs.shape)
    for ii, c in enumerate(imzml.coordinates):
        _m, _i = imzml.getspectrum(ii)
        intensities += np.asarray(_i)
    return mzs, intensities / float(ii)


def recalibrate(aligned_fn, clustered_fn, imzml_out_fn, ref_mzs, max_delta=30, x0 = (0,0,0),
                stabilise=1, ppm_flag=True, plot=False, dump_fn=None):
    aligned_imzml = ImzMLParser.ImzMLParser(aligned_fn)
    clustered_imzml = ImzMLParser.ImzMLParser(clustered_fn)
    ms_mzs, ms_ints = mean_spectrum_aligned(clustered_imzml)
    shift_model, data, p = get_shift_model(ms_mzs, ms_ints, ref_mzs, x0 = x0,
                             ppm_flag=ppm_flag, max_delta_ppm=max_delta,
                             plot=plot, stabilise=stabilise)
    if not dump_fn is None:
        dump_var = {
            "ref": [ref_mzs['mz'], ref_mzs['count']],
            "aligned_mean_spec": [ms_mzs, ms_ints],
            "cal_points": [data[0], data[1]],
            "polyfit": p
        }
        import pickle
        pickle.dump(dump_var, open(dump_fn, 'wb'))
    with ImzMLWriter.ImzMLWriter(imzml_out_fn) as imzml_out:
        for ii, c in enumerate(aligned_imzml.coordinates):
            mzs, counts = map(np.asarray, aligned_imzml.getspectrum(ii))
            mzs = mzs+shift_model(mzs)
            imzml_out.addSpectrum(mzs, counts, coords=(c[0], c[1], 1))


def _get_neighbours(graph, node):
    nodes = list(graph.predecessors(node)) + list(graph.neighbors(node))
    return nodes


def calculate_node_offset(graph):
    # Set an offset per node equal to the average of its edges
    # !! will modify in place
    for node in graph.nodes:
        weights = []
        offsets = []
        for node2 in _get_neighbours(graph, node):
            offsets.append(graph.nodes[node2]['offset'])
            try:
                weights.append(-1 * graph.edges[node2, node]['weight'])
            except KeyError:
                weights.append(graph.edges[node, node2]['weight'])
        av_weight = np.mean(np.asarray(offsets) - np.asarray(weights))
        graph.nodes[node]['offset'] = float(av_weight)
    return graph


def adjust_graph(graph, reps=750):
        for node in graph.nodes:
            graph.nodes[node].setdefault('offset', 0.)
        for ii in range(reps):
            graph = calculate_node_offset(graph)
        return graph


def write_tmp_imzml(graph, imzml_fn, imzml_out_fn, ppm=True ):
    imzml = ImzMLParser.ImzMLParser(imzml_fn)
    with ImzMLWriter.ImzMLWriter(imzml_out_fn) as imzml_out:
        for ii, c in enumerate(imzml.coordinates):
            mzs, counts = map(np.asarray, imzml.getspectrum(ii))
            if ppm:
                delta_mz = 1e-6*mzs*float(graph.nodes[str(ii)]['offset'])
            else:
                delta_mz = float(graph.nodes[str(ii)]['offset'])
            mzs = mzs - delta_mz
            imzml_out.addSpectrum(mzs, counts, coords=(c[0], c[1], 1))


def align(imzml_fn, tmp_fn, ppm=True):
    graph = shift_graph.create_mass_shift_graph(imzml_fn=imzml_fn, num_cores = 6, ppm=ppm)
    graph = adjust_graph(graph)
    write_tmp_imzml(graph, imzml_fn, tmp_fn)
    return graph


def cluster(imzml_fn, imzml_out_fn, imzb_fn = "./tmp.imzb", cluster_ppm = 0.2):
    # cluster to get m/z axis
    ret = subprocess.check_call(["/home/palmer/miniconda2/envs/py-sm/bin/ims", "convert", imzml_fn, imzb_fn])
    assert ret==0, 'subprocess returned non zero value {}'.format(ret)
    imzb = ImzbReader(imzb_fn)
    dbscan = imzb.dbscan(eps=lambda mz: mz * cluster_ppm * 1e-6)
    bins = np.sort(np.concatenate([dbscan.left, dbscan.right]))
    # Rebin and write
    centroid_mzs = dbscan['mean']
    imzml = ImzMLParser.ImzMLParser(imzml_fn)
    with ImzMLWriter.ImzMLWriter(imzml_out_fn) as imzml_out:
        for ii, c in enumerate(imzml.coordinates):
            mzs, counts = map(np.asarray, imzml.getspectrum(ii))
            counts = np.bincount(np.digitize(mzs, bins), weights=counts, minlength=len(bins))
            imzml_out.addSpectrum(centroid_mzs, counts[1::2], coords=(c[0], c[1], 1))
    print(ii)

def pipeline(imzml_fn, imzml_recal_fn, config, metadata, tmp_fn = "tmp.imzML", ppm_flag=True, align_flag=True, dump_fn=None):
#    config = json.load(open('../metaspace_es_config.json'))
    if align_flag:
        align(imzml_fn, tmp_fn, ppm_flag)
    else:
        # spoof the alignment
        subprocess.check_call(["cp", imzml_fn, tmp_fn])
        subprocess.check_call(["cp", imzml_fn.replace('.imzML','.ibd'), tmp_fn.replace('.imzML','.ibd')])

    cluster(tmp_fn, tmp_fn.replace(".imzML", "_cl.imzML"), tmp_fn.replace(".imzML", ".imzb"))
    ref_mzs = pd.concat(
            get_reference_mzs(polarity=metadata['polarity'], tissue_type=metadata['organ'],
                              source=metadata['source'], database=metadata['database'],
                              fdr = 0.1, config=config, min_count=5)
        , axis=1)
    print("number of reference peaks: ", ref_mzs.shape)
    align_fn = tmp_fn
    cluster_fn = tmp_fn.replace(".imzML", "_cl.imzML")
    recalibrate(align_fn, cluster_fn, imzml_recal_fn, ref_mzs,
                ppm_flag=True,
                max_delta=30,
                x0=(0,0,0),
                stabilise=1,
                dump_fn=dump_fn
                )