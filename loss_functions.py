import numpy as np
from utils import compute_pairwise_distances, load_target_rdfs
from scipy.stats import wasserstein_distance as emd
from scipy.spatial.distance import jensenshannon
from extract_metrics.get_self_assembly_metrics import radial_distribution_function_from_video
import config
from tqdm import tqdm
from pprint import pprint
import matplotlib.pyplot as plt

def RMSE(a, b):
    """
    Calculate root mean square error between two one dimensional arrays 
    a: (m, 1)
    b: (m, 1)
    """
    assert a.shape == b.shape
    # compute mean square error between a and b
    return np.sqrt(np.mean((a - b) ** 2)) 

# use MAE for loss function since robust to outliers possible in cell segmentation
def MAE(a, b):
    """
    Calculate mean absolute error between two one dimensional arrays 
    a: (m, 1)
    b: (m, 1)
    """
    assert a.shape == b.shape
    return np.mean(np.abs(a - b))

def MSE(a, b):
    """
    Calculate mean square error between two one dimensional arrays 
    a: (m, 1)
    b: (m, 1)
    """
    assert a.shape == b.shape
    return np.mean((a - b) ** 2)

def compute_histogram(distances, normalize:bool=True):
    """
    Compute histogram of pairwise distances.
    Args:
        distances: either a single distance matrix or a list of distance matrices
    Returns:
        if single matrix: histogram values and bins
        if list of matrices: list of normalized histogram values and bins
    """
    if isinstance(distances, list):
        histograms = []
        for dist in distances:
            # Extract only the upper triangular elements (not including diagonal)
            upper_triangular_flat = dist[np.triu_indices(dist.shape[0], k=1)]
            # get histogram of upper triangular part
            hist, bins = np.histogram(upper_triangular_flat, bins=config.BIN_WIDTH, range=(0, config.R_MAX))
            if normalize:
                hist_normalized = hist / np.sum(hist)
                histograms.append(hist_normalized)
            else:
                histograms.append(hist)
        return histograms, bins
    else:
        # Extract only the upper triangular elements (not including diagonal)
        upper_triangular_flat = distances[np.triu_indices(distances.shape[0], k=1)]
        # get histogram of upper triangular part
        hist, bins = np.histogram(upper_triangular_flat, bins=config.BIN_WIDTH, range=(0, config.R_MAX))
        if normalize:
            hist_normalized = hist / np.sum(hist)
            return hist_normalized, bins
        else:
            return hist, bins

def compute_avg_pairwise_distance(pairwise_distances:list[np.ndarray]):
    """
    Compute average pairwise distance across all frames.
    Note: we need to remove diagonal elements and only choose upper or lower triangular part
    Args:
        pairwise_distances: list of numpy arrays of shape (num_boids, num_boids)
    Returns:
        numpy array of shape (num_steps,)
    """
    avg_pairwise_distances = []
    for pairwise_distance in pairwise_distances:
        # remove diagonal elements
        pairwise_distance_flat = pairwise_distance[np.triu_indices(pairwise_distance.shape[0], k=1)]
        # compute average
        avg_pairwise_distances.append(np.mean(pairwise_distance_flat))
    return np.array(avg_pairwise_distances)

def loss_pairwise_distance(positions:np.ndarray, pairwise_distances_video:list[np.ndarray], per_frame:bool=False, normalize:bool=False):
    """
    Calculate loss based on between pairwise distance of boids and video.
    computes MAE between average of pairwise distances of boids and video.
    """
    pairwise_distances = compute_pairwise_distances(positions)

    avg_pairwise_distances = compute_avg_pairwise_distance(pairwise_distances)
    avg_pairwise_distances_video = compute_avg_pairwise_distance(pairwise_distances_video)
    num_steps = len(pairwise_distances)
    loss = []
    for t in range(num_steps):
        # compute MAE between averages of pairwise distances
        loss.append(MAE(avg_pairwise_distances[t], avg_pairwise_distances_video[t])) 
    if per_frame:
        return loss
    else:
        # compute mean across all frames
        return np.mean(loss)

def loss_rdf(positions:np.ndarray, per_frame:bool=False):
    """
    Calculate loss based on JS distance between RDF of boids and video.
    """
    pairwise_distances = compute_pairwise_distances(positions)
    num_steps = len(pairwise_distances)
    # convert to dict with frame index as key and distance matrix as value
    pairwise_distances_dict = {i: pairwise_distances[i] for i in range(num_steps)}
    gr_sim, r_sim, N_pairs_sim = radial_distribution_function_from_video(pairwise_distances_dict, config.WIDTH, config.HEIGHT, config.BIN_WIDTH)
    # get target rdf from utils
    g_r_target, radii, N_pairs_target = load_target_rdfs()
    loss = []
    for t in range(num_steps):
        loss.append(jensenshannon(gr_sim[t], g_r_target[t]))
    if per_frame:
        return loss
    else:
        # compute mean across all frames
        return np.mean(loss)