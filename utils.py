import numpy as np
import config
from scipy.spatial.distance import pdist, squareform
from numba import jit, float64, int64
from scipy.signal import find_peaks

def load_initial_positions(path=config.INITIAL_POSITIONS_PATH):
    """
    Load initial positions and apply subset selection
    
    Args:
        path: path to CSV file with initial positions
    Returns:
        initial_positions: numpy array of shape (num_boids, 2)
    """
    # Load initial positions
    initial_positions = np.loadtxt(path, delimiter=",", skiprows=1)
    
    print(f"Loaded {initial_positions.shape[0]} boids")
    
    return initial_positions


def load_pairwise_distances(path=config.PAIRWISE_DISTANCES_PATH):
    """
    Load pairwise distances from video
    Returns:
        pairwise_distances: list of numpy arrays of shape (num_boids, num_boids) of length num_frames in the video
    """
    pairwise_distances = np.load(path, allow_pickle=True).item()
    # sort by frame number
    pairwise_distances = {k: pairwise_distances[k] for k in sorted(pairwise_distances.keys())}
    # return as list of numpy arrays
    return list(pairwise_distances.values())

def compute_distance_matrix(positions):
    """
    Compute pairwise distance matrix for positions
    
    Args:
        positions: numpy array of shape (num_boids, 2)
        
    Returns:
        D: distance matrix of shape (num_boids, num_boids)
    """
    # Fast computation using scipy's pdist and squareform
    D = squareform(pdist(positions))
    return D

def compute_pairwise_distances(positions:np.ndarray):
    """
    Compute pairwise distances for each frame sequentially
    
    Args:
        positions: numpy array of shape (num_boids, num_steps, 2)
        
    Returns:
        pairwise_distances: list of numpy arrays of shape (num_boids, num_boids)
    """
    pairwise_distances = []
    for t in range(positions.shape[1]):  # Iterate over time steps
        frame_positions = positions[:, t, :]  # Get positions for this frame
        distances = compute_distance_matrix(frame_positions)
        pairwise_distances.append(distances)
    return pairwise_distances

def load_target_rdfs(g_r_path=config.G_R_LIST_PATH, 
                    radii_path=config.RADII_LIST_PATH,
                    n_pairs_path=config.N_PAIRS_LIST_PATH):
    """
    Load precomputed target RDFs without frame slicing
    
    Returns:
        g_r_array: numpy array of shape (num_frames, max_bins)
        radii: numpy array of shape (max_bins,)
        N_pairs_array: numpy array of shape (num_frames,)
    """
    # Load data
    g_r_dict = np.load(g_r_path, allow_pickle=True).item()
    radii = np.load(radii_path, allow_pickle=True)  # Load single radii array of shape (max_bins,)
    N_pairs_dict = np.load(n_pairs_path, allow_pickle=True).item()

    # convert dict to numpy array of shape (num_frames, max_bins)
    frame_indices = g_r_dict.keys()
    # sort frame indices
    frame_indices = sorted(frame_indices)
    gr_array = np.zeros((len(frame_indices), len(g_r_dict[frame_indices[0]])))
    N_pairs_array = np.zeros((len(frame_indices), len(N_pairs_dict[frame_indices[0]])))
    for i, frame_idx in enumerate(frame_indices):
        gr_array[i, :] = g_r_dict[frame_idx]
        N_pairs_array[i, :] = N_pairs_dict[frame_idx]
    return gr_array, radii, N_pairs_array

def get_first_peak_from_rdf(g_r):
    """
    Get the first peak from the RDF
    """
    return np.argmax(g_r)

if __name__ == "__main__":
    pairwise_distances = load_pairwise_distances()
    print(len(pairwise_distances))
    print(pairwise_distances[0].shape)
