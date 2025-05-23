import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from scipy.spatial.distance import pdist, squareform
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib import cm
from tqdm import tqdm

def compute_cell_pairwise_distances(masks):
    """
    Compute pairwise distances between cells in each frame using Cellpose masks.
    
    Parameters:
    -----------
    masks : list
        List of masks as arrays
    subsample_rate : int
        Subsample rate for the frames to process
        
    Returns:
    --------
    dict
        Dictionary where keys are frame indices and values are distance matrices
    """
    # Dictionary to store results
    all_distances = {}
    
    # Process each frame
    for frame_idx, mask in tqdm(enumerate(masks), total=len(masks)):
        print(f"Processing frame {frame_idx}")
        
        # Find unique cell labels (excluding background with value 0)
        cell_labels = np.unique(mask)[1:]  # Skip 0 which is background
        n_cells = len(cell_labels)
        
        if n_cells == 0:
            print(f"No cells found in frame {frame_idx}")
            all_distances[frame_idx] = np.zeros((0, 0))
            continue
        
        # Extract centroids
        centroids = []
        # Sort cell labels to ensure consistent ordering
        cell_labels = sorted(cell_labels)
        for label in cell_labels:
            # Get mask for this specific cell
            cell_mask = (mask == label)
            # Calculate centroid coordinates
            centroid = center_of_mass(cell_mask)
            centroids.append(centroid)
        
        centroids = np.array(centroids)
        
        # Compute pairwise distances between all centroids
        if n_cells > 1:
            # pdist returns condensed distance matrix, squareform converts to square form
            distance_matrix = squareform(pdist(centroids, metric='euclidean'))
        else:
            distance_matrix = np.zeros((1, 1))
        
        # Store the distance matrix
        all_distances[frame_idx] = distance_matrix
    
    return all_distances

if __name__ == "__main__":
    # Path to the masks.npy file
    masks_file = "masks_cyto_None_bw_longer.npy"

    masks = np.load(masks_file, allow_pickle=True)

    # get first 10 masks
    # masks = masks[]

    # Compute distances
    distances = compute_cell_pairwise_distances(masks)
    
    # Save all distance matrices to a single file
    np.save(f"all_distances_bw_longer.npy", distances)
    