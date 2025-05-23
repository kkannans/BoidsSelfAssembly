"""
Given the change in pairwise distances between cells
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import matplotlib.animation as animation

def calculate_rdf_from_distance_matrix(D, area, dr, r_max=None):
    """
    Calculates the 2D Radial Distribution Function (g(r)) from a distance matrix.

    Args:
        D (np.ndarray): A square N x N symmetric matrix where D[i, j] is the
                        distance between particle i and particle j. Diagonal
                        elements should be 0.
        area (float): The total area (in distance units squared, e.g., pixels^2)
                      in which the N particles are distributed.
        dr (float): The width of the histogram bins for r.
        r_max (float, optional): The maximum distance to calculate g(r) for.
                                 If None, defaults to half the maximum distance
                                 found in D, or dr if max distance is 0.

    Returns:
        tuple: A tuple containing:
            - g_r (np.ndarray): The calculated g(r) values for each bin. Returns
                              empty array if N < 2.
            - radii (np.ndarray): The center distances r for each bin. Returns
                                empty array if N < 2.
            - N_pairs (np.ndarray): The raw number of pairs counted in each bin.
                                    Returns empty array if N < 2.
    """
    N = D.shape[0]
    # Need at least 2 particles to calculate pairwise distances
    if N < 2:
        # Return empty arrays consistent with expected output types
        return np.array([]), np.array([]), np.array([])

    # 1. Extract unique pairwise distances (upper triangle, excluding diagonal)
    distances = D[np.triu_indices(N, k=1)]

    # Should not happen if N >= 2, but check just in case
    if distances.size == 0:
         return np.array([]), np.array([]), np.array([])

    # Use provided r_max or calculate from distances
    current_r_max = r_max
    if current_r_max is None:
        max_dist = distances.max()
        # Ensure r_max is positive even if max_dist is 0 (e.g., all particles coincident)
        current_r_max = max(max_dist / 2.0, dr)

    # 2. Histogram the distances
    # Ensure n_bins is at least 1
    n_bins = int(np.ceil(current_r_max / dr))
    if n_bins == 0: n_bins = 1
    actual_r_max = n_bins * dr # The actual upper limit of the histogram range

    # Use numpy.histogram to count pairs in each bin
    N_pairs, bin_edges = np.histogram(distances, bins=n_bins, range=(0, actual_r_max))

    # Calculate bin centers (radii)
    radii = bin_edges[:-1] + dr / 2.0

    # 3. Normalize to get g(r)
    g_r = np.zeros_like(N_pairs, dtype=float)

    for k in range(n_bins):
        r_lower = bin_edges[k]
        r_upper = bin_edges[k+1]
        # Area of the annulus (shell) dA = pi*(r_upper^2 - r_lower^2)
        dA = np.pi * (r_upper**2 - r_lower**2)
        if dA == 0: # Avoid division by zero if dr is tiny or r is 0
            # Should not happen with proper binning, but safeguard
            continue
        # for non-periodic and non-square frames, g(k) = (2 * N_pairs(k) * A) / (N*(N-1) * dA)
        g_r[k] = (2.0 * N_pairs[k] * area) / (N * (N - 1) * dA)

    return g_r, radii, N_pairs

def radial_distribution_function_from_video(distances:dict, frame_width:int, frame_height:int, bin_size:float):
    """
    Calculate the radial distribution function of the distances
    
    Args:
        distances: Dictionary with frame indices as keys and distance matrices as values
        frame_width: Width of the video frame
        frame_height: Height of the video frame
        bin_size: Width of the histogram bins for r
        
    Returns:
        g_r_list: Dictionary with frame indices as keys and g(r) arrays as values
        radii: Common array of radii values used for all frames
        N_pairs_list: Dictionary with frame indices as keys and N_pairs arrays as values
    """
    # since it is non-periodic and non-square, use min(width, height)/2 as r_max
    r_max = min(frame_width, frame_height) / 2
    
    # Calculate the common bin edges and radii that will be used for all frames
    n_bins = int(np.ceil(r_max / bin_size))
    if n_bins == 0: n_bins = 1
    actual_r_max = n_bins * bin_size
    bin_edges = np.linspace(0, actual_r_max, n_bins + 1)
    radii = bin_edges[:-1] + bin_size / 2.0
    
    g_r_list = {}
    N_pairs_list = {}
    
    for frame_idx in distances.keys():
        # get the distance matrix for the current frame
        distance_matrix = distances[frame_idx]
        # get the number of cells in the current frame
        num_cells = distance_matrix.shape[0]
        # get the radial distribution function
        area = frame_width * frame_height
        g_r, _, N_pairs = calculate_rdf_from_distance_matrix(distance_matrix, area, bin_size, r_max)
        g_r_list[frame_idx] = g_r
        N_pairs_list[frame_idx] = N_pairs
    
    return g_r_list, radii, N_pairs_list

if __name__ == "__main__":
    # load pairwise distances from file that has the dictonary with frame index as key and distance matrix as value
    
    all_distances = np.load('./all_distances_bw_longer.npy', allow_pickle=True).item()

    # get width and height from the first frame of video ./bw_longer.mp4
    cap = cv2.VideoCapture('./bw_longer.mp4')
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bin_size = 5
    # get the radial distribution function
    g_r_list, radii, N_pairs_list = radial_distribution_function_from_video(all_distances, video_width, video_height, bin_size)

    # save the radial distribution function
    np.save('g_r_list.npy', g_r_list)
    np.save('radii.npy', radii)  # Save single radii array instead of dictionary
    np.save('N_pairs_list.npy', N_pairs_list)
    print(f"Frame dimensions: {video_width}x{video_height}")
    print(f"Maximum distance (r_max): {min(video_width, video_height)/2}")
    print(f"bin size: {bin_size}")
    print(f"Number of bins: {len(radii)}")
    print(f"Radii range: [{radii[0]}, {radii[-1]}]")
