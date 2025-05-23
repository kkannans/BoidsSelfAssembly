import numpy as np
from utils import load_pairwise_distances, compute_pairwise_distances, load_target_rdfs
import config
from loss_functions import loss_pairwise_distance, loss_rdf


def fitness_pairwise_distance(positions: np.ndarray, pairwise_distances_video: list[np.ndarray], normalize:bool=True):
    """
    Calculate fitness based on earth mover's distance between pairwise distance histograms of boids and video.
    
    Args:
        positions: numpy array of shape (num_boids, num_steps, 2)
        pairwise_distances_video: list of numpy arrays of shape (num_boids, num_boids)
        normalize: whether to normalize the fitness
    Returns:
        fitness: scalar fitness value (to be maximized)
    """
    loss_pairwise_distance_value = loss_pairwise_distance(positions, pairwise_distances_video, normalize=normalize)
    fitness = -loss_pairwise_distance_value
    return fitness

    
def fitness_rdf(positions:np.ndarray):
    """
    Calculate fitness in terms of js distance between rdf of boids and target rdf
    
    Args:
        positions: numpy array of shape (num_boids, num_steps, 2)
    Returns:
        fitness: scalar fitness value (to be maximized)
    """
    loss_rdf_value = loss_rdf(positions)
    fitness = -loss_rdf_value
    return fitness

def fitness_per_frame_pairwise_distance(positions:np.ndarray, pairwise_distances_video:list[np.ndarray], normalize:bool=True):
    """
    Calculate fitness in terms of pairwise distance between boids and video.
    per frame fitness is fitness of pairwise distance at that frame
    mae = mean(abs(avg_pairwise_distance[t] - avg_pairwise_distance_video[t]))
    fitness is 1 - mae / diagonal_length
    Args:
        positions: numpy array of shape (num_boids, num_steps, 2)
        pairwise_distances_video: list of numpy arrays of shape (num_boids, num_boids)
        normalize: whether to normalize the fitness
    Returns:
        fitness_per_frame: list of fitness values for each frame
    """
    losses = loss_pairwise_distance(positions, pairwise_distances_video, per_frame=True, normalize=normalize)
    fitness_per_frame = []
    for t in range(len(losses)):
        fitness = losses[t]
        fitness_per_frame.append(fitness)
    return fitness_per_frame
    

def fitness_per_frame_rdf(positions: np.ndarray):
    """
    Calculate fitness in terms of js distance between rdf of boids and target rdf for each frame.
    
    Args:
        positions: numpy array of shape (num_boids, num_steps, 2)
    Returns:
        fitness_per_frame: list of fitness values for each frame
    """
    loss_per_frame_rdfs = loss_rdf(positions, per_frame=True)
    fitness_per_frame = []
    for t in range(len(loss_per_frame_rdfs)):
        fitness = loss_per_frame_rdfs[t]
        fitness_per_frame.append(fitness)
    return fitness_per_frame
