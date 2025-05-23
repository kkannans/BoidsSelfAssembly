# Configuration for boid simulation and evolution
import numpy as np
import time
import os
import multiprocessing

# Simulation settings
WIDTH = 1920  # in pixels
HEIGHT = 1460  # in pixels
AREA = WIDTH * HEIGHT
NUM_BOIDS = 691  # Number of boids to simulate
WEIGHTS_PER_BOID = 3  # Number of weights per boid

# Boid size settings
BOID_SIZE_PIXELS = 10  # Size of each boid in pixels
MAX_BOID_WEIGHT = 1.0
MIN_BOID_WEIGHT = -1.0
VIEW_RADIUS_MIN = 10
VIEW_RADIUS_MAX = int(WIDTH/2.0)
VIEW_RADIUS = 40

# Convert measurements to boid size units
MAX_SPEED = 2 # 2 pixels per timestep
MIN_SPEED = 0  # Minimum boid speed
TIMESTEP = 1.0  # Simulation timestep

# RDF calculation settings
BIN_WIDTH = 5
R_MAX = WIDTH/2 # Maximum distance for RDF in boid size units

# Evolution settings
GENERATIONS = 2  # Number of generations
POPULATION_SIZE = 2  # Population size
EVAL_INTERVAL = 1  # Evaluate every n frames
RANDOM_SEED = 1 # Random seed for reproducibility
NUM_PROCESSES = multiprocessing.cpu_count()  # Number of processes for parallel fitness evaluation

# GA settings
GA_MUTATION_SCALE = 0.01  # mutation scale for GA
GA_MUTATION_RATE = 0.5  # mutation rate for GA
GA_CROSSOVER_RATE = 0.5  # crossover rate for GA
GA_TOURNAMENT_SIZE = 2  # tournament size for GA

# ES settings
ES_MUTATION_SCALE = 0.01  # mutation scale for ES
ES_MUTATION_RATE = 0.5  # mutation rate for ES
ES_MUTATION_SCALE_VIEW_RADIUS = 0.1  # mutation scale for ES view radius
ES_LAMBDA = POPULATION_SIZE  # number of children for ES
ES_MU = ES_LAMBDA // 2  # number of parents for ES 

# AFPO settings
AFPO_MUTATION_SCALE = 0.01  # mutation rate for AFPO
AFPO_MUTATION_SCALE_VIEW_RADIUS = 0.01  # mutation rate for AFPO view radius
AFPO_MUTATION_RATE = 0.5  # mutation rate for AFPO

# CMA-ES settings
CMAES_MUTATION_SCALE = 0.01  # mutation scale for CMA-ES
CMAES_MUTATION_SCALE_VIEW_RADIUS = 0.1  # mutation scale for CMA-ES view radius
CMAES_MUTATION_RATE = 0.5  # mutation rate for CMA-ES
CMAES_LAMBDA = POPULATION_SIZE  # number of children for CMA-ES
CMAES_MU = CMAES_LAMBDA // 2  # number of parents for CMA-ES

# Time slicing settings for original videos
START_FRAME = 0  # Starting frame index
END_FRAME = 576  # Ending frame index (None means use all frames) # Training we use 576 frames
FRAME_SKIP = 1  # Number of frames to skip between samples
FPS = 10  # Frames per second of the original video

# Data paths
INITIAL_POSITIONS_PATH = "./extract_metrics/initial_positions.csv"
G_R_LIST_PATH = "./extract_metrics/g_r_list.npy"
RADII_LIST_PATH = "./extract_metrics/radii.npy"
N_PAIRS_LIST_PATH = "./extract_metrics/N_pairs_list.npy"
OUTPUT_DIR = f"./evolution_s{RANDOM_SEED}_afpo"  # Will be updated based on seed
VIDEO_PATH = "./extract_metrics/bw_longer.mp4"
PAIRWISE_DISTANCES_PATH = "./extract_metrics/all_distances_bw_longer.npy"
