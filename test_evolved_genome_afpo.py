"""
Given experiment folder name, containing path to seed folder that contains population.pkl,
load the population,
get best_pwd, best_rdf and best of both from population
load config.json
initialize hypothesis space from config
load initial velocities from initial_velocities.npy
simulate for each of three genomes in population for all frames in test set which is 721 - config["END_FRAME"] + 1

compute pairwise distance loss and rdf loss per frame for each of three genomes 
compute total loss based on fitness logic (1 - sqrt( (1+ losses_evolved_genome_pwd)^2 + (1+ losses_evolved_genome_rdf)^2))

"""
#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import pickle
import json
from boids import Boid
from simulation import BoidSimulation
import config
from utils import load_initial_positions
from loss_functions import loss_pairwise_distance, loss_rdf
from hypothesis_space import HypothesisSpace
from experiment1 import Experiment1
from experiment4 import Experiment4
from tqdm import tqdm
from typing import List, Dict, Tuple

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Test evolved genome performance')
    parser.add_argument('-exp', '--experiment_folder_name', type=str, required=False,
                       help='Name of experiment folder')
    parser.add_argument('-ev', '--evolved_folder_name', type=str, required=False,
                       help='Name of evolved folder')
    return parser.parse_args()

def get_waist(population: List[Dict]) -> List[Dict]:
    """Find the waist of the population (individuals closest to maximum values in both objectives)."""
    # Find maximum values for both fitness metrics
    max_pwd = max(p['fitness_pwd'] for p in population)
    max_rdf = max(p['fitness_rdf'] for p in population)
    
    # Calculate distances for all individuals
    distances = []
    for p in population:
        # Calculate normalized distance to maximum point
        norm_pwd = (max_pwd - p['fitness_pwd']) / max_pwd
        norm_rdf = (max_rdf - p['fitness_rdf']) / max_rdf
        distance = np.sqrt(norm_pwd**2 + norm_rdf**2)
        distances.append((distance, p))
    
    # Sort by distance and get the closest
    distances.sort(key=lambda x: x[0])
    min_distance = distances[0][0]
    waist = [p for dist, p in distances if dist == min_distance]
    return waist[0]

def get_best_individuals(population: List[Dict]) -> Tuple[Dict, Dict]:
    """Find the best individuals for each fitness metric."""
    best_pwd = max(population, key=lambda x: x['fitness_pwd'])
    best_rdf = max(population, key=lambda x: x['fitness_rdf'])
    return best_pwd, best_rdf

def load_best_genomes(folder_path):
    """Load best_pwd, best_rdf and best_combined from file"""
    if not os.path.exists(folder_path):
        print(f"Genome file not found: {folder_path}")
        sys.exit(1)
    # get population from population.pkl
    population_path = os.path.join(folder_path, 'population.pkl')
    with open(population_path, 'rb') as f:
        population = pickle.load(f)
    best_pwd, best_rdf = get_best_individuals(population)
    best_combined = get_waist(population)
    return best_pwd, best_rdf, best_combined

def load_config_from_folder(folder_path):
    """Load config.json from folder"""
    config_path = os.path.join(folder_path, 'config.json')
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return None
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    return config_dict

def test_trial(initial_positions, evolved_folder_name, test_performance_folder):
    # Create output directory with proper path joining
    output_dir = os.path.join(test_performance_folder, os.path.basename(evolved_folder_name))
    os.makedirs(output_dir, exist_ok=True)
    
    num_boids = int(initial_positions.shape[0])
    # Load initial velocities from file
    initial_velocities = np.load(os.path.join(evolved_folder_name, 'initial_velocities.npy'), allow_pickle=True)
    assert initial_velocities.shape == (num_boids, 2)
    
    # Load config
    evolved_config = load_config_from_folder(evolved_folder_name)
    if evolved_config is None:
            print("Failed to load evolved config")
            return 1
        
    best_pwd, best_rdf, best_combined = load_best_genomes(evolved_folder_name)
    # Get time slicing settings from evolved config
    training_frames = evolved_config["time_slicing_settings"]["END_FRAME"] + 1
    num_frames = 721  # Total number of frames
    test_frames = num_frames - training_frames
    
    print(f"\nSimulation settings:")
    print(f"  Total boids: {initial_positions.shape[0]}")
    print(f"  Training frames: {training_frames}")
    print(f"  Test frames: {test_frames}")

    # Initialize hypothesis space based on config
    experiment = evolved_config["experiment_settings"]["EXPERIMENT_NUMBER"]
    if experiment == 1:
        view_radius = config.VIEW_RADIUS
        hypothesis_space = Experiment1(initial_positions, initial_velocities, view_radius)
    elif experiment == 2:
        hypothesis_space = Experiment2(initial_positions, initial_velocities)
    elif experiment == 3:
        hypothesis_space = Experiment3(initial_positions, initial_velocities)
    elif experiment == 4:
        hypothesis_space = Experiment4(initial_positions, initial_velocities)
    else:
        raise ValueError(f"Invalid experiment number: {experiment}")

    # for each of best_pwd, best_rdf and best_combined, simulate boids for all frames
    positions_best_pwd = hypothesis_space.simulate_boid_positions(best_pwd['genome'], num_steps=num_frames)
    positions_best_rdf = hypothesis_space.simulate_boid_positions(best_rdf['genome'], num_steps=num_frames)
    positions_best_combined = hypothesis_space.simulate_boid_positions(best_combined['genome'], num_steps=num_frames)

    # save positions to file
    np.save(os.path.join(output_dir, 'positions_best_pwd.npy'), positions_best_pwd)
    np.save(os.path.join(output_dir, 'positions_best_rdf.npy'), positions_best_rdf)
    np.save(os.path.join(output_dir, 'positions_best_combined.npy'), positions_best_combined)

    # compute losses for all frames
    # best_pwd
    pairwise_distance_losses_best_pwd = np.array(loss_pairwise_distance(positions_best_pwd, hypothesis_space.pairwise_distances_video, per_frame=True))
    rdf_losses_best_pwd = np.array(loss_rdf(positions_best_pwd, per_frame=True))
    losses_total_best_pwd = np.sqrt((1 + pairwise_distance_losses_best_pwd)**2 + (1 + rdf_losses_best_pwd)**2) - 1 # since fitness = -loss
    # best_rdf
    pairwise_distance_losses_best_rdf = np.array(loss_pairwise_distance(positions_best_rdf, hypothesis_space.pairwise_distances_video, per_frame=True))
    rdf_losses_best_rdf = np.array(loss_rdf(positions_best_rdf, per_frame=True))
    losses_total_best_rdf = np.sqrt((1 + pairwise_distance_losses_best_rdf)**2 + (1 + rdf_losses_best_rdf)**2) - 1 # since fitness = -loss
    # best_combined
    pairwise_distance_losses_best_combined = np.array(loss_pairwise_distance(positions_best_combined, hypothesis_space.pairwise_distances_video, per_frame=True))
    rdf_losses_best_combined = np.array(loss_rdf(positions_best_combined, per_frame=True))
    losses_total_best_combined = np.sqrt((1 + pairwise_distance_losses_best_combined)**2 + (1 + rdf_losses_best_combined)**2) - 1 # since fitness = -loss
    # save losses to file for each of best_pwd, best_rdf and best_combined
    np.save(os.path.join(output_dir, 'pairwise_distance_losses_best_pwd.npy'), pairwise_distance_losses_best_pwd)
    np.save(os.path.join(output_dir, 'rdf_losses_best_pwd.npy'), rdf_losses_best_pwd)
    np.save(os.path.join(output_dir, 'losses_total_best_pwd.npy'), losses_total_best_pwd)
    np.save(os.path.join(output_dir, 'pairwise_distance_losses_best_rdf.npy'), pairwise_distance_losses_best_rdf)
    np.save(os.path.join(output_dir, 'rdf_losses_best_rdf.npy'), rdf_losses_best_rdf)
    np.save(os.path.join(output_dir, 'losses_total_best_rdf.npy'), losses_total_best_rdf)
    np.save(os.path.join(output_dir, 'pairwise_distance_losses_best_combined.npy'), pairwise_distance_losses_best_combined)
    np.save(os.path.join(output_dir, 'rdf_losses_best_combined.npy'), rdf_losses_best_combined)
    np.save(os.path.join(output_dir, 'losses_total_best_combined.npy'), losses_total_best_combined)

    print(f"Best PWD fitness: {best_pwd['fitness_pwd']}")
    print(f"Best RDF fitness: {best_rdf['fitness_rdf']}")
    combined = 1 - np.sqrt((1 + best_pwd['fitness_pwd'])**2 + (1 + best_rdf['fitness_rdf'])**2)
    print(f"Best Combined fitness: {combined}")
    print(f"\nResults saved to {output_dir}")

def main():
    args = parse_arguments()
    if args.experiment_folder_name is None:
        experiment_folder_name = None
    else:
        experiment_folder_name = args.experiment_folder_name.split("/")[-1]
    if args.evolved_folder_name is None:
        evolved_folder_name = None
    else:
        evolved_folder_name = args.evolved_folder_name.split("/")[-1]
    if experiment_folder_name is None:
        test_performance_folder = f"test_performance_{evolved_folder_name}"
    else:
        test_performance_folder = f"test_performance_{experiment_folder_name}"

    os.makedirs(test_performance_folder, exist_ok=True)

    initial_positions = load_initial_positions(config.INITIAL_POSITIONS_PATH)
    if experiment_folder_name is None:
        evolved_folder_path = args.evolved_folder_name
        test_trial(initial_positions, evolved_folder_path, test_performance_folder)
    else:
        # check if experiment_folder_name exists
        if not os.path.exists(experiment_folder_name):
            print(f"Experiment folder not found: {experiment_folder_name}")
            return 1
        else:
            evolved_folders = os.listdir(args.experiment_folder_name)
            pbar = tqdm(evolved_folders)
            for evolved_folder in pbar:
                pbar.set_description(f"Testing evolved folder: {evolved_folder}")
                # pass full path to test_trial
                test_trial(initial_positions, os.path.join(args.experiment_folder_name, evolved_folder), test_performance_folder)
    print("Test completed successfully")

if __name__ == "__main__":
    main()

