"""
Given folder name,
get seed and initial_velocities from folder name/initial_velocities.npy
get config from folder name/config.json
initialize hypothesis space from config
simulate boids for all frames (721)

compute pairwise distance loss and rdf loss per frame for random genome
compute total loss based on fitness logic (1 - sqrt( (1+ losses_random_genome_pwd)^2 + (1+ losses_random_genome_rdf)^2))

save positions and losses to ./test_performance/args.folder_name/
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt
from hypothesis_space import HypothesisSpace
from simulation import BoidSimulation
from boids import Boid
from loss_functions import loss_pairwise_distance, loss_rdf
from utils import load_pairwise_distances, load_target_rdfs, load_initial_positions
import config
import argparse
from experiment1 import Experiment1
from experiment4 import Experiment4
from tqdm import tqdm

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Test random genome performance')
    parser.add_argument('-exp', '--experiment_folder_name', type=str, required=False,
                       help='Name of experiment folder')
    parser.add_argument('-ev', '--evolved_folder_name', type=str, required=False,
                       help='Name of evolved folder')
    return parser.parse_args()

def load_config_from_folder(folder_path):
    """Load config.json from folder"""
    config_path = os.path.join(folder_path, 'config.json')
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return None
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    return config_dict

# from generate_random_individual in evolutionary_strategy.py
def generate_random_individual(experiment_number):
    """
    Generate a random individual based on experiment number
    
    Returns:
        dict: A new individual with random genome and fitness
    """
    # initialize based on the experiment
    if experiment_number == 1:
        # Experiment 1: 3 weights for all boids
        genome = np.random.uniform(config.MIN_BOID_WEIGHT, config.MAX_BOID_WEIGHT, 3)
    elif experiment_number == 2:
        # Experiment 2: 3 weights per boid
        genome_size = config.NUM_BOIDS * 3
        genome = np.random.uniform(config.MIN_BOID_WEIGHT, config.MAX_BOID_WEIGHT, genome_size)
    elif experiment_number == 3:
        # Experiment 3: 3 weights + view radius per boid
        genome_size = config.NUM_BOIDS * 4
        genome = np.random.uniform(config.MIN_BOID_WEIGHT, config.MAX_BOID_WEIGHT, genome_size)
        # Set view radius values to be within appropriate bounds
        for i in range(3, genome_size, 4):
            genome[i] = np.random.uniform(config.VIEW_RADIUS_MIN, config.VIEW_RADIUS_MAX)
    elif experiment_number == 4:
        # Experiment 4: 4 weights + view radius per boid
        genome_size = 4
        genome = np.random.uniform(config.MIN_BOID_WEIGHT, config.MAX_BOID_WEIGHT, genome_size)
        # Set view radius values to be within appropriate bounds
        genome[3] = np.random.uniform(0.01, 0.5)
    else:
        raise ValueError(f"Invalid experiment number: {experiment_number}")

    return genome

def test_trial(initial_positions, evolved_folder_name, test_performance_folder):
    # Create output directory with proper path joining
    output_dir = os.path.join(test_performance_folder, os.path.basename(evolved_folder_name))
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if directory is empty
    if os.listdir(output_dir):
        print(f"Skipping {output_dir} as it is not empty")
        return
    
    num_boids = int(initial_positions.shape[0])
    # Load initial velocities from file
    initial_velocities = np.load(os.path.join(evolved_folder_name, 'initial_velocities.npy'), allow_pickle=True)
    assert initial_velocities.shape == (num_boids, 2)
    
    # Load config
    evolved_config = load_config_from_folder(evolved_folder_name)
    if evolved_config is None:
        print("Failed to load evolved config")
        return 1
    
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
        hypothesis_space = Experiment1(initial_positions, initial_velocities, config.VIEW_RADIUS)
    elif experiment == 4:
        hypothesis_space = Experiment4(initial_positions, initial_velocities)
    else:
        raise ValueError(f"Invalid experiment number: {experiment}")

    # Initialize arrays to store results for all trials
    num_trials = 30
    all_pairwise_distance_losses = np.zeros((num_trials, num_frames))
    all_rdf_losses = np.zeros((num_trials, num_frames))
    all_positions = np.zeros((num_trials, num_frames, num_boids, 2))
    
    # Run multiple trials with different random genomes
    for trial in tqdm(range(num_trials), desc="Running trials"):
        # Get random genome
        random_genome = generate_random_individual(experiment)
        
        # Simulate boids for all frames
        positions = hypothesis_space.simulate_boid_positions(random_genome, num_steps=num_frames)
        # Store positions in original format (num_boids, num_frames, 2)
        all_positions[trial] = np.transpose(positions, (1, 0, 2))
        
        # Compute losses using loss functions
        # Calculate per-frame losses
        pairwise_distance_losses = np.array(loss_pairwise_distance(positions, hypothesis_space.pairwise_distances_video, per_frame=True))
        rdf_losses = np.array(loss_rdf(positions, per_frame=True))
        # Store results
        all_pairwise_distance_losses[trial] = pairwise_distance_losses
        all_rdf_losses[trial] = rdf_losses
    
    # Save results
    print("Saving results...")
    np.save(os.path.join(output_dir, 'positions_random_genome.npy'), all_positions)
    np.save(os.path.join(output_dir, 'pairwise_distance_losses_random_genome.npy'), all_pairwise_distance_losses)
    np.save(os.path.join(output_dir, 'rdf_losses_random_genome.npy'), all_rdf_losses)
    
    # Create and save plot
    plt.figure(figsize=(12, 6))
    frames = np.arange(num_frames)
    
    # Plot training frames in blue
    plt.plot(frames[:training_frames], np.mean(all_pairwise_distance_losses[:, :training_frames], axis=0), 
            color='blue', label='Training Frames (PWD)')
    plt.plot(frames[:training_frames], np.mean(all_rdf_losses[:, :training_frames], axis=0), 
            color='green', label='Training Frames (RDF)')
    
    # Plot test frames in red
    plt.plot(frames[training_frames:], np.mean(all_pairwise_distance_losses[:, training_frames:], axis=0), 
            color='red', label='Test Frames (PWD)')
    plt.plot(frames[training_frames:], np.mean(all_rdf_losses[:, training_frames:], axis=0), 
            color='orange', label='Test Frames (RDF)')
    
    # Add vertical line at training/test boundary
    plt.axvline(x=training_frames, color='gray', linestyle='--', 
                label='Training/Test Split')
    
    plt.title('Average Losses per Frame (30 Random Genomes)')
    plt.xlabel('Frame Number')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'random_genome_losses.png'))
    plt.close()
    
    print(f"\nResults saved to {output_dir}")
    print(f"Average PWD loss (training): {np.mean(all_pairwise_distance_losses[:, :training_frames]):.6f}")
    print(f"Average PWD loss (test): {np.mean(all_pairwise_distance_losses[:, training_frames:]):.6f}")
    print(f"Average RDF loss (training): {np.mean(all_rdf_losses[:, :training_frames]):.6f}")
    print(f"Average RDF loss (test): {np.mean(all_rdf_losses[:, training_frames:]):.6f}")

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
        test_performance_folder = f"test_performance_random_genome_{evolved_folder_name}"
    else:
        test_performance_folder = f"test_performance_random_genome_{experiment_folder_name}"

    os.makedirs(test_performance_folder, exist_ok=True)

    initial_positions = load_initial_positions(config.INITIAL_POSITIONS_PATH)

    if experiment_folder_name is None:
        test_trial(initial_positions, evolved_folder_name, test_performance_folder)
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