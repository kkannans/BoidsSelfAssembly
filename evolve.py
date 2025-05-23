#!/usr/bin/env python
# Main script to run boid evolution
import numpy as np
import os
import sys
import json
import argparse
import signal
import time
import config
from hypothesis_space import HypothesisSpace
from experiment1 import Experiment1
from experiment4 import Experiment4
from afpo import AFPO
from afpo_elitism import AFPOElite
from utils import load_initial_positions, load_pairwise_distances, load_target_rdfs
import pickle

def is_running_in_slurm():
    """Check if the script is running in a SLURM environment"""
    return 'SLURM_JOB_ID' in os.environ

# Set flush value based on SLURM environment
FLUSH = is_running_in_slurm()

class GracefulExit:
    def __init__(self):
        self.should_exit = False
        self.original_sigint_handler = signal.getsignal(signal.SIGINT)
        self.original_sigterm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        print("\nReceived interrupt signal. Saving checkpoint and exiting gracefully...")
        self.should_exit = True
        
        # Restore original handlers
        signal.signal(signal.SIGINT, self.original_sigint_handler)
        signal.signal(signal.SIGTERM, self.original_sigterm_handler)
        
        # Re-raise the signal to ensure proper cleanup
        signal.raise_signal(signum)

def parse_arguments():
    """Parse command-line arguments"""
    # Pre-process sys.argv to handle = format
    processed_argv = []
    for arg in sys.argv[1:]:
        if '=' in arg:
            # Split on first = only
            key, value = arg.split('=', 1)
            processed_argv.extend([key, value])
        else:
            processed_argv.append(arg)
    
    parser = argparse.ArgumentParser(description='Evolve boid behaviors to match RDF patterns')
    
    parser.add_argument('-s', '--seed', type=int, default=config.RANDOM_SEED,
                      help='Random seed for reproducibility (default: from config)')
    # experiment number
    parser.add_argument('-e', '--experiment', type=int, default=1,
                      help='Experiment number 1,2,3 (default: 1)')
    # algorithm
    parser.add_argument('-a', '--algorithm', type=str, default="afpo",
                      help='Algorithm to use: afpo (default: afpo)')
    # generations
    parser.add_argument('-g', '--generations', type=int, default=config.GENERATIONS,
                      help='Number of generations to run (default: from config)')
    
    return parser.parse_args(processed_argv)

def make_config_dict(args):
    """Save current config settings to JSON file"""
    config_dict = {
        "simulation_settings": {
            "WIDTH": config.WIDTH,
            "HEIGHT": config.HEIGHT,
            "AREA": config.AREA,
            "VIEW_RADIUS": config.VIEW_RADIUS,
            "MAX_SPEED": config.MAX_SPEED,
            "MIN_SPEED": config.MIN_SPEED,
            "TIMESTEP": config.TIMESTEP
        },
        "rdf_settings": {
            "BIN_WIDTH": config.BIN_WIDTH,
            "R_MAX": config.R_MAX
        },
        "evolution_settings": {
            "GENERATIONS": args.generations,
            "POPULATION_SIZE": config.POPULATION_SIZE,
            "EVAL_INTERVAL": config.EVAL_INTERVAL,
            "RANDOM_SEED": args.seed
        },
        "time_slicing_settings": {
            "START_FRAME": config.START_FRAME,
            "END_FRAME": config.END_FRAME,
            "FRAME_SKIP": config.FRAME_SKIP,
            "FPS": config.FPS
        },
        "data_paths": {
            "INITIAL_POSITIONS_PATH": config.INITIAL_POSITIONS_PATH,
            "G_R_LIST_PATH": config.G_R_LIST_PATH,
            "RADII_LIST_PATH": config.RADII_LIST_PATH,
            "N_PAIRS_LIST_PATH": config.N_PAIRS_LIST_PATH,
            "OUTPUT_DIR": config.OUTPUT_DIR
        },
        "experiment_settings": {
            "EXPERIMENT_NUMBER": args.experiment
        },
        "algorithm_settings": {
            "ALGORITHM": args.algorithm
        }
    }
    
    # Add algorithm-specific settings
    if args.algorithm == "afpo":
        config_dict["algorithm_settings"].update({
            "AFPO_MUTATION_SCALE": config.AFPO_MUTATION_SCALE,
            "AFPO_MUTATION_RATE": config.AFPO_MUTATION_RATE
        })
    elif args.algorithm == "es":
        config_dict["algorithm_settings"].update({
            "ES_MUTATION_SCALE": config.ES_MUTATION_SCALE,
            "ES_MUTATION_RATE": config.ES_MUTATION_RATE,
            "ES_LAMBDA": config.ES_LAMBDA,
            "ES_MU": config.ES_MU
        })
    elif args.algorithm == "ga":
        config_dict["algorithm_settings"].update({
            "GA_MUTATION_RATE": config.GA_MUTATION_RATE,
            "GA_CROSSOVER_RATE": config.GA_CROSSOVER_RATE
        })
    elif args.algorithm == "cmaes":
        config_dict["algorithm_settings"].update({
            "CMAES_SIGMA": config.CMAES_SIGMA,
            "CMAES_POPULATION_SIZE": config.CMAES_POPULATION_SIZE
        })
    else:
        raise ValueError(f"Unsupported algorithm in evolve.py: {args.algorithm}")
    
    return config_dict

def load_checkpoint(output_dir):
    """Load checkpoint if it exists"""
    population_checkpoint_path = os.path.join(output_dir, 'population.pkl')
    config_path = os.path.join(output_dir, 'config.json')
    metrics_path = os.path.join(output_dir, 'metrics.npy')
    
    if not all(os.path.exists(p) for p in [population_checkpoint_path, config_path, metrics_path]):
        print(f"Checkpoint not found in {output_dir}")
        return None, None, None
    
    # Load population
    with open(population_checkpoint_path, 'rb') as f:
        population = pickle.load(f)
    
    # Load config
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Load metrics
    metrics = np.load(metrics_path, allow_pickle=True).item()
    
    return population, config_dict, metrics

def main():
    # Initialize graceful exit handler
    exit_handler = GracefulExit()
    print("Initialized graceful exit handler", flush=FLUSH)
    
    # Parse arguments
    args = parse_arguments()
    print(f"Parsed arguments: {args}", flush=FLUSH)

    config.RANDOM_SEED = args.seed
    # set output directory based on seed, experiment, and algorithm
    config.OUTPUT_DIR = f"./evolution_s{args.seed}_e{args.experiment}_{args.algorithm}"
    print(f"Output directory set to: {config.OUTPUT_DIR}", flush=FLUSH)
    
    # Create output directories
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    print("Created output directory", flush=FLUSH)
    
    np.random.seed(config.RANDOM_SEED)
    print(f"Set random seed to: {config.RANDOM_SEED}", flush=FLUSH)
    
    # Load initial positions
    print("Loading initial positions...", flush=FLUSH)
    initial_positions = load_initial_positions(config.INITIAL_POSITIONS_PATH)
    print("Loaded initial positions", flush=FLUSH)
    
    # check if initial velocities exist
    if os.path.exists(os.path.join(config.OUTPUT_DIR, 'initial_velocities.npy')):
        print("Loading initial velocities...", flush=FLUSH)
        initial_velocities = np.load(os.path.join(config.OUTPUT_DIR, 'initial_velocities.npy'))
        print("Loaded initial velocities", flush=FLUSH)
    else:
        # Initialize initial velocities with random seed
        print("Initializing velocities...", flush=FLUSH)
        initial_velocities = np.random.uniform(-0.01, 0.01, (config.NUM_BOIDS, 2))
            # save to file
        np.save(os.path.join(config.OUTPUT_DIR, 'initial_velocities.npy'), initial_velocities)
        print(f"Saved initial velocities to {config.OUTPUT_DIR}/initial_velocities.npy", flush=FLUSH)
    
    # Initialize experiment
    print(f"Initializing experiment {args.experiment}...", flush=FLUSH)
    if args.experiment == 1:
        hypothesis_space = Experiment1(initial_positions, initial_velocities, config.VIEW_RADIUS)
    elif args.experiment == 4:
        hypothesis_space = Experiment4(initial_positions, initial_velocities)
    else:
        raise ValueError(f"Invalid experiment number: {args.experiment}")
    print(f"Initialized hypothesis space for experiment {args.experiment}", flush=FLUSH)
    
    # Initialize algorithm with hypothesis space
    print(f"Initializing algorithm {args.algorithm}...", flush=FLUSH)
    if args.algorithm == "afpo":
        algorithm = AFPO(hypothesis_space, config.OUTPUT_DIR)
    elif args.algorithm == "afpo_elite":
        algorithm = AFPOElite(hypothesis_space, config.OUTPUT_DIR)
    print(f"Initialized algorithm {args.algorithm}", flush=FLUSH)
    
    print("Checking for checkpoint...", flush=FLUSH)
    population, config_dict, metrics = load_checkpoint(config.OUTPUT_DIR)
    if population is not None and config_dict is not None and metrics is not None:
        print(f"Found checkpoint in {config.OUTPUT_DIR}", flush=FLUSH)
        # Load metrics first to determine generation
        start_generation = max(metrics.keys()) + 1
        print(f"Will continue from generation {start_generation}", flush=FLUSH)
        
        # Load checkpoint into algorithm
        if algorithm.load_checkpoint():
            print(f"Successfully loaded checkpoint from generation {start_generation - 1}", flush=FLUSH)
        else:
            print("Failed to load checkpoint, starting fresh", flush=FLUSH)
            start_generation = 0
    else:
        print("No checkpoint found, starting fresh", flush=FLUSH)
        start_generation = 0
    
    # create config dict from args and save to file
    print("Saving config...", flush=FLUSH)
    config_dict = make_config_dict(args)
    with open(os.path.join(config.OUTPUT_DIR, 'config.json'), 'w') as f:
        json.dump(config_dict, f)
    print("Saved config", flush=FLUSH)
    
    # Run evolution
    print(f"Starting evolution for {args.generations} generations...", flush=FLUSH)
    timings_per_generation = []
    for generation in range(start_generation, args.generations):
        if exit_handler.should_exit:
            print("\nSaving checkpoint before exit...", flush=FLUSH)
            algorithm.save_checkpoint()
            print("Checkpoint saved. Exiting gracefully.", flush=FLUSH)
            return 0
            
        print(f"Running generation {generation}", flush=FLUSH)
        start_time = time.time()
        algorithm.evolve(generation)
        end_time = time.time()
        timings_per_generation.append(end_time - start_time)
    print("Evolution complete", flush=FLUSH)
    # save timings per generation
    np.save(os.path.join(config.OUTPUT_DIR, 'timings_per_generation.npy'), timings_per_generation)
    print(f"Saved timings per generation to {config.OUTPUT_DIR}/timings_per_generation.npy", flush=FLUSH)

    return 0

if __name__ == '__main__':
    sys.exit(main())