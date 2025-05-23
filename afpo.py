# Evolutionary algorithm for optimizing boid behaviors using AFPO
import numpy as np
from scipy.spatial.distance import jensenshannon
import os
import json
import pickle
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
import config
from boids import Boid
from evolutionary_algorithm import EvolutionaryAlgorithm, calculate_genomic_diversity
import random
from copy import deepcopy
from hypothesis_space import HypothesisSpace
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class AFPO(EvolutionaryAlgorithm):
    """Age-Fitness Pareto Optimization implementation with three objectives: age, pwd fitness, and rdf fitness"""
    
    def __init__(self, hypothesis_space:HypothesisSpace, output_dir:str=None):
        """
        Initialize AFPO
        
        Args:
            config_dict: dictionary of configuration parameters
            keys:
                population_size: size of population
                genome_dim: dimension of the genome (should be config.NUM_BOIDS*3)
                generations: number of generations to run
                evaluate_fitness: function to evaluate fitness
        """
        super().__init__(hypothesis_space, output_dir)
        self.mutation_scale = config.AFPO_MUTATION_SCALE
        self.mutation_rate = config.AFPO_MUTATION_RATE
        self.mutation_scale_view_radius = config.AFPO_MUTATION_SCALE_VIEW_RADIUS
        self.index = 0

        # set bounds for genome values based on the experiment
        if self.experiment == 3:
            self.lower_bound = -1
            self.upper_bound = 1
            self.view_radius_lower_bound = config.VIEW_RADIUS_MIN
            self.view_radius_upper_bound = config.VIEW_RADIUS_MAX
        elif self.experiment == 4:
            self.lower_bound = -1
            self.upper_bound = 1
            self.view_radius_lower_bound = config.VIEW_RADIUS_MIN
            self.view_radius_upper_bound = config.VIEW_RADIUS_MAX
        else:
            self.lower_bound = -1
            self.upper_bound = 1

        self.metrics = {}
        self.current_generation = 0
        self.best_individual_ever = None
        # Try to load from checkpoint first
        if self.load_checkpoint():
            logger.info(f"Successfully loaded checkpoint from generation {self.current_generation}")
        else:
            # initialize the population if no checkpoint found
            logger.info(f"Initializing population with {self.population_size} individuals")
            self.initialize_population()

    # override
    def evaluate_fitness(self, genome:np.ndarray):
        """Evaluate fitness of an individual"""
        start_time = time.time()
        fitness_pwd, fitness_rdf = self.hypothesis_space.multi_objective_fitness_function(genome, self.num_steps)
        end_time = time.time()
        logger.info(f"Individual child fitness evaluation time: {end_time - start_time:.2f}s")
        return fitness_pwd, fitness_rdf

    # override
    def evaluate_population_fitness(self):
        """Evaluate fitness of all individuals sequentially or in parallel"""
        start_time = time.time()
        
        # Skip evaluation for individuals that already have fitness
        unevaluated = [ind for ind in self.population if ind['fitness_pwd'] is None or ind['fitness_rdf'] is None]
        # sequential processing since simulation time is not too long < 1 minute
        for ind in unevaluated:
            ind['fitness_pwd'], ind['fitness_rdf'] = self.evaluate_fitness(ind['genome'])
        
        end_time = time.time()
        logger.info(f"Fitness evaluation time: {end_time - start_time:.2f}s")
        return
    
    def initialize_population(self):
        """
        initialize the population with random individuals
        """
        self.population = []
        
        # create random individuals for population size
        for i in range(self.population_size):
            self.population.append(self.generate_random_individual())
            # increment index (afpo non-dominance tie-breaker)
            self.index += 1
        # evaluate fitness of all individuals
        self.evaluate_population_fitness()
        # print view radius of each individual line by line
        for ind in self.population:
            logger.info(f"View radius: {ind['genome'][3::4]}")
        logger.info(f"Evaluated fitness of all individuals post initialization")
        # initialize best individual
        self.best_individual_ever = self.best_individual()
        
    # override
    def generate_random_individual(self):
        """
        Generate a random individual for the population.
        
        Returns:
            dict: A new individual with random genome and fitness
        """
        # initialize based on the experiment
        if self.experiment == 1:
            # Experiment 1: 3 weights for all boids
            genome = np.random.uniform(self.lower_bound, self.upper_bound, self.genome_size)
        elif self.experiment == 2:
            # Experiment 2: 3 weights per boid
            genome = np.random.uniform(self.lower_bound, self.upper_bound, self.genome_size)
        elif self.experiment == 3:
            # Experiment 3: 3 weights + view radius per boid
            genome = np.random.uniform(self.lower_bound, self.upper_bound, self.genome_size)
            # Set view radius values to be within appropriate bounds
            for i in range(3, self.genome_size, 4):
                genome[i] = np.random.uniform(self.view_radius_lower_bound, self.view_radius_upper_bound)
        elif self.experiment == 4:
            # Experiment 4: 4 weights + view radius per boid
            genome = np.random.uniform(self.lower_bound, self.upper_bound, self.genome_size)
            # Set view radius values to be within appropriate bounds
            for i in range(3, self.genome_size, 4):
                genome[i] = np.random.uniform(0.01, 0.5)
        else:
            raise ValueError(f"Invalid experiment number: {self.experiment}")
        
        return {
            'genome': genome,
            'fitness_pwd': None,
            'fitness_rdf': None,
            'age': 0,
            'index': self.index
        }
    
    # specific to AFPO
    def is_dominated(self, individual):
        """
        A solution A dominates solution B if:
         - A is not worse than B in all objectives
         - A is strictly better than B in at least one objective
         - Tie-breaker for deterministic behavior: if A and B are the same in all objectives, then A dominates B if A has a lower index
        """
        for other in self.population:
            if other['index'] == individual['index']:
                continue

            not_worse_in_all = (
                other['fitness_pwd'] >= individual['fitness_pwd'] and
                other['fitness_rdf'] >= individual['fitness_rdf'] and
                other['age'] <= individual['age']
            )
            strictly_better_in_one = (
                other['fitness_pwd'] > individual['fitness_pwd'] or
                other['fitness_rdf'] > individual['fitness_rdf'] or
                other['age'] < individual['age']
            )

            if not_worse_in_all and strictly_better_in_one:
                return True

            # tie-breaker for deterministic behavior
            if (
                np.isclose(other['fitness_pwd'], individual['fitness_pwd'], rtol=1e-8, atol=1e-8) and
                np.isclose(other['fitness_rdf'], individual['fitness_rdf'], rtol=1e-8, atol=1e-8) and
                other['age'] == individual['age'] and
                other['index'] < individual['index']
            ):
                return True

        return False

    # override
    def mutate(self, parent):
        """
        Mutate the parent
        """
        # copy parent for mutation
        child = deepcopy(parent)
        # mutate by adding random noise to the genome
        # choose mask based on mutation rate
        if self.experiment == 4:
            # Create separate masks for weights and view radius
            weight_mask = np.ones(self.genome_size, dtype=bool)
            weight_mask[3::4] = False  # Exclude view radius positions
            weight_mask = weight_mask & (np.random.rand(self.genome_size) < self.mutation_rate)

            view_radius_mask = np.zeros(self.genome_size, dtype=bool)
            vr_indices = np.arange(3, self.genome_size, 4)
            view_radius_mask[vr_indices] = np.random.rand(len(vr_indices)) < self.mutation_rate

            # Apply mutations with different scales
            weight_noise = np.random.uniform(-self.mutation_scale, self.mutation_scale, self.genome_size)
            view_radius_noise = np.random.uniform(-self.mutation_scale_view_radius, self.mutation_scale_view_radius, self.genome_size)

            child['genome'][weight_mask] += weight_noise[weight_mask]
            child['genome'][view_radius_mask] += view_radius_noise[view_radius_mask]
        else:
            mask = np.random.rand(self.genome_size) < self.mutation_rate
            noise = np.random.uniform(-self.mutation_scale, self.mutation_scale, self.genome_size)
            child['genome'][mask] += noise[mask]
        # clip genome to be within bounds self.lower_bound and self.upper_bound
        child['genome'] = np.clip(child['genome'], self.lower_bound, self.upper_bound)
        # if experiment 4, clip view radius to be within bounds 0.01 and 1.0
        if self.experiment == 4:
            child['genome'][3::4] = np.clip(child['genome'][3::4], 0.01, 1.0)
        # set fitness to None
        child['fitness_pwd'] = None
        child['fitness_rdf'] = None
        return child

    # override
    def best_individual(self):
        """
        Return the best individual as the individual with the highest combined fitness
        """
        return max(self.population, key=lambda x: (x['fitness_pwd'] + x['fitness_rdf']))
    
    # override
    def save_checkpoint(self):
        """
        save the population as .pkl file
        """
        checkpoint_path = os.path.join(self.output_dir, 'population.pkl')
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(self.population, f)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    # override
    def save_best_genome(self):
        """
        save the best genome
        """
        best_genome_path = os.path.join(self.output_dir, 'best_genome.pkl')
        with open(best_genome_path, 'wb') as f:
            pickle.dump(self.best_individual(), f)
        logger.info(f"Saved best genome to {best_genome_path}")
    
    # override
    def load_checkpoint(self):
        """
        load the population from .pkl file
        
        Returns:
            bool: True if checkpoint was successfully loaded, False otherwise
        """
        try:
            checkpoint_path = os.path.join(self.output_dir, 'population.pkl')
            if not os.path.exists(checkpoint_path):
                logger.info(f"No checkpoint file found at {checkpoint_path} (this is normal for a new run)")
                return False
                
            with open(checkpoint_path, 'rb') as f:
                self.population = pickle.load(f)
                
            # Update index to be one more than the highest index in population
            if self.population:
                self.index = max(ind['index'] for ind in self.population) + 1
                logger.info(f"Loaded checkpoint with {len(self.population)} individuals")
                logger.info(f"Next index: {self.index}")
                return True
            else:
                logger.info("Checkpoint file is empty")
                return False
                
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            return False

    # specific to AFPO
    def pareto_front(self):
        """
        return the pareto front
        """
        return [individual for individual in self.population if not self.is_dominated(individual)]
    
    # specific to AFPO
    def save_metrics(self, save_dir):
        """
        save the metrics to folder
        """
        # create save_dir if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract genomes from population for diversity calculation
        population_genomes = np.array([individual['genome'] for individual in self.population])
        
        # Extract fitnesses and ages from population
        fitnesses_pwd = np.array([individual['fitness_pwd'] for individual in self.population])
        fitnesses_rdf = np.array([individual['fitness_rdf'] for individual in self.population])
        ages = np.array([individual['age'] for individual in self.population])
        
        # Get Pareto front indices and ensure they are integers
        pareto_front = np.array([i for i, ind in enumerate(self.population) if not self.is_dominated(ind)], dtype=int)
        
        # compute metrics for current generation
        self.metrics[self.current_generation] = { 
            'algorithm': 'afpo',
            'fitnesses_pwd': fitnesses_pwd.tolist(),
            'fitnesses_rdf': fitnesses_rdf.tolist(),
            'ages': ages.tolist(),
            'pareto_front': pareto_front.tolist(),
            'best_fitness_pwd': self.best_individual()['fitness_pwd'],
            'best_fitness_rdf': self.best_individual()['fitness_rdf'],
            "genomic_diversity": calculate_genomic_diversity(population_genomes),
        }
        
        # Save metrics as numpy file for compatibility with plotter
        metrics_path = os.path.join(save_dir, 'metrics.npy')
        np.save(metrics_path, self.metrics)
        
        # Also save as pickle for backup
        pickle_path = os.path.join(save_dir, 'metrics_bkup.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.metrics, f)

    # specific to AFPO
    def plot_metrics(self, save_dir):
        """
        plot metrics for the current generation.
        
        Args:
            save_dir: Directory to save the plots
        """

        # plot best fitness for each of the objectives over generations
        fig, ax = plt.subplots(1,2, figsize=(10, 5))
        ax[0].plot(self.metrics.keys(), [self.metrics[gen]['best_fitness_pwd'] for gen in self.metrics.keys()])
        ax[0].set_xlabel('Generation')
        ax[0].set_ylabel('Best Fitness (PWD)')
        ax[0].set_title('Best Fitness over Generations')
        ax[1].plot(self.metrics.keys(), [self.metrics[gen]['best_fitness_rdf'] for gen in self.metrics.keys()])
        ax[1].set_xlabel('Generation')
        ax[1].set_ylabel('Best Fitness (RDF)')
        ax[1].set_title('Best Fitness over Generations')
        # tight layout
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'best_fitness_over_generations.png'), dpi=300)
        plt.close()

        # plot pareto front (f1 vs f2)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.metrics[self.current_generation]['fitnesses_pwd'], self.metrics[self.current_generation]['fitnesses_rdf'], 'o')
        ax.set_xlabel('f1 (PWD)')
        ax.set_ylabel('f2 (RDF)')
        ax.set_title('Pareto Front')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'pareto_front.png'), dpi=300)
        plt.close()

        # plot genomic diversity over generations
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.metrics.keys(), [self.metrics[gen]['genomic_diversity'] for gen in self.metrics.keys()])
        ax.set_xlabel('Generation')
        ax.set_ylabel('Genomic Diversity')
        ax.set_title('Genomic Diversity over Generations')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'genomic_diversity_over_generations.png'), dpi=300)
        plt.close()

    # override
    def evolve(self, start_generation, generations=1):
        """
        Evolve the population for a given number of generations
        
        Args:
            start_generation: the generation to start from
            generations: the number of generations to evolve (default: 1)
        """
        for generation in range(start_generation, start_generation + generations):
            # Update current generation
            self.current_generation = generation
            
            # Identify non-dominated individuals (Pareto front)
            parents = []
            # get all individuals that are not dominated
            for candidate in self.population:
                is_dominated = self.is_dominated(candidate)
                if not is_dominated:
                    parents.append(candidate)
        
            # if parents are empty, add a random individual
            if not parents:
                logger.debug("[DEBUG] No non-dominated individuals found")
                for ind in self.population:
                    logger.debug(f"→ Index={ind['index']} Fitness_pwd={ind['fitness_pwd']} Fitness_rdf={ind['fitness_rdf']} Age={ind['age']}")
                    logger.debug(f"→ is_dominated: {self.is_dominated(ind)}")
                exit()
            
            # increment age of surviving parents
            for parent in parents:
                parent['age'] += 1
        
            # create new population with parents and offspring
            new_population = []
            # keep all parents
            new_population.extend(parents)
            
            # fill the rest with offspring of parents
            while len(new_population) < self.population_size - 1:
                parent = self.random_selection(parents)
                child = self.mutate(parent)
                child['age'] = 0  # reset age for new offspring
                child['fitness_pwd'], child['fitness_rdf'] = self.evaluate_fitness(child['genome'])
                child['index'] = self.index  # assign new index
                self.index += 1  # increment index counter
                new_population.append(child)
        
            # add one random new individual to maintain diversity
            random_individual = self.generate_random_individual()
            random_individual['age'] = 0
            random_individual['fitness_pwd'], random_individual['fitness_rdf'] = self.evaluate_fitness(random_individual['genome'])
            random_individual['index'] = self.index  # assign new index
            self.index += 1  # increment index counter
            new_population.append(random_individual)
            
            # replace old population
            self.population = new_population

            # checkpoint
            self.save_checkpoint()

            # save metrics
            self.save_metrics(self.output_dir)

            # plot metrics
            self.plot_metrics(self.output_dir)
            
        # return the best individual
        return self.best_individual()

    def random_selection(self, parents):
        """
        Select a random parent from the population
        
        Args:
            parents: list of parent individuals
            
        Returns:
            selected_parent: randomly selected parent
        """
        return random.choice(parents)
    
    def plot_progress(self, best_genome, generation):
        """
        Plot the evolution progress
        
        Args:
            best_genome: the best genome in the current generation
            generation: current generation number
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Plot metrics
        self.plot_metrics(self.output_dir)
        
        # Save best genome
        self.save_best_genome()

    