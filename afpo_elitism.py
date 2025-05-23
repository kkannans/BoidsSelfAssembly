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

class AFPOElite(EvolutionaryAlgorithm):
    """Age-Fitness Pareto Optimization implementation with three objectives: age, pwd fitness, and rdf fitness"""
    
    def __init__(self, hypothesis_space:HypothesisSpace, output_dir:str=None):
        """
        Initialize AFPOElite
        
        Args:
            hypothesis_space: Hypothesis space to evolve
            output_dir: Directory to save outputs
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
            # Set initial best individual
            self.best_individual_ever = self.best_individual()

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
        if self.experiment == 4:
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

            # Check if other is not worse in all objectives
            not_worse_in_all = (
                other['fitness_pwd'] >= individual['fitness_pwd'] and
                other['fitness_rdf'] >= individual['fitness_rdf'] and
                other['age'] <= individual['age']
            )
            
            # Check if other is strictly better in at least one objective
            strictly_better_in_one = (
                other['fitness_pwd'] > individual['fitness_pwd'] or
                other['fitness_rdf'] > individual['fitness_rdf'] or
                other['age'] < individual['age']
            )

            # If other is not worse in all objectives and strictly better in at least one, then it dominates
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
    def best_individual(self, metric='fitness_total'):
        """
        Return the best individual based on the specified metric
        default is fitness_total
        """
        if metric == 'fitness_pwd':
            return max(self.population, key=lambda x: x['fitness_pwd'])
        elif metric == 'fitness_rdf':
            return max(self.population, key=lambda x: x['fitness_rdf'])
        elif metric == 'fitness_total':
            return self.get_waist()
        else:
            raise ValueError(f"Invalid metric: {metric}")
    
    # override
    def save_checkpoint(self):
        """
        Save the population, metrics, and best individual to disk
        """
        try:
            # Save population
            checkpoint_path = os.path.join(self.output_dir, 'population.pkl')
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(self.population, f)
            
            # Save best individual
            best_genome_path = os.path.join(self.output_dir, 'best_genome.pkl')
            with open(best_genome_path, 'wb') as f:
                pickle.dump(self.best_individual_ever, f)
                
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            logger.info(f"Saved best genome to {best_genome_path}")
            
            return True
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            return False

    # override
    def load_checkpoint(self):
        """
        Load the population, metrics, and current generation from disk
        
        Returns:
            bool: True if checkpoint was successfully loaded, False otherwise
        """
        try:
            checkpoint_path = os.path.join(self.output_dir, 'population.pkl')
            metrics_path = os.path.join(self.output_dir, 'metrics.npy')
            
            if not os.path.exists(checkpoint_path):
                logger.info(f"No checkpoint file found at {checkpoint_path}")
                return False
                
            # Load population
            with open(checkpoint_path, 'rb') as f:
                self.population = pickle.load(f)
            
            # Load metrics to determine current generation
            if os.path.exists(metrics_path):
                self.metrics = np.load(metrics_path, allow_pickle=True).item()
                # Set current generation to the last generation in metrics
                if self.metrics:
                    self.current_generation = max(self.metrics.keys())
                    logger.info(f"Restored current generation to {self.current_generation} from metrics")
            
            # Load best individual ever if it exists
            best_genome_path = os.path.join(self.output_dir, 'best_genome.pkl')
            if os.path.exists(best_genome_path):
                with open(best_genome_path, 'rb') as f:
                    self.best_individual_ever = pickle.load(f)
                    logger.info(f"Loaded best individual ever from checkpoint")
            else:
                # If no best individual ever found, set it to the best in current population
                self.best_individual_ever = self.best_individual()
                logger.info(f"Set best individual ever from current population")
            
            # Update index to be one more than the highest index in population
            if self.population:
                self.index = max(ind['index'] for ind in self.population) + 1
                logger.info(f"Loaded checkpoint with {len(self.population)} individuals")
                logger.info(f"Current generation: {self.current_generation}")
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
        Return the pareto front (non-dominated individuals)
        """
        return [individual for individual in self.population if not self.is_dominated(individual)]
    
    # specific to AFPO
    def save_metrics(self, save_dir):
        """
        Save the metrics to folder
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
            'best_fitness_pwd': self.best_individual('fitness_pwd')['fitness_pwd'],
            'best_fitness_rdf': self.best_individual('fitness_rdf')['fitness_rdf'],
            "genomic_diversity": calculate_genomic_diversity(population_genomes),
            "best_pwd_genome": self.best_individual('fitness_pwd')['genome'],
            "best_rdf_genome": self.best_individual('fitness_rdf')['genome'],
            "best_total_genome": self.best_individual('fitness_total')['genome']
        }
        
        # Save metrics as numpy file
        metrics_path = os.path.join(save_dir, 'metrics.npy')
        np.save(metrics_path, self.metrics)
        
        # Also save as pickle for backup
        pickle_path = os.path.join(save_dir, 'metrics_bkup.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.metrics, f)

    # specific to AFPO
    def plot_metrics(self, save_dir):
        """
        Plot metrics for the current generation.
        
        Args:
            save_dir: Directory to save the plots
        """
        # Create save_dir if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Plot best fitness for each of the objectives over generations
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # PWD fitness
        ax[0].plot(list(self.metrics.keys()), [self.metrics[gen]['best_fitness_pwd'] for gen in self.metrics.keys()], 
                   color='blue')
        ax[0].set_xlabel('Generation')
        ax[0].set_ylabel('Best Fitness (PWD)')
        ax[0].set_title('PWD Fitness over Generations')
        
        # RDF fitness
        ax[1].plot(list(self.metrics.keys()), [self.metrics[gen]['best_fitness_rdf'] for gen in self.metrics.keys()], 
                   color='green')
        ax[1].set_xlabel('Generation')
        ax[1].set_ylabel('Best Fitness (RDF)')
        ax[1].set_title('RDF Fitness over Generations')
        
        # Tight layout
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'best_fitness_over_generations.png'), dpi=300)
        plt.close()

        # Plot detailed Pareto front analysis
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Extract fitness values
        fitnesses_pwd = [p['fitness_pwd'] for p in self.population]
        fitnesses_rdf = [p['fitness_rdf'] for p in self.population]
        
        # Find maximum values
        max_pwd = max(fitnesses_pwd)
        max_rdf = max(fitnesses_rdf)
        
        # Plot all individuals
        ax.scatter(fitnesses_pwd, fitnesses_rdf, c='gray', alpha=0.3, label='Population')
        
        # Get Pareto front
        pareto_front = [p for p in self.population if not self.is_dominated(p)]
        pareto_pwd = [p['fitness_pwd'] for p in pareto_front]
        pareto_rdf = [p['fitness_rdf'] for p in pareto_front]
        ax.scatter(pareto_pwd, pareto_rdf, c='blue', alpha=0.7, label='Pareto Front')
        
        # Get waist
        waist = self.get_waist()
        ax.scatter(waist['fitness_pwd'], waist['fitness_rdf'], 
                  c='green', marker='s', s=100, label='Best PWD and RDF')
        
        # Get best individuals
        best_pwd = self.best_individual('fitness_pwd')
        best_rdf = self.best_individual('fitness_rdf')
        ax.scatter(best_pwd['fitness_pwd'], best_pwd['fitness_rdf'], 
                  c='red', marker='*', s=200, label='Best PWD')
        ax.scatter(best_rdf['fitness_pwd'], best_rdf['fitness_rdf'], 
                  c='purple', marker='*', s=200, label='Best RDF')
        
        # Plot maximum values
        ax.scatter(max_pwd, max_rdf, c='black', marker='x', s=200, label='Max Values')

        # Load and plot regression baseline losses as fitness values
        regression_losses = {
            "linear_rdf": np.load('./regression_rdf_bw_longer/losses_linear_regression.npy'),
            "linear_pwd": np.load('./regression_pwd_bw_longer/linear_model_avg_pairwise_distances_losses.npy'),
        }
        
        # Convert losses to fitness values (negative of losses)
        regression_fitness = {
            "linear_rdf": -regression_losses["linear_rdf"],
            "linear_pwd": -regression_losses["linear_pwd"],
        }
        
        # compute mean only until END_FRAME
        regression_fitness = {
            "linear_rdf": np.mean(regression_fitness["linear_rdf"][:config.END_FRAME]),
            "linear_pwd": np.mean(regression_fitness["linear_pwd"][:config.END_FRAME]),
        }
        
        # Debug prints
        logger.info(f"Regression fitness values - RDF: {regression_fitness['linear_rdf']}, PWD: {regression_fitness['linear_pwd']}")
        logger.info(f"Population fitness range - RDF: [{min(fitnesses_rdf)}, {max(fitnesses_rdf)}], PWD: [{min(fitnesses_pwd)}, {max(fitnesses_pwd)}]")
        
        # Plot regression baselines
        colors = {'linear_rdf': 'red', 'linear_pwd': 'blue'}
        ax.axhline(y=regression_fitness["linear_rdf"], color=colors["linear_rdf"], linestyle='--', 
                    label=f'Linear RDF Regression')
        ax.axvline(x=regression_fitness["linear_pwd"], color=colors["linear_pwd"], linestyle='--', 
                    label=f'Linear PWD Regression')
        
        # Set labels and title
        ax.set_xlabel('PWD Fitness')
        ax.set_ylabel('RDF Fitness')
        ax.set_title(f'Pareto Front Analysis (Generation {self.current_generation})')
        
        # Add legend
        ax.legend()
        # place in bottom left corner
        ax.legend(loc='lower left')
        
        # Set axis limits to show full range including regression lines
        x_min = min(min(fitnesses_pwd), regression_fitness["linear_pwd"])
        x_max = max(max(fitnesses_pwd), regression_fitness["linear_pwd"])
        y_min = min(min(fitnesses_rdf), regression_fitness["linear_rdf"])
        y_max = max(max(fitnesses_rdf), regression_fitness["linear_rdf"])
        
        # Add some padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        ax.set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'pareto_front_analysis.png'), dpi=300)
        plt.close()

        # Plot genomic diversity over generations
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(list(self.metrics.keys()), [self.metrics[gen]['genomic_diversity'] for gen in self.metrics.keys()])
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
        # Use current generation (from checkpoint) if valid, otherwise use start_generation
        current_gen = max(start_generation, self.current_generation)
        logger.info(f"Starting evolution from generation {current_gen}")
        
        for generation in range(current_gen, current_gen + generations):
            start_time = time.time()
            
            # Update current generation
            self.current_generation = generation
            
            # Select elite individuals - top 2 in each fitness metric, regardless of age
            elite_individuals = []
            
            # Sort by PWD fitness and take top 2
            sorted_by_pwd = sorted(self.population, key=lambda x: x['fitness_pwd'], reverse=True)
            elite_individuals.extend(sorted_by_pwd[:2])
            
            # Sort by RDF fitness and take top 2
            sorted_by_rdf = sorted(self.population, key=lambda x: x['fitness_rdf'], reverse=True)
            elite_individuals.extend(sorted_by_rdf[:2])
            
            # Remove duplicates while preserving order
            seen_indices = set()
            unique_elites = []
            for elite in elite_individuals:
                if elite['index'] not in seen_indices:
                    seen_indices.add(elite['index'])
                    unique_elites.append(elite)
            
            elite_individuals = unique_elites
            logger.info(f"Selected {len(elite_individuals)} unique elite individuals")
            
            # Identify non-dominated individuals (Pareto front) for parent selection
            parents = []
            # First, get all non-dominated individuals
            for candidate in self.population:
                is_dominated = self.is_dominated(candidate)
                if not is_dominated:
                    parents.append(candidate)
            
            # Add elite individuals to parents if they're not already there
            parent_indices = {p['index'] for p in parents}
            for elite in elite_individuals:
                if elite['index'] not in parent_indices:
                    parents.append(elite)
            
            # Calculate target size for Pareto front
            target_pareto_size = self.population_size - len(elite_individuals) - 1  # -1 for random individual
            
            if len(parents) > target_pareto_size:
                # Find min and max values for each objective
                pwd_values = [ind['fitness_pwd'] for ind in parents]
                rdf_values = [ind['fitness_rdf'] for ind in parents]
                min_pwd, max_pwd = min(pwd_values), max(pwd_values)
                min_rdf, max_rdf = min(rdf_values), max(rdf_values)
                
                # Calculate distance to ideal point (top right corner) for each individual
                distances = []
                for ind in parents:
                    # Normalize objectives to [0,1] range
                    norm_pwd = (ind['fitness_pwd'] - min_pwd) / (max_pwd - min_pwd) if max_pwd != min_pwd else 0
                    norm_rdf = (ind['fitness_rdf'] - min_rdf) / (max_rdf - min_rdf) if max_rdf != min_rdf else 0
                    # Calculate distance to ideal point (1,1)
                    distance = np.sqrt((1 - norm_pwd)**2 + (1 - norm_rdf)**2)
                    distances.append((distance, ind))
                
                # Sort by distance (closest to ideal point first)
                distances.sort(key=lambda x: x[0])
                # Take top target_pareto_size individuals
                parents = [ind for _, ind in distances[:target_pareto_size]]
            
            # if parents are empty, add a random individual
            if not parents:
                logger.error("[ERROR] No non-dominated individuals found")
                for ind in self.population:
                    logger.debug(f"→ Index={ind['index']} Fitness_pwd={ind['fitness_pwd']} Fitness_rdf={ind['fitness_rdf']} Age={ind['age']}")
                    logger.debug(f"→ is_dominated: {self.is_dominated(ind)}")
                exit()
            
            # increment age of surviving parents
            for parent in parents:
                parent['age'] += 1
        
            # create new population starting with elite individuals and parents
            new_population = []
            # keep all elite individuals and parents
            new_population.extend(elite_individuals)
            new_population.extend(parents)
            
            # Remove duplicates from new_population
            seen_indices = set()
            unique_population = []
            for ind in new_population:
                if ind['index'] not in seen_indices:
                    seen_indices.add(ind['index'])
                    unique_population.append(ind)
            
            new_population = unique_population
            parents_for_reproduction = unique_population
            
            # Log detailed population composition
            num_elites = len(elite_individuals)
            num_pareto = len(parents)
            num_unique = len(new_population)
            logger.info(f"Parents composition: {num_elites} elites + {num_pareto} pareto front = {num_unique} unique individuals")
            
            # fill the rest with offspring of non-dominated parents and elite individuals
            while len(new_population) < self.population_size - 1:
                parent = self.random_selection(parents_for_reproduction)
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
            logger.info(f"Final population size: {len(self.population)}")
            
            # Update best individual if necessary
            current_best = self.best_individual()
            if self.best_individual_ever is None or (current_best['fitness_pwd'] + current_best['fitness_rdf']) > (self.best_individual_ever['fitness_pwd'] + self.best_individual_ever['fitness_rdf']):
                self.best_individual_ever = deepcopy(current_best)
                logger.info(f"New best individual found! PWD={self.best_individual_ever['fitness_pwd']:.4f}, RDF={self.best_individual_ever['fitness_rdf']:.4f}")

            # checkpoint
            self.save_checkpoint()

            # save metrics
            self.save_metrics(self.output_dir)

            # plot metrics
            self.plot_metrics(self.output_dir)
            
            end_time = time.time()
            logger.info(f"Generation {generation} completed in {end_time - start_time:.2f}s")
            
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

    def save_best_genome(self):
        """
        Save the best genome found so far
        """
        best_genome_path = os.path.join(self.output_dir, 'best_genome.pkl')
        with open(best_genome_path, 'wb') as f:
            pickle.dump(self.best_individual_ever, f)
        logger.info(f"Saved best genome to {best_genome_path}")

    def get_waist(self, population=None):
        """
        Find the waist of the population (individuals closest to maximum values in both objectives).
        """
        if population is None:
            population = self.population
            
        # Find maximum values for both fitness metrics
        max_pwd = max(p['fitness_pwd'] for p in population)
        max_rdf = max(p['fitness_rdf'] for p in population)
        
        # Calculate distances for all individuals
        distances = []
        for p in population:
            # Calculate normalized distance to maximum point
            eps = 1e-9 # for numerical stability
            norm_pwd = (max_pwd - p['fitness_pwd']) / (max_pwd + eps)
            norm_rdf = (max_rdf - p['fitness_rdf']) / (max_rdf + eps)
            distance = np.sqrt(norm_pwd**2 + norm_rdf**2)
            distances.append((distance, p))
        
        # Sort by distance and get the closest
        distances.sort(key=lambda x: x[0])
        min_distance = distances[0][0]
        waist = [p for dist, p in distances if dist == min_distance]
        
        return waist[0]  # Return first individual if multiple have same distance