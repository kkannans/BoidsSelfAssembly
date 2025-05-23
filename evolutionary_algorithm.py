from abc import ABC, abstractmethod
import numpy as np
import random
from copy import deepcopy
import config
import os
import json
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import time
import pickle
from hypothesis_space import HypothesisSpace

def calculate_genomic_diversity(population_genomes):
    """
    Calculate the average pairwise distance between genomes in the population
    
    Args:
        population_genomes: numpy array of shape (population_size, genome_dim)
        
    Returns:
        diversity: average pairwise distance between genomes
    """
    # Calculate pairwise distances between all genomes
    distances = []
    for i in range(len(population_genomes)):
        for j in range(i+1, len(population_genomes)):
            dist = np.linalg.norm(population_genomes[i] - population_genomes[j])
            distances.append(dist)
    
    return np.mean(distances)

class EvolutionaryAlgorithm(ABC):
    """Abstract base class for evolutionary algorithms"""
    
    def __init__(self, hypothesis_space:HypothesisSpace, output_dir:str=None):
        """
        Initialize the evolutionary algorithm with a hypothesis space
        
        Args:
            hypothesis_space: hypothesis space to evolve
        """
        self.hypothesis_space = hypothesis_space
        self.population_size = config.POPULATION_SIZE
        self.num_steps = config.END_FRAME - config.START_FRAME + 1 # define from config
        self.generations = config.GENERATIONS
        self.genome_size = self.hypothesis_space.genome_size
        self.output_dir = output_dir
        self.experiment = self.hypothesis_space.experiment

    def best_individual(self):
        """
        Return the best individual in the population
        """
        return max(self.population, key=lambda x: x['fitness'])
    
    @abstractmethod
    def initialize_population(self):
        """Initialize the population"""
        pass
    
    @abstractmethod
    def generate_random_individual(self):
        """Generate a random individual"""
        pass

    @abstractmethod
    def mutate(self, individual):
        """Mutate an individual"""
        pass

    @abstractmethod
    def evolve(self, current_generation:int, num_generations:int):
        """
        Evolve the population for given number of generations
        
        Args:
            current_generation: current generation number
            num_generations: number of generations to evolve
        """
        pass

    def evaluate_fitness(self, genome:np.ndarray):
        """Evaluate fitness of an individual"""
        start_time = time.time()
        fitness = self.hypothesis_space.fitness_function(genome, self.num_steps)
        end_time = time.time()
        print(f"Individual fitness evaluation time: {end_time - start_time:.2f}s")
        return fitness

    def evaluate_population_fitness(self):
        """Evaluate fitness of all individuals sequentially or in parallel"""
        start_time = time.time()
        
        # Skip evaluation for individuals that already have fitness
        unevaluated = [ind for ind in self.population if ind['fitness'] is None]
        # sequential processing since simulation time is not too long < 1 minute
        for ind in unevaluated:
            ind['fitness'] = self.evaluate_fitness(ind['genome'])
        
        end_time = time.time()
        print(f"Fitness evaluation time: {end_time - start_time:.2f}s")
        return

    @abstractmethod
    def save_metrics(self, output_dir:str):
        pass

    @abstractmethod
    def save_checkpoint(self):
        pass

    @abstractmethod
    def load_checkpoint(self):
        pass
    
    @abstractmethod
    def random_selection(self, parents):
        pass
    
    @abstractmethod
    def save_best_genome(self, generation, best_genome):
        pass