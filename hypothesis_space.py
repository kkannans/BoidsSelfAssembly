# Evolutionary algorithm for optimizing boid behaviors
import numpy as np
import os
import json
import time
import config
import psutil
from simulation import BoidSimulation
from fitness_function import fitness_pairwise_distance, fitness_rdf
from boids import Boid
from abc import ABC, abstractmethod
from utils import load_pairwise_distances

class HypothesisSpace(ABC):
    """Class for evolving boid behaviors in different hypothesis spaces"""
    
    def __init__(self, initial_positions:np.ndarray, initial_velocities:np.ndarray):
        """
        Define the hypothesis space given initial positions and velocities
        Uses BoidSimulation to simulate boid positions
        Args:
            initial_positions: numpy array of shape (num_boids, 2)
            initial_velocities: numpy array of shape (num_boids, 2)
        """
        self.initial_positions = initial_positions
        self.initial_velocities = initial_velocities
        self.num_boids = initial_positions.shape[0]

        self.pairwise_distances_video = load_pairwise_distances()
        
        self.experiment = None # to be set by child classes
        self.genome_size = None # to be set by child classes
        
    @abstractmethod
    def initialize_boids(self, genome:np.ndarray):
        """
        Initialize boids with weights from genome 
        To be implemented by child classes with specific hypothesis spaces
        Args:
            genome: numpy array of shape (num_boids * weights_per_boid,)
        Returns:
            boids: list of Boid objects
        """
        pass
    
    def simulate_boid_positions(self, genome:np.ndarray, num_steps:int=None):
        """
        Simulate boid positions for given genome
        
        Args:
            genome: numpy array of shape (num_boids * weights_per_boid,)
            num_steps: number of simulation steps
            
        Returns:
            positions: numpy array of shape (num_boids, num_steps, 2)
        """
        if num_steps is None:
            num_steps = config.END_FRAME - config.START_FRAME + 1 # fall back if none provided
            
        # Check if num_boids is valid
        if self.num_boids <= 0:
            raise ValueError(f"Invalid number of boids: {self.num_boids}")
            
        simulation = BoidSimulation()
        boids = self.initialize_boids(genome) # implemented in child classes    
        
        # Initialize array to store all positions
        positions = np.zeros((self.num_boids, num_steps, 2), dtype=np.float64)
        
        # Check if positions array was properly initialized
        if positions.shape != (self.num_boids, num_steps, 2):
            raise ValueError(f"Positions array has incorrect shape: {positions.shape}, expected {(self.num_boids, num_steps, 2)}")
        
        # Run simulation and store positions
        for t in range(num_steps):
            # Store current positions
            current_positions = np.array([boid.position for boid in boids], dtype=np.float64)
            if current_positions.shape != (self.num_boids, 2):
                raise ValueError(f"Current positions have incorrect shape: {current_positions.shape}, expected {(self.num_boids, 2)}")
            positions[:, t, :] = current_positions
            
            # Update simulation
            simulation.update(boids)
            
        return positions
    
    def fitness_function(self, genome:np.ndarray, num_steps:int):
        """
        Calculate fitness of a genome
        
        Args:
            genome: numpy array of shape (num_boids * weights_per_boid,)
        Returns:
            fitness: scalar fitness value (to be maximized)
            combination of pairwise distance and rdf
        """
        # simulate boid positions
        positions = self.simulate_boid_positions(genome, num_steps)
        pd_fitness = fitness_pairwise_distance(positions, self.pairwise_distances_video)
        rdf_fitness = fitness_rdf(positions)
        f1_squared_plus_1 = pd_fitness**2 + 1
        f2_squared_plus_1 = rdf_fitness**2 + 1
        product = f1_squared_plus_1 * f2_squared_plus_1
        fitness = 1 - np.sqrt(product)
        return fitness
    
    def multi_objective_fitness_function(self, genome:np.ndarray, num_steps:int):
        """
        Calculate fitness of a genome
        """
        positions = self.simulate_boid_positions(genome, num_steps)
        pd_fitness = fitness_pairwise_distance(positions, self.pairwise_distances_video)
        rdf_fitness = fitness_rdf(positions)
        return pd_fitness, rdf_fitness
            