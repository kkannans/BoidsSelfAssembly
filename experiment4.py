import numpy as np
from boids import Boid
from hypothesis_space import HypothesisSpace
from evolutionary_algorithm import EvolutionaryAlgorithm
import config

class Experiment4(HypothesisSpace):
    """
    Experiment 4:(hypothesis space 4)
    - Genome: 3 total (3 weights per boid for separation, alignment, and cohesion) + view_radius scale 
    """
    # modify args to include view_radius for experiment 1 hyperparameter search
    def __init__(self, initial_positions:np.ndarray, initial_velocities:np.ndarray):
        """
        Initialize the hypothesis space with target data
        
        Args:
            initial_positions: numpy array of shape (num_boids, 2)
            initial_velocities: numpy array of shape (num_boids, 2)
            view_radius: int
        """
        super().__init__(initial_positions, initial_velocities)
        # Set genome dimension for hypothesis space 1 (experiment 1)
        self.genome_size = 4
        self.experiment = 4
        
    # concrete implementation of abstract method
    def initialize_boids(self, genome):
        """
        Initialize boids with weights from genome
        
        Args:
            genome: numpy array of shape (self.genome_size,)
            view_radius: int
        Returns:
            boids: list of Boid objects
        """
        assert genome.shape == (self.genome_size,)
        boids = []
        assert self.initial_velocities.shape == (self.num_boids, 2)
        for i, position in enumerate(self.initial_positions):
            boid = Boid(position, self.initial_velocities[i])
            boid.separation_weight = genome[0]
            boid.alignment_weight = genome[1]
            boid.cohesion_weight = genome[2]
            boid.view_radius = np.abs(genome[3]) * config.VIEW_RADIUS_MAX
            boids.append(boid)
        return boids