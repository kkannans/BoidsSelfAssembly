# Boid simulation logic
import numpy as np
import config
from numba import jit, float64, int64

@jit(nopython=True)
def compute_forces(positions, velocities, separation_weights, alignment_weights, 
                  cohesion_weights, view_radii):
    """
    Compute all forces for all boids at once using Numba JIT
    
    Args:
        positions: numpy array of shape (num_boids, 2) in pixels
        velocities: numpy array of shape (num_boids, 2) in pixels per timestep
        separation_weights: numpy array of shape (num_boids,)
        alignment_weights: numpy array of shape (num_boids,)
        cohesion_weights: numpy array of shape (num_boids,)
        view_radii: numpy array of shape (num_boids,)
    
    Returns:
        Tuple of (separation_forces, alignment_forces, cohesion_forces)
        Each is a numpy array of shape (num_boids, 2) in pixels per timestep
    """
    num_boids = positions.shape[0]
    separation_forces = np.zeros((num_boids, 2))
    alignment_forces = np.zeros((num_boids, 2))
    cohesion_forces = np.zeros((num_boids, 2))
    
    # Sequential processing
    for i in range(num_boids):
        # Find neighbors
        separation_sum = np.zeros(2)
        alignment_sum = np.zeros(2)
        cohesion_sum = np.zeros(2)
        count = 0
        
        for j in range(num_boids):
            if i == j:
                continue
                
            # Calculate distance using vectorized operations
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance < view_radii[i]:
                # Separation
                if distance > 0:
                    separation_sum[0] -= dx / distance
                    separation_sum[1] -= dy / distance
                
                # Alignment
                alignment_sum += velocities[j]
                
                # Cohesion
                cohesion_sum += positions[j]
                
                count += 1
        
        if count > 0:
            # Normalize and apply weights
            separation_forces[i] = separation_sum * separation_weights[i]
            alignment_forces[i] = (alignment_sum / count - velocities[i]) * alignment_weights[i]
            cohesion_forces[i] = ((cohesion_sum / count) - positions[i]) * cohesion_weights[i]
    
    return separation_forces, alignment_forces, cohesion_forces

class BoidSimulation:
    """Simulation for a flock of boids"""
    def __init__(self):
        """Initialize the simulation with default parameters"""
        self.max_speed = config.MAX_SPEED
        self.min_speed = config.MIN_SPEED

    def update(self, boids):
        """Update all boids in a single step"""
        
        # Pre-compute all positions and velocities as numpy arrays for vectorization
        positions = np.array([boid.position for boid in boids])
        velocities = np.array([boid.velocity for boid in boids])
        separation_weights = np.array([boid.separation_weight for boid in boids])
        alignment_weights = np.array([boid.alignment_weight for boid in boids])
        cohesion_weights = np.array([boid.cohesion_weight for boid in boids])
        view_radii = np.array([boid.view_radius for boid in boids])
        
        # Use optimized computation of forces
        separation_forces, alignment_forces, cohesion_forces = compute_forces(
            positions, velocities, separation_weights, alignment_weights, cohesion_weights, view_radii
        )
        
        # Update each boid
        for i in range(len(boids)):
            # Apply forces
            boids[i].velocity += (separation_forces[i] + alignment_forces[i] + cohesion_forces[i])
            
            # Enforce speed limits
            boids[i].limit_speed(self.min_speed, self.max_speed)
            
            # Update position and resolve wall collisions
            boids[i].update()