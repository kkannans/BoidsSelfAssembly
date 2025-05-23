# Boid class for simulation
import numpy as np
import math
import random
import config

class Boid:
    """Represents a single boid in the simulation"""
    def __init__(self, position, velocity, width=None, height=None, wall_behavior='bounce'):
        """
        Initialize a boid with position and velocity
        
        Args:
            position: numpy array of shape (2,) for x,y coordinates in pixels
            velocity: numpy array of shape (2,) for vx,vy components in boid size units per timestep
            width: width of the simulation area in pixels
            height: height of the simulation area in pixels
            wall_behavior: how boids behave at boundaries ('wrap', 'bounce', 'change_direction')
        """
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        
        # Default behavior weights - can be overridden
        self.separation_weight = 1.0
        self.alignment_weight = 1.0
        self.cohesion_weight = 1.0
        self.view_radius = None
        # Environment boundaries
        self.width = width
        self.height = height
        self.wall_behavior = wall_behavior
        
        # For tracking positions over time
        self.states = [{'position': self.position.copy(), 'velocity': self.velocity.copy()}]
        
    
    def limit_speed(self, min_speed=config.MIN_SPEED, max_speed=config.MAX_SPEED):
        """Enforce minimum and maximum speed limits in boid size units per timestep"""
        speed = np.linalg.norm(self.velocity)
        # Apply speed limits
        if speed > max_speed:
            self.velocity = self.velocity * (max_speed / speed)     
        elif speed < min_speed and speed > 0:  # Only scale up if velocity is non-zero
            self.velocity = self.velocity * (min_speed / speed)
        elif speed == 0:  # If velocity is zero, set a random direction with min_speed
            angle = random.uniform(0, 2 * math.pi)
            self.velocity = np.array([math.cos(angle), math.sin(angle)]) * min_speed
    
    def update(self):
        # Update position based on change in velocity  
        self.position += self.velocity

        # Wall behavior
        if self.width is not None and self.height is not None:
            if self.wall_behavior == 'wrap':
                # Wrap around edges
                self.position[0] = self.position[0] % self.width
                self.position[1] = self.position[1] % self.height
                
            elif self.wall_behavior == 'bounce':
                # Bounce off edges by reversing velocity component
                if self.position[0] < 0:
                    self.position[0] = 0
                    self.velocity[0] *= -1
                if self.position[0] > self.width:
                    self.position[0] = self.width
                    self.velocity[0] *= -1
                if self.position[1] < 0:
                    self.position[1] = 0
                    self.velocity[1] *= -1
                if self.position[1] > self.height:
                    self.position[1] = self.height
                    self.velocity[1] *= -1
                    
            elif self.wall_behavior == 'change_direction':
                # Change to a random direction when hitting walls
                if (self.position[0] < 0 or self.position[0] > self.width or 
                    self.position[1] < 0 or self.position[1] > self.height):
                    # Keep position within bounds
                    self.position[0] = np.clip(self.position[0], 0, self.width)
                    self.position[1] = np.clip(self.position[1], 0, self.height)
                    # Set random new direction
                    angle = random.uniform(0, 2 * math.pi)
                    speed = np.linalg.norm(self.velocity)
                    self.velocity = np.array([math.cos(angle), math.sin(angle)]) * speed

        # Update states
        self.states.append({'position': self.position.copy(), 'velocity': self.velocity.copy()})