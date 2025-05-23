import pygame
import numpy as np
import argparse
import config
import pandas as pd
from simulation import BoidSimulation
from utils import load_initial_positions
from boids import Boid

class Slider:
    def __init__(self, x, y, width, height, min_val, max_val, initial_val, text):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.text = text
        self.active = False
        self.font = pygame.font.SysFont(None, 24)
        
    def draw(self, screen):
        # Draw slider background
        pygame.draw.rect(screen, (200, 200, 200), self.rect)
        
        # Calculate handle position
        handle_x = self.rect.x + (self.value - self.min_val) / (self.max_val - self.min_val) * self.rect.width
        handle_rect = pygame.Rect(handle_x - 5, self.rect.y - 5, 10, self.rect.height + 10)
        pygame.draw.rect(screen, (100, 100, 100), handle_rect)
        
        # Draw text
        text_surface = self.font.render(f"{self.text}: {self.value:.2f}", True, (0, 0, 0))
        screen.blit(text_surface, (self.rect.x, self.rect.y - 25))
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.active = False
        elif event.type == pygame.MOUSEMOTION and self.active:
            # Update value based on mouse position
            rel_x = event.pos[0] - self.rect.x
            self.value = self.min_val + (rel_x / self.rect.width) * (self.max_val - self.min_val)
            self.value = max(self.min_val, min(self.max_val, self.value))

class Button:
    def __init__(self, x, y, width, height, text, color=(100, 100, 100), hover_color=(150, 150, 150)):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.is_hovered = False
        self.font = pygame.font.SysFont(None, 24)
        
    def draw(self, screen):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, (0, 0, 0), self.rect, 2)  # Border
        
        text_surface = self.font.render(self.text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False

class Dropdown:
    def __init__(self, x, y, width, height, options, initial_index=0):
        self.rect = pygame.Rect(x, y, width, height)
        self.options = options
        self.selected_index = initial_index
        self.is_open = False
        self.font = pygame.font.SysFont(None, 24)
        self.option_height = 30
        self.hover_index = -1
        
    def draw(self, screen):
        # Draw main button
        pygame.draw.rect(screen, (200, 200, 200), self.rect)
        pygame.draw.rect(screen, (0, 0, 0), self.rect, 2)
        
        # Draw selected option with view radius
        view_radius = self.options[self.selected_index][3]
        text = f"Radius: {view_radius:.1f}"
        text_surface = self.font.render(text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
        
        # Draw dropdown if open
        if self.is_open:
            dropdown_rect = pygame.Rect(
                self.rect.x, 
                self.rect.y + self.rect.height,
                self.rect.width,
                len(self.options) * self.option_height
            )
            pygame.draw.rect(screen, (240, 240, 240), dropdown_rect)
            pygame.draw.rect(screen, (0, 0, 0), dropdown_rect, 2)
            
            for i, option in enumerate(self.options):
                option_rect = pygame.Rect(
                    self.rect.x,
                    self.rect.y + self.rect.height + i * self.option_height,
                    self.rect.width,
                    self.option_height
                )
                
                # Highlight hovered option
                if i == self.hover_index:
                    pygame.draw.rect(screen, (200, 200, 200), option_rect)
                
                view_radius = option[3]
                text = f"Radius: {view_radius:.1f}"
                text_surface = self.font.render(text, True, (0, 0, 0))
                text_rect = text_surface.get_rect(center=option_rect.center)
                screen.blit(text_surface, text_rect)
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.is_open = not self.is_open
                return True
            elif self.is_open:
                # Check if click is in dropdown area
                dropdown_rect = pygame.Rect(
                    self.rect.x,
                    self.rect.y + self.rect.height,
                    self.rect.width,
                    len(self.options) * self.option_height
                )
                if dropdown_rect.collidepoint(event.pos):
                    # Calculate which option was clicked
                    relative_y = event.pos[1] - (self.rect.y + self.rect.height)
                    clicked_index = relative_y // self.option_height
                    if 0 <= clicked_index < len(self.options):
                        self.selected_index = clicked_index
                        self.is_open = False
                        return True
                else:
                    self.is_open = False
        elif event.type == pygame.MOUSEMOTION and self.is_open:
            # Update hover index
            dropdown_rect = pygame.Rect(
                self.rect.x,
                self.rect.y + self.rect.height,
                self.rect.width,
                len(self.options) * self.option_height
            )
            if dropdown_rect.collidepoint(event.pos):
                relative_y = event.pos[1] - (self.rect.y + self.rect.height)
                self.hover_index = relative_y // self.option_height
            else:
                self.hover_index = -1
        return False

def initialize_boids(num_boids, area_size=1000, positions=None):
    """Initialize boids with positions from file or random positions if not provided"""
    boids = []
    if positions is None:
        # Create random positions if none provided
        positions = np.array([
            [np.random.uniform(0, area_size), np.random.uniform(0, area_size)]
            for _ in range(num_boids)
        ])
    
    for position in positions:
        # Create random 2D velocity in boid size units
        velocity = np.array([
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(-0.1, 0.1)
        ])
        # Pass width and height for wall boundary conditions
        boids.append(Boid(position, velocity, width=area_size, height=area_size, wall_behavior='bounce'))
    return boids

def load_weight_genomes():
    """Load weight genomes from CSV file"""
    try:
        weights_df = pd.read_csv('all_weights.csv')
        # Filter for best_combined type only
        weights_df = weights_df[weights_df['type'] == 'best_combined']
        # Filter for view radius between 18 and 40
        weights_df = weights_df[(weights_df['view_radius'] > 18) & (weights_df['view_radius'] < 40)]
        # Sort by view radius in descending order
        weights_df = weights_df.sort_values('view_radius', ascending=False)
        # Extract the relevant columns in the correct order
        weights = weights_df[['w_s', 'w_a', 'w_c', 'view_radius']].values
        return weights
    except Exception as e:
        print(f"Error loading weight genomes: {e}")
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Simulate boids with given weights')
    parser.add_argument('--separation_weight', type=float, default=1.0, help='Weight for separation force')
    parser.add_argument('--alignment_weight', type=float, default=1.0, help='Weight for alignment force')
    parser.add_argument('--cohesion_weight', type=float, default=1.0, help='Weight for cohesion force')
    parser.add_argument('--num_boids', type=int, default=config.NUM_BOIDS, help='Number of boids to simulate')
    parser.add_argument('--width', type=int, default=config.WIDTH, help='Width of the simulation window')
    parser.add_argument('--height', type=int, default=config.HEIGHT, help='Height of the simulation window')
    parser.add_argument('--show_view_radius', action='store_true', help='Show view radius around boids')
    args = parser.parse_args()

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height))
    pygame.display.set_caption("Boid Simulation")
    clock = pygame.time.Clock()

    # Load weight genomes
    weight_genomes = load_weight_genomes()
    current_genome_index = 0

    # Create genome dropdown if genomes are available
    genome_dropdown = None
    if weight_genomes is not None:
        genome_dropdown = Dropdown(
            x=args.width - 150,  # Position on the right side
            y=10,
            width=140,
            height=30,
            options=weight_genomes,
            initial_index=current_genome_index
        )

    # Load initial positions from file
    initial_positions = load_initial_positions()

    # Initialize simulation
    simulation = BoidSimulation()
    print(f"Initializing boids with positions from file: {initial_positions is not None}")
    boids = initialize_boids(args.num_boids, args.width, initial_positions)
    original_boids = initialize_boids(args.num_boids, args.width, initial_positions)  # Store original state for reset

    # Create sliders
    slider_width = 200
    slider_height = 20
    slider_x = 10
    slider_y_start = 50
    slider_spacing = 50

    separation_slider = Slider(slider_x, slider_y_start, slider_width, slider_height,
                             config.MIN_BOID_WEIGHT, config.MAX_BOID_WEIGHT, args.separation_weight, "Separation")
    alignment_slider = Slider(slider_x, slider_y_start + slider_spacing, slider_width, slider_height,
                            config.MIN_BOID_WEIGHT, config.MAX_BOID_WEIGHT, args.alignment_weight, "Alignment")
    cohesion_slider = Slider(slider_x, slider_y_start + 2 * slider_spacing, slider_width, slider_height,
                           config.MIN_BOID_WEIGHT, config.MAX_BOID_WEIGHT, args.cohesion_weight, "Cohesion")
    view_radius_slider = Slider(slider_x, slider_y_start + 3 * slider_spacing, slider_width, slider_height,
                              config.VIEW_RADIUS_MIN, config.VIEW_RADIUS_MAX, config.VIEW_RADIUS, "View Radius")

    # Simulation state
    is_running = False
    frame_count = 0
    font = pygame.font.SysFont(None, 24)

    # Main simulation loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_v:
                    args.show_view_radius = not args.show_view_radius  # Toggle view radius display
                elif event.key == pygame.K_r:  # Reset
                    print("Reset pressed - reloading positions from file")
                    try:
                        df = pd.read_csv(config.INITIAL_POSITIONS_PATH)
                        initial_positions = df[['x', 'y']].values
                        print(f"Successfully loaded {len(initial_positions)} positions for reset")
                    except Exception as e:
                        print(f"Error loading positions for reset: {e}")
                        initial_positions = None
                    boids = initialize_boids(args.num_boids, args.width, initial_positions)
                    is_running = False
                    frame_count = 0
                elif event.key == pygame.K_s:  # Start
                    is_running = True
                elif event.key == pygame.K_e:  # End
                    is_running = False
        
        # Handle slider events
        separation_slider.handle_event(event)
        alignment_slider.handle_event(event)
        cohesion_slider.handle_event(event)
        view_radius_slider.handle_event(event)
        
        # Handle genome dropdown events
        if genome_dropdown is not None:
            if genome_dropdown.handle_event(event):
                # Update weights when a new genome is selected
                weights = weight_genomes[genome_dropdown.selected_index]
                separation_slider.value = weights[0]
                alignment_slider.value = weights[1]
                cohesion_slider.value = weights[2]
                view_radius_slider.value = weights[3]

        # Update boid parameters from sliders
        for boid in boids:
            boid.separation_weight = separation_slider.value
            boid.alignment_weight = alignment_slider.value
            boid.cohesion_weight = cohesion_slider.value
            boid.view_radius = int(view_radius_slider.value)

        # Update simulation only if running
        if is_running:
            simulation.update(boids)
            frame_count += 1
            # Stop simulation after 721 frames but keep window open
            if frame_count >= 721:
                is_running = False
                print(f"Simulation stopped after {frame_count} frames")

        # Draw
        screen.fill((255, 255, 255))  # White background
        
        # Draw boids
        for boid in boids:
            # Draw view radius if enabled
            if args.show_view_radius:
                pygame.draw.circle(screen, (200, 200, 200), 
                                 (int(boid.position[0]), int(boid.position[1])), 
                                 int(boid.view_radius), 1)
            
            # Draw boid as a circle with size based on BOID_SIZE_PIXELS
            pygame.draw.circle(screen, (0, 0, 255), 
                             (int(boid.position[0]), int(boid.position[1])), 
                             config.BOID_SIZE_PIXELS // 2)
            
            # Draw velocity vector (scaled for visibility)
            end_pos = boid.position + boid.velocity * 10
            pygame.draw.line(screen, (255, 0, 0),
                           (int(boid.position[0]), int(boid.position[1])),
                           (int(end_pos[0]), int(end_pos[1])), 1)

        # Draw sliders
        separation_slider.draw(screen)
        alignment_slider.draw(screen)
        cohesion_slider.draw(screen)
        view_radius_slider.draw(screen)
        
        # Draw genome dropdown if available
        if genome_dropdown is not None:
            genome_dropdown.draw(screen)

        # Draw frame counter
        frame_text = font.render(f"frame = {frame_count}", True, (0, 0, 0))
        screen.blit(frame_text, (10, 10))

        # Update display
        pygame.display.flip()
        clock.tick(60)  # 60 FPS

    pygame.quit()

if __name__ == "__main__":
    main()
