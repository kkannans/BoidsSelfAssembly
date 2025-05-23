import sys
import os
# Add parent directory to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import matplotlib.pyplot as plt
import config
from typing import List, Dict, Tuple

def load_best_genomes(folder_path):
    """Load best_pwd, best_rdf and best_combined from population.pkl"""
    population_path = os.path.join(folder_path, 'population.pkl')
    if not os.path.exists(population_path):
        raise FileNotFoundError(f"Genome file not found: {population_path}")
    with open(population_path, 'rb') as f:
        population = pickle.load(f)
    best_pwd = max(population, key=lambda x: x['fitness_pwd'])
    best_rdf = max(population, key=lambda x: x['fitness_rdf'])
    best_combined = get_waist(population)
    return best_pwd, best_rdf, best_combined, population

def get_waist(population: List[Dict]) -> Dict:
    """Find the waist of the population (individuals closest to maximum values in both objectives)."""
    # Find maximum values for both fitness metrics
    max_pwd = max(p['fitness_pwd'] for p in population)
    max_rdf = max(p['fitness_rdf'] for p in population)
    
    # Calculate distances for all individuals
    distances = []
    for p in population:
        # Calculate normalized distance to maximum point
        norm_pwd = (max_pwd - p['fitness_pwd']) / max_pwd
        norm_rdf = (max_rdf - p['fitness_rdf']) / max_rdf
        distance = np.sqrt(norm_pwd**2 + norm_rdf**2)
        distances.append((distance, p))
    
    # Sort by distance and get the closest
    distances.sort(key=lambda x: x[0])
    min_distance = distances[0][0]
    waist = [p for dist, p in distances if dist == min_distance]
    return waist[0]

def is_dominated(point: Tuple[float, float], points: List[Tuple[float, float]]) -> bool:
    """Check if a point is dominated by any other point in the set."""
    for other in points:
        if other[0] > point[0] and other[1] > point[1]:
            return True
    return False

def get_pareto_front(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Find the Pareto front from a set of points."""
    pareto_front = []
    for point in points:
        if not is_dominated(point, points):
            pareto_front.append(point)
    return sorted(pareto_front, key=lambda x: x[0])

def plot_radius_distributions(base_dir):
    # Create output directory if it doesn't exist
    output_dir = "ALIFE_2025"
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = ['fitness_pwd', 'fitness_rdf', 'best_combined']
    radii_by_metric = {m: [] for m in metrics}
    all_points = []
    best_global_point = None
    best_local_point = None
    max_global = float('-inf')
    max_local = float('-inf')
    
    # Create a single figure with one main plot
    fig, ax2 = plt.subplots(figsize=(4, 4))
    
    # Create an inset axes for the Pareto front plot with adjusted position
    ax1 = fig.add_axes([0.5, 0.5, 0.43, 0.43])  # [left, bottom, width, height]
    
    for trial in os.listdir(base_dir):
        trial_path = os.path.join(base_dir, trial)
        if not os.path.isdir(trial_path):
            continue
        try:
            best_pwd, best_rdf, best_combined, population = load_best_genomes(trial_path)
            # Collect data for box plot
            radii_by_metric['fitness_pwd'].extend(best_pwd['genome'][3::4] * config.VIEW_RADIUS_MAX)
            radii_by_metric['fitness_rdf'].extend(best_rdf['genome'][3::4] * config.VIEW_RADIUS_MAX)
            radii_by_metric['best_combined'].extend(best_combined['genome'][3::4] * config.VIEW_RADIUS_MAX)
            
            # Collect points for Pareto front and update max values
            for p in population:
                all_points.append((p['fitness_pwd'], p['fitness_rdf']))
                max_global = max(max_global, p['fitness_pwd'])
                max_local = max(max_local, p['fitness_rdf'])
            
            # Update best points if better
            if best_global_point is None or best_pwd['fitness_pwd'] > best_global_point[0]:
                best_global_point = (best_pwd['fitness_pwd'], best_pwd['fitness_rdf'])
            if best_local_point is None or best_rdf['fitness_rdf'] > best_local_point[1]:
                best_local_point = (best_rdf['fitness_pwd'], best_rdf['fitness_rdf'])
            
        except Exception as e:
            print(f"Skipping {trial}: {e}")

    # Calculate waist point across all points
    distances = []
    for point in all_points:
        norm_global = (max_global - point[0]) / max_global
        norm_local = (max_local - point[1]) / max_local
        distance = np.sqrt(norm_global**2 + norm_local**2)
        distances.append((distance, point))
    
    distances.sort(key=lambda x: x[0])
    min_distance = distances[0][0]
    waist_points = [p for dist, p in distances if dist == min_distance]
    best_combined_point = waist_points[0]  # Take the first point if there are multiple with same distance

    # Plot Pareto front in the inset axes
    pareto_front = get_pareto_front(all_points)
    pareto_x = [p[0] for p in pareto_front]
    pareto_y = [p[1] for p in pareto_front]
    
    ax1.plot(pareto_x, pareto_y, 'r-', linewidth=1)
    ax1.scatter(pareto_x, pareto_y, c='red', s=10, alpha=0.7)
    
    # Plot best points with compact markers and minimal annotation
    ax1.scatter(best_global_point[0], best_global_point[1], c='blue', s=20, marker='*')
    ax1.scatter(best_local_point[0], best_local_point[1], c='green', s=20, marker='*')
    ax1.scatter(best_combined_point[0], best_combined_point[1], c='purple', s=20, marker='*')
    
    # Fix: Use r"$F_{\text{global}}$" format for proper LaTeX rendering
    ax1.set_xlabel(r"$F_{\text{global}}$", fontsize=10)
    ax1.set_ylabel(r"$F_{\text{local}}$", fontsize=10)
    
    # Create a separate legend with minimal text
    legend_elements = [
        plt.Line2D([0], [0], color='r', lw=1, label='Pareto'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='blue', markersize=8, label=r'$F_{\text{global}}$'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='green', markersize=8, label=r'$F_{\text{local}}$'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='purple', markersize=8, label=r'$F_{\text{combined}}$')
    ]
    
    # Place legend inside the plot with small font size and tight layout
    ax1.legend(handles=legend_elements, fontsize=10, loc='lower left', 
               frameon=True, framealpha=0.8, edgecolor='none',
               handlelength=1.5, handleheight=1.5, handletextpad=0.5,
               borderaxespad=0.3, borderpad=0.4, labelspacing=0.3)
    
    # Make tick labels smaller
    ax1.tick_params(axis='both', which='major', labelsize=10)
    
    # Store original axis limits
    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()
    
    # Add arrow heads using arrow
    # X-axis arrow
    ax1.arrow(xmax-0.1*(xmax-xmin), 0, 0.1*(xmax-xmin), 0, 
             head_width=0.02, head_length=0.02, fc='black', ec='black')
    # Y-axis arrow
    ax1.arrow(0, ymax-0.1*(ymax-ymin), 0, 0.1*(ymax-ymin),
             head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    # Restore original axis limits
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    
    # Plot box plot in the main axis
    data = [radii_by_metric[m] for m in metrics]
    # Fix: Update boxplot labels with proper LaTeX formatting
    box = ax2.boxplot(data, labels=[r"$F_{\text{global}}$", r"$F_{\text{local}}$", r"$F_{\text{combined}}$"])
    
    # Get the positions of the boxes
    box_positions = [box['boxes'][i].get_path().vertices[:, 0].mean() for i in range(len(metrics))]
    
    # Add text of mean and std above the boxplot, with smaller font
    for i, m in enumerate(metrics):
        mean = np.mean(radii_by_metric[m])
        std = np.std(radii_by_metric[m])
        # Position text at y=30
        ax2.text(box_positions[i], 30, f'μ={mean:.1f}\nσ={std:.1f}', 
                fontsize=6, ha='center', va='bottom',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
                
    ax2.set_ylabel(r"View radius ($r_{v}$)", fontsize=10)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    
    # Add a border to the inset plot to make it stand out
    for spine in ax1.spines.values():
        spine.set_edgecolor('gray')
        spine.set_linewidth(0.8)
    
    # Remove extra space and tighten layout
    plt.tight_layout()
    
    # Update save paths to use output_dir
    plt.savefig(os.path.join(output_dir, "figure_radius_distribution_exp4.png"), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "figure_radius_distribution_exp4.pdf"), bbox_inches='tight')

# usage
if __name__ == "__main__":
    experiment_dir = "../evolution_e4_afpo_pop50_gen50_new_pwd"
    plot_radius_distributions(experiment_dir)