import sys
import os
# Add parent directory to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

import config

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

def plot_weight_distributions(base_dir):
    # Create output directory if it doesn't exist
    output_dir = "ALIFE_2025"
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = ['fitness_pwd', 'fitness_rdf', 'best_combined']
    weights_by_metric = {m: {'w_s': [], 'w_a': [], 'w_c': []} for m in metrics}
    
    # Create a single figure with one main plot
    fig, ax = plt.subplots(figsize=(8, 4))
    
    for trial in os.listdir(base_dir):
        trial_path = os.path.join(base_dir, trial)
        if not os.path.isdir(trial_path):
            continue
        try:
            best_pwd, best_rdf, best_combined, population = load_best_genomes(trial_path)
            # Collect data for box plot - weights are at indices 0,1,2
            weights_by_metric['fitness_pwd']['w_s'].append(best_pwd['genome'][0])
            weights_by_metric['fitness_pwd']['w_a'].append(best_pwd['genome'][1])
            weights_by_metric['fitness_pwd']['w_c'].append(best_pwd['genome'][2])
            
            weights_by_metric['fitness_rdf']['w_s'].append(best_rdf['genome'][0])
            weights_by_metric['fitness_rdf']['w_a'].append(best_rdf['genome'][1])
            weights_by_metric['fitness_rdf']['w_c'].append(best_rdf['genome'][2])
            
            weights_by_metric['best_combined']['w_s'].append(best_combined['genome'][0])
            weights_by_metric['best_combined']['w_a'].append(best_combined['genome'][1])
            weights_by_metric['best_combined']['w_c'].append(best_combined['genome'][2])
            
        except Exception as e:
            print(f"Skipping {trial}: {e}")

    # Plot box plots in the main axis
    positions = []
    data = []
    labels = []
    
    # Create positions for 9 boxes (3 weights × 3 metrics)
    weight_labels = [r'$w_s$', r'$w_a$', r'$w_c$']
    for i, metric in enumerate(metrics):
        for j, weight in enumerate(['w_s', 'w_a', 'w_c']):
            positions.append(i * 4 + j + 1)  # Space between metric groups
            data.append(weights_by_metric[metric][weight])
            labels.append(weight_labels[j])
    
    # Create the boxplot
    box = ax.boxplot(data, positions=positions, labels=labels)
    
    # Add metric labels
    metric_positions = [2, 6, 10]  # Center positions for each metric group
    for i, metric in enumerate(metrics):
        ax.text(metric_positions[i], ax.get_ylim()[0], 
                r"$F_{\text{" + metric.split('_')[1] + "}}$",
                ha='center', va='top', fontsize=10)
    
    # Add mean and std values above each box
    for i, d in enumerate(data):
        mean = np.mean(d)
        std = np.std(d)
        ax.text(positions[i], ax.get_ylim()[1], 
                f'μ={mean:.2f}\nσ={std:.2f}',
                fontsize=6, ha='center', va='bottom',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
    
    ax.set_ylabel("Weight values", fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    
    # Save figures with proper path handling
    plt.savefig(os.path.join(output_dir, "figure_weight_distribution_exp4.png"), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "figure_weight_distribution_exp4.pdf"), bbox_inches='tight')

def plot_weight_magnitudes(base_dir):
    # Create output directory if it doesn't exist
    output_dir = "ALIFE_2025"
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = ['fitness_pwd', 'fitness_rdf', 'best_combined']
    weights_by_metric = {m: {'w_s': [], 'w_a': [], 'w_c': []} for m in metrics}
    
    # Create a single figure with one main plot
    fig, ax = plt.subplots(figsize=(8, 4))
    
    for trial in os.listdir(base_dir):
        trial_path = os.path.join(base_dir, trial)
        if not os.path.isdir(trial_path):
            continue
        try:
            best_pwd, best_rdf, best_combined, population = load_best_genomes(trial_path)
            # Collect data for box plot - weights are at indices 0,1,2
            weights_by_metric['fitness_pwd']['w_s'].append(abs(best_pwd['genome'][0]))
            weights_by_metric['fitness_pwd']['w_a'].append(abs(best_pwd['genome'][1]))
            weights_by_metric['fitness_pwd']['w_c'].append(abs(best_pwd['genome'][2]))
            
            weights_by_metric['fitness_rdf']['w_s'].append(abs(best_rdf['genome'][0]))
            weights_by_metric['fitness_rdf']['w_a'].append(abs(best_rdf['genome'][1]))
            weights_by_metric['fitness_rdf']['w_c'].append(abs(best_rdf['genome'][2]))
            
            weights_by_metric['best_combined']['w_s'].append(abs(best_combined['genome'][0]))
            weights_by_metric['best_combined']['w_a'].append(abs(best_combined['genome'][1]))
            weights_by_metric['best_combined']['w_c'].append(abs(best_combined['genome'][2]))
            
        except Exception as e:
            print(f"Skipping {trial}: {e}")

    # Plot box plots in the main axis
    positions = []
    data = []
    labels = []
    
    # Create positions for 9 boxes (3 weights × 3 metrics)
    weight_labels = [r'$w_s$', r'$w_a$', r'$w_c$']
    for i, metric in enumerate(metrics):
        for j, weight in enumerate(['w_s', 'w_a', 'w_c']):
            positions.append(i * 4 + j + 1)  # Space between metric groups
            data.append(weights_by_metric[metric][weight])
            labels.append(weight_labels[j])
    
    # Create the boxplot
    box = ax.boxplot(data, positions=positions, labels=labels)
    
    # Add metric labels
    metric_positions = [2, 6, 10]  # Center positions for each metric group
    for i, metric in enumerate(metrics):
        ax.text(metric_positions[i], ax.get_ylim()[0], 
                r"$F_{\text{" + metric.split('_')[1] + "}}$",
                ha='center', va='top', fontsize=10)
    
    # Add mean and std values above each box
    for i, d in enumerate(data):
        mean = np.mean(d)
        std = np.std(d)
        ax.text(positions[i], ax.get_ylim()[1], 
                f'μ={mean:.2f}\nσ={std:.2f}',
                fontsize=6, ha='center', va='bottom',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
    
    ax.set_ylabel("Weight magnitudes", fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    
    # Save figures with proper path handling
    plt.savefig(os.path.join(output_dir, "figure_weight_magnitudes_exp4.png"), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "figure_weight_magnitudes_exp4.pdf"), bbox_inches='tight')

def plot_weights_by_radius(base_dir):
    # Create output directory if it doesn't exist
    output_dir = "ALIFE_2025"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize dictionary to store weights by radius
    weights_by_radius = {}  # radius -> {'w_s': [], 'w_a': [], 'w_c': []}
    
    # Collect data from all trials
    trial_count = 0
    for trial in os.listdir(base_dir):
        trial_path = os.path.join(base_dir, trial)
        if not os.path.isdir(trial_path):
            continue
        try:
            best_pwd, best_rdf, best_combined, population = load_best_genomes(trial_path)
            
            # Get view radius from best combined solution and multiply by VIEW_RADIUS_MAX
            view_radius = best_combined['genome'][3] * config.VIEW_RADIUS_MAX
            radius_key = round(view_radius, 2)  # Round to 2 decimal places for grouping
            
            # Initialize radius group if not exists
            if radius_key not in weights_by_radius:
                weights_by_radius[radius_key] = {'w_s': [], 'w_a': [], 'w_c': []}
            
            # Add weights for this radius
            weights_by_radius[radius_key]['w_s'].append(best_combined['genome'][0])
            weights_by_radius[radius_key]['w_a'].append(best_combined['genome'][1])
            weights_by_radius[radius_key]['w_c'].append(best_combined['genome'][2])
            
            trial_count += 1
            
        except Exception as e:
            print(f"Skipping {trial}: {e}")
    
    print(f"Total trials processed: {trial_count}")
    
    # Sort radii for consistent plotting
    sorted_radii = sorted(weights_by_radius.keys())
    
    # Create a single figure
    plt.figure(figsize=(10, 5))
    
    # Plot data for each weight type
    weight_data = {
        'w_s': ('blue', r'$w_s$'),
        'w_a': ('green', r'$w_a$'),
        'w_c': ('red', r'$w_c$')
    }
    
    # Calculate global y-axis limits
    all_means = []
    all_stds = []
    for weight_type, (color, label) in weight_data.items():
        means = []
        stds = []
        for radius in sorted_radii:
            data = weights_by_radius[radius][weight_type]
            means.append(np.mean(data))
            stds.append(np.std(data))
        all_means.extend(means)
        all_stds.extend(stds)
    
    y_min = min(np.array(all_means) - np.array(all_stds))
    y_max = max(np.array(all_means) + np.array(all_stds))
    y_margin = (y_max - y_min) * 0.1
    plt.ylim(y_min - y_margin, y_max + y_margin)
    
    # Plot all weights on the same axis
    for weight_type, (color, label) in weight_data.items():
        means = []
        stds = []
        
        for radius in sorted_radii:
            data = weights_by_radius[radius][weight_type]
            means.append(np.mean(data))
            stds.append(np.std(data))
        
        # Plot mean values with error bars
        plt.errorbar(sorted_radii, means, yerr=stds, fmt='o-', color=color, 
                    capsize=5, label=label, markersize=6)
    
    # Add vertical line for average cell radius
    plt.axvline(x=15.78, color='black', linestyle='--', alpha=0.7, linewidth=2)
    
    plt.xlabel(r"View radius ($r_v$) (pixels)", fontsize=18)
    plt.ylabel("Weights", fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.legend(loc='best', fontsize=18)
    
    # Set x-axis to start at 15.78
    plt.xlim(15.78, plt.xlim()[1])
    
    plt.tight_layout()
    
    # Save figures with proper path handling
    plt.savefig(os.path.join(output_dir, "figure_weight_distribution_by_radius_exp4.png"), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "figure_weight_distribution_by_radius_exp4.pdf"), bbox_inches='tight')
    plt.close()

# usage
if __name__ == "__main__":
    experiment_dir = "../evolution_e4_afpo_pop50_gen50_new_pwd"
    plot_weight_distributions(experiment_dir)
    plot_weight_magnitudes(experiment_dir)
    plot_weights_by_radius(experiment_dir)