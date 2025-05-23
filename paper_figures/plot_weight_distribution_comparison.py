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
    
    # Save figures
    plt.savefig("ALIFE_2025/figure_weight_distribution_exp1.png", dpi=600, bbox_inches='tight')
    plt.savefig("ALIFE_2025/figure_weight_distribution_exp1.pdf", bbox_inches='tight')


def plot_weight_distributions_comparison(exp1_dir, exp4_dir):
    """Plot weight distributions from experiment 1 and 4 in a single figure, only for best combined fitness."""
    # Create output directory if it doesn't exist
    output_dir = "ALIFE_2025"
    os.makedirs(output_dir, exist_ok=True)
    
    experiments = {
        'HS1': exp1_dir,
        'HS2': exp4_dir
    }
    
    # Create a single figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for all experiments
    all_data = []
    positions = []
    labels = []
    
    # First collect all data
    exp_data = {}
    for exp_name, exp_dir in experiments.items():
        weights = {'w_s': [], 'w_a': [], 'w_c': []}
        
        for trial in os.listdir(exp_dir):
            trial_path = os.path.join(exp_dir, trial)
            if not os.path.isdir(trial_path):
                continue
            try:
                _, _, best_combined, _ = load_best_genomes(trial_path)
                weights['w_s'].append(best_combined['genome'][0])
                weights['w_a'].append(best_combined['genome'][1])
                weights['w_c'].append(best_combined['genome'][2])
            except Exception as e:
                print(f"Skipping {trial}: {e}")
        
        exp_data[exp_name] = weights
    
    # Now organize data by weight type
    weight_types = ['w_s', 'w_a', 'w_c']
    for weight_type in weight_types:
        for exp_name in experiments.keys():
            all_data.append(exp_data[exp_name][weight_type])
            positions.append(len(all_data))
            labels.append(exp_name)
    
    # Create the boxplot
    box = ax.boxplot(all_data, positions=positions, labels=labels)
    
    # Add weight type labels
    weight_labels = [r'$w_s$', r'$w_a$', r'$w_c$']
    for i, label in enumerate(weight_labels):
        pos = (positions[i*2] + positions[i*2+1]) / 2
        ax.text(pos, ax.get_ylim()[0], label, 
                ha='center', va='top', fontsize=20)
    
    # Add mean and std values above each box
    for i, d in enumerate(all_data):
        mean = np.mean(d)
        std = np.std(d)
        # Position text higher for better visibility
        ax.text(positions[i], ax.get_ylim()[1] * 0.95, 
                f'μ={mean:.2f}\nσ={std:.2f}',
                fontsize=16, ha='center', va='bottom',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
    
    # Add vertical lines to separate weight types
    for i in range(1, 3):
        ax.axvline(x=i*2 + 0.5, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_ylabel("Weights", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0, ha='right')
    
    # Add legend
    legend_text = [
        'HS1: $(w_s,w_a,w_c,r_v=40)$',
        'HS2: $(w_s,w_a,w_c,r_v)$'
    ]
    ax.legend(legend_text, loc='lower left', fontsize=18)
    
    # Adjust y-axis limits to accommodate text
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max * 1.15)
    
    plt.tight_layout()
    
    # Save figures with the created directory
    plt.savefig(os.path.join(output_dir, "figure_weight_distribution_comparison.png"), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "figure_weight_distribution_comparison.pdf"), bbox_inches='tight')

# usage
if __name__ == "__main__":
    experiment1_dir = "../evolution_e1_afpo_pop50_gen50_new_pwd"
    experiment4_dir = "../evolution_e4_afpo_pop50_gen50_new_pwd"
    
    # Plot weight distributions side by side
    plot_weight_distributions_comparison(experiment1_dir, experiment4_dir)