import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
# Add parent directory to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from typing import List, Dict, Tuple
from plot_distribution_best_weights_exp4 import load_best_genomes, get_waist

def analyze_weight_dominance_by_radius(base_dir, min_radius=15, max_radius=45):
    """
    Analyze how often and by how much each weight dominates the other two
    between the specified radius range.
    
    Args:
        base_dir: Directory containing evolution experiment results
        min_radius: Minimum radius to consider (default: 15)
        max_radius: Maximum radius to consider (default: 45)
    
    Returns:
        Dictionary containing analysis results
    """
    # Initialize data structures
    radius_weight_data = {}  # radius -> {'w_s': [], 'w_a': [], 'w_c': []}
    dominance_stats = {
        'w_s_dominates': {'count': 0, 'magnitudes': []},
        'w_a_dominates': {'count': 0, 'magnitudes': []},
        'w_c_dominates': {'count': 0, 'magnitudes': []},
        'no_clear_dominance': {'count': 0}
    }
    
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
            
            # Only consider radii within the specified range
            if min_radius <= view_radius <= max_radius:
                radius_key = round(view_radius, 2)  # Round to 2 decimal places for grouping
                
                # Initialize radius group if not exists
                if radius_key not in radius_weight_data:
                    radius_weight_data[radius_key] = {'w_s': [], 'w_a': [], 'w_c': []}
                
                # Add weights for this radius
                radius_weight_data[radius_key]['w_s'].append(best_combined['genome'][0])
                radius_weight_data[radius_key]['w_a'].append(best_combined['genome'][1])
                radius_weight_data[radius_key]['w_c'].append(best_combined['genome'][2])
                
                trial_count += 1
                
        except Exception as e:
            print(f"Skipping {trial}: {e}")
    
    print(f"Total trials processed within radius range [{min_radius}, {max_radius}]: {trial_count}")
    
    # Analyze dominance for each radius group
    for radius, weights in radius_weight_data.items():
        # Calculate mean weights for this radius
        mean_ws = np.mean(weights['w_s'])
        mean_wa = np.mean(weights['w_a'])
        mean_wc = np.mean(weights['w_c'])
        
        # Calculate absolute values for magnitude comparison
        abs_ws = abs(mean_ws)
        abs_wa = abs(mean_wa)
        abs_wc = abs(mean_wc)
        
        # Determine which weight dominates
        max_weight = max(abs_ws, abs_wa, abs_wc)
        
        if abs_ws == max_weight and abs_ws > abs_wa and abs_ws > abs_wc:
            # w_s dominates
            dominance_stats['w_s_dominates']['count'] += 1
            # Calculate magnitude of dominance (difference from second highest)
            second_highest = max(abs_wa, abs_wc)
            dominance_magnitude = abs_ws - second_highest
            dominance_stats['w_s_dominates']['magnitudes'].append(dominance_magnitude)
            
        elif abs_wa == max_weight and abs_wa > abs_ws and abs_wa > abs_wc:
            # w_a dominates
            dominance_stats['w_a_dominates']['count'] += 1
            # Calculate magnitude of dominance
            second_highest = max(abs_ws, abs_wc)
            dominance_magnitude = abs_wa - second_highest
            dominance_stats['w_a_dominates']['magnitudes'].append(dominance_magnitude)
            
        elif abs_wc == max_weight and abs_wc > abs_ws and abs_wc > abs_wa:
            # w_c dominates
            dominance_stats['w_c_dominates']['count'] += 1
            # Calculate magnitude of dominance
            second_highest = max(abs_ws, abs_wa)
            dominance_magnitude = abs_wc - second_highest
            dominance_stats['w_c_dominates']['magnitudes'].append(dominance_magnitude)
            
        else:
            # No clear dominance (ties or very close values)
            dominance_stats['no_clear_dominance']['count'] += 1
    
    return dominance_stats, radius_weight_data

def print_analysis_results(dominance_stats):
    """Print detailed analysis results."""
    print("\n" + "="*60)
    print("WEIGHT DOMINANCE ANALYSIS RESULTS")
    print("="*60)
    
    total_dominance_cases = sum([
        dominance_stats['w_s_dominates']['count'],
        dominance_stats['w_a_dominates']['count'],
        dominance_stats['w_c_dominates']['count']
    ])
    
    print(f"\nTotal cases analyzed: {total_dominance_cases + dominance_stats['no_clear_dominance']['count']}")
    print(f"Cases with clear dominance: {total_dominance_cases}")
    print(f"Cases with no clear dominance: {dominance_stats['no_clear_dominance']['count']}")
    
    print("\nDominance Breakdown:")
    print("-" * 40)
    
    for weight_type in ['w_s_dominates', 'w_a_dominates', 'w_c_dominates']:
        count = dominance_stats[weight_type]['count']
        magnitudes = dominance_stats[weight_type]['magnitudes']
        
        if count > 0:
            avg_magnitude = np.mean(magnitudes)
            max_magnitude = np.max(magnitudes)
            min_magnitude = np.min(magnitudes)
            
            print(f"{weight_type.replace('_', ' ').title()}:")
            print(f"  Count: {count} ({count/total_dominance_cases*100:.1f}%)")
            print(f"  Average dominance magnitude: {avg_magnitude:.4f}")
            print(f"  Maximum dominance magnitude: {max_magnitude:.4f}")
            print(f"  Minimum dominance magnitude: {min_magnitude:.4f}")
        else:
            print(f"{weight_type.replace('_', ' ').title()}: 0 cases")
    
    # Find the most dominant weight type
    max_count = 0
    most_dominant = None
    for weight_type in ['w_s_dominates', 'w_a_dominates', 'w_c_dominates']:
        if dominance_stats[weight_type]['count'] > max_count:
            max_count = dominance_stats[weight_type]['count']
            most_dominant = weight_type
    
    if most_dominant:
        print(f"\nMost dominant weight: {most_dominant.replace('_', ' ').title()}")
        print(f"Dominates in {max_count} out of {total_dominance_cases} cases ({max_count/total_dominance_cases*100:.1f}%)")

def plot_dominance_analysis(dominance_stats, radius_weight_data, min_radius=15, max_radius=45):
    """Create visualization of the dominance analysis."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data for plotting
    weight_types = ['w_s_dominates', 'w_a_dominates', 'w_c_dominates']
    labels = ['$w_s$', '$w_a$', '$w_c$']
    colors = ['blue', 'green', 'red']
    
    # Get counts and magnitude data for box plots
    counts = [dominance_stats[wt]['count'] for wt in weight_types]
    magnitude_data = []
    for wt in weight_types:
        magnitudes = dominance_stats[wt]['magnitudes']
        if magnitudes:
            magnitude_data.append(magnitudes)
        else:
            magnitude_data.append([0])
    
    # Create dual y-axis plot
    ax2 = ax.twinx()
    
    # Color for each axis
    left_axis_color = "black"  # Weight magnitudes
    right_axis_color = "darkorange"  # Number of cases
    
    # Boxplots (left axis)
    bp = ax.boxplot(magnitude_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Weight magnitude', fontsize=18, color=left_axis_color)
    ax.set_xlabel('Weights', fontsize=18)
    ax.tick_params(axis='y', colors=left_axis_color, labelsize=18)
    ax.spines['left'].set_color(left_axis_color)
    
    # Scatter plot (right axis) - circles only, no line
    line = ax2.plot(range(1, len(labels) + 1), counts, 'o', color=right_axis_color,
                    markersize=8, label='Number of cases')
    ax2.set_ylabel('Number of cases dominating', fontsize=18, color=right_axis_color)
    ax2.tick_params(axis='y', colors=right_axis_color, labelsize=18)
    ax2.spines['right'].set_color(right_axis_color)
    
    # Add count labels on the line
    for i, count in enumerate(counts):
        ax2.text(i + 1, count + 0.5, str(count), ha='center', va='bottom', fontweight='bold', fontsize=18, color=right_axis_color)
    
    # Set y-axis limits
    all_magnitudes = [item for sublist in magnitude_data for item in sublist]
    ax.set_ylim(0, max(all_magnitudes) * 1.2)
    ax2.set_ylim(0, max(counts) * 1.2)

    # Set x ticks font size
    ax.tick_params(axis='x', labelsize=18)
    ax2.tick_params(axis='x', labelsize=18)
    
    # Remove title as requested
    plt.tight_layout()
    
    # Save figure
    os.makedirs("ALIFE_2025", exist_ok=True)
    plt.savefig("ALIFE_2025/figure_weight_dominance_analysis_exp4.png", dpi=300, bbox_inches='tight')
    plt.savefig("ALIFE_2025/figure_weight_dominance_analysis_exp4.pdf", bbox_inches='tight')
    plt.close()

def main():
    """Main function to run the analysis."""
    experiment_dir = "../evolution_e4_afpo_pop50_gen50_new_pwd"
    
    print("Analyzing weight dominance in experiment 4...")
    print(f"Radius range: [15, 45]")
    print(f"Experiment directory: {experiment_dir}")
    
    # Run the analysis
    dominance_stats, radius_weight_data = analyze_weight_dominance_by_radius(
        experiment_dir, min_radius=15, max_radius=45
    )
    
    # Print results
    print_analysis_results(dominance_stats)
    
    # Create visualization
    plot_dominance_analysis(dominance_stats, radius_weight_data, min_radius=15, max_radius=45)
    
    print("\nAnalysis complete! Visualizations saved to ALIFE_2025/")

if __name__ == "__main__":
    main()
