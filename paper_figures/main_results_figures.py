import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import gridspec
from scipy import stats

def compute_test_statistics(data1, data2, prediction_horizon=20):
    """
    Computes statistical significance of the difference between data1 and data2
    at accumulated time points t=5, t=10, t=15, and t=20.
    
    Args:
        data1: 2D array (num_trials, prediction_horizon) - 30 trials of model losses
        data2: 1D array (prediction_horizon) - 1 control loss series
        prediction_horizon: int - number of time steps
        
    Returns:
        Dictionary containing p-values and t-statistics for each time horizon
    """
    # Initialize arrays for results
    p_vals = np.zeros(prediction_horizon)
    t_stats = np.zeros(prediction_horizon)
    
    # For each time point, we want to compute the accumulated loss up to that point
    accumulation_points = [5, 10, 15, 20]
    accumulation_results = {
        'p_vals': [],
        't_stats': [],
        'time_points': [],
        'difference': [],
        'effect_size': []
    }
    
    for time_point in accumulation_points:
        if time_point > prediction_horizon:
            continue
            
        # Calculate accumulated losses for each trial up to this time point
        accumulated_model_losses = np.sum(data1[:, :time_point], axis=1)  # Shape: (num_trials,)
        
        # Calculate accumulated control loss up to this time point
        accumulated_control_loss = np.sum(data2[:time_point])
        
        # Create array of control values (same accumulated value for comparison to each trial)
        control_values = np.full(accumulated_model_losses.shape, accumulated_control_loss)
        
        # Use two-sided Welch's t-test
        t_stat, p_val = stats.ttest_ind(accumulated_model_losses, control_values, equal_var=False)
        
        # Calculate effect size (Cohen's d) using pooled standard deviation
        n1 = len(accumulated_model_losses)
        n2 = len(control_values)
        
        # Calculate means
        mean1 = np.mean(accumulated_model_losses)
        mean2 = np.mean(control_values)
        
        # Calculate variances
        var1 = np.var(accumulated_model_losses, ddof=1)
        var2 = np.var(control_values, ddof=1)
        
        # Calculate pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        # Calculate Cohen's d
        if pooled_std == 0:
            # If standard deviation is 0, check if means are different
            effect_size = np.sign(mean1 - mean2) * float('inf') if mean1 != mean2 else 0
        else:
            effect_size = (mean1 - mean2) / pooled_std
        
        # Store values in the corresponding time point
        accumulation_results['time_points'].append(time_point)
        accumulation_results['p_vals'].append(p_val)
        accumulation_results['t_stats'].append(t_stat)
        accumulation_results['difference'].append(mean1 - mean2)
        accumulation_results['effect_size'].append(effect_size)
        
    # Create a more detailed results dictionary
    results = {
        'p_vals': p_vals,
        't_stats': t_stats,
        'accumulation_results': accumulation_results
    }
    
    return results
    

def compute_test_statistics_model_vs_random(data1, data2, prediction_horizon=20):
    """
    Computes statistical significance of the difference between two sets of model data
    where both inputs are 2D arrays of shape (num_trials, prediction_horizon).
    
    Args:
        data1: 2D array (num_trials, prediction_horizon) - First model's losses
        data2: 2D array (num_trials, prediction_horizon) - Second model's losses
        prediction_horizon: int - number of time steps
        
    Returns:
        Dictionary containing p-values and t-statistics for each time horizon
    """
    # Initialize arrays for results
    p_vals = np.zeros(prediction_horizon)
    t_stats = np.zeros(prediction_horizon)
    
    # For each time point, we want to compute the accumulated loss up to that point
    accumulation_points = [5, 10, 15, 20]
    accumulation_results = {
        'p_vals': [],
        't_stats': [],
        'time_points': [],
        'difference': [],
        'effect_size': []
    }
    
    for time_point in accumulation_points:
        if time_point > prediction_horizon:
            continue
            
        # Calculate accumulated losses for each trial up to this time point
        accumulated_model1_losses = np.sum(data1[:, :time_point], axis=1)  # Shape: (num_trials,)
        accumulated_model2_losses = np.sum(data2[:, :time_point], axis=1)  # Shape: (num_trials,)
        
        # Use two-sided Welch's t-test
        t_stat, p_val = stats.ttest_ind(accumulated_model1_losses, accumulated_model2_losses, equal_var=False)
        
        # Calculate effect size (Cohen's d) using pooled standard deviation
        n1 = len(accumulated_model1_losses)
        n2 = len(accumulated_model2_losses)
        
        # Calculate means
        mean1 = np.mean(accumulated_model1_losses)
        mean2 = np.mean(accumulated_model2_losses)
        
        # Calculate variances
        var1 = np.var(accumulated_model1_losses, ddof=1)
        var2 = np.var(accumulated_model2_losses, ddof=1)
        
        # Calculate pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        # Calculate Cohen's d
        if pooled_std == 0:
            # If standard deviation is 0, check if means are different
            effect_size = np.sign(mean1 - mean2) * float('inf') if mean1 != mean2 else 0
        else:
            effect_size = (mean1 - mean2) / pooled_std
        
        # Store values in the corresponding time point
        accumulation_results['time_points'].append(time_point)
        accumulation_results['p_vals'].append(p_val)
        accumulation_results['t_stats'].append(t_stat)
        accumulation_results['difference'].append(mean1 - mean2)
        accumulation_results['effect_size'].append(effect_size)
        
    # Create a more detailed results dictionary
    results = {
        'p_vals': p_vals,
        't_stats': t_stats,
        'accumulation_results': accumulation_results
    }
    
    return results

def plot_model_data_vs_regression_baseline(model_mean, model_std, total_regression, 
                                         total_random_genome_mean, total_random_genome_std, 
                                         regression_test_statistics, random_test_statistics, save_path, break_restart_y):
    # Create figure with custom gridspec
    fig = plt.figure(figsize=(6, 4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3], hspace=0.09)
    
    # Create two axes
    ax_top = plt.subplot(gs[0])
    ax_bottom = plt.subplot(gs[1])
    
    # Use steps 1 to 20 for x-axis
    x_values = np.arange(1, 21)
    
    # Plot in both axes
    for ax in [ax_top, ax_bottom]:
        # Plot model with std
        ax.plot(x_values, model_mean, label='Evolved hypothesis', linewidth=1.5, color='blue')
        ax.fill_between(x_values, 
                        model_mean - model_std, 
                        model_mean + model_std, 
                        alpha=0.2, 
                        color='blue')
        
        # Plot regression baseline
        ax.plot(x_values, total_regression, label='Regression', linewidth=1.5, color='red')
        
        # Plot random genome with std
        ax.plot(x_values, total_random_genome_mean, label='Random', linewidth=1.5, color='green')
        ax.fill_between(x_values,
                        total_random_genome_mean - total_random_genome_std,
                        total_random_genome_mean + total_random_genome_std,
                        alpha=0.2,
                        color='green')
        
        # Add vertical lines for time ranges
        for t in [5, 10, 15]:
            ax.axvline(x=t, color='gray', linestyle='--', alpha=0.5)
    
    # Set y-axis limits
    ax_top.set_ylim(break_restart_y, max(np.max(model_mean + model_std), 
                np.max(total_regression),
                np.max(total_random_genome_mean + total_random_genome_std)) + 5)
    ax_bottom.set_ylim(0, 40)
    
    # Set x-axis limits and ticks
    for ax in [ax_top, ax_bottom]:
        ax.set_xlim(1, 20)
        ax.set_xticks([0, 5, 10, 15, 20])
    
    # Hide x-axis labels for top subplot
    ax_top.tick_params(labelbottom=False)
    
    # Add diagonal lines to indicate broken axis
    d = .015  # how big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    
    # Add legend to top plot
    ax_top.legend(fontsize=9, loc='upper left', bbox_to_anchor=(0.02, 0.98))
    
    # Get p-values from accumulated test statistics
    if 'accumulation_results' in regression_test_statistics and 'accumulation_results' in random_test_statistics:
        regression_accum = regression_test_statistics['accumulation_results']
        random_accum = random_test_statistics['accumulation_results']
        regression_p_vals = regression_accum['p_vals']
        random_p_vals = random_accum['p_vals']
        time_points = regression_accum['time_points']
    else:
        # For backward compatibility
        time_ranges = [(0, 5), (5, 10), (10, 15), (15, 20)]
        regression_p_vals = []
        random_p_vals = []
        for start, end in time_ranges:
            avg_p = np.mean(regression_test_statistics['p_vals'][start:end])
            regression_p_vals.append(avg_p)
            if 'p_vals' in random_test_statistics:
                avg_p = np.mean(random_test_statistics['p_vals'][start:end])
                random_p_vals.append(avg_p)
            else:
                random_p_vals.append(avg_p)  # Use same p-values if random test statistics not available
        time_points = [5, 10, 15, 20]
    
    # Add p-value and effect size annotations
    y_pos_p = 35  # Position for p-value text
    y_pos_p_random = 32  # Position for random p-value text
    y_pos_p_regression = 29  # Position for regression p-value text
    
    for i, t in enumerate(time_points):
        if i >= len(regression_p_vals) or i >= len(random_p_vals):
            continue
            
        x_pos = t - 2.5 if t > 5 else t - 1  # Center in each segment
        
        # Format p-value with stars for significance levels
        def format_p_value(p_val):
            if p_val < 0.001:
                return f'p<0.001***'
            elif p_val < 0.01:
                return f'p<0.01**'
            elif p_val < 0.05:
                return f'p<0.05*'
            else:
                return f'p={p_val:.3f}'
            
        # Random comparison (green)
        ax_bottom.text(x_pos, y_pos_p_random, format_p_value(random_p_vals[i]), 
                      ha='center', va='bottom', fontsize=8, color='green',
                      bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        # Regression comparison (red)
        ax_bottom.text(x_pos, y_pos_p_regression, format_p_value(regression_p_vals[i]), 
                      ha='center', va='bottom', fontsize=8, color='red',
                      bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # Add shared labels
    fig.text(0.5, 0.001, 'Prediction steps', ha='center', fontsize=12)
    fig.text(0.02, 0.5, r'$L_{combined}$', va='center', rotation='vertical', fontsize=12)
    
    # Adjust layout manually with reduced margins
    plt.subplots_adjust(left=0.1, right=0.99, bottom=0.1, top=0.99, hspace=0.001)
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()