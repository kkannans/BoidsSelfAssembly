"""
Experiment4:

fig1: Test set performance

Total loss vs test prediction steps (0 - 20)
Control 1: Random hypothesis setting.
Control 2: Time series regression model

fig 2: Pareto front showing best pwd, rdf and best of both.

"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from scipy import stats
import re
from main_results_figures import plot_model_data_vs_regression_baseline, compute_test_statistics, compute_test_statistics_model_vs_random

# load test losses from test_performance folder corresponding to experiment 1 
test_performance_folder = "test_performance_evolution_e4_afpo_pop50_gen50_new_pwd"
experiment_folder_name = "../evolution_e4_afpo_pop50_gen50_new_pwd"
output_dir = "./ALIFE_2025"

def parse_folder_name(folder_name):
    match = re.match(r'evolution_s(\d+)_e(\d+)_(.+)', folder_name)
    if match:
        return int(match.group(2)), match.group(3), int(match.group(1))
    return None, None, None

def load_model_losses(folder_path):
    models = ["best_combined"]
    loss_names = ["pairwise_distance_losses", "rdf_losses"]
    model_losses = {}
    
    # Initialize model_losses structure
    for model in models:
        model_losses[model] = {}
        for loss_name in loss_names:
            model_losses[model][loss_name] = []
    
    # Search through all seed folders
    for folder in os.listdir(folder_path):
        if folder.startswith('evolution_s'):
            experiment, algorithm, seed = parse_folder_name(folder)
            if experiment is not None:  # Valid folder name
                seed_folder = os.path.join(folder_path, folder)
                for model in models:
                    for loss_name in loss_names:
                        path = os.path.join(seed_folder, f"{loss_name}_{model}.npy")
                        if os.path.exists(path):
                            losses_array = np.load(path)
                            model_losses[model][loss_name].append(losses_array)
    
    # Convert lists to numpy arrays but don't compute mean yet
    for model in models:
        for loss_name in loss_names:
            if model_losses[model][loss_name]:  # If we found any data
                model_losses[model][loss_name] = np.array(model_losses[model][loss_name])
                print(f"Loaded {loss_name} for {model}, shape: {model_losses[model][loss_name].shape}")
            else:
                raise ValueError(f"No data found for {model} {loss_name} in any seed folder")
    
    return model_losses

def load_regression_baseline_losses():
    regression_baseline_losses = {
        "linear_rdf": np.load('../regression_rdf_bw_longer/losses_linear_regression.npy'),
        "linear_pwd": np.load('../regression_models_bw_longer/linear_model_avg_pairwise_distances_losses.npy'),
        "quadratic_pwd": np.load('../regression_models_bw_longer/quadratic_model_avg_pairwise_distances_losses.npy'),
        "svr_pwd": np.load('../regression_models_bw_longer/svr_model_avg_pairwise_distances_losses.npy'),
    }
    return regression_baseline_losses


def load_random_genome_losses(experiment_folder_name):
    random_test_performance_folder = "./test_performance_random_genome_evolution_e4_afpo_elite_new_pwd"
    assert os.path.exists(random_test_performance_folder), f"Random test performance folder {random_test_performance_folder} does not exist"
    all_rdf_losses = []
    all_pairwise_distance_losses = []
    for trial_folder in os.listdir(random_test_performance_folder):
        # load losses 
        pairwise_distance_losses = np.load(os.path.join(random_test_performance_folder, trial_folder, 'pairwise_distance_losses_random_genome.npy'))
        rdf_losses = np.load(os.path.join(random_test_performance_folder, trial_folder, 'rdf_losses_random_genome.npy'))
        all_rdf_losses.append(rdf_losses)
        all_pairwise_distance_losses.append(pairwise_distance_losses)
    all_pairwise_distance_losses = np.array(all_pairwise_distance_losses)
    all_rdf_losses = np.array(all_rdf_losses)
    # shape should be (30, 30, 721)
    assert all_pairwise_distance_losses.shape == (30, 30, 721)
    assert all_rdf_losses.shape == (30, 30, 721)
    random_genome_losses = {
        "pairwise_distance_losses": all_pairwise_distance_losses,
        "rdf_losses": all_rdf_losses,
    }
    return random_genome_losses


def get_mean_and_std_of_random_genome_losses(random_genome_losses):
    pairwise_distance_losses = {
        "mean": [],
        "std": [],
    }
    rdf_losses = {
        "mean": [],
        "std": [],
    }
    for seed in range(random_genome_losses["pairwise_distance_losses"].shape[0]):
        pairwise_distance_losses["mean"].append(np.mean(random_genome_losses["pairwise_distance_losses"][seed], axis=0))
        assert pairwise_distance_losses["mean"][seed].shape == (721,)
        rdf_losses["mean"].append(np.mean(random_genome_losses["rdf_losses"][seed], axis=0))
        assert rdf_losses["mean"][seed].shape == (721,)
        pairwise_distance_losses["std"].append(np.std(random_genome_losses["pairwise_distance_losses"][seed], axis=0))
        assert pairwise_distance_losses["std"][seed].shape == (721,)
        rdf_losses["std"].append(np.std(random_genome_losses["rdf_losses"][seed], axis=0))
        assert rdf_losses["std"][seed].shape == (721,)

    return {
        "pairwise_distance_losses": pairwise_distance_losses,
        "rdf_losses": rdf_losses,
    }

def compute_total_loss_for_regression_losses(regression_losses):
    linear_rdf = regression_losses["linear_rdf"]
    linear_pwd = regression_losses["linear_pwd"]
    print(f"Regression shapes - RDF: {linear_rdf.shape}, PWD: {linear_pwd.shape}")
    total_losses = linear_pwd + linear_rdf
    return total_losses

def compute_total_loss_for_random_genome_losses(random_genome_losses):
    pairwise_distance_losses = np.array(random_genome_losses["pairwise_distance_losses"]["mean"])
    rdf_losses = np.array(random_genome_losses["rdf_losses"]["mean"])

    # compute mean across seeds for each loss
    pairwise_distance_losses = np.mean(pairwise_distance_losses, axis=0)
    rdf_losses = np.mean(rdf_losses, axis=0)
    
    # compute std across seeds for each loss
    pairwise_distance_losses_std = np.std(np.array(random_genome_losses["pairwise_distance_losses"]["mean"]), axis=0)
    rdf_losses_std = np.std(np.array(random_genome_losses["rdf_losses"]["mean"]), axis=0)
    
    # for each seed, compute the total loss
    total_losses = pairwise_distance_losses + rdf_losses
    total_losses_std = pairwise_distance_losses_std + rdf_losses_std
    
    return total_losses, total_losses_std

def get_plot_data(model_data, regression_losses, random_genome_losses, test_horizon, model_type):
    # Plot total loss (computed from individual components)
    pwd_model = model_data[model_type]["pairwise_distance_losses"]  # shape: (num_seeds, sequence_length)
    rdf_model = model_data[model_type]["rdf_losses"]  # shape: (num_seeds, sequence_length)
    print(f"Model shapes - PWD: {pwd_model.shape}, RDF: {rdf_model.shape}")
    
    # Compute total loss without normalization
    total_model = pwd_model + rdf_model
    
    # Only take test period data
    total_model = total_model[:, config.END_FRAME:config.END_FRAME + test_horizon]
    print(f"Total model shape after slicing: {total_model.shape}")
    
    # Compute mean and std across seeds
    model_mean = np.mean(total_model, axis=0)
    model_std = np.std(total_model, axis=0)
    
    total_regression = compute_total_loss_for_regression_losses(regression_losses)
    total_regression = total_regression[config.END_FRAME:config.END_FRAME + test_horizon]
    
    # Compute total loss for random genome
    total_random_genome, total_random_genome_std = compute_total_loss_for_random_genome_losses(random_genome_losses)
    total_random_genome = total_random_genome[config.END_FRAME:config.END_FRAME + test_horizon]
    total_random_genome_std = total_random_genome_std[config.END_FRAME:config.END_FRAME + test_horizon]
    
    if len(model_mean) == 0 or len(total_regression) == 0 or len(total_random_genome) == 0:
        raise ValueError("Empty arrays after slicing. Check END_FRAME and test_horizon values.")
    
    # Compute statistics comparing model to regression
    regression_test_statistics = compute_test_statistics(total_model, total_regression, test_horizon)
    
    # Compute statistics comparing model to random genome
    # Convert random genome data to match model data shape
    random_genome_data = np.array(random_genome_losses["pairwise_distance_losses"]["mean"]) + np.array(random_genome_losses["rdf_losses"]["mean"])
    random_genome_data = random_genome_data[:, config.END_FRAME:config.END_FRAME + test_horizon]
    random_test_statistics = compute_test_statistics_model_vs_random(total_model, random_genome_data, test_horizon)
    
    return model_mean, model_std, total_regression, total_random_genome, total_random_genome_std, regression_test_statistics, random_test_statistics

def main():
    # Load model losses
    model_data = load_model_losses(test_performance_folder)
    
    # Load regression baseline losses
    regression_losses = load_regression_baseline_losses()
    
    # Load random genome losses
    random_genome_losses = load_random_genome_losses(experiment_folder_name)
    
    # Get mean and std of random genome losses across 30 trials for each seed 
    random_genome_losses = get_mean_and_std_of_random_genome_losses(random_genome_losses)
    
    # only plot best_combined
    model_mean, model_std, total_regression, total_random_genome, total_random_genome_std, regression_test_statistics, random_test_statistics = get_plot_data(
        model_data, regression_losses, random_genome_losses, 20, "best_combined")
    
    # Print the accumulated statistics results for regression comparison
    if 'accumulation_results' in regression_test_statistics:
        accum = regression_test_statistics['accumulation_results']
        print("\nAccumulated Statistical Test Results (vs Regression):")
        print("=====================================")
        for i, t in enumerate(accum['time_points']):
            print(f"Time horizon t={t}:")
            print(f"  p-value: {accum['p_vals'][i]:.6f}")
            print(f"  t-statistic: {accum['t_stats'][i]:.6f}")
            print(f"  Effect size (d): {accum['effect_size'][i]:.6f}")
            print(f"  Mean difference: {accum['difference'][i]:.6f}")
            sig_level = "***" if accum['p_vals'][i] < 0.001 else "**" if accum['p_vals'][i] < 0.01 else "*" if accum['p_vals'][i] < 0.05 else "ns"
            print(f"  Significance: {sig_level}")
            print()
    
    # Print the accumulated statistics results for random genome comparison
    if 'accumulation_results' in random_test_statistics:
        accum = random_test_statistics['accumulation_results']
        print("\nAccumulated Statistical Test Results (vs Random Genome):")
        print("=====================================")
        for i, t in enumerate(accum['time_points']):
            print(f"Time horizon t={t}:")
            print(f"  p-value: {accum['p_vals'][i]:.6f}")
            print(f"  t-statistic: {accum['t_stats'][i]:.6f}")
            print(f"  Effect size (d): {accum['effect_size'][i]:.6f}")
            print(f"  Mean difference: {accum['difference'][i]:.6f}")
            sig_level = "***" if accum['p_vals'][i] < 0.001 else "**" if accum['p_vals'][i] < 0.01 else "*" if accum['p_vals'][i] < 0.05 else "ns"
            print(f"  Significance: {sig_level}")
            print()
    
    save_path = os.path.join(output_dir, "figure3.png")
    plot_model_data_vs_regression_baseline(model_mean, model_std, total_regression, total_random_genome, 
                                         total_random_genome_std, regression_test_statistics, random_test_statistics, save_path, break_restart_y=250)
if __name__ == "__main__":
    main()

