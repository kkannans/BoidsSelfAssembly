import os
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

def get_all_weights(base_dir):
    all_weights = []
    for trial in os.listdir(base_dir):
        trial_path = os.path.join(base_dir, trial)
        trial_id = trial.split('_')[1][1:]
        if not os.path.isdir(trial_path):
            continue
        try:
            best_pwd, best_rdf, best_combined, population = load_best_genomes(trial_path)
            # Add all best genomes
            all_weights.append({
                'trial': trial_id,
                'type': 'best_pwd',
                'genome': best_pwd['genome']
            })
            all_weights.append({
                'trial': trial_id,
                'type': 'best_rdf',
                'genome': best_rdf['genome']
            })
            all_weights.append({
                'trial': trial_id,
                'type': 'best_combined',
                'genome': best_combined['genome']
            })
            print(f"Processed {trial}")
        except Exception as e:
            print(f"Skipping {trial}: {e}")
    return all_weights

# usage
if __name__ == "__main__":
    experiment_dir = "./evolution_e4_afpo_elite_new_pwd"
    all_weights = get_all_weights(experiment_dir)
    
    # save as csv with headers
    with open('all_weights.csv', 'w') as f:
        f.write("trial,type,w_s,w_a,w_c,view_radius\n")
        for weight_data in all_weights:
            genome = weight_data['genome']
            f.write(f"{weight_data['trial']},{weight_data['type']},{genome[0]},{genome[1]},{genome[2]},{genome[3]*config.VIEW_RADIUS_MAX}\n")