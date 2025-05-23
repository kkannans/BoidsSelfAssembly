# BoidsSelfAssembly: Evolving Agents to Generate and Falsify Hypotheses of Biological Self-Assembly

[![Data](https://img.shields.io/badge/Data-Available-blue)](https://youtu.be/kZpef3o-z0Y)

## Overview

This repository contains code that implements the pipeline for **"Evolving agents to generate and falsify hypotheses of biological self-assembly"**. We introduce a novel approach that uses evolutionary algorithms to evolve Boids (agent-based models) to match biological self-assembly dynamics observed in video data, generating falsifiable hypotheses that can reveal potential mechanisms of biological self-assembly.

## Quick Start

### Environment Setup
```bash
chmod +x install_env.sh
./install_env.sh
source ./BoidsEnv/bin/activate
```

### Dataset
The biological self-assembly video dataset shows *Xenopus* cells forming multicellular assemblies over 721 frames:
- **Video**: [Biological Self-Assembly Dataset](https://youtu.be/kZpef3o-z0Y)
- **Resolution**: 1 pixel = 1.6 μm 
- **Duration**: ~6 hours (30-second intervals)

## Pipeline Overview

The pipeline consists of four main stages:

1. **Video Processing**: Extract cell positions using CellPose segmentation
2. **Metric Computation**: Quantify self-assembly using global and local metrics
3. **Hypothesis Evolution**: Evolve Boids parameters to match biological dynamics
4. **Hypothesis Testing**: Evaluate evolved models on unseen data

## Usage

### 1. Extract Metrics from Video

Navigate to the `extract_metrics/` directory:

#### Cell Segmentation
```bash
python3 segment_cells.py
```

#### Visualize Segmentation Results
```bash
python3 visualize_masks_over_video.py
```

#### Compute Self-Assembly Metrics
```bash
# Global metric: average pairwise distances
python3 compute_pairwise_distances.py

# Local metric: radial distribution function
python3 get_self_assembly_metrics.py
```

![Alt text](./metrics_viz/invitro_rdf.gif)

### 2. Evolve Hypotheses

Configure experiments in `config.py`, then run evolution:

#### Hypothesis Space 1 (3 parameters: ws, wa, wc)
```bash
python3 evolve.py -s=1 -e=1 -a=afpo
```

#### Hypothesis Space 2 (4 parameters: ws, wa, wc, rv)
```bash
python3 evolve.py -s=1 -e=4 -a=afpo_elite
```

### 3. Test Evolved Hypotheses

#### Test Evolved Models
```bash
python3 test_evolved_genome_afpo.py -exp=evolution_e1_afpo_pop50_gen50_new_pwd
python3 test_evolved_genome_afpo.py -exp=evolution_e4_afpo_pop50_gen50_new_pwd
```

#### Test Random Controls
```bash
python3 test_random_genome.py -exp=evolution_e1_afpo_pop50_gen50_new_pwd
python3 test_random_genome.py -exp=evolution_e4_afpo_pop50_gen50_new_pwd
```

#### Test Regression Models for prediction of pairwise distances and radial distribution function
```bash
python3 test_regression_pwd.py
python3 test_regression_rdf.py
```



### 4. Generate Paper Figures

place the test_performance* folders in the paper_figures/ folder before running the following scripts.

```bash
cd paper_figures/
python3 plot_test_result_exp1_paper.py  # Hypothesis Space 1 results
python3 plot_test_result_exp2_paper.py  # Hypothesis Space 2 results
```
## Visualization

### Interactive Boids Simulation
```bash
# Extract evolved parameters
python3 get_all_weights_from_experiment4.py

# Run interactive simulation
python3 simulate_exp4_replay.py
```
This initializes the simulation with initial positions from the video and uses initial velocities from the evolved config. The simulation is run in a pygame window.

**Controls:**
- `s`: Start simulation
- `e`: End simulation  
- `r`: Reset simulation
- Dropdown: Select different evolved genomes labelled by view radius

### Example Evolved Hypotheses

![Alt text](./evolved_boidh2vsIV.gif)

**Hypothesis 2** (sparse clusters):
- Parameters: wa=0.17, wc=0.03, ws=0.52, rv=26.78
- [Video demonstration](https://youtu.be/uxuxgarkpY8)

**Hypothesis 2** (compact structures):
- Parameters: wa=0.90, wc=0.55, ws=0.79, rv=24.77
- [Video demonstration](https://youtu.be/Rb2VR0SI-HE)

## Code Structure

```
├── extract_metrics/          # Video processing and metric computation
├── paper_figures/           # Figure generation scripts
├── config.py               # Experiment configuration
├── evolve.py              # Main evolutionary optimization script
├── experiment1.py         # Hypothesis Space 1 (3D)
├── experiment4.py         # Hypothesis Space 2 (4D)
├── afpo.py               # Age-Fitness Pareto Optimization
├── afpo_elitism.py      # Age-Fitness Pareto Optimization with Elitism
├── boids.py              # Boids implementation
├── simulation.py         # Simulation engine
├── fitness_function.py   # Evolution fitness functions
├── loss_functions.py     # Testing loss functions
├── regressions.py        # Baseline regression models
└── utils.py              # Utility functions
```

## Requirements

- Python 3.8+
- CellPose for cell segmentation
- NumPy, SciPy for numerical computation
- Pygame for visualization
- See `install_env.sh` for complete dependencies

## License

[GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)

## Contact

For questions about the implementation or paper, please contact xxx (change after review process)
