# BoidsSelfAssembly

**Paper:** “Evolving agents to generate and falsify hypotheses of biological self-assembly” \
**Authors:** Krishna Kannan Srinivasan^1, Nam Le^1, Joshua Bongard^1, Douglas Blackiston^2 \
**Affiliations:** 1. University of Vermont; 2. Tufts University \
**Contact:** kkannans@uvm.edu \
**Code:** https://github.com/kkannans/BoidsSelfAssembly \
**Data:** https://youtu.be/kZpef3o-z0Y (Credit: Douglas Blackiston, Tufts University)


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
- **Video**: [Biological self-assembly dataset](https://youtu.be/kZpef3o-z0Y) Credit: Douglas Blackiston, Tufts University.
- **Resolution**: 1 pixel = 1.6 μm 
- **Duration**: ~6 hours (30-second intervals)

## Pipeline Overview

The pipeline consists of four main stages:

1. **Video processing**: Extract cell positions using CellPose segmentation
2. **Metric computation**: Quantify self-assembly using global and local metrics
3. **Hypothesis evolution**: Evolve Boids parameters to match biological dynamics
4. **Hypothesis testing**: Evaluate evolved models on unseen data

## Usage

### 1. Extract metrics from video

Navigate to the `extract_metrics/` directory:

#### Cell segmentation
```bash
python3 segment_cells.py
```

#### Visualize segmentation results
```bash
python3 visualize_masks_over_video.py
```

#### Compute self-assembly metrics
```bash
# Global metric: average pairwise distances
python3 compute_pairwise_distances.py

# Local metric: radial distribution function
python3 get_self_assembly_metrics.py
```

![Alt text](./metrics_viz/invitro_rdf.gif)

### 2. Evolve hypotheses

Configure experiments in `config.py`, then run evolution:

#### Hypothesis space 1 (3 parameters: ws, wa, wc)
```bash
python3 evolve.py -s=1 -e=1 -a=afpo
```

#### Hypothesis space 2 (4 parameters: ws, wa, wc, rv)
```bash
python3 evolve.py -s=1 -e=4 -a=afpo_elite
```

### 3. Test evolved hypotheses

#### Test evolved models
```bash
python3 test_evolved_genome_afpo.py -exp=evolution_e1_afpo_pop50_gen50_new_pwd
python3 test_evolved_genome_afpo.py -exp=evolution_e4_afpo_pop50_gen50_new_pwd
```

#### Test random controls
```bash
python3 test_random_genome.py -exp=evolution_e1_afpo_pop50_gen50_new_pwd
python3 test_random_genome.py -exp=evolution_e4_afpo_pop50_gen50_new_pwd
```

#### Test regression models for prediction of pairwise distances and radial distribution function
```bash
python3 test_regression_pwd.py
python3 test_regression_rdf.py
```



### 4. Generate paper figures

place the test_performance* folders in the paper_figures/ folder before running the following scripts.

```bash
cd paper_figures/
python3 plot_test_result_exp1_paper.py  # Hypothesis Space 1 results
python3 plot_test_result_exp2_paper.py  # Hypothesis Space 2 results
```
## Visualization

### Interactive boids simulation
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

### Example evolved hypotheses

![Alt text](./evolved_boidh2vsIV.gif)

**Hypothesis 2** (sparse clusters):
- Parameters: wa=0.17, wc=0.03, ws=0.52, rv=26.78
- [Video demonstration](https://youtu.be/uxuxgarkpY8)

**Hypothesis 2** (compact structures):
- Parameters: wa=0.90, wc=0.55, ws=0.79, rv=24.77
- [Video demonstration](https://youtu.be/Rb2VR0SI-HE)

## Code structure

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

MIT License — see `LICENSE`.
