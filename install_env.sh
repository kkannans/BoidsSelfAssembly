#!/bin/bash
# Create and set up virtual environment for boid simulation
# Create a virtual environment
python3.11 -m venv BoidsEnv

# Activate the virtual environment
# On Windows:
# venv\Scripts\activate
# On macOS/Linux:
source BoidsEnv/bin/activate

# Install required packages
pip install -r requirements.txt

