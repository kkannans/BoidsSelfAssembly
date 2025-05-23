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

# Create a simple script to run the simulation
echo "echo 'Running boid simulation...'"
echo "python boid_simulation.py"

# Print success message
echo "Virtual environment has been set up successfully!"
echo "To activate the environment:"
echo "  On Windows: venv\\Scripts\\activate"
echo "  On macOS/Linux: source venv/bin/activate"
echo ""
echo "After activation, run your boid simulation with: python boid_simulation.py"