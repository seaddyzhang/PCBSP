GPU-PCBSP-Scheduling
This repository contains the simulation source code for the paper: "Multidimensional Feature Modeling and Dynamic Priority Scheduling Optimization for GPU Jobs"

📋 Requirements
Python 3.8+
No third-party libraries required (Only standard libraries: math, random, json, etc.)
🚀 Quick Start
Clone the repository:
git clone https://github.com/your-username/GPU-PCBSP-Scheduling.gitcd GPU-PCBSP-Scheduling
Generate the synthetic workload and cluster environment:
bash

python data_generator.py
Run the simulation experiments (FCFS, Static, PCBP, and Ablation studies):
bash

python run_experiment.py
📂 Project Structure
config.py: Configuration parameters (cluster size, task properties, weights).
data_generator.py: Generates synthetic workloads (Poisson arrival, lognormal duration) and saves to JSON.
simulator.py: Core discrete event simulation engine and three scheduling strategies.
run_experiment.py: Main script to run baseline and ablation experiments.
📊 Results
Running the code will output the simulation results table comparing FCFS, Static Priority, and the proposed PCBP method, aligning with the data presented in the paper.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.