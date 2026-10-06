# PCBSP
This repository contains the simulation source code for the paper: **"Multidimensional Feature Modeling and Dynamic Priority Scheduling Optimization for GPU Jobs"**

## 📋 Requirements
- Python 3.8+
- No third-party libraries required (Only standard libraries: `math`, `random`, `json`, etc.)


## 🚀 Quick Start
1. Clone the repository:
```bash
git clone https://github.com/seaddyzhang/PCBSP.git
cd PCBSP
```

2. Generate the synthetic workload and cluster environment:
```bash
python data_generator.py
```

3. Run the simulation experiments (FCFS, Static, PCBP, and Ablation studies):
```bash
python run_experiment.py
```


## 📂 Project Structure
- `config.py`: Configuration parameters (cluster size, task properties, priority weights).
- `data_generator.py`: Generates synthetic workloads (Poisson arrival, lognormal duration) and saves them to JSON files.
- `simulator.py`: Core discrete event simulation engine and the implementation of the three scheduling strategies (FCFS, Static Priority, PCBP).
- `run_experiment.py`: Main script to run baseline comparisons and ablation studies.

## 📊 Results
Running the code will output a summary table of the simulation results comparing FCFS, Static Priority, and the proposed PCBP method, aligning with the data presented in the paper.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.