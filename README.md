# MAX-SAT dNN: A Differentiable Neural Network for Maximum Satisfiability

This repository implements a **novel neural network architecture** for solving the **Maximum Satisfiability (MAX-SAT)** problem. The network is built directly from a CNF formula and minimises an objective whose global optimum (zero) corresponds to a fully satisfying assignment.

The project includes:
- A PyTorch model (`Maxsat_dNN.py`) with sparse hidden layers derived from the CNF.
- A training/evaluation pipeline (`eval.py`, `helpers.py`) supporting batch restarts, early stopping, and a two‑stage decoding strategy (randomised rounding + greedy hill‑climb + optional brute‑force refinement).
- Extensive experiments on the **UF100‑430** benchmark (1000 random 3‑SAT instances) covering:
  - Hyperparameter optimisation with **Optuna**
  - Ablation study of **weight initialisation** strategies
  - Comparison of **learning rate schedulers**

---

## 📁 Repository Structure
- Maxsat_dNN.py # Core model definition
- eval.py # Main training/evaluation loop
- helpers.py # CNF loading, decoding, and refinement utilities
- weight_study.py # Initialisation methods for θ_var
- requirements.txt # Python dependencies
- notebooks/ # Jupyter notebooks for experiments
    - hyperparameter_tuning.ipynb
    - initilization_ablation.ipynb
    - Scheduler_experiments.ipynb
- testing/ test outputs from notebook runs and model development
- results/  aggregated results, figures, and best parameters

## Model Architecture (Brief)
The network defines a continuous objective:
text

f(θ) = α * Σ_i ReLU(1 - |2θ_i - 1|) + β * Σ_j ReLU(θ_{cj} - Σ_{lit} ...) - γ * Σ_j ReLU(θ_{cj} - 0.5)

    Block 1 encourages integrality (θ_i ∈ {0,1})

    Block 2 penalises unsatisfied clauses

    Block 3 rewards satisfied clauses

All hidden weights are sparse and constructed directly from the clause‑variable incidence, making the network size proportional to the formula size.


## 🚀 Installation

Clone the repository and install the required packages:

```bash
git clone <repo-url>
cd MaxSAT-dNN
pip install -r requirements.txt
```

## 📜 License

This project is released under the MIT License. See the LICENSE file for details.
## 🤝 Acknowledgments

    - PySAT for CNF parsing and exact MaxSAT solving.
    - Optuna for hyperparameter optimisation.
    - The SATLIB benchmark suite (UF100‑430) for experimental validation.
    - VLSAT benchmark suite for scalability testing. 
    - Gurobi Optimizer version 12.0.3
    - CP-SAT OR Tools version 9.15.6755
