import re
from pysat.formula import CNF, WCNF
from pysat.examples.rc2 import RC2
import torch
from typing import List, Optional, Tuple
import numpy as np

# Handles syntax issues with some CNF files from SATLIB. 
def load_cnf_file(filepath: str):
    with open(filepath, 'r') as f:
        lines = []
        for ln in f:
            ln = ln.strip()
            if ln.startswith('%'):
                break
            if ln.startswith(('c', 'p')) or re.match(r'^((-?\d+)\s+)*0$', ln):
                lines.append(ln)
    
    cnf = CNF(from_string='\n'.join(lines) + '\n')
    return cnf

# Used to verify optimum for unsatisfiable instances. Uses RC2 (exact MaxSAT solver). 
def get_optimum(cnf_file_path):
    cnf = load_cnf_file(cnf_file_path)
    wcnf = WCNF()
    wcnf.nv = cnf.nv
    for cl in cnf.clauses:
        wcnf.append(cl, weight=1)

    # Solve and retrieve cost
    with RC2(wcnf) as rc2:
        rc2.compute()
        cost = rc2.cost          # always an int after compute()
    total_clauses = len(wcnf.soft)
    max_satisfied = total_clauses - cost

    print(f"File: {cnf_file_path}")
    print(f"Total Clauses: {total_clauses}")
    print(f"Verified Min Violations (Global Optimum): {cost}")
    print(f"Verified Max Satisfied Clauses: {max_satisfied}")
    return max_satisfied

# Naieve assignment recovery from dNN. 
def count_satisfied(cnf: CNF, theta_var: torch.Tensor) -> int:
    """
    Count satisfied clauses given a 1‑D theta_var assignment (CPU tensor).
    Literal l > 0: satisfied if theta[|l|-1] > 0.5.
    Literal l < 0: satisfied if theta[|l|-1] <= 0.5.
    """
    tv = theta_var.detach().cpu().float()
    satisfied = 0
    for clause in cnf.clauses:
        if any(
            (lit > 0 and tv[lit - 1] > 0.5) or (lit < 0 and tv[-lit - 1] <= 0.5)
            for lit in clause
        ):
            satisfied += 1
    return satisfied


def plot_convergence(histories: List[List[float]], name: str, out_path: str) -> None:
    """Save convergence plot (requires matplotlib)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        for i, h in enumerate(histories):
            ax.plot(h, linewidth=1.0, alpha=0.7, label=f'Init {i}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Objective')
        ax.set_title(f'MaxSAT dNN - {name}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    except ImportError:
        pass  # matplotlib not installed – skip plotting


# ------------------------------------------------------------
# Fast clause counter for a Boolean numpy array
# ------------------------------------------------------------
def _count_satisfied_from_bool(assign: np.ndarray, cnf) -> int:
    """assign: bool array of length n_vars, True/1 = variable True"""
    sat = 0
    for clause in cnf.clauses:
        for lit in clause:
            idx = abs(lit) - 1
            if (lit > 0 and assign[idx]) or (lit < 0 and not assign[idx]):
                sat += 1
                break
    return sat

# ------------------------------------------------------------
# Greedy 1‑flip hill‑climb
# ------------------------------------------------------------
def greedy_hill_climb(assign: np.ndarray, cnf) -> Tuple[np.ndarray, int]:
    """
    Perform 1‑flip local search until no single flip increases satisfied clauses.
    Returns (improved assignment, satisfaction count).
    """
    assign = assign.copy()
    n = len(assign)
    sat = _count_satisfied_from_bool(assign, cnf)
    improved = True
    while improved:
        improved = False
        for i in range(n):
            assign[i] = not assign[i]
            new_sat = _count_satisfied_from_bool(assign, cnf)
            if new_sat > sat:
                sat = new_sat
                improved = True
            else:
                assign[i] = not assign[i]   # revert
    return assign, sat

# ------------------------------------------------------------
# Main replacement for count_satisfied
# ------------------------------------------------------------
def count_satisfied_batch(
    cnf,
    theta_var: torch.Tensor,     # (n_vars,) or (n_vars, B)
    n_samples_per_restart: int = 200,
    return_assignment: bool = False,
) -> int:
    """
    Best‑of‑(B × K) randomised rounding + greedy 1‑flip on the winner.

    Args:
        cnf: pysat CNF object.
        theta_var: continuous parameters θ_i ∈ [0,1].
                   If 1D, treated as a batch of size 1.
        n_samples_per_restart: K – number of randomised roundings per restart column.
        return_assignment: if True, also returns the final Boolean assignment (numpy array).

    Returns:
        satisfied (int) : number of satisfied clauses after greedy hill‑climb.
        (if return_assignment) also best_assign (np.ndarray).
    """
    tv = theta_var.detach().cpu().float()
    if tv.ndim == 1:
        tv = tv.unsqueeze(1)               # (n_vars, 1)
    n_vars, B = tv.shape

    tv_np = tv.numpy()                     # (n_vars, B)

    best_sat = -1
    best_assign = None

    # 1. Best‑of‑(B × K) randomised rounding
    for b in range(B):
        probs = tv_np[:, b]                # shape (n_vars,)
        # Generate K assignments for this restart
        # Sampling a (n_vars, K) matrix of random uniforms and threshold
        r = np.random.rand(n_vars, n_samples_per_restart)  # (n_vars, K)
        assignments = (probs[:, None] > r)                  # (n_vars, K) bool
        # Evaluate each assignment
        for k in range(n_samples_per_restart):
            assign = assignments[:, k]
            sat = _count_satisfied_from_bool(assign, cnf)
            if sat > best_sat:
                best_sat = sat
                best_assign = assign.copy()

    # 2. Greedy hill‑climb on the overall best assignment
    best_assign, final_sat = greedy_hill_climb(best_assign, cnf)

    if return_assignment:
        return final_sat, best_assign
    return final_sat

def refine_with_bruteforce(
    assign: np.ndarray,          # Boolean assignment from count_satisfied_batch
    theta_vec: torch.Tensor,     # best_theta, shape (n_vars,) – network's belief
    cnf,                         # PySAT CNF
    confidence_threshold: float = 0.15,
    max_free_vars: int = 18,
) -> Tuple[np.ndarray, int]:
    """
    Fix variables where network is very confident (|θ - 0.5| ≥ threshold),
    then exhaustively brute‑force the remaining (uncertain) ones.

    Returns (improved_assignment, new_satisfied_count).
    If the number of uncertain variables exceeds max_free_vars, falls back
    to the original assignment with a warning.
    """
    tv = theta_vec.detach().cpu().float().numpy()
    n = len(tv)
    uncertain = np.abs(tv - 0.5) < confidence_threshold
    n_uncertain = np.sum(uncertain)
    #print(f"Uncertain variables: {n_uncertain}")
    if n_uncertain == 0:
        # already fully confident – nothing to brute‑force
        return assign.copy(), _count_satisfied_from_bool(assign, cnf)

    if n_uncertain > max_free_vars:
        import warnings
        warnings.warn(
            f"Too many uncertain variables ({n_uncertain} > {max_free_vars}). "
            "Falling back to original assignment."
        )
        return assign.copy(), _count_satisfied_from_bool(assign, cnf)

    # Base assignment: keep the greedy/climbed values for the confident variables
    base = assign.copy()
    # Indices of the uncertain variables
    free_idx = np.where(uncertain)[0]

    best_sat = -1
    best_full = None

    # Enumerate all 2^{n_uncertain} combinations
    for bits in range(1 << n_uncertain):
        trial = base.copy()
        # Decode bits into True/False for each free variable
        for i, idx in enumerate(free_idx):
            trial[idx] = bool((bits >> i) & 1)
        sat = _count_satisfied_from_bool(trial, cnf)
        if sat > best_sat:
            best_sat = sat
            best_full = trial.copy()

    # Optionally, run a greedy hill‑climb on the exhaustive result
    # (it should already be locally optimal, but cheap insurance)
    best_full, best_sat = greedy_hill_climb(best_full, cnf)
    return best_full, best_sat
### Hyper parameter optimization
