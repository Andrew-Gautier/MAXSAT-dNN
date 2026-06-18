#!/usr/bin/env python3
"""
Minimal MaxSAT dNN evaluator.

"""
import argparse
import csv
import json
import time
from pathlib import Path
from typing import List, Optional
import torch
from pysat.formula import CNF 
from Maxsat_dNN import MaxsatdNN
from helpers import load_cnf_file, count_satisfied_batch, plot_convergence, refine_with_bruteforce, count_satisfied


# Main training loop for MaxSAT dNN. Returns a dict of results and training history.
def run_maxsat_dNN(
    cnf: CNF,
    epochs: int,
    lr: float,
    batch_size: int,
    grad_clip: float,
    eta_min: float,
    patience: int,
    weight_decay: float,
    random_noise: float,
    warmup_epochs: int, 
    restart_period: int,
    device: torch.device,
    dtype: torch.dtype,
    restart_samples: int,
    confidence_threshold: float,
    max_free_vars: int,
) -> dict:

    solver = MaxsatdNN(cnf, batch_size=batch_size, random_noise=random_noise).to(device=device, dtype=dtype)

    # Freeze clause weights to 1.0 (unweighted MaxSAT)
    with torch.no_grad():
        solver.theta_clause.weight.fill_(1.0)

    optimizer = torch.optim.AdamW([solver.theta_var.weight], lr=lr, weight_decay=weight_decay)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = restart_period, T_mult=1, eta_min=eta_min)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])


    histories: List[List[float]] = [[] for _ in range(batch_size)]
    best_obj = float('inf')
    best_theta: Optional[torch.Tensor] = None
    no_improve = 0

    t_start = time.perf_counter()

    for epoch in range(epochs):
        optimizer.zero_grad()
        obj = solver()                     # (batch_size,) if batch_size>1 else scalar
        if batch_size > 1:
            obj.sum().backward()
        else:
            obj.backward()

        torch.nn.utils.clip_grad_norm_([solver.theta_var.weight], grad_clip)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            solver.theta_var.weight.data.clamp_(0.0, 1.0)

            if batch_size > 1:
                obj_vals = obj.detach().cpu().tolist()
                best_col = int(obj.argmin().item())
                col_min = obj_vals[best_col]
                for c in range(batch_size):
                    histories[c].append(obj_vals[c])
            else:
                col_min = obj.item()
                histories[0].append(col_min)

            if col_min < best_obj - 1e-6:
                best_obj = col_min
                no_improve = 0
                if batch_size > 1:
                    best_theta = solver.theta_var.weight[:, best_col].clone().cpu()
                else:
                    best_theta = solver.theta_var.weight.clone().cpu()
            else:
                no_improve += 1

            if patience > 0 and no_improve >= patience:
                break

    train_time = time.perf_counter() - t_start

    if best_theta is None:
        best_theta = (solver.theta_var.weight[:, 0] if batch_size > 1
                    else solver.theta_var.weight).clone().cpu()
        
    print(f"Satisfaction prior to repeated randomised rounding: {count_satisfied(cnf, best_theta)} / {len(cnf.clauses)}")
    # ------------------------------------------------------------
    # Step 1: best‑of‑(B × K) randomised rounding + greedy climb
    # ------------------------------------------------------------
    counting_start = time.perf_counter()
    satisfied_batch, best_bool = count_satisfied_batch(
        cnf, solver.theta_var.weight,
        n_samples_per_restart=restart_samples,
        return_assignment=True
    )
    counting_time = time.perf_counter() - counting_start
    print(f"Satisfaction after randomised rounding + greedy climb: {satisfied_batch} / {len(cnf.clauses)}")
    # ------------------------------------------------------------
    # Step 2: post‑hoc freeze + brute‑force on uncertain variables (if enabled)
    # ------------------------------------------------------------
    brute_start = time.perf_counter()
    best_bool_refined, satisfied_refined = refine_with_bruteforce(
        best_bool, best_theta, cnf,
        confidence_threshold=confidence_threshold,
        max_free_vars=max_free_vars
    )
    brute_time = time.perf_counter() - brute_start
    brute_time = 0.0
    satisfied_refined = satisfied_batch
    best_bool_refined = best_bool

    # Choose the final satisfaction count (refinement can never decrease it)
    final_satisfied = satisfied_refined
    final_assignment = best_bool_refined   # numpy bool array

    # Total decoding time = randomised search + brute‑force refinement
    total_decoding_time = counting_time + brute_time

    # ------------------------------------------------------------
    # Return dictionary
    # ------------------------------------------------------------
    
    final_satisfied = int(satisfied_refined)
    total_clauses = int(len(cnf.clauses))
    sat_rate = float(final_satisfied / total_clauses) if total_clauses > 0 else 0.0
    # counting time and decoding time will be set to 0 if those steps are disabled, so total_time will just be train_time in that case
    return {
        'best_objective': best_obj,          # training loss minimum
        'num_satisfied': final_satisfied,
        'total_clauses': total_clauses,
        'satisfaction_rate': sat_rate,
        'fully_satisfied': final_satisfied == total_clauses,
        'total_epochs': sum(len(h) for h in histories) // max(batch_size, 1),
        'histories': histories,
        'train_time_s': train_time,
        'counting_time_s': counting_time,       # randomised rounding + 1st greedy
        'brute_force_time_s': brute_time,       # freeze + exhaustive search
        'total_decoding_time_s': total_decoding_time,
        'total_time_s': train_time + total_decoding_time,
    }

# ----------------------------------------------------------------------
# Process a single CNF file
# ----------------------------------------------------------------------

def maxsat_dNN_single_cnf(
    cnf_path: str,
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """Load one CNF, run MaxSAT, save per‑instance outputs, return summary row."""
    name = Path(cnf_path).stem
    cnf = load_cnf_file(cnf_path)

    # Hyperparameters used in this run
    params = {
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_clip": args.grad_clip,
        "eta_min": args.eta_min,
        "patience": args.patience,
        "weight_decay": args.weight_decay,
        "random_noise": args.random_noise,
        "warmup_epochs": args.warmup_epochs,
        "restart_period": args.restart_period,
        "restart_samples": args.restart_samples,
        "device": str(device),
        "dtype": str(dtype),
    }

    result = run_maxsat_dNN(
        cnf=cnf,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_clip=args.grad_clip,
        eta_min=args.eta_min,
        patience=args.patience,
        weight_decay=args.weight_decay,
        random_noise=args.random_noise,
        warmup_epochs=args.warmup_epochs,
        restart_period=args.restart_period,
        device=device,
        dtype=dtype,
        restart_samples=args.restart_samples,
        confidence_threshold=args.confidence_threshold,
        max_free_vars=args.max_free_vars,
    )

    # Save per‑instance data
    inst_dir = output_dir / name
    inst_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    with open(inst_dir / "config.json", "w") as f:
        json.dump(params, f, indent=2)

    # Convergence plot (assumes plot_convergence exists)
    plot_convergence(result["histories"], name, str(inst_dir / "convergence.png"))

    # CSV with all epoch data
    with open(inst_dir / "epoch_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["init", "epoch", "objective"])
        for init_idx, hist in enumerate(result["histories"]):
            for ep, val in enumerate(hist):
                writer.writerow([init_idx, ep, f"{val:.8f}"])

    return {
        "instance": name,
        "num_vars": cnf.nv,
        "num_clauses": len(cnf.clauses),
        "best_objective": result["best_objective"],
        "num_satisfied": result["num_satisfied"],
        "satisfaction_rate": round(result["satisfaction_rate"], 6),
        "fully_satisfied": result["fully_satisfied"],
        "total_epochs": result["total_epochs"],
        "params": params,                # <-- list/dict of all configuration parameters
        "train_time_s": round(result["train_time_s"], 3),
        "counting_time_s": round(result["counting_time_s"], 3),
        "brute_force_time_s": round(result["brute_force_time_s"], 3),
        "total_decoding_time_s": round(result["total_decoding_time_s"], 3),
        "total_time_s": round(result["total_time_s"], 3),
    }
