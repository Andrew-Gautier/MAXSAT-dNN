import torch
import torch.nn as nn
from pysat.formula import CNF

def initialize_weights(method: str, cnf: CNF, num_variables: int) -> nn.Parameter:
    if method == "literal_frequency":  
        all_lits = torch.tensor([abs(lit) for clause in cnf.clauses for lit in clause], dtype=torch.long)
        var_counts = torch.bincount(all_lits - 1, minlength=num_variables).float()
        max_count = var_counts.max()
        theta_weight = 1 - var_counts / max_count + torch.rand(num_variables) / 10
        return nn.Parameter(theta_weight)
    elif method == "random_uniform":
        return nn.Parameter(torch.rand(num_variables, dtype=torch.float16))
    elif method == "jw_heuristic":
        # Jeroslow-Wang initialization
        weights = torch.zeros(num_variables)
        for clause in cnf.clauses:
            for lit in clause:
                var_idx = abs(lit) - 1
                weights[var_idx] += 2 ** -len(clause)
        return nn.Parameter(weights / weights.max())
    elif method == "clause_degree":
        # Weight by number of distinct clauses
        var_degrees = torch.zeros(num_variables)
        for clause in cnf.clauses:
            unique_vars = set(abs(lit) for lit in clause)
            for var_id in unique_vars:
                var_idx = var_id - 1
                var_degrees[var_idx] += 1
        max_degree = var_degrees.max()
        theta_weight = 1 - var_degrees / max_degree + torch.rand(num_variables) / 10
        return nn.Parameter(theta_weight)
    elif method == "polarity_aware":
        # Reward variables with balanced positive/negative occurrences
        pos_count = torch.zeros(num_variables)
        neg_count = torch.zeros(num_variables)
        
        for clause in cnf.clauses:
            for lit in clause:
                var_idx = abs(lit) - 1
                if lit > 0:
                    pos_count[var_idx] += 1
                else:
                    neg_count[var_idx] += 1
        
        # Balance metric: 1 - |pos-neg|/(pos+neg)
        total = pos_count + neg_count
        balance = 1 - torch.abs(pos_count - neg_count) / (total + 1e-8)  # Avoid division by zero
        theta_weight = balance + torch.rand(num_variables) / 10
        return nn.Parameter(theta_weight)