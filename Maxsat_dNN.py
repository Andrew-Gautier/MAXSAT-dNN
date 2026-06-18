"""
MaxSAT dNN 
======================================
Encodes the Maximum Satisfiability problem as a continuous minimization
over θ_var ∈ [0,1]^n (and optionally θ_clause ∈ [0,1]^m for weighted variants).
All hidden-layer and output-layer parameters are fixed by construction from the
CNF structure; only theta_var are optimized. For unweighted MAX-SAT.
To support weighted MAX-SAT, the caller can set theta_clause to the desired weights.

Architecture:
  Input: [θ_var | θ_clause] ∈ R^(n+m)
  Hidden layer: (n + 2m) × (n+m) sparse CSR, ~(n + 2m + Σ|clause|) non-zeros
    Block 1 (n rows):   integrality — ReLU(1 - |2θ_i - 1|)
    Block 2 (m rows):   clause satisfaction — ReLU(θ_cj - Σliterals)
    Block 3 (m rows):   clause counting     — ReLU(θ_cj - 0.5)
  Output: dense 1D weight vector (n+2m,); result is a scalar (or B-vector in batch mode).
  Optimal value f* = 0 iff all clauses are satisfied.

Hidden and output weights are stored as register_buffer. cuSPARSE requires
same-dtype operands, so autocast(enabled=False) guards the SpMM call.

dtype note:
  bfloat16 / float16 are only supported on CUDA devices.
  CPU requires float32; passing a non-float32 dtype with a CPU device raises ValueError.
"""

import warnings
import torch
import torch.nn as nn
from torch import Tensor
from pysat.formula import CNF

warnings.filterwarnings('ignore', message='Sparse CSR tensor support is in beta state')
warnings.filterwarnings('ignore', message='Sparse invariant checks')


class ConstrainedElemMultiply(nn.Module):
    """Learnable parameter vector clamped to [lower_bound, upper_bound] on every forward pass."""
    def __init__(self, size: int, lower_bound=0, upper_bound=1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(size, dtype=torch.float32))
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def forward(self) -> Tensor:
        self.weight.data = self.weight.data.clamp(self.lower_bound, self.upper_bound)
        return self.weight


# ---------------------------------------------------------------------------
# Weight initializers
# ---------------------------------------------------------------------------
"""Initialize theta_var weights corresponding to variable occurrence frequency in the CNF."""
def generate_theta_var_weight(cnf: CNF) -> torch.Tensor:
    all_lits = torch.tensor([abs(lit) for clause in cnf.clauses for lit in clause], dtype=torch.long)
    var_counts = torch.bincount(all_lits - 1, minlength=cnf.nv).float()
    max_count = var_counts.max()
    theta_weight = 1 - var_counts / max_count + torch.rand(cnf.nv) / 10
    return theta_weight

"""Initialize theta_clause weights to all 1.0 (fixed for unweighted MAX-SAT)."""
def generate_theta_clause_weight(num_clauses: int) -> torch.Tensor:
    return torch.ones(num_clauses, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Sparse hidden-weight builder
# ---------------------------------------------------------------------------

def generate_hidden_weights_sparse(cnf: CNF, num_variables: int, num_clauses: int) -> torch.Tensor:
    """
    Build the hidden-layer weight matrix as a sparse CSR tensor.

    Shape: (n + 2m) × (n + m) with ~(n + 2m + Σ|clause|) non-zero entries:
      Block 1 (rows 0..n-1):       diagonal 2.0 on variable columns  — integrality
      Block 2 (rows n..n+m-1):     +1 on clause column, ±1 per literal — satisfaction
      Block 3 (rows n+m..n+2m-1):  +1 on clause column               — counting
    """
    n, m = num_variables, num_clauses

    # Collect COO entries: (row, col, value)
    rows, cols, vals = [], [], []

    # Block 1: diagonal, weight = 2.0
    for i in range(n):
        rows.append(i); cols.append(i); vals.append(2.0)

    # Block 2: clause satisfaction
    # z_j = θ_cj - Σ_{pos lit} θ_i + Σ_{neg lit} θ_i − (#neg in clause)
    # bias handles the −(#neg) term; here we record variable ±1 and the clause +1
    for j, clause in enumerate(cnf.clauses):
        row2 = n + j
        rows.append(row2); cols.append(n + j); vals.append(1.0)   # θ_cj column
        for lit in clause:
            vi = abs(lit) - 1
            v = -1.0 if lit > 0 else 1.0
            rows.append(row2); cols.append(vi); vals.append(v)

    # Block 3: clause counting — z_j = θ_cj − 0.5  (bias handles −0.5)
    for j in range(m):
        rows.append(n + m + j); cols.append(n + j); vals.append(1.0)

    out_features = n + 2 * m
    in_features = n + m
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.tensor(vals, dtype=torch.float32)

    # coalesce() sums duplicate (row, col) pairs (e.g. self-loop variables)
    sparse_w = torch.sparse_coo_tensor(indices, values, size=(out_features, in_features)).coalesce()
    return sparse_w.to_sparse_csr()


def generate_hidden_biases(cnf: CNF, num_variables: int, num_clauses: int) -> torch.Tensor:
    n, m = num_variables, num_clauses
    b = torch.zeros(n + 2 * m, dtype=torch.float32)
    # Block 1: integrality bias
    b[:n] = -1.0
    # Block 2: bias = −(number of negative literals in clause j)
    for j, clause in enumerate(cnf.clauses):
        neg_count = sum(1 for lit in clause if lit < 0)
        b[n + j] = -float(neg_count)
    # Block 3: counting bias
    b[n + m:] = -0.5
    return b


def generate_output_weights(num_variables: int, num_clauses: int) -> torch.Tensor:
    """Return a 1-D output weight vector of shape (n + 2m,)."""
    n, m = num_variables, num_clauses
    w = torch.zeros(n + 2 * m, dtype=torch.float32)
    w[0:n] = float(n)        # Block 1: penalise non-integral variables
    w[n:n + m] = float(n)    # Block 2: penalise unsatisfied clauses
    w[n + m:] = -1.0         # Block 3: reward satisfied clauses
    return w


# ---------------------------------------------------------------------------
# MaxSAT dNN model
# ---------------------------------------------------------------------------

class MaxsatdNN(nn.Module):
    """
    Differentiable Neural Network for MAX-SAT.

    The objective function is minimised as more clauses are satisfied;
    f* = 0 implies all clauses are satisfied.

    Args:
        cnf:               PySAT CNF formula.
        device:            Target device. Defaults to CUDA if available, else CPU.
        dtype:             Floating-point dtype. Must be torch.float32 on CPU.
                           bfloat16 / float16 require a CUDA device.
        batch_size:        Number of independent restarts sharing frozen weights.
                           When > 1, theta_var is (n, B) and theta_clause is (m, B).
    """

    def __init__(
        self,
        cnf: CNF,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        batch_size: int = 1,
        random_noise: float = 0.1
    ) -> None:
        super().__init__()
        self.cnf = cnf
        self.num_variables = cnf.nv
        self.num_clauses = len(cnf.clauses)
        self.batch_size = batch_size
        self.random_noise = random_noise

        # Resolve device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device)
        self.dtype = dtype

        # Enforce float32 on CPU (sparse CSR + non-float32 is unsupported on CPU)
        if self.device.type == 'cpu' and dtype != torch.float32:
            raise ValueError(
                f"MaxsatdNN: dtype={dtype} is not supported on CPU. "
                "Use dtype=torch.float32 on CPU, or move to a CUDA device for bfloat16/float16."
            )

        n, m = self.num_variables, self.num_clauses

        # ------------------------------------------------------------------
        # Input layer — only theta_var is trained by default;
        # theta_clause is typically frozen at 1.0 by the caller (unweighted MAX-SAT).
        # ------------------------------------------------------------------
        self.theta_var = ConstrainedElemMultiply(n)
        self.theta_clause = ConstrainedElemMultiply(m)

        base_var = generate_theta_var_weight(cnf).to(dtype)
        base_clause = generate_theta_clause_weight(m).to(dtype)

        if batch_size > 1:
            # (size, B): each column is a independently noise-perturbed restart
            var_cols = [
                (base_var + torch.rand(n, dtype=dtype) * self.random_noise).clamp(0.0, 1.0)
                for _ in range(batch_size)
            ]
            clause_cols = [
                (base_clause + torch.rand(m, dtype=dtype) * self.random_noise).clamp(0.0, 1.0)
                for _ in range(batch_size)
            ]
            self.theta_var.weight = nn.Parameter(torch.stack(var_cols, dim=1))       # (n, B)
            self.theta_clause.weight = nn.Parameter(torch.stack(clause_cols, dim=1)) # (m, B)
        else:
            self.theta_var.weight.data = base_var
            self.theta_clause.weight.data = base_clause

        # ------------------------------------------------------------------
        # Hidden layer — sparse CSR weight + dense bias, both frozen buffers
        # ------------------------------------------------------------------
        hidden_w = generate_hidden_weights_sparse(cnf, n, m).to(dtype)
        hidden_b = generate_hidden_biases(cnf, n, m).to(dtype)
        self.register_buffer('hidden_weight', hidden_w)
        self.register_buffer('hidden_bias', hidden_b)

        # ------------------------------------------------------------------
        # Output layer — dense 1-D weight vector, frozen buffer
        # ------------------------------------------------------------------
        out_w = generate_output_weights(n, m).to(dtype)
        self.register_buffer('output_weight', out_w)

        self.activation = nn.ReLU()

        # Move entire model (buffers + parameters) to target device
        self.to(self.device)

    def forward(self) -> Tensor:
        n, m = self.num_variables, self.num_clauses

        # Build input: (n+m,) unbatched  or  (n+m, B) batched
        x = torch.cat([self.theta_var(), self.theta_clause()], dim=0).to(self.dtype)
        batched = x.dim() == 2  # True when batch_size > 1

        # cuSPARSE requires same-dtype operands; autocast(enabled=False) prevents
        # an outer AMP context from overriding the SpMM output dtype.
        with torch.amp.autocast('cuda', enabled=False):
            if batched:
                h = torch.sparse.mm(self.hidden_weight, x)                        # (n+2m, B)
            else:
                h = torch.sparse.mm(self.hidden_weight, x.unsqueeze(1)).squeeze(1) # (n+2m,)

        if batched:
            h = h + self.hidden_bias.unsqueeze(1)  # broadcast over B
        else:
            h = h + self.hidden_bias

        # Block 1: ReLU(1 - |2θ_i - 1|)  — 0 iff θ_i ∈ {0, 1}
        b1 = self.activation(1.0 - torch.abs(h[:n]))

        # Block 2: ReLU(θ_cj - Σ literal contributions)  — 0 iff clause j satisfied
        b2 = self.activation(h[n:n + m])

        # Block 3: ReLU(θ_cj - 0.5)  — counts satisfied clauses
        b3 = self.activation(h[n + m:])

        combined = torch.cat([b1, b2, b3], dim=0)  # (n+2m,) or (n+2m, B)

        if batched:
            return self.output_weight @ combined   # (B,)
        return combined @ self.output_weight       # scalar

