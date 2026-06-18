from pysat.formula import CNF
from torch import monitor
from Maxsat_dNN import MaxsatdNN
from helpers import count_satisfied
from ortools.sat.python import cp_model
import psutil
import time
import gc
from typing import Dict, List, Optional
import threading
import os
from enum import Enum
import argparse
import csv
import os.path
import torch
import gurobipy as gp
import json
import subprocess
import torch
import sys
from statistics import mean, median, stdev

### SolverStatus is used to easily map CP-SAT status of solving cnf instances. 
class cpsat_status(Enum):
    OPTIMAL = cp_model.OPTIMAL
    FEASIBLE = cp_model.FEASIBLE
    INFEASIBLE = cp_model.INFEASIBLE
    UNKNOWN = -1

    @classmethod
    def from_cp_sat(cls, status):
        for s in cls:
            if s.value == status:
                return s.name
        return cls.UNKNOWN.name
    
class Benchmark:
    def __init__(self):
        pass

    def _get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 2)
    def _peak_memory_monitor(self, interval: float = 0.1):
        """
        Context manager that monitors peak RSS memory usage in a background thread.
        Returns an object with a `.peak` attribute containing the maximum MB observed.
        """
        class _Monitor:
            def __init__(self, benchmark, interval):
                self.benchmark = benchmark
                self.interval = interval
                self.peak = 0.0
                self._stop = threading.Event()
                self._thread = None

            def _sample(self):
                while not self._stop.is_set():
                    mem = self.benchmark._get_memory_usage()
                    if mem > self.peak:
                        self.peak = mem
                    time.sleep(self.interval)

            def __enter__(self):
                self._thread = threading.Thread(target=self._sample)
                self._thread.daemon = True
                self._thread.start()
                return self

            def __exit__(self, *args):
                self._stop.set()
                if self._thread is not None:
                    self._thread.join(timeout=1.0)  # wait for thread to finish

        return _Monitor(self, interval)
    def _force_garbage_collection(self):
        gc.collect()
        gc.collect()
        time.sleep(0.5)
        gc.collect()

    def run_MAXSAT_cpu(self, cnf: CNF, num_threads: int = 1, time_limit: float = 1.0) -> Dict:
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)   
        self._force_garbage_collection()
        baseline_mem = self._get_memory_usage()
        monitor = self._peak_memory_monitor(interval=0.05)  # faster sampling for dNN
        with monitor:
            self._force_garbage_collection()
            baseline_mem = self._get_memory_usage()
            PARAMS = dict(
                epochs         = 1000,
                lr             = 0.09727746954141563,
                batch_size     = 4,
                grad_clip      = 1.0,
                eta_min        = 0.0,
                patience       = 20,
                weight_decay   = 5.210905999973664e-05,
                random_noise   = 0.05,   # per-column noise added on top of base init
                warmup_epochs  = 10,
                restart_period = 70,
                restart_samples= 50,
                device=torch.device('cpu'),
                dtype=torch.float32,
                )

            solver = MaxsatdNN(cnf, batch_size=PARAMS["batch_size"], random_noise=PARAMS["random_noise"]).to(device=PARAMS["device"], dtype=PARAMS["dtype"])

            optimizer = torch.optim.AdamW([solver.theta_var.weight], lr=PARAMS["lr"], weight_decay=PARAMS["weight_decay"])
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=PARAMS["warmup_epochs"])
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = PARAMS["restart_period"], T_mult=1, eta_min=PARAMS["eta_min"])
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[PARAMS["warmup_epochs"]])


            histories: List[List[float]] = [[] for _ in range(PARAMS["batch_size"])]
            best_obj = float('inf')
            best_theta: Optional[torch.Tensor] = None
            no_improve = 0
            start_time = time.perf_counter()

            for epoch in range(PARAMS["epochs"]):
                if time_limit is not None:
                    elapsed = time.perf_counter() - start_time
                    if elapsed >= time_limit:
                        break
                optimizer.zero_grad()
                obj = solver()                     # (batch_size,) if batch_size>1 else scalar
                if PARAMS["batch_size"] > 1:
                    obj.sum().backward()
                else:
                    obj.backward()

                torch.nn.utils.clip_grad_norm_([solver.theta_var.weight], PARAMS["grad_clip"])
                optimizer.step()
                scheduler.step()

                with torch.no_grad():
                    solver.theta_var.weight.data.clamp_(0.0, 1.0)

                    if PARAMS["batch_size"] > 1:
                        obj_vals = obj.detach().cpu().tolist()
                        best_col = int(obj.argmin().item())
                        col_min = obj_vals[best_col]
                        for c in range(PARAMS["batch_size"]):
                            histories[c].append(obj_vals[c])
                    else:
                        col_min = obj.item()
                        histories[0].append(col_min)

                    if col_min < best_obj - 1e-6:
                        best_obj = col_min
                        no_improve = 0
                        if PARAMS["batch_size"] > 1:
                            best_theta = solver.theta_var.weight[:, best_col].clone().cpu()
                        else:
                            best_theta = solver.theta_var.weight.clone().cpu()
                    else:
                        no_improve += 1

                    if PARAMS["patience"] > 0 and no_improve >= PARAMS["patience"]:
                        break

            if best_theta is None:
                best_theta = (solver.theta_var.weight[:, 0] if PARAMS["batch_size"] > 1
                            else solver.theta_var.weight).clone().cpu()
                
            end_time = time.perf_counter()
            solve_time = end_time - start_time
            end_mem = self._get_memory_usage()
            total_mem_delta = end_mem - baseline_mem
            peak_mem = monitor.peak
            satisfied = count_satisfied(cnf, best_theta)
            total_clauses = len(cnf.clauses)
            
            results = { 
                'solver': 'MaxSatdNN',
                'time (seconds)': solve_time,  # seconds
                'memory_usage (MB)': total_mem_delta,
                'peak_memory_usage (MB)': peak_mem, 
                'status': 'FEASIBLE' if satisfied == total_clauses else 'UNKNOWN',
                'num_vars': cnf.nv,  # Number of variables in CNF
                'num_clauses': len(cnf.clauses),
                'satisfied_clauses': satisfied,
                'satisfaction_ratio': satisfied / total_clauses if total_clauses > 0 else 0,
                'solved': satisfied == total_clauses,
                'time_budget': None,
                'threads_used': num_threads,
            }
            # Clean up
            del solver
            return results
    
    def run_cpsat_maxsat(self, cnf: CNF, time_limit_seconds=60, num_threads: int =1) -> Dict:
        self._force_garbage_collection()
        baseline_mem = self._get_memory_usage()
        monitor = self._peak_memory_monitor(interval=0.1)
        with monitor:
            model = cp_model.CpModel()
            all_vars = {abs(lit) for clause in cnf.clauses for lit in clause}
            variables = {var: model.NewBoolVar(f'x_{var}') for var in all_vars}
            
            clause_indicators = []
            for i, clause in enumerate(cnf.clauses):
                if not clause:
                    continue
                ind = model.NewBoolVar(f'clause_{i}_sat')
                clause_indicators.append(ind)
                lits = [variables[abs(l)] if l > 0 else variables[abs(l)].Not() for l in clause]
                # Reified equivalence: ind == (sum(lits) >= 1)
                # satisfied side
                model.Add(sum(lits) >= 1).OnlyEnforceIf(ind)
                # unsatisfied side
                model.Add(sum(lits) == 0).OnlyEnforceIf(ind.Not())

            if clause_indicators:
                model.Maximize(sum(clause_indicators))

            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = time_limit_seconds
            solver.parameters.num_search_workers = num_threads

            start = time.perf_counter()
            status = solver.Solve(model)
            end = time.perf_counter()
            solve_time = end - start

            mapped_status = cpsat_status.from_cp_sat(status)
            
            # Evaluate clause satisfaction regardless of status if any solution values exist
            solution = {var: solver.Value(variables[var]) for var in variables}
            satisfied_count = sum(
                1 for clause in cnf.clauses if any(
                    (lit > 0 and solution.get(abs(lit), 0) == 1) or
                    (lit < 0 and solution.get(abs(lit), 0) == 0)
                    for lit in clause
                )
            )
            satisfaction_ratio = satisfied_count / len(cnf.clauses) if cnf.clauses else 0.0

            end_mem = self._get_memory_usage()
            total_mem_delta = end_mem - baseline_mem
            peak_mem = monitor.peak
            results = {
                'solver': 'CP-SAT-MAXSAT',
                'time (seconds)': solve_time,
                'memory_usage (MB)': total_mem_delta,
                'peak_memory_usage (MB)': peak_mem,
                'status': mapped_status,
                'num_vars': cnf.nv,
                'num_clauses': len(cnf.clauses),
                'satisfied_clauses': satisfied_count,
                'satisfaction_ratio': satisfaction_ratio,
                'solved': mapped_status in ['OPTIMAL', 'FEASIBLE'] and satisfaction_ratio == 1.0,
                'time_budget': time_limit_seconds,
                'threads_used': num_threads,
            }
            self._force_garbage_collection()
            return results


    
    def run_gurobi_maxsat(self, cnf: CNF, time_limit_seconds=60, num_threads: int =1) -> Dict:
        self._force_garbage_collection()
        baseline_mem = self._get_memory_usage()
        monitor = self._peak_memory_monitor(interval=0.1)
        with monitor:
            # Setup Gurobi model
            with gp.Env(empty=True) as env:
                env.setParam('OutputFlag', 0)  # Disable console output
                env.setParam('Threads', num_threads)
                env.setParam('TimeLimit', time_limit_seconds)
                env.start()
                
                model = gp.Model("MaxSAT", env=env)
                
                all_vars = {abs(lit) for clause in cnf.clauses for lit in clause}
                variables = {var: model.addVar(vtype=gp.GRB.BINARY, name=f'x_{var}') for var in all_vars}
                
                # Add clauses as constraints with slack variables
                clause_satisfied = []
                for i, clause in enumerate(cnf.clauses):
                    if not clause:
                        continue
                    
                    # Create a slack variable for this clause (1 if satisfied, 0 if not)
                    slack = model.addVar(vtype=gp.GRB.BINARY, name=f'slack_{i}')
                    clause_satisfied.append(slack)
                    
                    # Add constraint: at least one literal must be true OR slack must be 1
                    clause_literals = []
                    for lit in clause:
                        var = variables[abs(lit)]
                        if lit > 0:
                            clause_literals.append(var)
                        else:
                            clause_literals.append(1 - var)  # Negation: 1 - x
                    
                    # The sum of satisfied literals plus slack must be at least 1
                    # If all literals are false (sum=0), then slack must be 1 (clause unsatisfied)
                    model.addConstr(gp.quicksum(clause_literals) + slack >= 1, name=f'clause_{i}')
                
                # Objective: minimize number of unsatisfied clauses (slack == 1)
                model.setObjective(gp.quicksum(clause_satisfied), gp.GRB.MINIMIZE)
                
                # Solve the model
                start_time = time.perf_counter()
                model.optimize()
                end_time = time.perf_counter()
                solve_time = end_time - start_time
                
                # Map Gurobi status
                status_map = {
                    gp.GRB.OPTIMAL: 'OPTIMAL',
                    gp.GRB.SUBOPTIMAL: 'FEASIBLE',  # Time limit reached but feasible solution found
                    gp.GRB.INFEASIBLE: 'INFEASIBLE',
                    gp.GRB.UNBOUNDED: 'UNKNOWN',
                    gp.GRB.INF_OR_UNBD: 'UNKNOWN',
                    gp.GRB.TIME_LIMIT: 'FEASIBLE' if model.SolCount > 0 else 'UNKNOWN',
                }
                mapped_status = status_map.get(model.status, 'UNKNOWN')
                
                # Extract solution and calculate satisfaction
                satisfied_count = 0
                satisfaction_ratio = 0.0
                solution = {}
                
                if mapped_status in ['OPTIMAL', 'FEASIBLE'] and model.SolCount > 0:
                    # Get variable assignments
                    solution = {var: round(variables[var].X) for var in variables}
                    
                    # Count satisfied clauses directly from the CNF
                    satisfied_count = sum(
                        1 for clause in cnf.clauses if any(
                            (lit > 0 and solution[abs(lit)] == 1) or 
                            (lit < 0 and solution[abs(lit)] == 0) 
                            for lit in clause if abs(lit) in solution
                        )
                    )
                    satisfaction_ratio = satisfied_count / len(cnf.clauses) if cnf.clauses else 0.0
            
            end_mem = self._get_memory_usage()
            total_mem_delta = end_mem - baseline_mem
            peak_mem = monitor.peak

            results = {
                'solver': 'Gurobi-MAXSAT',
                'time (seconds)': solve_time,
                'memory_usage (MB)': total_mem_delta,
                'peak_memory_usage (MB)': peak_mem,
                'status': mapped_status,
                'num_vars': cnf.nv,
                'num_clauses': len(cnf.clauses),
                'satisfied_clauses': satisfied_count,
                'satisfaction_ratio': satisfaction_ratio,
                'solved': mapped_status == 'OPTIMAL' and satisfaction_ratio == 1.0,
                'time_budget': time_limit_seconds,
                'threads_used': num_threads,
            }
            
            # Cleanup
            model.dispose()
            env.dispose()
            self._force_garbage_collection()
            return results


    def run_benchmark(self, cnf: CNF, solvers: List[str], time_limit_seconds=60, num_threads=1, cnf_file_path=None):
        print("=" * 80)
        print("COMPREHENSIVE MEMORY BENCHMARK")
        print("=" * 80)
        
        cnf_filename = os.path.basename(cnf_file_path) if cnf_file_path else "unknown"
        results = []
        for solver_name in solvers:
            print("\n" + "="*40)
            print(f"BENCHMARKING {solver_name} on {cnf_filename}")
            print("="*40)
            if solver_name == 'MaxSatdNN':
                res = self.run_MAXSAT_cpu(cnf, num_threads, time_limit_seconds)
            elif solver_name == 'CP-SAT-MAXSAT':
                res = self.run_cpsat_maxsat(cnf, time_limit_seconds, num_threads)
            elif solver_name == 'Gurobi-MAXSAT':
                res = self.run_gurobi_maxsat(cnf, time_limit_seconds, num_threads)
            else:
                continue  # Unknown solver, skip
            res['cnf_file'] = cnf_filename
            results.append(res)
        return results

    # --- Subprocess isolated execution for a single solver ---
    @staticmethod
    def run_solver_isolated(script_path: str, solver_name: str, cnf_file: str, time_limit: float, threads: int) -> Dict:
        """Invoke this script in a subprocess for a single solver and return the result dict."""
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(threads)
        env["MKL_NUM_THREADS"] = str(threads) 
        
        cmd = [sys.executable, script_path, '--solvers', solver_name, '--time_limit', str(time_limit), '--threads', str(threads), '--output', '/dev/null', '--json_single', cnf_file]
        completed = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if completed.returncode != 0:
            return {
                'solver': solver_name,
                'error': f' subprocess failed: {completed.stderr.strip()}',
                'time (seconds)': 0,
                'memory_usage (MB)': None,
                'peak_memory_usage (MB)': None,
                'status': 'ERROR',
                'num_vars': 0,
                'num_clauses': 0,
                'satisfied_clauses': 0,
                'satisfaction_ratio': 0.0,
                'solved': False,
                'time_budget': time_limit,
                'threads_used': threads,
                'cnf_file': os.path.basename(cnf_file)
            }
        try:
            data = json.loads(completed.stdout.strip())
            return data
        except json.JSONDecodeError:
            return {
                'solver': solver_name,
                'error': 'invalid JSON from child',
                'time (seconds)': 0,
                'memory_usage (MB)': None,
                'peak_memory_usage (MB)': None,
                'status': 'ERROR',
                'num_vars': 0,
                'num_clauses': 0,
                'satisfied_clauses': 0,
                'satisfaction_ratio': 0.0,
                'solved': False,
                'time_budget': time_limit,
                'threads_used': threads,
                'cnf_file': os.path.basename(cnf_file)
            }

    @staticmethod
    def aggregate_repetitions(result_list: List[Dict], repetitions: int) -> Dict:
        """Aggregate numeric fields across repetitions (mean, median, stdev) without
        overwriting integral counts like satisfied_clauses."""
        if not result_list:
            return {}
        numeric_fields = [
            'time (seconds)', 'memory_usage (MB)', 'peak_memory_usage (MB)', 'satisfaction_ratio'
        ]  # exclude satisfied_clauses from averaging in-place
        aggregated = result_list[0].copy()

        # Preserve an integer satisfied_clauses (use median of ints if you prefer)
        sat_vals = [r['satisfied_clauses'] for r in result_list if 'satisfied_clauses' in r]
        if sat_vals:
            aggregated['satisfied_clauses'] = int(round(median(sat_vals)))
            aggregated['satisfied_clauses_mean'] = mean(sat_vals)
            aggregated['satisfied_clauses_stdev'] = stdev(sat_vals) if len(sat_vals) > 1 else 0.0

        for field in numeric_fields:
            vals = [r[field] for r in result_list if r.get(field) is not None]
            if vals:
                aggregated[f'{field}'] = mean(vals)
                aggregated[f'{field}__median'] = median(vals)
                aggregated[f'{field}__stdev'] = stdev(vals) if len(vals) > 1 else 0.0

        statuses = {r.get('status') for r in result_list}
        aggregated['status'] = statuses.pop() if len(statuses) == 1 else 'VARIED'
        aggregated['repetitions'] = repetitions
        return aggregated

    def print_results(self, results: List[Dict]):
        for result in results:
            print(f"Solver: {result['solver']}")
            print(f"Problem size: {result['num_vars']} variables, {result['num_clauses']} clauses")
            print(f"Total time: {result['time (seconds)']:.6f} seconds")      
            print(f"Total memory usage: {result['memory_usage (MB)']:.2f} MB" if result['memory_usage (MB)'] else "Total memory usage: N/A")
            print(f"Peak memory usage: {result['peak_memory_usage (MB)']:.2f} MB" if result['peak_memory_usage (MB)'] else "Peak memory usage: N/A")
            # Solution quality
            print(f"Satisfied clauses: {result.get('satisfied_clauses', 'N/A')} / {result['num_clauses']}")
            print(f"Satisfaction ratio: {result['satisfaction_ratio']:.4f}")
            print(f"Solved: {'Yes' if result['solved'] else 'No'}")
            # Status and configuration
            if 'status' in result:
                print(f"Status: {result['status']}")
            print(f"Threads used: {result['threads_used']}")
            if result['time_budget']:
                print(f"Time budget: {result['time_budget']} seconds")
            print("-" * 80)

    def results_to_csv(self, results, csv_file):
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, 'a', newline='') as f:
            fieldnames = list(results[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for result in results:
                writer.writerow(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MaxSAT benchmark on specified solvers.')
    parser.add_argument('--solvers', nargs='+', default=['CP-SAT', 'MaxSatdNN', 'Gurobi'], help='List of solvers to benchmark')
    parser.add_argument('cnf_file', help='Path to DIMACS CNF file')
    parser.add_argument('--time_limit', type=float, default=0.05, help='Time limit in seconds')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads')
    parser.add_argument('--output', default='benchmark_results.csv', help='Output CSV file')
    parser.add_argument('--repetitions', type=int, default=1, help='Run each solver N times and average results')
    parser.add_argument('--isolate', action='store_true', help='Run each solver in a fresh subprocess for memory isolation')
    parser.add_argument('--json_single', action='store_true', help=argparse.SUPPRESS)  # internal flag for child process single JSON output
    args = parser.parse_args()
    
    cnf = CNF(from_file=args.cnf_file)
    
    benchmark = Benchmark()

    # Child process mode: run requested solver and output ONLY JSON (no banners)
    if args.json_single:
        solver_name = args.solvers[0] if args.solvers else None
        if solver_name == 'MaxSatdNN':
            res = benchmark.run_MAXSAT_cpu(cnf, args.threads, args.time_limit)
        elif solver_name == 'CP-SAT-MAXSAT':
            res = benchmark.run_cpsat_maxsat(cnf, args.time_limit, args.threads)
        elif solver_name == 'Gurobi-MAXSAT':
            res = benchmark.run_gurobi_maxsat(cnf, args.time_limit, args.threads)
        else:
            res = {'error': f'Unknown solver {solver_name}'}
        if isinstance(res, dict):
            res['cnf_file'] = os.path.basename(args.cnf_file)
        print(json.dumps(res))
        sys.exit(0)

    # Parent aggregation mode
    if args.isolate or args.repetitions > 1:
        aggregated_results = []
        for solver_name in args.solvers:
            rep_results = []
            for i in range(args.repetitions):
                r = Benchmark.run_solver_isolated(os.path.abspath(sys.argv[0]), solver_name, args.cnf_file, args.time_limit, args.threads)
                rep_results.append(r)
            agg = Benchmark.aggregate_repetitions(rep_results, args.repetitions)
            aggregated_results.append(agg)
        benchmark.print_results(aggregated_results)
        benchmark.results_to_csv(aggregated_results, args.output)
        print(f"Aggregated results appended to {args.output}")
    else:
        results = benchmark.run_benchmark(cnf, args.solvers, time_limit_seconds=args.time_limit, num_threads=args.threads, cnf_file_path=args.cnf_file)
        benchmark.print_results(results)
        benchmark.results_to_csv(results, args.output)
        print(f"Results appended to {args.output}")