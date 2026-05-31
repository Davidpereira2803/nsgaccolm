import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from evox.algorithms import NSGA2
from evox.core import Problem
from evox.workflows import EvalMonitor, StdWorkflow
from helper_fn.due_dates_computation import compute_due_dates, estimate_average_completion_time
from helper_fn.flowshop import generate_taillard_processing_times, taillard_get_nb_jobs, taillard_get_nb_machines

def parse_args():
    parser = argparse.ArgumentParser(description="Run NSGA-II with EvoX on GPU for the Taillard flow shop problem.")
    parser.add_argument("inst_id", type=int, help="Taillard instance id, for example 51")
    parser.add_argument("comment", nargs="?", default="", help="Optional comment for the run")
    parser.add_argument("--time-limit", type=float, default=120.0, help="Wall-clock time limit in seconds")
    parser.add_argument("--population-size", type=int, default=300, help="NSGA-II population size")
    parser.add_argument("--pareto-output", default="output/pareto.txt", help="Output file for Pareto solutions")
    parser.add_argument("--all-output", default="output/all_solutions.txt", help="Output file for the final population")
    parser.add_argument("--plot", action="store_true", help="Show a Pareto plot at the end")
    parser.add_argument("--db-path", default="", help="SQLite DB path to save all solutions (optional)")
    parser.add_argument("--run-id", default="nsga", help="Run identifier stored in the DB")
    parser.add_argument("--max-solutions", type=int, default=100_000,
                        help="Max unique solutions to accumulate for the DB (default 100 000)")
    return parser.parse_args()

def select_device():
    # Prefer GPU if available, otherwise fall back to CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class FlowShopBiObjectiveProblem(Problem):
    def __init__(self, processing_times, due_dates, device):
        super().__init__()
        # Store tensors on the chosen device for batched GPU evaluation
        self.device = device
        self.processing_times = torch.as_tensor(processing_times, dtype=torch.float32, device=device)
        self.due_dates = torch.as_tensor(due_dates, dtype=torch.float32, device=device)
        self.n_jobs = int(self.processing_times.shape[1])
        self.n_machines = int(self.processing_times.shape[0])
        # Reorder to index by job first for easier batching: shape (n_jobs, n_machines)
        self.processing_times_by_job = self.processing_times.transpose(0, 1).contiguous()

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        # Convert continuous decision variabes into a permutation via argsort
        permutation = torch.argsort(x, dim=1)
        batch_size = permutation.shape[0]

        # Gather processing times per permutation: shape (batch_size, n_jobs, n_machines)
        proc = self.processing_times_by_job[permutation]
        # Completion table with extra padding row/col to simplifiy recurrence
        completion = torch.zeros(
            (batch_size, self.n_jobs + 1, self.n_machines + 1),
            dtype=proc.dtype,
            device=proc.device,
        )

        # Sweep along anti-diagonals to compute completion times in parallel
        for diagonal in range(self.n_jobs + self.n_machines - 1):
            start_job = max(0, diagonal - (self.n_machines - 1))
            end_job = min(self.n_jobs - 1, diagonal)
            job_indices = torch.arange(start_job, end_job + 1, device=proc.device)
            machine_indices = diagonal - job_indices

            proc_diag = proc[:, job_indices, machine_indices]
            prev_job = completion[:, job_indices, machine_indices + 1]
            prev_machine = completion[:, job_indices + 1, machine_indices]

            completion[:, job_indices + 1, machine_indices + 1] = proc_diag + torch.maximum(prev_job, prev_machine)

        # Objective 1: makespan (last completion)
        makespan = completion[:, self.n_jobs, self.n_machines]
        # Objective 2: total tardiness computed from final machine completion times per job
        final_machine_completion = completion[:, 1:, self.n_machines]
        tardiness = torch.clamp(final_machine_completion - self.due_dates[permutation], min=0.0).sum(dim=1)

        return torch.stack((makespan, tardiness), dim=1)

def pareto_mask(fitness: torch.Tensor) -> torch.Tensor:
    # Compute non-dominated mask for a fitness matrix of shape (N,2)
    dominated = (
        (fitness[:, None, :] >= fitness[None, :, :]).all(dim=2)
        & (fitness[:, None, :] > fitness[None, :, :]).any(dim=2)
    )
    return ~dominated.any(dim=0)

def update_pareto_archive(archive_solutions: torch.Tensor | None, archive_fitness: torch.Tensor | None, candidate_solutions: torch.Tensor, candidate_fitness: torch.Tensor):
    # Merge archive and candidates, then keep only non-dominated entries
    candidate_solutions = candidate_solutions.detach().cpu()
    candidate_fitness = candidate_fitness.detach().cpu()

    if archive_solutions is None or archive_fitness is None:
        combined_solutions = candidate_solutions
        combined_fitness = candidate_fitness
    else:
        combined_solutions = torch.cat((archive_solutions, candidate_solutions), dim=0)
        combined_fitness = torch.cat((archive_fitness, candidate_fitness), dim=0)

    if combined_fitness.numel() == 0:
        return combined_solutions, combined_fitness

    mask = pareto_mask(combined_fitness)
    return combined_solutions[mask], combined_fitness[mask]

def run_evox_nsga2(inst_id, comment, time_limit, population_size, max_solutions=100_000):
    device = select_device()
    if device.type == "cuda":
        print(f"Using GPU device: {torch.cuda.get_device_name(device)}")
    else:
        print("CUDA is not available; EvoX will run on CPU.")

    # Problem setup using Taillard instance generator and due-dates heuristics
    n_jobs = taillard_get_nb_jobs(inst_id)
    n_machines = taillard_get_nb_machines(inst_id)
    processing_times = np.array(generate_taillard_processing_times(inst_id))
    p_value = estimate_average_completion_time(processing_times)
    due_dates = compute_due_dates(processing_times, p_value, R=0.4, T=0.2)

    problem = FlowShopBiObjectiveProblem(processing_times, due_dates, device=device)
    algorithm = NSGA2(
        pop_size=population_size,
        n_objs=2,
        lb=torch.zeros(n_jobs, device=device),
        ub=torch.ones(n_jobs, device=device),
        device=device,
    )
    monitor = EvalMonitor(multi_obj=True, full_fit_history=False, full_sol_history=False)
    workflow = StdWorkflow(algorithm, problem, monitor, device=device)

    workflow.init_step()
    start_time = time.time()
    steps = 0
    archive_solutions = None
    archive_fitness = None
    # Collect unique decoded permutations with their objective values
    all_unique: dict[tuple[int, ...], tuple[int, float]] = {}
    while time.time() - start_time < time_limit:
        workflow.step()
        steps += 1
        latest_solutions = monitor.get_latest_solution()
        latest_fitness = monitor.get_latest_fitness()
        archive_solutions, archive_fitness = update_pareto_archive(
            archive_solutions,
            archive_fitness,
            latest_solutions,
            latest_fitness,
        )
        # Accumulate unique permutations up to a configured cap
        if len(all_unique) < max_solutions:
            perms = decode_permutations(latest_solutions)
            fit_list = latest_fitness.detach().cpu().tolist()
            for perm_list, (ms, tt) in zip(perms, fit_list):
                key = tuple(perm_list)
                if key not in all_unique:
                    all_unique[key] = (int(round(ms)), float(tt))
                    if len(all_unique) >= max_solutions:
                        break

    latest_solutions = monitor.get_latest_solution()
    latest_fitness = monitor.get_latest_fitness()
    # Use archive if available, otherwise fall back to last population
    pareto_solutions = archive_solutions if archive_solutions is not None else latest_solutions.detach().cpu()
    pareto_fitness = archive_fitness if archive_fitness is not None else latest_fitness.detach().cpu()

    workflow.final_step()

    print(f"Instance: {inst_id}")
    if comment:
        print(f"Comment: {comment}")
    print(f"Jobs: {n_jobs}, Machines: {n_machines}")
    print(f"Time limit: {time_limit:.1f} seconds")
    print(f"Workflow steps completed: {steps}")
    print(f"Final population size: {latest_solutions.shape[0]}")
    print(f"Pareto front size: {pareto_solutions.shape[0]}")

    return {
        "device": device,
        "monitor": monitor,
        "latest_solutions": latest_solutions,
        "latest_fitness": latest_fitness,
        "pareto_solutions": pareto_solutions,
        "pareto_fitness": pareto_fitness,
        "all_unique": all_unique,
        "inst_id": inst_id,
    }

def decode_permutations(solutions: torch.Tensor):
    # Convert continuous representation into discrete permutations
    if solutions.numel() == 0:
        return []
    permutations = torch.argsort(solutions, dim=1)
    return permutations.detach().cpu().tolist()

def unique_permutations(permutations):
    # Preserve original order while removing duplicates
    unique = []
    seen = set()
    for sequence in permutations:
        key = tuple(sequence)
        if key in seen:
            continue
        seen.add(key)
        unique.append(sequence)
    return unique

def save_pareto_sequences(pareto_solutions, filename="output/pareto.txt"):
    # Write unique decoded Pareto solutions to a text file, one permutation per line
    permutations = unique_permutations(decode_permutations(pareto_solutions))
    with open(filename, "w", encoding="utf-8") as handle:
        for sequence in permutations:
            handle.write(" ".join(map(str, sequence)) + "\n")

def save_all_solutions(latest_solutions, filename="output/all_solutions.txt"):
    # Write unique decoded final population solutions to a text file, one permutation per line
    permutations = unique_permutations(decode_permutations(latest_solutions))
    with open(filename, "w", encoding="utf-8") as handle:
        for sequence in permutations:
            handle.write(" ".join(map(str, sequence)) + "\n")

def plot_pareto_front(latest_fitness, pareto_fitness):
    # Simple scatter plot comparing final population to Pareto front in objective space
    all_objs = latest_fitness.detach().cpu().numpy()
    pareto_objs = pareto_fitness.detach().cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.scatter(all_objs[:, 0], all_objs[:, 1], s=18, color="blue", alpha=0.5, label="Final population")
    plt.scatter(pareto_objs[:, 0], pareto_objs[:, 1], s=30, color="red", label="Pareto front")
    plt.xlabel("Makespan")
    plt.ylabel("Total Tardiness")
    plt.title("EvoX NSGA-II Pareto Front")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    args = parse_args()
    result = run_evox_nsga2(args.inst_id, args.comment, args.time_limit, args.population_size, args.max_solutions)

    save_pareto_sequences(result["pareto_solutions"], args.pareto_output)
    save_all_solutions(result["latest_solutions"], args.all_output)

    if args.db_path:
        from db import init_db, save_solutions, compute_and_update_ranks
        init_db(args.db_path)
        all_unique = result["all_unique"]
        solutions_list = [(perm, ms, tt) for perm, (ms, tt) in all_unique.items()]
        n_saved = save_solutions(args.db_path, solutions_list, args.run_id, args.inst_id)
        print(f"Saved {n_saved} unique solutions to {args.db_path} (run_id={args.run_id})")
        print(f"Computing Pareto front ranks for {n_saved} solutions...")
        n_fronts = compute_and_update_ranks(args.db_path, args.run_id, args.inst_id)
        print(f"Computed {n_fronts} Pareto fronts")

    if args.plot:
        plot_pareto_front(result["latest_fitness"], result["pareto_fitness"])

if __name__ == "__main__":
    main()