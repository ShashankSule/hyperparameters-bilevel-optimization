"""Run the YAML-configured biexponential recovery sweep."""
# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
import traceback
from itertools import product
from typing import Any

import torch

from data import make_lambda_grid, make_synthetic_observation, make_time_grid
from optimization import ep_gradient_descent_mu, gauss_newton_mu, gradient_descent_mu
from utils import (
    append_jsonl_row,
    count_jsonl_rows,
    finalize_json_array,
    load_completed_keys,
    load_config,
    make_result_key,
    prepare_jsonl_output,
    result_key,
    resolve_experiment,
    save_config_snapshot,
    summarize_solver_history,
)


SOLVERS = {
    "gd": gradient_descent_mu,
    "gn": gauss_newton_mu,
    "ep": ep_gradient_descent_mu,
}
WORKER_CONFIG: dict[str, Any] | None = None
WORKER_T: torch.Tensor | None = None


def parse_args() -> argparse.Namespace:
    """Parse the config path. All other options live in YAML."""
    parser = argparse.ArgumentParser(description="Run the biexponential recovery sweep.")
    parser.add_argument("--config", required=True, help="Path to the YAML simulation config.")
    return parser.parse_args()


def make_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Enumerate one scalar task per method-level output row."""
    experiment = config["experiment"]
    methods = list(config["solver"].get("methods", SOLVERS.keys()))
    lambda0_values = make_lambda_grid(config["lambda0"]).tolist()
    lambda1_values = make_lambda_grid(config["lambda1"]).tolist()

    tasks = []
    task_index = 0
    for job_index, (c1, lambda0, lambda1, noise_type, realization) in enumerate(
        product(
            experiment["c1_values"],
            lambda0_values,
            lambda1_values,
            experiment["noise_types"],
            range(int(experiment["n_realizations"])),
        )
    ):
        seed = int(experiment.get("base_seed", 0)) + job_index
        for method in methods:
            tasks.append(
                {
                    "task_index": task_index,
                    "job_index": job_index,
                    "method": method,
                    "seed": seed,
                    "realization": int(realization),
                    "noise_type": noise_type,
                    "c0_true": float(experiment["c0"]),
                    "c1_true": float(c1),
                    "lambda0_true": float(lambda0),
                    "lambda1_true": float(lambda1),
                }
            )
            task_index += 1
    return tasks


def task_result_key(task: dict[str, Any]) -> tuple[Any, ...]:
    """Return the resume key for a method-level task before it has run."""
    return make_result_key(
        c1_true=task["c1_true"],
        lambda0_true=task["lambda0_true"],
        lambda1_true=task["lambda1_true"],
        noise_type=task["noise_type"],
        realization=task["realization"],
        method=task["method"],
    )


def filter_completed_tasks(
    tasks: list[dict[str, Any]],
    completed: set[tuple[Any, ...]],
) -> list[dict[str, Any]]:
    """Drop tasks whose output rows already exist in the JSONL file."""
    return [task for task in tasks if task_result_key(task) not in completed]


def init_worker(config: dict[str, Any]) -> None:
    """Initialize worker-local config and torch thread count."""
    global WORKER_CONFIG, WORKER_T

    WORKER_CONFIG = config
    parallel_config = config.get("parallel") or {}
    torch_num_threads = int(parallel_config.get("torch_num_threads", 1))
    if torch_num_threads > 0:
        torch.set_num_threads(torch_num_threads)
    WORKER_T = make_time_grid(config)


def get_worker_state() -> tuple[dict[str, Any], torch.Tensor]:
    """Return initialized worker config and time grid."""
    if WORKER_CONFIG is None or WORKER_T is None:
        raise RuntimeError("Worker has not been initialized.")
    return WORKER_CONFIG, WORKER_T


def make_failed_row(task: dict[str, Any], exc: Exception, runtime_seconds: float) -> dict[str, Any]:
    """Build a JSON-serializable row for a failed task."""
    return {
        "experiment_id": task["experiment_id"],
        "task_index": int(task["task_index"]),
        "job_index": int(task["job_index"]),
        "method": task["method"],
        "noise_type": task["noise_type"],
        "seed": int(task["seed"]),
        "realization": int(task["realization"]),
        "c0_true": float(task["c0_true"]),
        "c1_true": float(task["c1_true"]),
        "lambda0_true": float(task["lambda0_true"]),
        "lambda1_true": float(task["lambda1_true"]),
        "runtime_seconds": float(runtime_seconds),
        "n_outer_iterations": 0,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
    }


def run_one_task(task: dict[str, Any]) -> dict[str, Any]:
    """Run one method-level recovery task and return one output row."""
    config, t = get_worker_state()
    experiment = config["experiment"]
    solver_config = config["solver"]

    c_true = torch.tensor([task["c0_true"], task["c1_true"]], dtype=t.dtype, device=t.device)
    lam_true = torch.tensor(
        [task["lambda0_true"], task["lambda1_true"]],
        dtype=t.dtype,
        device=t.device,
    )
    x_star = torch.cat([c_true, lam_true])
    observation_rng = np.random.default_rng(int(task["seed"]))
    g_true, y = make_synthetic_observation(
        c_true,
        lam_true,
        t,
        task["noise_type"],
        float(experiment["sigma"]),
        rng=observation_rng,
    )

    start_time = time.perf_counter()
    hist = run_solver(task["method"], y, t, x_star, solver_config)
    runtime_seconds = time.perf_counter() - start_time

    return make_result_row(
        method=task["method"],
        task=task,
        hist=hist,
        t=t,
        x_star=x_star,
        g_true=g_true,
        runtime_seconds=runtime_seconds,
    )


def run_one_task_safe(task: dict[str, Any]) -> dict[str, Any]:
    """Run a task, converting exceptions into failed result rows."""
    start_time = time.perf_counter()
    try:
        return run_one_task(task)
    except Exception as exc:
        runtime_seconds = time.perf_counter() - start_time
        return make_failed_row(task, exc, runtime_seconds)


def iter_result_rows(
    tasks: list[dict[str, Any]],
    config: dict[str, Any],
):
    """Yield result rows from serial or multiprocessing execution."""
    parallel_config = config.get("parallel") or {}
    enabled = bool(parallel_config.get("enabled", True))
    num_workers = int(parallel_config.get("num_workers", mp.cpu_count()))
    chunksize = int(parallel_config.get("chunksize", 1))
    maxtasksperchild = parallel_config.get("maxtasksperchild")
    if maxtasksperchild is not None:
        maxtasksperchild = int(maxtasksperchild)

    if not enabled or num_workers <= 1:
        init_worker(config)
        for task in tasks:
            yield run_one_task_safe(task)
        return

    with mp.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(config,),
        maxtasksperchild=maxtasksperchild,
    ) as pool:
        yield from pool.imap_unordered(run_one_task_safe, tasks, chunksize=chunksize)


def run_solver(method: str, y: torch.Tensor, t: torch.Tensor, x_star: torch.Tensor, solver_config: dict[str, Any]):
    """Dispatch one configured outer solver."""
    common_kwargs = {
        "mu_init": float(solver_config["mu_init"]),
        "n_steps": int(solver_config["outer_steps"]),
        "lower_max_iter": int(solver_config.get("lower_max_iter", 300)),
        "lower_tol": float(solver_config.get("lower_tol", 1e-9)),
        "progress": bool(solver_config.get("progress", False)),
    }

    if method == "gd":
        return SOLVERS[method](
            y,
            t,
            lr=float(solver_config["gd_lr"]),
            **common_kwargs,
        )
    if method == "gn":
        return SOLVERS[method](y, t, **common_kwargs)
    if method == "ep":
        return SOLVERS[method](
            y,
            t,
            x_star=x_star,
            beta=float(solver_config["ep_beta"]),
            lr=float(solver_config["ep_lr"]),
            **common_kwargs,
        )
    raise ValueError(f"Unknown solver method: {method!r}")


def make_result_row(
    *,
    method: str,
    task: dict[str, Any],
    hist: dict[str, list],
    t: torch.Tensor,
    x_star: torch.Tensor,
    g_true: torch.Tensor,
    runtime_seconds: float,
) -> dict[str, Any]:
    """Build one flat JSON-serializable result row."""
    return {
        "experiment_id": task["experiment_id"],
        "task_index": int(task["task_index"]),
        "job_index": int(task["job_index"]),
        "method": method,
        "noise_type": task["noise_type"],
        "seed": int(task["seed"]),
        "realization": int(task["realization"]),
        "c0_true": x_star[0].item(),
        "c1_true": x_star[1].item(),
        "lambda0_true": x_star[2].item(),
        "lambda1_true": x_star[3].item(),
        "runtime_seconds": float(runtime_seconds),
        "n_outer_iterations": len(hist["mu"]),
        "status": "ok",
        **summarize_solver_history(hist, t, x_star, g_true),
    }


def attach_experiment_id(tasks: list[dict[str, Any]], experiment_id: str) -> list[dict[str, Any]]:
    """Add the experiment ID to each task row without mutating the input list."""
    return [{**task, "experiment_id": experiment_id} for task in tasks]


def print_startup_summary(
    *,
    experiment_id: str,
    experiment_dir: Any,
    total_tasks: int,
    completed_rows: int,
    pending_tasks: int,
    config: dict[str, Any],
) -> None:
    """Print a concise summary before dispatching work."""
    parallel_config = config.get("parallel") or {}
    output = config["output"]
    print(
        f"[exp {experiment_id}] output_dir={experiment_dir} "
        f"total={total_tasks} completed={completed_rows} pending={pending_tasks} "
        f"workers={parallel_config.get('num_workers', mp.cpu_count())} "
        f"jsonl={output['jsonl_path']}"
    )


def format_result_summary(row: dict[str, Any], finished: int, total: int, ok_count: int, failed_count: int) -> str:
    """Create one short parent-process result summary line."""
    prefix = (
        f"[exp {row['experiment_id']}] {finished}/{total} "
        f"ok={ok_count} failed={failed_count} "
        f"method={row['method']} noise={row['noise_type']} "
        f"c1={row['c1_true']:.3g} "
        f"lambda=({row['lambda0_true']:.4g},{row['lambda1_true']:.4g})"
    )
    if row.get("status") == "ok":
        return (
            f"{prefix} final_sol={row['final_solution_error']:.3e} "
            f"final_fwd={row['final_forward_error']:.3e} "
            f"best_sol={row['best_solution_error']:.3e}"
        )
    return f"{prefix} status=failed error={row.get('error_type')}: {row.get('error_message')}"


def print_final_summary(
    *,
    experiment_id: str,
    runtime_seconds: float,
    rows_written: int,
    ok_count: int,
    failed_count: int,
    config: dict[str, Any],
) -> None:
    """Print a concise summary after all rows have been handled."""
    output = config["output"]
    final_json = output["json_path"] if bool(output.get("finalize_json", False)) else "not requested"
    print(
        f"[exp {experiment_id}] finished rows_written={rows_written} "
        f"ok={ok_count} failed={failed_count} runtime={runtime_seconds:.1f}s "
        f"jsonl={output['jsonl_path']} json={final_json}"
    )


def run_simulation(config: dict[str, Any]) -> None:
    """Run the configured sweep and write method-level JSON rows."""
    experiment_id, experiment_dir, config = resolve_experiment(config)
    output = config["output"]
    solver_config = config["solver"]
    tracking = config.get("tracking") or {}

    output_jsonl = output["jsonl_path"]
    resume = bool(output.get("resume", True))
    prepare_jsonl_output(output_jsonl, resume=resume)
    config_snapshot_name = tracking.get("config_snapshot_name", "config.yaml")
    save_config_snapshot(config, experiment_dir / config_snapshot_name)

    methods = list(solver_config.get("methods", SOLVERS.keys()))
    unknown_methods = sorted(set(methods) - set(SOLVERS))
    if unknown_methods:
        raise ValueError(f"Unknown solver methods in config: {unknown_methods}")

    completed = load_completed_keys(output_jsonl) if resume else set()
    all_tasks = attach_experiment_id(make_tasks(config), experiment_id)
    tasks = filter_completed_tasks(all_tasks, completed)
    total_pending = len(tasks)
    summary_every = max(1, int(tracking.get("summary_every", 1)))
    start_time = time.perf_counter()
    rows_written = 0
    ok_count = 0
    failed_count = 0

    print_startup_summary(
        experiment_id=experiment_id,
        experiment_dir=experiment_dir,
        total_tasks=len(all_tasks),
        completed_rows=count_jsonl_rows(output_jsonl) if resume else 0,
        pending_tasks=total_pending,
        config=config,
    )

    for row in iter_result_rows(tasks, config):
        append_jsonl_row(output_jsonl, row)
        completed.add(result_key(row))
        rows_written += 1
        if row.get("status") == "ok":
            ok_count += 1
        else:
            failed_count += 1
        if rows_written % summary_every == 0 or rows_written == total_pending:
            print(format_result_summary(row, rows_written, total_pending, ok_count, failed_count))

    if bool(output.get("finalize_json", False)):
        finalize_json_array(output_jsonl, output["json_path"])
    print_final_summary(
        experiment_id=experiment_id,
        runtime_seconds=time.perf_counter() - start_time,
        rows_written=rows_written,
        ok_count=ok_count,
        failed_count=failed_count,
        config=config,
    )


def main() -> None:
    mp.freeze_support()
    args = parse_args()
    run_simulation(load_config(args.config))


if __name__ == "__main__":
    main()
