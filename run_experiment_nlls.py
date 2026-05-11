"""Run the direct constrained NLLS biexponential recovery sweep."""
# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
import traceback
from itertools import product
from pathlib import Path
from typing import Any

import torch

from data import (
    biexponential_three_param,
    biexponential_three_param_T2,
    make_lambda_grid,
    make_synthetic_observation_three_param,
    make_synthetic_observation_three_param_T2,
    make_time_grid,
)
from optimization import solve_direct_nlls
from utils import (
    append_jsonl_row,
    count_jsonl_rows,
    finalize_json_array,
    load_config,
    prepare_jsonl_output,
    resolve_experiment,
    save_config_snapshot,
)

WORKER_CONFIG: dict[str, Any] | None = None
WORKER_T: torch.Tensor | None = None


def parse_args() -> argparse.Namespace:
    """Parse the config path. All experiment options live in YAML."""
    parser = argparse.ArgumentParser(description="Run the direct NLLS recovery sweep.")
    parser.add_argument("--config", required=True, help="Path to the YAML simulation config.")
    return parser.parse_args()


def direct_result_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Return the resume key for a direct-NLLS output row."""
    use_inverses = bool(row.get("use_inverses", False))
    if "param1_true" in row and "param2_true" in row:
        param1_true = row["param1_true"]
        param2_true = row["param2_true"]
    elif use_inverses:
        param1_true = row["T21_true"]
        param2_true = row["T22_true"]
    else:
        param1_true = row["lambda1_true"]
        param2_true = row["lambda2_true"]
    return (
        use_inverses,
        float(row["c1_true"]),
        float(param1_true),
        float(param2_true),
        int(row["realization"]),
        int(row["init_index"]),
    )


def make_direct_result_key(
    *,
    use_inverses: bool,
    c1_true: float,
    param1_true: float,
    param2_true: float,
    realization: int,
    init_index: int,
) -> tuple[Any, ...]:
    """Build a resume key before a direct-NLLS task has run."""
    return (
        bool(use_inverses),
        float(c1_true),
        float(param1_true),
        float(param2_true),
        int(realization),
        int(init_index),
    )


def load_completed_direct_keys(output_path: str | Path) -> set[tuple[Any, ...]]:
    """Load completed direct-NLLS row keys from an append-only JSONL result file."""
    output_path = Path(output_path)
    if not output_path.exists():
        return set()

    completed = set()
    with output_path.open("r") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                completed.add(direct_result_key(json.loads(line)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {output_path}") from exc
    return completed


def use_inverses(config: dict[str, Any]) -> bool:
    """Return whether this run optimizes T2 parameters instead of lambdas."""
    return bool(config["experiment"].get("use_inverses", False))


def parameterization(config_or_flag: dict[str, Any] | bool) -> str:
    """Return the active parameterization label."""
    flag = use_inverses(config_or_flag) if isinstance(config_or_flag, dict) else bool(config_or_flag)
    return "T2" if flag else "lambda"


def parameter_field_names(config_or_flag: dict[str, Any] | bool) -> tuple[str, str]:
    """Return the active output field roots."""
    flag = use_inverses(config_or_flag) if isinstance(config_or_flag, dict) else bool(config_or_flag)
    return ("T21", "T22") if flag else ("lambda1", "lambda2")


def parameter_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the active ordered-pair range config."""
    return config["T2"] if use_inverses(config) else config["lambda"]


def make_parameter_pairs(config: dict[str, Any]) -> list[tuple[float, float]]:
    """Build all ordered parameter pairs from a shared grid with p1 <= p2."""
    values = make_lambda_grid(parameter_config(config)).tolist()
    return [(float(param1), float(param2)) for i, param1 in enumerate(values) for param2 in values[i:]]


def make_lambda_pairs(config: dict[str, Any]) -> list[tuple[float, float]]:
    """Build all lambda pairs from a shared grid with lambda1 <= lambda2."""
    return make_parameter_pairs(config)


def make_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Enumerate one task per true triplet, noisy realization, and random initialization."""
    experiment = config["experiment"]
    n_realizations = int(experiment["n_realizations"])
    n_initializations = int(experiment["n_initializations"])
    base_seed = int(experiment.get("base_seed", 0))
    init_seed_offset = int(experiment.get("init_seed_offset", 1_000_000))
    inverse_mode = use_inverses(config)
    param1_name, param2_name = parameter_field_names(inverse_mode)
    param_pairs = make_parameter_pairs(config)

    tasks = []
    task_index = 0
    job_index = 0
    for c1_true, (param1_true, param2_true), realization in product(
        experiment["c1_values"],
        param_pairs,
        range(n_realizations),
    ):
        observation_seed = base_seed + job_index
        for init_index in range(n_initializations):
            task = {
                "task_index": task_index,
                "job_index": job_index,
                "use_inverses": inverse_mode,
                "parameterization": parameterization(inverse_mode),
                "c1_true": float(c1_true),
                "param1_true": float(param1_true),
                "param2_true": float(param2_true),
                f"{param1_name}_true": float(param1_true),
                f"{param2_name}_true": float(param2_true),
                "realization": int(realization),
                "init_index": int(init_index),
                "observation_seed": int(observation_seed),
                "init_seed": int(base_seed + init_seed_offset + task_index),
            }
            tasks.append(task)
            task_index += 1
        job_index += 1
    return tasks


def task_result_key(task: dict[str, Any]) -> tuple[Any, ...]:
    """Return the resume key for a task before it has run."""
    return make_direct_result_key(
        use_inverses=task["use_inverses"],
        c1_true=task["c1_true"],
        param1_true=task["param1_true"],
        param2_true=task["param2_true"],
        realization=task["realization"],
        init_index=task["init_index"],
    )


def filter_completed_tasks(
    tasks: list[dict[str, Any]],
    completed: set[tuple[Any, ...]],
) -> list[dict[str, Any]]:
    """Drop tasks whose output rows already exist in the JSONL file."""
    return [task for task in tasks if task_result_key(task) not in completed]


def attach_experiment_id(tasks: list[dict[str, Any]], experiment_id: str) -> list[dict[str, Any]]:
    """Add the experiment ID to each task row without mutating the input list."""
    return [{**task, "experiment_id": experiment_id} for task in tasks]


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
    param1_name, param2_name = parameter_field_names(task["use_inverses"])
    row = {
        "experiment_id": task["experiment_id"],
        "task_index": int(task["task_index"]),
        "job_index": int(task["job_index"]),
        "use_inverses": bool(task["use_inverses"]),
        "parameterization": task["parameterization"],
        "c1_true": float(task["c1_true"]),
        "param1_true": float(task["param1_true"]),
        "param2_true": float(task["param2_true"]),
        "realization": int(task["realization"]),
        "init_index": int(task["init_index"]),
        "observation_seed": int(task["observation_seed"]),
        "init_seed": int(task["init_seed"]),
        "runtime_seconds": float(runtime_seconds),
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
    }
    row[f"{param1_name}_true"] = float(task["param1_true"])
    row[f"{param2_name}_true"] = float(task["param2_true"])
    return row


def run_one_task(task: dict[str, Any]) -> dict[str, Any]:
    """Run one direct-NLLS start and return one output row."""
    config, t = get_worker_state()
    experiment = config["experiment"]
    solver_config = config["solver"]
    param_config = parameter_config(config)
    sigma = float(experiment["sigma"])

    if task["use_inverses"]:
        g_true, y = make_synthetic_observation_three_param_T2(
            task["c1_true"],
            task["param1_true"],
            task["param2_true"],
            t,
            sigma,
            seed=int(task["observation_seed"]),
        )
    else:
        g_true, y = make_synthetic_observation_three_param(
            task["c1_true"],
            task["param1_true"],
            task["param2_true"],
            t,
            sigma,
            seed=int(task["observation_seed"]),
        )

    start_time = time.perf_counter()
    fit = solve_direct_nlls(
        y,
        t,
        seed=int(task["init_seed"]),
        lambda_min=float(param_config["min"]),
        lambda_max=float(param_config["max"]),
        max_nfev=int(solver_config.get("max_nfev", 300)),
        ftol=float(solver_config.get("ftol", 1e-10)),
        xtol=float(solver_config.get("xtol", 1e-10)),
        gtol=float(solver_config.get("gtol", 1e-10)),
        use_inverses=bool(task["use_inverses"]),
    )
    runtime_seconds = time.perf_counter() - start_time

    return make_result_row(
        task=task,
        fit=fit,
        t=t,
        g_true=g_true,
        sigma=sigma,
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


def make_result_row(
    *,
    task: dict[str, Any],
    fit: dict[str, Any],
    t: torch.Tensor,
    g_true: torch.Tensor,
    sigma: float,
    runtime_seconds: float,
) -> dict[str, Any]:
    """Build one flat JSON-serializable per-start result row."""
    x_init = fit["x_init"]
    x_hat = fit["x_hat"]
    inverse_mode = bool(task["use_inverses"])
    param1_name, param2_name = parameter_field_names(inverse_mode)
    if inverse_mode:
        g_hat = biexponential_three_param_T2(t, x_hat[0], x_hat[1], x_hat[2]).detach()
    else:
        g_hat = biexponential_three_param(t, x_hat[0], x_hat[1], x_hat[2]).detach()
    x_true = torch.tensor(
        [task["c1_true"], task["param1_true"], task["param2_true"]],
        dtype=t.dtype,
        device=t.device,
    )
    x_hat_t = torch.tensor(x_hat, dtype=t.dtype, device=t.device)
    parameter_rmse = ((x_hat_t - x_true) ** 2).mean().sqrt().item()
    forward_rmse = ((g_hat - g_true) ** 2).mean().sqrt().item()

    row = {
        "experiment_id": task["experiment_id"],
        "task_index": int(task["task_index"]),
        "job_index": int(task["job_index"]),
        "use_inverses": inverse_mode,
        "parameterization": task["parameterization"],
        "c1_true": float(task["c1_true"]),
        "param1_true": float(task["param1_true"]),
        "param2_true": float(task["param2_true"]),
        "sigma": float(sigma),
        "realization": int(task["realization"]),
        "init_index": int(task["init_index"]),
        "observation_seed": int(task["observation_seed"]),
        "init_seed": int(task["init_seed"]),
        "c1_init": float(x_init[0]),
        "param1_init": float(x_init[1]),
        "param2_init": float(x_init[2]),
        "runtime_seconds": float(runtime_seconds),
        "status": "ok",
        "solver_success": bool(fit["success"]),
        "solver_status": int(fit["status"]),
        "solver_message": str(fit["message"]),
        "nfev": int(fit["nfev"]),
        "njev": int(fit["njev"]) if fit["njev"] is not None else None,
        "optimality": float(fit["optimality"]),
        "active_mask": fit["active_mask"],
        "c1_hat": float(x_hat[0]),
        "param1_hat": float(x_hat[1]),
        "param2_hat": float(x_hat[2]),
        "rss": float(fit["rss"]),
        "cost": float(fit["cost"]),
        "parameter_rmse": float(parameter_rmse),
        "forward_rmse": float(forward_rmse),
    }
    row[f"{param1_name}_true"] = float(task["param1_true"])
    row[f"{param2_name}_true"] = float(task["param2_true"])
    row[f"{param1_name}_init"] = float(x_init[1])
    row[f"{param2_name}_init"] = float(x_init[2])
    row[f"{param1_name}_hat"] = float(x_hat[1])
    row[f"{param2_name}_hat"] = float(x_hat[2])
    return row


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


def print_startup_summary(
    *,
    experiment_id: str,
    experiment_dir: Path,
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
    param_label = row.get("parameterization", "lambda")
    prefix = (
        f"[exp {row['experiment_id']}] {finished}/{total} "
        f"ok={ok_count} failed={failed_count} "
        f"c1={row['c1_true']:.3g} "
        f"{param_label}=({row['param1_true']:.4g},{row['param2_true']:.4g}) "
        f"realization={row['realization']} init={row['init_index']}"
    )
    if row.get("status") == "ok":
        return (
            f"{prefix} rss={row['rss']:.3e} "
            f"param_rmse={row['parameter_rmse']:.3e} "
            f"fwd_rmse={row['forward_rmse']:.3e}"
        )
    return f"{prefix} status=failed error={row.get('error_type')}: {row.get('error_message')}"


def write_best_summary(jsonl_path: str | Path, best_path: str | Path) -> int:
    """Write one best-by-RSS row per true triplet and noisy realization."""
    best: dict[tuple[Any, ...], dict[str, Any]] = {}
    jsonl_path = Path(jsonl_path)
    best_path = Path(best_path)
    if not jsonl_path.exists():
        best_path.parent.mkdir(parents=True, exist_ok=True)
        best_path.write_text("[]\n")
        return 0

    with jsonl_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            if "param1_true" in row and "param2_true" in row:
                param1_true = row["param1_true"]
                param2_true = row["param2_true"]
            elif row.get("use_inverses", False):
                param1_true = row["T21_true"]
                param2_true = row["T22_true"]
            else:
                param1_true = row["lambda1_true"]
                param2_true = row["lambda2_true"]
            key = (
                row.get("use_inverses", False),
                row["c1_true"],
                param1_true,
                param2_true,
                row["realization"],
            )
            if key not in best or row["rss"] < best[key]["rss"]:
                best[key] = row

    rows = [best[key] for key in sorted(best)]
    best_path.parent.mkdir(parents=True, exist_ok=True)
    with best_path.open("w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
    return len(rows)


def print_final_summary(
    *,
    experiment_id: str,
    runtime_seconds: float,
    rows_written: int,
    ok_count: int,
    failed_count: int,
    best_rows: int,
    config: dict[str, Any],
) -> None:
    """Print a concise summary after all rows have been handled."""
    output = config["output"]
    final_json = output["json_path"] if bool(output.get("finalize_json", False)) else "not requested"
    best_json = output.get("best_json_path", "not requested")
    print(
        f"[exp {experiment_id}] finished rows_written={rows_written} "
        f"ok={ok_count} failed={failed_count} best_rows={best_rows} "
        f"runtime={runtime_seconds:.1f}s jsonl={output['jsonl_path']} "
        f"json={final_json} best_json={best_json}"
    )


def run_experiment_nlls(config: dict[str, Any]) -> None:
    """Run the configured direct-NLLS sweep and write per-start JSON rows."""
    experiment_id, experiment_dir, config = resolve_experiment(config)
    output = config["output"]
    tracking = config.get("tracking") or {}

    output_jsonl = output["jsonl_path"]
    output["best_json_path"] = str(experiment_dir / output.get("best_json_name", "nlls_best.json"))
    resume = bool(output.get("resume", True))
    prepare_jsonl_output(output_jsonl, resume=resume)
    config_snapshot_name = tracking.get("config_snapshot_name", "config.yaml")
    save_config_snapshot(config, experiment_dir / config_snapshot_name)

    completed = load_completed_direct_keys(output_jsonl) if resume else set()
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
        completed.add(direct_result_key(row))
        rows_written += 1
        if row.get("status") == "ok":
            ok_count += 1
        else:
            failed_count += 1
        if rows_written % summary_every == 0 or rows_written == total_pending:
            print(format_result_summary(row, rows_written, total_pending, ok_count, failed_count))

    if bool(output.get("finalize_json", False)):
        finalize_json_array(output_jsonl, output["json_path"])
    best_rows = write_best_summary(output_jsonl, output["best_json_path"])
    print_final_summary(
        experiment_id=experiment_id,
        runtime_seconds=time.perf_counter() - start_time,
        rows_written=rows_written,
        ok_count=ok_count,
        failed_count=failed_count,
        best_rows=best_rows,
        config=config,
    )


def main() -> None:
    mp.freeze_support()
    args = parse_args()
    run_experiment_nlls(load_config(args.config))


if __name__ == "__main__":
    main()
