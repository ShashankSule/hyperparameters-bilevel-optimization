"""Utility helpers for simulation config, metrics, and JSON result storage."""

from __future__ import annotations

import json
import importlib
import secrets
from copy import deepcopy
from pathlib import Path
from typing import Any

from data import biexponential


RESULT_KEY_FIELDS = (
    "c1_true",
    "lambda0_true",
    "lambda1_true",
    "noise_type",
    "realization",
    "method",
)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML simulation config."""
    yaml = import_yaml()

    with Path(config_path).open("r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return config


def import_yaml():
    """Import PyYAML with a clear error message for the active environment."""
    try:
        return importlib.import_module("yaml")
    except ImportError as exc:  # pragma: no cover - depends on the active environment.
        raise ImportError(
            "PyYAML is required to read config files. Install it with `pip install pyyaml`."
        ) from exc


def generate_experiment_id(n_bytes: int = 8) -> str:
    """Generate a unique hex experiment ID."""
    return secrets.token_hex(n_bytes)


def resolve_experiment(config: dict[str, Any]) -> tuple[str, Path, dict[str, Any]]:
    """
    Resolve experiment tracking paths and return a config copy.

    New runs get a fresh hex ID. Runs with tracking.experiment_id set reuse that
    ID and therefore resume from the same experiment directory.
    """
    resolved = deepcopy(config)
    tracking = resolved.setdefault("tracking", {})
    output = resolved.setdefault("output", {})

    experiment_id = tracking.get("experiment_id") or generate_experiment_id()
    tracking["experiment_id"] = experiment_id

    root_dir = Path(tracking.get("root_dir", "results/experiments"))
    experiment_dir = root_dir / experiment_id
    tracking["experiment_dir"] = str(experiment_dir)

    jsonl_name = output.get("jsonl_name", "rician_sweep.jsonl")
    json_name = output.get("json_name", "rician_sweep.json")
    output["jsonl_path"] = str(experiment_dir / jsonl_name)
    output["json_path"] = str(experiment_dir / json_name)

    return experiment_id, experiment_dir, resolved


def save_config_snapshot(config: dict[str, Any], output_path: str | Path) -> None:
    """Save a YAML snapshot of the resolved experiment config."""
    yaml = import_yaml()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def count_jsonl_rows(output_path: str | Path) -> int:
    """Count non-empty rows in a JSONL file."""
    output_path = Path(output_path)
    if not output_path.exists():
        return 0
    with output_path.open("r") as f:
        return sum(1 for line in f if line.strip())


def result_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Return the unique key for a completed method-level result row."""
    return tuple(row[field] for field in RESULT_KEY_FIELDS)


def make_result_key(
    *,
    c1_true: float,
    lambda0_true: float,
    lambda1_true: float,
    noise_type: str,
    realization: int,
    method: str,
) -> tuple[Any, ...]:
    """Build a result key before the full result row exists."""
    return (
        float(c1_true),
        float(lambda0_true),
        float(lambda1_true),
        noise_type,
        int(realization),
        method,
    )


def load_completed_keys(output_path: str | Path) -> set[tuple[Any, ...]]:
    """Load completed row keys from an append-only JSONL result file."""
    output_path = Path(output_path)
    if not output_path.exists():
        return set()

    completed = set()
    with output_path.open("r") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                completed.add(result_key(json.loads(line)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {output_path}") from exc
    return completed


def append_jsonl_row(output_path: str | Path, row: dict[str, Any]) -> None:
    """Append one table row to a newline-delimited JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def prepare_jsonl_output(output_path: str | Path, *, resume: bool) -> None:
    """Ensure the output directory exists, and truncate stale output when not resuming."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        output_path.write_text("")


def finalize_json_array(jsonl_path: str | Path, json_path: str | Path) -> None:
    """Materialize an append-only JSONL file as one JSON array file."""
    jsonl_path = Path(jsonl_path)
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with jsonl_path.open("r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    with json_path.open("w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)


def compute_iteration_metrics(
    hist: dict[str, list],
    t: Any,
    x_star: Any,
    g_true: Any,
) -> list[dict[str, float | int]]:
    """Compute parameter and forward recovery metrics for each outer iteration."""
    rows = []
    x_star = x_star.to(dtype=t.dtype, device=t.device)
    g_true = g_true.to(dtype=t.dtype, device=t.device)

    for iteration, (mu_k, v_k, xhat_k) in enumerate(zip(hist["mu"], hist["V"], hist["xhat"])):
        xhat_k = xhat_k.detach().to(dtype=t.dtype, device=t.device)
        g_hat_k = biexponential(t, xhat_k[:2], xhat_k[2:]).detach()

        solution_error = ((xhat_k - x_star) ** 2).mean().sqrt().item()
        forward_error = ((g_hat_k - g_true) ** 2).mean().sqrt().item()

        rows.append(
            {
                "iter": iteration,
                "mu": float(mu_k),
                "V": float(v_k),
                "c0_hat": xhat_k[0].item(),
                "c1_hat": xhat_k[1].item(),
                "lambda0_hat": xhat_k[2].item(),
                "lambda1_hat": xhat_k[3].item(),
                "solution_error": solution_error,
                "forward_error": forward_error,
            }
        )
    return rows


def summarize_solver_history(
    hist: dict[str, list],
    t: Any,
    x_star: Any,
    g_true: Any,
) -> dict[str, float | int]:
    """Summarize final metrics and best-by-solution-error metrics for one solver."""
    metrics = compute_iteration_metrics(hist, t, x_star, g_true)
    if not metrics:
        raise ValueError("Solver history is empty; cannot summarize metrics.")

    final = metrics[-1]
    best = min(metrics, key=lambda row: row["solution_error"])

    return {
        "final_iter": final["iter"],
        "final_mu": final["mu"],
        "final_V": final["V"],
        "final_c0_hat": final["c0_hat"],
        "final_c1_hat": final["c1_hat"],
        "final_lambda0_hat": final["lambda0_hat"],
        "final_lambda1_hat": final["lambda1_hat"],
        "final_solution_error": final["solution_error"],
        "final_forward_error": final["forward_error"],
        "best_solution_iter": best["iter"],
        "best_solution_mu": best["mu"],
        "best_solution_V": best["V"],
        "best_solution_c0_hat": best["c0_hat"],
        "best_solution_c1_hat": best["c1_hat"],
        "best_solution_lambda0_hat": best["lambda0_hat"],
        "best_solution_lambda1_hat": best["lambda1_hat"],
        "best_solution_error": best["solution_error"],
        "best_solution_forward_error": best["forward_error"],
    }


__all__ = [
    "append_jsonl_row",
    "compute_iteration_metrics",
    "finalize_json_array",
    "count_jsonl_rows",
    "generate_experiment_id",
    "load_completed_keys",
    "load_config",
    "make_result_key",
    "prepare_jsonl_output",
    "result_key",
    "resolve_experiment",
    "save_config_snapshot",
    "summarize_solver_history",
]
