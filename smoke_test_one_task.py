# Smoke test: run one task and verify first xhat differs from default_x_init when seeded
import numpy as np
import torch
from utils import load_config
from run_simulation import resolve_experiment, init_worker, get_worker_state, make_tasks
from run_simulation import run_solver
from optimization import default_x_init
from data import make_synthetic_observation


def main():
    config = load_config("experiment_config.yaml")
    exp_id, exp_dir, resolved = resolve_experiment(config)
    init_worker(resolved)
    cfg, t = get_worker_state()

    tasks = make_tasks(resolved)
    if not tasks:
        print("No tasks generated from config; aborting.")
        return
    task = tasks[0]

    c_true = torch.tensor([task["c0_true"], task["c1_true"]], dtype=t.dtype, device=t.device)
    lam_true = torch.tensor([task["lambda0_true"], task["lambda1_true"]], dtype=t.dtype, device=t.device)
    x_star = torch.cat([c_true, lam_true])

    observation_rng = np.random.default_rng(int(task.get("seed", 0)))
    g_true, y = make_synthetic_observation(
        c_true,
        lam_true,
        t,
        task.get("noise_type"),
        float(resolved["experiment"]["sigma"]),
        rng=observation_rng,
    )

    init_seed_offset = int(resolved["experiment"].get("init_seed_offset", 1_000_000))
    init_seed = int(task.get("seed", 0)) + init_seed_offset

    hist = run_solver(task["method"], y, t, x_star, resolved["solver"], init_seed=init_seed)

    first_xhat = hist["xhat"][0].detach().cpu()
    default = default_x_init(dtype=t.dtype, device="cpu").detach().cpu()

    print("task method:", task["method"]) 
    print("init_seed:", init_seed)
    print("default_x_init:", default.numpy())
    print("first_xhat:", first_xhat.numpy())
    print("allclose to default:", torch.allclose(first_xhat, default))


if __name__ == '__main__':
    main()
