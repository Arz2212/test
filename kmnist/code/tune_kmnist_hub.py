#!/usr/bin/env python3
"""Ray Tune + HyperOpt search for ONLY hub neuron parameters in kmnist_solver.py.

Search space (hub neurons only):
  tau_m_hab  = 4.87563
  v_thr_hab  = 0.249441
  tau_hub    = 4.39997
  hub_power  = 1.20313
  hub_w_init = 0.0503053

All other parameters in kmnist_solver.py remain fixed at their default values.
One trial = 1 training epoch on KMNIST (exact same training as kmnist_solver.py),
then test accuracy.
Objective: minimize 1 - accuracy.

Run (A100, 100 samples):
    python3 tune_kmnist_hub.py --samples 100 --device cuda
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import kmnist_solver  # noqa: E402

import ray  # noqa: E402
from hyperopt import hp  # noqa: E402
from ray import tune  # noqa: E402
from ray.tune.search.hyperopt import HyperOptSearch  # noqa: E402

# Baseline hub neuron parameters from kmnist_solver.py
BASELINE = {
    "tau_m_hab": 4.87563,
    "v_thr_hab": 0.249441,
    "tau_hub": 4.39997,
    "hub_power": 1.20313,
    "hub_w_init": 0.0503053,
}

INITIAL_POINTS_TO_EVALUATE = [
    BASELINE,
]

# Search space for ONLY the 5 hub neuron parameters
SEARCH_SPACE = {
    "tau_m_hab": hp.uniform("tau_m_hab", 2.0, 14.0),
    "v_thr_hab": hp.loguniform("v_thr_hab", math.log(0.05), math.log(1.0)),
    "tau_hub": hp.uniform("tau_hub", 1.2, 12.0),
    "hub_power": hp.uniform("hub_power", 0.6, 2.5),
    "hub_w_init": hp.loguniform("hub_w_init", math.log(0.008), math.log(0.25)),
}


def apply_config_to_solver(config: dict) -> None:
    """Update only the hub neuron parameters on kmnist_solver."""
    kmnist_solver.tau_m_hab = float(config["tau_m_hab"])
    kmnist_solver.v_thr_hab = float(config["v_thr_hab"])
    kmnist_solver.tau_hub = float(config["tau_hub"])
    kmnist_solver.hub_power = float(config["hub_power"])
    kmnist_solver.hub_w_init = float(config["hub_w_init"])


class BestProgressCallback(tune.Callback):
    def __init__(self, total_samples: int, t0: float):
        self.total_samples = total_samples
        self.t0 = t0
        self.done = 0
        self.best_acc = -1.0
        self.best_config = None

    def on_trial_result(self, iteration, trials, trial, result, **info):
        self.done += 1
        acc = float(result.get("accuracy", 0.0))
        tr_acc = float(result.get("train_acc", 0.0))
        elapsed_m = (time.time() - self.t0) / 60.0
        is_new_best = acc > self.best_acc
        if is_new_best:
            self.best_acc = acc
            self.best_config = dict(trial.config)
            cfg_str = ", ".join(f"{k}={v:.6g}" for k, v in self.best_config.items())
            print(
                f"[NEW BEST] Trial {self.done:3d}/{self.total_samples} | "
                f"test_acc={acc:.4f} (train={tr_acc:.4f}) | [{elapsed_m:.1f}m] | {cfg_str}",
                flush=True,
            )
        elif self.done % 5 == 0 or self.done == self.total_samples:
            print(
                f"[Progress] Trial {self.done:3d}/{self.total_samples} | "
                f"last_acc={acc:.4f} (train={tr_acc:.4f}) | best_acc={self.best_acc:.4f} | [{elapsed_m:.1f}m]",
                flush=True,
            )


def objective(config, data_refs, device_str, epochs):
    apply_config_to_solver(config)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    torch.manual_seed(kmnist_solver.seed)
    np.random.seed(kmnist_solver.seed)
    torch.set_num_threads(int(os.environ.get("CSDP_THREADS", "1")))

    X_tr = torch.from_numpy(ray.get(data_refs["X_tr"]).copy()).float()
    Y_tr = torch.from_numpy(ray.get(data_refs["Y_tr"]).copy()).float()
    X_te = torch.from_numpy(ray.get(data_refs["X_te"]).copy()).float()
    Y_te = torch.from_numpy(ray.get(data_refs["Y_te"]).copy()).float()

    tr_ld = DataLoader(
        TensorDataset(X_tr, Y_tr),
        batch_size=kmnist_solver.batch_size,
        shuffle=True,
        drop_last=True,
    )
    te_ld = DataLoader(
        TensorDataset(X_te, Y_te),
        batch_size=kmnist_solver.dev_batch_size,
        shuffle=False,
        drop_last=True,
    )

    model = kmnist_solver.CSDPSNN(device).to(device)

    tr_acc = 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(epochs):
            tr_acc = kmnist_solver.train_epoch(model, tr_ld, device)

    acc = kmnist_solver.evaluate(model, te_ld, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {"1-acc": 1.0 - acc, "accuracy": acc, "train_acc": tr_acc}


def main():
    p = argparse.ArgumentParser(
        description="Ray Tune + HyperOpt search for hub neuron parameters in kmnist_solver.py"
    )
    p.add_argument("--samples", type=int, default=100, help="number of trials")
    p.add_argument("--epochs", type=int, default=1, help="epochs per trial")
    p.add_argument(
        "--train-best-epochs",
        type=int,
        default=0,
        help="after tuning finishes, optionally run kmnist_solver for N epochs with best config",
    )
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="max parallel trials; 0 = auto",
    )
    p.add_argument("--initial-points", type=int, default=20)
    p.add_argument(
        "--gpus-per-trial",
        type=float,
        default=0.1,
        help="GPU per trial; 0.1 => 10 trials on one GPU",
    )
    p.add_argument("--cpus-per-trial", type=int, default=1)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--data-dir", default=str(SCRIPT_DIR.parents[1] / "data" / "kmnist"))
    args = p.parse_args()

    os.environ.setdefault("CSDP_THREADS", "1")

    data_dir = Path(args.data_dir).expanduser().resolve()
    required = ("trainX.npy", "trainY.npy", "testX.npy", "testY.npy")
    if not all((data_dir / name).exists() for name in required):
        from download_kmnist import ensure_kmnist

        ensure_kmnist(data_dir)

    n_cpus = int(os.environ.get("RAY_NUM_CPUS", os.cpu_count() or 1))
    n_gpus = int(
        os.environ.get(
            "RAY_NUM_GPUS",
            torch.cuda.device_count() if args.device != "cpu" else 0,
        )
    )
    gpus_per_trial = args.gpus_per_trial if n_gpus > 0 else 0.0
    cpus_per_trial = max(1, args.cpus_per_trial)

    by_cpu = max(1, n_cpus // cpus_per_trial)
    if gpus_per_trial > 0:
        by_gpu = max(1, int(n_gpus / gpus_per_trial))
        default_concurrent = max(1, min(by_gpu, by_cpu))
    else:
        default_concurrent = by_cpu
    max_concurrent = (
        args.max_concurrent if args.max_concurrent > 0 else default_concurrent
    )

    print(
        f"CPU: {n_cpus} | GPU: {n_gpus} | device: {args.device} | "
        f"gpu/trial: {gpus_per_trial} | cpu/trial: {cpus_per_trial} | "
        f"concurrent: {max_concurrent} | trials: {args.samples}",
        flush=True,
    )
    print(
        "Fixed parameters from kmnist_solver.py: "
        f"T={kmnist_solver.T}, dt={kmnist_solver.dt}, tau_m={kmnist_solver.tau_m}, "
        f"v_thr={kmnist_solver.v_thr}, tau_tr={kmnist_solver.tau_tr}, R_m={kmnist_solver.R_m}, "
        f"inh_R={kmnist_solver.inh_R}, eta={kmnist_solver.eta}, "
        f"weight_bound={kmnist_solver.weight_bound}, seed={kmnist_solver.seed}",
        flush=True,
    )

    python_path = str(SCRIPT_DIR)
    if os.environ.get("PYTHONPATH"):
        python_path += ":" + os.environ["PYTHONPATH"]

    runtime_env = {
        "env_vars": {"PYTHONPATH": python_path},
        "excludes": [
            "data",
            "ray_results",
            "__pycache__",
            "**/__pycache__",
            "*.pyc",
            ".git",
            "**/.git",
        ],
    }

    ray.init(
        num_cpus=max(n_cpus, max_concurrent * cpus_per_trial),
        num_gpus=n_gpus,
        ignore_reinit_error=True,
        include_dashboard=os.environ.get("RAY_DASHBOARD", "0") == "1",
        runtime_env=runtime_env,
    )

    data_refs = {
        "X_tr": ray.put(np.load(data_dir / "trainX.npy").astype(np.float32)),
        "Y_tr": ray.put(np.load(data_dir / "trainY.npy").astype(np.float32)),
        "X_te": ray.put(np.load(data_dir / "testX.npy").astype(np.float32)),
        "Y_te": ray.put(np.load(data_dir / "testY.npy").astype(np.float32)),
    }

    algo = HyperOptSearch(
        SEARCH_SPACE,
        metric="1-acc",
        mode="min",
        points_to_evaluate=INITIAL_POINTS_TO_EVALUATE,
        n_initial_points=args.initial_points,
        random_state_seed=kmnist_solver.seed,
    )

    t0 = time.time()
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                objective,
                data_refs=data_refs,
                device_str=args.device,
                epochs=args.epochs,
            ),
            {"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        ),
        tune_config=tune.TuneConfig(
            search_alg=algo,
            num_samples=args.samples,
            max_concurrent_trials=max_concurrent,
        ),
        run_config=tune.RunConfig(
            name="kmnist_solver_hub_only_hyperopt",
            storage_path=str(SCRIPT_DIR / "ray_results"),
            verbose=0,
            callbacks=[BestProgressCallback(args.samples, t0)],
        ),
    )

    results = tuner.fit()
    elapsed = time.time() - t0

    best = results.get_best_result(metric="1-acc", mode="min")
    print("\n" + "=" * 64, flush=True)
    print(f"Done in {elapsed / 60:.1f} min; trials: {args.samples}", flush=True)
    print("Best hub neuron configuration:", flush=True)
    for key in ("tau_m_hab", "v_thr_hab", "tau_hub", "hub_power", "hub_w_init"):
        print(f"  {key:12s} = {best.config[key]:.8g}", flush=True)
    print(f"Best train accuracy: {best.metrics.get('train_acc', 0.0):.4f}", flush=True)
    print(f"Best test accuracy:  {best.metrics['accuracy']:.4f}", flush=True)
    print(f"Best 1-acc:          {best.metrics['1-acc']:.4f}", flush=True)
    print("=" * 64, flush=True)

    out_json = SCRIPT_DIR / "kmnist_hub_best_params.json"
    out_json.write_text(
        json.dumps(
            {
                "samples": args.samples,
                "best_1ep_test_accuracy": float(best.metrics["accuracy"]),
                "best_1ep_train_accuracy": float(best.metrics.get("train_acc", 0.0)),
                "parameters": {
                    k: float(best.config[k])
                    for k in ("tau_m_hab", "v_thr_hab", "tau_hub", "hub_power", "hub_w_init")
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    ray.shutdown()

    if args.train_best_epochs > 0:
        print(
            f"\n=== RUNNING {args.train_best_epochs} EPOCHS WITH BEST HUB CONFIGURATION ===",
            flush=True,
        )
        apply_config_to_solver(best.config)
        eval_device = (
            "cuda"
            if (args.device == "auto" and torch.cuda.is_available())
            else args.device
        )
        _, best_ep = kmnist_solver.train_kmnist(
            data_dir=str(data_dir),
            epochs=args.train_best_epochs,
            device=eval_device,
        )
        print(
            f"\n=== FINAL {args.train_best_epochs}-EPOCH BEST TEST ACCURACY: {best_ep:.4f} ===",
            flush=True,
        )


if __name__ == "__main__":
    main()
