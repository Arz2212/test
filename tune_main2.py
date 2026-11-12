#!/usr/bin/env python3
"""Ray Tune + HyperOpt search for main2.py hub parameters.

Search space: hub_power, hub_w_init, tau_hub, tau_m_hab, v_thr_hab,
w_decay.
One trial = 1 training epoch on MNIST, then test accuracy.
Objective: minimize 1 - accuracy.

Run (A100):
    uv run --active python tune_main2.py --samples 1000 --device cuda

Quick smoke test:
    uv run --active python tune_main2.py --samples 1 --device cpu \
        --gpus-per-trial 0 --max-train-batches 1 --max-eval-batches 1
"""

from __future__ import annotations

import argparse
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

import main2  # noqa: E402

import ray  # noqa: E402
from hyperopt import hp  # noqa: E402
from ray import tune  # noqa: E402
from ray.tune.search.hyperopt import HyperOptSearch  # noqa: E402

# Baseline values supplied by the user. They are also the first trial.
BASELINE = {
    "hub_power": 1.20313,
    "hub_w_init": 0.0503053,
    "tau_hub": 4.39997,
    "tau_m_hab": 4.87563,
    "v_thr_hab": 0.249441,
    "w_decay": 5e-5,
}

# Ranges around the supplied baseline.
SEARCH_SPACE = {
    "hub_power": hp.uniform("hub_power", 0.8, 2.0),
    "hub_w_init": hp.loguniform("hub_w_init", math.log(0.01), math.log(0.15)),
    "tau_hub": hp.uniform("tau_hub", 2.0, 8.0),
    "tau_m_hab": hp.uniform("tau_m_hab", 2.0, 8.0),
    "v_thr_hab": hp.loguniform("v_thr_hab", math.log(0.08), math.log(0.5)),
    "w_decay": hp.uniform("w_decay", 0.0, 1e-4),
}


def _train_epoch(model, loader, device, max_batches=0):
    model.train()
    correct = total = 0
    for i, (xb, yb) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).float()
        B = xb.shape[0]

        yn = main2.neg_labels(yb, device)
        X_cat = torch.cat([xb, xb], dim=0)
        Y_cat = torch.cat([yb, yn], dim=0)
        mask = torch.cat(
            [torch.ones(B, 1, device=device), torch.zeros(B, 1, device=device)],
            dim=0,
        )

        yh, _, _ = model(X_cat, Y_cat, mask, learn=True)
        correct += (yh[:B].argmax(1) == yb.argmax(1)).sum().item()
        total += B
    return correct / max(total, 1)


@torch.no_grad()
def _evaluate(model, loader, device, max_batches=0):
    model.eval()
    correct = total = 0
    for i, (xb, yb) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        B = xb.shape[0]
        yh, _, _ = model(
            xb,
            torch.zeros(B, main2.out_dim, device=device),
            torch.ones(B, 1, device=device),
            learn=False,
        )
        correct += (yh.argmax(1) == yb.argmax(1)).sum().item()
        total += B
    return correct / max(total, 1)


def objective(config, data_refs, device_str, epochs, max_train_batches,
              max_eval_batches, zmax_init):
    # main2.py keeps these as module globals.
    main2.hub_power = float(config["hub_power"])
    main2.hub_w_init = float(config["hub_w_init"])
    main2.tau_hub = float(config["tau_hub"])
    main2.tau_m_hab = float(config["tau_m_hab"])
    main2.v_thr_hab = float(config["v_thr_hab"])
    main2.w_decay = float(config["w_decay"])

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    torch.manual_seed(main2.seed)
    np.random.seed(main2.seed)
    torch.set_num_threads(int(os.environ.get("CSDP_THREADS", "1")))

    X_tr = torch.from_numpy(ray.get(data_refs["X_tr"]).copy())
    Y_tr = torch.from_numpy(ray.get(data_refs["Y_tr"]).copy())
    X_te = torch.from_numpy(ray.get(data_refs["X_te"]).copy())
    Y_te = torch.from_numpy(ray.get(data_refs["Y_te"]).copy())

    pin = device.type == "cuda"
    tr_ld = DataLoader(
        TensorDataset(X_tr, Y_tr),
        batch_size=main2.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=pin,
    )
    te_ld = DataLoader(
        TensorDataset(X_te, Y_te),
        batch_size=main2.dev_batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=pin,
    )

    model = main2.CSDPSNN(device).to(device)
    # main2 starts zmax at 0; protect against 0/0 in hub_mod.
    if zmax_init > 0:
        model.zmax1.fill_(zmax_init)
        model.zmax2.fill_(zmax_init)

    for _ in range(epochs):
        _train_epoch(model, tr_ld, device, max_batches=max_train_batches)

    acc = _evaluate(model, te_ld, device, max_batches=max_eval_batches)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {"1-acc": 1.0 - acc, "accuracy": acc}


def main():
    p = argparse.ArgumentParser(
        description="Ray Tune + HyperOpt search for main2.py hub parameters"
    )
    p.add_argument("--samples", type=int, default=1000, help="number of trials")
    p.add_argument("--epochs", type=int, default=1, help="epochs per trial")
    p.add_argument("--max-concurrent", type=int, default=10,
                   help="max parallel trials; 0 = auto")
    p.add_argument("--initial-points", type=int, default=100)
    p.add_argument("--gpus-per-trial", type=float, default=0.1,
                   help="GPU per trial; 0.1 => 10 trials on one GPU")
    p.add_argument("--cpus-per-trial", type=int, default=1)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--data-dir", default=str(main2.data_dir))
    p.add_argument("--max-train-batches", type=int, default=0,
                   help="0 = full epoch; useful for smoke tests")
    p.add_argument("--max-eval-batches", type=int, default=0,
                   help="0 = full test set; useful for smoke tests")
    p.add_argument("--zmax-init", type=float, default=1.0,
                   help="initial zmax; 0 keeps main2's original behavior")
    args = p.parse_args()

    os.environ.setdefault("CSDP_THREADS", "1")

    data_dir = Path(args.data_dir).expanduser().resolve()
    required = ("trainX.npy", "trainY.npy", "testX.npy", "testY.npy")
    if not all((data_dir / name).exists() for name in required):
        from download_mnist import ensure_mnist
        ensure_mnist(data_dir)

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
    max_concurrent = args.max_concurrent if args.max_concurrent > 0 else default_concurrent

    print(
        f"CPU: {n_cpus} | GPU: {n_gpus} | device: {args.device} | "
        f"gpu/trial: {gpus_per_trial} | cpu/trial: {cpus_per_trial} | "
        f"concurrent: {max_concurrent} | trials: {args.samples}"
    )

    python_path = str(SCRIPT_DIR)
    if os.environ.get("PYTHONPATH"):
        python_path += ":" + os.environ["PYTHONPATH"]

    runtime_env = {
        "env_vars": {"PYTHONPATH": python_path},
        # Do not upload MNIST/checkpoints to every Ray worker.
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
        num_cpus=n_cpus,
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
        points_to_evaluate=[BASELINE],
        n_initial_points=args.initial_points,
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                objective,
                data_refs=data_refs,
                device_str=args.device,
                epochs=args.epochs,
                max_train_batches=args.max_train_batches,
                max_eval_batches=args.max_eval_batches,
                zmax_init=args.zmax_init,
            ),
            {"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        ),
        tune_config=tune.TuneConfig(
            search_alg=algo,
            num_samples=args.samples,
            max_concurrent_trials=max_concurrent,
        ),
        run_config=tune.RunConfig(
            name="mnist_main2_hub_hyperopt",
            storage_path=str(SCRIPT_DIR / "ray_results"),
            verbose=1,
        ),
    )

    t0 = time.time()
    results = tuner.fit()
    elapsed = time.time() - t0

    best = results.get_best_result(metric="1-acc", mode="min")
    print("\n" + "=" * 64)
    print(f"Done in {elapsed / 60:.1f} min; trials: {args.samples}")
    print("Best hub configuration:")
    for key, value in best.config.items():
        print(f"  {key:12s} = {value:.8g}")
    print(f"Best test accuracy: {best.metrics['accuracy']:.4f}")
    print(f"Best 1-acc:         {best.metrics['1-acc']:.4f}")
    print("=" * 64)

    ray.shutdown()


if __name__ == "__main__":
    main()
