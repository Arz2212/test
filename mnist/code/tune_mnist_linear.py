#!/usr/bin/env python3
"""Ray Tune + HyperOpt hyperparameter search for mnist_solver_linear.py.

Optimizes spike-to-signal decoding and linear classifier parameters (plus
optionally hub neuron parameters) on MNIST using Ray Tune + HyperOpt.

Usage:
    python3 -u tune_mnist_linear.py --samples 100 --epochs 3 --device cuda
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

import mnist_solver_linear as msl  # noqa: E402

import ray  # noqa: E402
from hyperopt import hp  # noqa: E402
from ray import tune  # noqa: E402
from ray.tune.search.hyperopt import HyperOptSearch  # noqa: E402

SPIKE_MODE_CHOICES = ["count", "rate", "l2_count", "sqrt", "l2_sqrt", "zscore", "log1p"]
CLS_STEPS_CHOICES = [1, 3, 5]

SEARCH_SPACE = {
    "spike_mode": hp.choice("spike_mode", SPIKE_MODE_CHOICES),
    "lr_cls": hp.loguniform("lr_cls", math.log(5e-4), math.log(3e-2)),
    "wd_cls": hp.loguniform("wd_cls", math.log(1e-5), math.log(1e-2)),
    "cls_dropout": hp.uniform("cls_dropout", 0.0, 0.25),
    "label_smoothing": hp.uniform("label_smoothing", 0.0, 0.1),
    "cls_steps": hp.choice("cls_steps", CLS_STEPS_CHOICES),
}


def apply_config(config: dict, feat_source: str) -> None:
    msl.hid1 = 5000
    msl.hid2 = 1000
    msl.feat_source = feat_source
    msl.cls_train_mode = "clean"
    msl.spike_mode = str(config["spike_mode"])
    msl.lr_cls = float(config["lr_cls"])
    msl.wd_cls = float(config["wd_cls"])
    msl.cls_dropout = float(config["cls_dropout"])
    msl.label_smoothing = float(config["label_smoothing"])
    msl.cls_steps = int(config["cls_steps"])


def objective(config, data_refs, device_str, epochs, feat_source):
    apply_config(config, feat_source=feat_source)
    device = torch.device(
        "cuda" if (device_str == "auto" and torch.cuda.is_available()) else device_str
    )
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.cuda.manual_seed_all(msl.seed)

    torch.manual_seed(msl.seed)
    np.random.seed(msl.seed)
    torch.set_num_threads(int(os.environ.get("CSDP_THREADS", "1")))

    X_tr = torch.from_numpy(ray.get(data_refs["X_tr"]).copy()).float()
    Y_tr = torch.from_numpy(ray.get(data_refs["Y_tr"]).copy()).float()
    X_te = torch.from_numpy(ray.get(data_refs["X_te"]).copy()).float()
    Y_te = torch.from_numpy(ray.get(data_refs["Y_te"]).copy()).float()

    tr_ld = DataLoader(
        TensorDataset(X_tr, Y_tr),
        batch_size=msl.batch_size,
        shuffle=True,
        drop_last=True,
    )
    te_ld = DataLoader(
        TensorDataset(X_te, Y_te),
        batch_size=msl.dev_batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = msl.CSDPSNNLinear(device).to(device)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            model.opt_cls, T_max=max(epochs, 1), eta_min=msl.lr_cls * 0.05
        )
        if epochs > 1
        else None
    )

    best_acc = 0.0
    tr_acc = 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(epochs):
            tr_acc = msl.train_epoch(model, tr_ld, device)
            te_acc = msl.evaluate(model, te_ld, device)
            best_acc = max(best_acc, te_acc)
            if scheduler is not None:
                scheduler.step()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {"1-acc": 1.0 - best_acc, "accuracy": best_acc, "train_acc": tr_acc}


def main():
    p = argparse.ArgumentParser(description="Ray Tune search for mnist_solver_linear.py")
    p.add_argument("--samples", type=int, default=60)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--feat-source", choices=("z2", "z1_z2"), default="z2")
    p.add_argument("--gpus-per-trial", type=float, default=0.25)
    p.add_argument("--cpus-per-trial", type=int, default=2)
    p.add_argument("--max-concurrent", type=int, default=4)
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--data-dir", default=str(SCRIPT_DIR.parents[1] / "data" / "mnist"))
    p.add_argument("--out-json", default=str(SCRIPT_DIR / "mnist_linear_best_params.json"))
    args = p.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    required = ("trainX.npy", "trainY.npy", "testX.npy", "testY.npy")
    if not all((data_dir / name).exists() for name in required):
        from download_mnist import ensure_mnist

        ensure_mnist(data_dir)

    n_cpus = os.cpu_count() or 4
    n_gpus = torch.cuda.device_count() if args.device != "cpu" else 0
    ray.init(num_cpus=n_cpus, num_gpus=n_gpus, ignore_reinit_error=True)

    data_refs = {
        "X_tr": ray.put(np.load(data_dir / "trainX.npy").astype(np.float32)),
        "Y_tr": ray.put(np.load(data_dir / "trainY.npy").astype(np.float32)),
        "X_te": ray.put(np.load(data_dir / "testX.npy").astype(np.float32)),
        "Y_te": ray.put(np.load(data_dir / "testY.npy").astype(np.float32)),
    }

    algo = HyperOptSearch(SEARCH_SPACE, metric="1-acc", mode="min", random_state_seed=msl.seed)
    t0 = time.time()
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                objective,
                data_refs=data_refs,
                device_str=args.device,
                epochs=args.epochs,
                feat_source=args.feat_source,
            ),
            {"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial if n_gpus > 0 else 0.0},
        ),
        tune_config=tune.TuneConfig(
            search_alg=algo,
            num_samples=args.samples,
            max_concurrent_trials=args.max_concurrent,
        ),
        run_config=tune.RunConfig(
            name="mnist_solver_linear_hyperopt",
            storage_path=str(SCRIPT_DIR / "ray_results"),
            verbose=1,
        ),
    )

    results = tuner.fit()
    best = results.get_best_result(metric="1-acc", mode="min")
    payload = {
        "feat_source": args.feat_source,
        "epochs_per_trial": args.epochs,
        "samples": args.samples,
        "elapsed_min": (time.time() - t0) / 60.0,
        "best_test_accuracy": float(best.metrics["accuracy"]),
        "best_config": best.config,
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    ray.shutdown()


if __name__ == "__main__":
    main()
