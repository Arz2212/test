#!/usr/bin/env python3
"""KMNIST 100%-local Conv+FC CSDP-SNN with preserved Hub-Neuron modulation.

Uses the EXACT same architecture, hub-neuron logic, and hyperparameters as
mnist_solver_conv_hub.py (zero dataset-specific tuning or augmentation).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Reuse the exact same model & local training routines as mnist_solver_conv_hub.py
MNIST_CODE_DIR = Path(__file__).resolve().parents[2] / "mnist" / "code"
if str(MNIST_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(MNIST_CODE_DIR))

from mnist_solver_conv_hub import (  # noqa: E402
    ConvCSDPHubSNN,
    T,
    batch_size,
    c1_ch,
    c2_ch,
    dev_batch_size,
    eta,
    eta_min_ratio,
    eta_out,
    evaluate,
    hid_fc,
    num_epochs,
    seed,
    train_epoch,
)

default_data_dir = str(Path(__file__).resolve().parents[2] / "data" / "kmnist")


def load_data(data_dir: str, bs: int, dev_bs: int) -> tuple[DataLoader, DataLoader]:
    p = Path(data_dir).expanduser()
    required = ("trainX.npy", "trainY.npy", "testX.npy", "testY.npy")
    if not all((p / name).exists() for name in required):
        from download_kmnist import ensure_kmnist

        ensure_kmnist(p)
    X_tr = torch.from_numpy(np.load(p / "trainX.npy")).float()
    Y_tr = torch.from_numpy(np.load(p / "trainY.npy")).float()
    X_te = torch.from_numpy(np.load(p / "testX.npy")).float()
    Y_te = torch.from_numpy(np.load(p / "testY.npy")).float()
    tr_ld = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=bs, shuffle=True, drop_last=True)
    te_ld = DataLoader(TensorDataset(X_te, Y_te), batch_size=dev_bs, shuffle=False, drop_last=False)
    return tr_ld, te_ld


def main() -> None:
    parser = argparse.ArgumentParser(description="KMNIST 100%-local Conv+FC CSDP-Hub-SNN")
    parser.add_argument("--data-dir", default=default_data_dir)
    parser.add_argument("--epochs", type=int, default=num_epochs)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--dev-batch-size", type=int, default=dev_batch_size)
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--save-json", default="")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tr_ld, te_ld = load_data(args.data_dir, args.batch_size, args.dev_batch_size)
    model = ConvCSDPHubSNN(device, amp_bf16=(not args.no_bf16))

    print(
        f"KMNIST ConvCSDPHubSNN (100% local, preserved hub logic) | "
        f"c1={c1_ch}(14x14->7x7) -> c2={c2_ch}(7x7) -> fc={hid_fc} | "
        f"T={T}, epochs={args.epochs}, seed={args.seed}, device={device}",
        flush=True,
    )

    t0 = time.time()
    best = 0.0
    best_ep = 0
    history = []
    for ep in range(1, args.epochs + 1):
        frac = 0.5 * (1.0 + math.cos(math.pi * (ep - 1) / max(args.epochs - 1, 1)))
        scale = eta_min_ratio + (1.0 - eta_min_ratio) * frac
        for pg in model.opt_hidden.param_groups:
            pg["lr"] = eta * scale
        for pg in model.opt_out.param_groups:
            pg["lr"] = eta_out * scale

        tr_acc, diag = train_epoch(model, tr_ld, device)
        te_acc = evaluate(model, te_ld, device)
        if te_acc > best:
            best = te_acc
            best_ep = ep
        elapsed = time.time() - t0
        print(
            f"Epoch {ep:2d}/{args.epochs}  Train: {tr_acc:.4f}  Test: {te_acc:.4f}  "
            f"Best: {best:.4f} (ep {best_ep})  [{elapsed / 60:.2f}m] | "
            f"r=({diag['r1']:.3f},{diag['r2']:.3f},{diag['rfc']:.3f}) "
            f"h=({diag['h1']:.2f},{diag['h2']:.2f},{diag['hfc']:.2f}) "
            f"zm=({diag['zm1']:.2f},{diag['zm2']:.2f},{diag['zmfc']:.2f})",
            flush=True,
        )
        history.append({"epoch": ep, "train": tr_acc, "test": te_acc, **diag})

    print(f"\nDone in {(time.time() - t0) / 60:.2f} min.  Best test accuracy: {best:.4f} (ep {best_ep})", flush=True)
    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_json).write_text(
            json.dumps({"dataset": "kmnist", "best": best, "best_ep": best_ep, "seed": args.seed, "history": history}, indent=2)
        )


if __name__ == "__main__":
    main()
