#!/usr/bin/env python3
"""CIFAR-10 spiking classifier trained with backpropagation.

No CSDP, no local Hebbian updates, no hub neurons, no lateral synapses.
Just a standard feed-forward SNN trained with BPTT + surrogate gradients.

Architecture:
    input: 3 channels × 32×32 = 3072
    Linear(3072 -> 1024) -> LIF
    Linear(1024 -> 1024) -> LIF
    Linear(1024 -> 1024) -> LIF
    Linear(1024 -> 128)  -> LIF
    Linear(128 -> 10)    -> LIF
    loss = cross-entropy on the mean output spike rate over T steps

Input encoding:
    one Bernoulli neuron per color-channel pixel (no copies). At every time
    step each input neuron spikes with probability equal to the normalized
    pixel value. The mean spike rate over T steps approximates the pixel
    value (rate/frequency coding).

Run:
    uv run --active python cifar10_backprop.py --device cuda --epochs 30

Smoke test:
    uv run --active python cifar10_backprop.py --device cpu \
        --T 2 --batch-size 2 --max-train-batches 1 --max-test-batches 1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional, neuron, surrogate
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────── fixed sizes ─────────────────────────────────────
HIDDEN = (1024, 1024, 1024, 128)
OUT_DIM = 10
CIFAR_CHANNELS = 3
CIFAR_SIZE = 32

DEFAULT_T = 50
DEFAULT_TAU = 2.0

DEFAULT_BATCH_SIZE = 128
DEFAULT_TEST_BATCH_SIZE = 256
DEFAULT_EPOCHS = 30
DEFAULT_LR = 1e-3
SEED = 1234

data_dir = "/root/snn/data/cifar10"


class CIFAR10SNN(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN, out_dim=OUT_DIM, tau=DEFAULT_TAU):
        super().__init__()

        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(
                neuron.LIFNode(
                    tau=tau,
                    v_threshold=1.0,
                    v_reset=0.0,
                    surrogate_function=surrogate.ATan(),
                    detach_reset=True,
                )
            )
            prev = h

        self.hidden = nn.Sequential(*layers)
        self.fc_out = nn.Linear(prev, out_dim)
        self.lif_out = neuron.LIFNode(
            tau=tau,
            v_threshold=1.0,
            v_reset=0.0,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
        )

    def forward_step(self, x_t):
        """One time step. LIF state is kept between calls."""
        s = self.hidden(x_t)
        return self.lif_out(self.fc_out(s))


def flatten_input(x):
    """(B, 3, 32, 32) -> (B, 3072) probabilities."""
    return x.reshape(x.shape[0], -1)


def spike_encode(probs):
    """Bernoulli rate coding: one binary spike per input neuron per step."""
    return (torch.rand_like(probs) < probs).float()


def load_data(data_dir=data_dir, batch_size=DEFAULT_BATCH_SIZE,
              test_batch_size=DEFAULT_TEST_BATCH_SIZE, num_workers=0):
    data_dir = Path(data_dir).expanduser()
    X_tr = torch.from_numpy(np.load(data_dir / "trainX.npy")).float()
    Y_tr = torch.from_numpy(np.load(data_dir / "trainY.npy")).float()
    X_te = torch.from_numpy(np.load(data_dir / "testX.npy")).float()
    Y_te = torch.from_numpy(np.load(data_dir / "testY.npy")).float()

    train_loader = DataLoader(
        TensorDataset(X_tr, Y_tr),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        TensorDataset(X_te, Y_te),
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, device, T,
                    max_batches=0, print_every=10):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = len(loader)
    t0 = time.time()

    for i, (xb, yb) in enumerate(loader):
        if max_batches and i >= max_batches:
            break

        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        B = xb.shape[0]

        probs = flatten_input(xb)
        functional.reset_net(model)

        logits = torch.zeros(B, OUT_DIM, device=device)
        for _ in range(T):
            x_t = spike_encode(probs)
            logits = logits + model.forward_step(x_t)
        logits = logits / T

        loss = F.cross_entropy(logits, yb.argmax(dim=1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * B
        correct += (logits.argmax(dim=1) == yb.argmax(dim=1)).sum().item()
        total += B

        if print_every and (i + 1) % print_every == 0:
            print(
                f"  batch {i + 1:3d}/{n_batches}  "
                f"loss: {total_loss / total:.4f}  "
                f"acc: {correct / total:.4f}  "
                f"[{(time.time() - t0) / 60:.1f}m]",
                end="\r",
                flush=True,
            )

    print()
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device, T, max_batches=0):
    model.eval()
    correct = 0
    total = 0

    for i, (xb, yb) in enumerate(loader):
        if max_batches and i >= max_batches:
            break

        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        B = xb.shape[0]

        probs = flatten_input(xb)
        functional.reset_net(model)

        logits = torch.zeros(B, OUT_DIM, device=device)
        for _ in range(T):
            x_t = spike_encode(probs)
            logits = logits + model.forward_step(x_t)
        logits = logits / T

        correct += (logits.argmax(dim=1) == yb.argmax(dim=1)).sum().item()
        total += B

    model.train()
    return correct / max(total, 1)


def train_cifar10(
    data_dir=data_dir,
    epochs=DEFAULT_EPOCHS,
    device="cpu",
    T=DEFAULT_T,
    tau=DEFAULT_TAU,
    batch_size=DEFAULT_BATCH_SIZE,
    test_batch_size=DEFAULT_TEST_BATCH_SIZE,
    lr=DEFAULT_LR,
    max_train_batches=0,
    max_test_batches=0,
    num_workers=0,
):
    device = torch.device(device) if isinstance(device, str) else device
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    train_loader, test_loader = load_data(
        data_dir, batch_size, test_batch_size, num_workers
    )

    in_dim = CIFAR_CHANNELS * CIFAR_SIZE * CIFAR_SIZE
    model = CIFAR10SNN(in_dim, tau=tau).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Device: {device} | T: {T} | input: {in_dim} "
        f"(3 channels × 32×32) | hidden: {HIDDEN} | params: {n_params / 1e6:.2f}M"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )

    print(
        "Test accuracy before training: "
        f"{evaluate(model, test_loader, device, T, max_test_batches):.4f}"
    )

    best = 0.0
    elapsed = 0.0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, T,
            max_batches=max_train_batches,
        )
        test_acc = evaluate(
            model, test_loader, device, T,
            max_batches=max_test_batches,
        )
        scheduler.step()

        best = max(best, test_acc)
        elapsed = time.time() - t0
        print(
            f"Epoch {ep:2d}/{epochs}  "
            f"Train loss: {train_loss:.4f}  Train: {train_acc:.4f}  "
            f"Test: {test_acc:.4f}  Best: {best:.4f}  "
            f"[{elapsed / 60:.1f}m]"
        )

    print(f"\nDone in {elapsed / 60:.1f} min.  Best test accuracy: {best:.4f}")
    return model, best


def main():
    parser = argparse.ArgumentParser(
        description="CIFAR-10 SNN trained with backprop (no CSDP)"
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda / cuda:0 / cpu",
    )
    parser.add_argument("--data-dir", default=data_dir)
    parser.add_argument("--T", type=int, default=DEFAULT_T, help="time steps")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU, help="LIF tau")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--test-batch-size", type=int, default=DEFAULT_TEST_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    required = ("trainX.npy", "trainY.npy", "testX.npy", "testY.npy")
    if not all((Path(args.data_dir) / name).exists() for name in required):
        from download_cifar10 import ensure_cifar10
        ensure_cifar10(args.data_dir)

    train_cifar10(
        data_dir=args.data_dir,
        epochs=args.epochs,
        device=args.device,
        T=args.T,
        tau=args.tau,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        lr=args.lr,
        max_train_batches=args.max_train_batches,
        max_test_batches=args.max_test_batches,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
