#!/usr/bin/env python3
"""MNIST CSDP-SNN solver with spiking output layer connected to BOTH hidden layers (z1 + z2).

Extends mnist_solver.py by connecting the spiking output layer (lif_out) to both
hidden spiking layers via C2 (hid1 -> out_dim) and C3 (hid2 -> out_dim), matching
the original CSDP architecture (summation(C2.outputs, C3.outputs)):
    zy = lif_out(z1 @ C2 + z2 @ C3)
"""

from pathlib import Path
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, layer, functional
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────── fixed hyperparameters ───────────────────────────
T = 50
dt = 3.0
tau_m = 100.0
v_thr = 0.055
tau_tr = 13.0
R_m = 1.0
inh_R = 0.01
goodness_thr = 10.0
eta = 0.002
weight_bound = 1.0
seed = 1234

tau_m_hab = 4.87563
v_thr_hab = 0.249441
tau_hub = 4.39997
hub_power = 1.20313
hub_w_init = 0.0503053

in_dim = 784
out_dim = 10
hid1 = 5000
hid2 = 1000

batch_size = 500
dev_batch_size = 1000
num_epochs = 30

data_dir = str(Path(__file__).resolve().parents[2] / "data" / "mnist")


class CSDPSNNSpikingZ1Z2(torch.nn.Module):
    def __init__(self, device, amp_fp16=False):
        super().__init__()
        self.device = device
        self.amp_fp16 = amp_fp16 and (device.type == "cuda")
        tau_sj = tau_m / dt

        # Hidden layer 1
        self.W1 = nn_param(in_dim, hid1)
        self.M1 = nn_param(hid1, hid1)
        self.V2y = nn_param(out_dim, hid1)

        # Hidden layer 2
        self.W2 = nn_param(hid1, hid2)
        self.M2 = nn_param(hid2, hid2)
        self.V3y = nn_param(out_dim, hid2)

        # Spiking output layer connected to BOTH z1 (C2) and z2 (C3)
        self.C2 = nn_param(hid1, out_dim)
        self.C3 = nn_param(hid2, out_dim)

        self.lif1 = neuron.LIFNode(tau=tau_sj, v_threshold=v_thr, v_reset=0.0).to(device)
        self.lif2 = neuron.LIFNode(tau=tau_sj, v_threshold=v_thr, v_reset=0.0).to(device)
        self.lif_out = neuron.LIFNode(tau=tau_sj, v_threshold=v_thr, v_reset=0.0).to(device)

        self.hub_fc1 = layer.Linear(hid1, 1, bias=False).to(device)
        self.hub_fc2 = layer.Linear(hid2, 1, bias=False).to(device)
        self.hub_lif1 = neuron.LIFNode(
            tau=tau_m_hab, v_threshold=v_thr_hab, v_reset=0.0
        ).to(device)
        self.hub_lif2 = neuron.LIFNode(
            tau=tau_m_hab, v_threshold=v_thr_hab, v_reset=0.0
        ).to(device)

        torch.nn.init.constant_(self.hub_fc1.weight, hub_w_init)
        torch.nn.init.constant_(self.hub_fc2.weight, hub_w_init)

        self.hub_fc1.weight.requires_grad_(False)
        self.hub_fc2.weight.requires_grad_(False)
        self.register_buffer("zmax1", torch.tensor(1.0))
        self.register_buffer("zmax2", torch.tensor(1.0))

        self.to(device)
        self.opt_all = torch.optim.Adam(
            [self.W1, self.M1, self.V2y, self.W2, self.M2, self.V3y, self.C2, self.C3],
            lr=eta,
            fused=(device.type == "cuda"),
        )

    def hub_mod(self, z, hub, phase, zmax_buf):
        torch.maximum(zmax_buf, hub.max(), out=zmax_buf)
        factor = (hub / zmax_buf) ** hub_power - phase
        return factor * z

    def forward(self, x, y, mask, learn=True):
        B = x.shape[0]
        dv = self.device
        functional.reset_net(self.lif1)
        functional.reset_net(self.lif2)
        functional.reset_net(self.lif_out)
        functional.reset_net(self.hub_lif1)
        functional.reset_net(self.hub_lif2)

        tr1 = torch.zeros(B, hid1, device=dv)
        tr2 = torch.zeros(B, hid2, device=dv)
        tr_out = torch.zeros(B, out_dim, device=dv)
        spk_cnt = torch.zeros(B, out_dim, device=dv)
        z1_prev = torch.zeros(B, hid1, device=dv)
        z2_prev = torch.zeros(B, hid2, device=dv)
        hub1 = torch.zeros(B, 1, device=dv)
        hub2 = torch.zeros(B, 1, device=dv)
        d = dt / tau_tr
        decay_hub = 1.0 / tau_hub

        for _ in range(T):
            z0 = (torch.rand(B, in_dim, device=dv) < x).float()

            # layer 1
            if self.amp_fp16:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    I1 = (
                        (z0 @ self.W1) * R_m
                        - (z1_prev @ self.M1) * inh_R
                        + (y @ self.V2y) * R_m
                    )
                I1 = I1.float()
            else:
                I1 = (
                    (z0 @ self.W1) * R_m
                    - (z1_prev @ self.M1) * inh_R
                    + (y @ self.V2y) * R_m
                )
            z1 = self.lif1(I1)
            tr1 = tr1 + d * (z1 - tr1)

            h1_spk = self.hub_lif1(self.hub_fc1(z1))
            hub1 = hub1 + decay_hub * (-hub1 + h1_spk.float())
            mod1 = self.hub_mod(tr1, hub1, mask, self.zmax1)

            # layer 2
            if self.amp_fp16:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    I2 = (
                        (z1 @ self.W2) * R_m
                        - (z2_prev @ self.M2) * inh_R
                        + (y @ self.V3y) * R_m
                    )
                I2 = I2.float()
            else:
                I2 = (
                    (z1 @ self.W2) * R_m
                    - (z2_prev @ self.M2) * inh_R
                    + (y @ self.V3y) * R_m
                )
            z2 = self.lif2(I2)
            tr2 = tr2 + d * (z2 - tr2)

            h2_spk = self.hub_lif2(self.hub_fc2(z2))
            hub2 = hub2 + decay_hub * (-hub2 + h2_spk.float())
            mod2 = self.hub_mod(tr2, hub2, mask, self.zmax2)

            # spiking output layer connected to BOTH z1 (C2) and z2 (C3)
            if self.amp_fp16:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    I_out = z1 @ self.C2 + z2 @ self.C3
                zy = self.lif_out(I_out.float())
            else:
                zy = self.lif_out(z1 @ self.C2 + z2 @ self.C3)
            tr_out = tr_out + d * (zy - tr_out)
            spk_cnt = spk_cnt + zy
            dmu = 2.0 * (y - tr_out) * mask

            if learn:
                mod1_rm = mod1 * R_m
                mod1_inh = mod1 * inh_R
                mod2_rm = mod2 * R_m
                mod2_inh = mod2 * inh_R
                if self.amp_fp16:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        self.W1.grad = (z0.T @ mod1_rm).float()
                        self.M1.grad = (z1_prev.T @ mod1_inh).float()
                        self.V2y.grad = (y.T @ mod1_rm).float()
                        self.W2.grad = (z1.T @ mod2_rm).float()
                        self.M2.grad = (z2_prev.T @ mod2_inh).float()
                        self.V3y.grad = (y.T @ mod2_rm).float()
                        self.C2.grad = (-(z1.T @ dmu)).float()
                        self.C3.grad = (-(z2.T @ dmu)).float()
                else:
                    self.W1.grad = z0.T @ mod1_rm
                    self.M1.grad = z1_prev.T @ mod1_inh
                    self.V2y.grad = y.T @ mod1_rm
                    self.W2.grad = z1.T @ mod2_rm
                    self.M2.grad = z2_prev.T @ mod2_inh
                    self.V3y.grad = y.T @ mod2_rm
                    self.C2.grad = -(z1.T @ dmu)
                    self.C3.grad = -(z2.T @ dmu)
                self.opt_all.step()
                self.W1.data.clamp_(-weight_bound, weight_bound)
                self.M1.data.fill_diagonal_(0.0)
                self.M1.data.clamp_(0, weight_bound)
                self.V2y.data.clamp_(-weight_bound, weight_bound)
                self.W2.data.clamp_(-weight_bound, weight_bound)
                self.M2.data.fill_diagonal_(0.0)
                self.M2.data.clamp_(0, weight_bound)
                self.V3y.data.clamp_(-weight_bound, weight_bound)
                self.C2.data.clamp_(-weight_bound, weight_bound)
                self.C3.data.clamp_(-weight_bound, weight_bound)

            z1_prev = z1
            z2_prev = z2

        return F.softmax(spk_cnt / T, dim=1), spk_cnt, tr_out


def nn_param(*shape):
    p = torch.nn.Parameter(torch.empty(*shape))
    torch.nn.init.uniform_(p, -1, 1)
    return p


def neg_labels(y_onehot, device):
    B = y_onehot.shape[0]
    correct = y_onehot.argmax(dim=1)
    neg = torch.randint(0, out_dim, (B,), device=device)
    neg[neg == correct] = (neg[neg == correct] + 1) % out_dim
    return F.one_hot(neg, out_dim).float()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        B = xb.shape[0]
        yh, _, _ = model(
            xb,
            torch.zeros(B, out_dim, device=device),
            torch.ones(B, 1, device=device),
            learn=False,
        )
        correct += (yh.argmax(dim=1) == yb.argmax(dim=1).to(device)).sum().item()
        total += B
    model.train()
    return correct / max(total, 1)


@torch.no_grad()
def train_epoch(model, loader, device):
    model.train()
    correct = total = 0
    n_batches = len(loader)
    for i, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.float().to(device)
        B = xb.shape[0]
        yn = neg_labels(yb, device)

        X_cat = torch.cat([xb, xb], dim=0)
        Y_cat = torch.cat([yb, yn], dim=0)
        mask = torch.cat(
            [torch.ones(B, 1, device=device), torch.zeros(B, 1, device=device)],
            dim=0,
        )

        yh, _, _ = model(X_cat, Y_cat, mask, learn=True)
        batch_correct = (
            yh[:B].argmax(dim=1) == yb.argmax(dim=1).to(device)
        ).sum().item()
        correct += batch_correct
        total += B

        print(
            f"  Batch {i + 1:3d}/{n_batches}  acc: {batch_correct / B:.4f}  "
            f"avg: {correct / total:.4f}",
            end="\r",
            flush=True,
        )

    print()
    return correct / max(total, 1)


def load_data(data_dir=data_dir, batch_size=batch_size, dev_batch_size=dev_batch_size):
    data_dir = Path(data_dir).expanduser()
    required = ("trainX.npy", "trainY.npy", "testX.npy", "testY.npy")
    if not all((data_dir / name).exists() for name in required):
        from download_mnist import ensure_mnist

        ensure_mnist(data_dir)

    X_tr = torch.from_numpy(np.load(data_dir / "trainX.npy")).float()
    Y_tr = torch.from_numpy(np.load(data_dir / "trainY.npy")).float()
    X_te = torch.from_numpy(np.load(data_dir / "testX.npy")).float()
    Y_te = torch.from_numpy(np.load(data_dir / "testY.npy")).float()

    tr_ld = DataLoader(
        TensorDataset(X_tr, Y_tr),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    te_ld = DataLoader(
        TensorDataset(X_te, Y_te),
        batch_size=dev_batch_size,
        shuffle=False,
        drop_last=True,
    )
    return tr_ld, te_ld


def train_mnist(
    data_dir=data_dir,
    epochs=num_epochs,
    device="cpu",
    batch_size=batch_size,
    dev_batch_size=dev_batch_size,
    amp_fp16=False,
):
    device = torch.device(device) if isinstance(device, str) else device
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr_ld, te_ld = load_data(data_dir, batch_size, dev_batch_size)
    model = CSDPSNNSpikingZ1Z2(device, amp_fp16=amp_fp16).to(device)

    t0 = time.time()
    print(f"Test accuracy: {evaluate(model, te_ld, device):.4f} ", flush=True)

    best = 0.0
    elapsed = 0.0
    for ep in range(1, epochs + 1):
        tr_acc = train_epoch(model, tr_ld, device)
        te_acc = evaluate(model, te_ld, device)
        best = max(best, te_acc)
        elapsed = time.time() - t0
        print(
            f"Epoch {ep:2d}/{epochs}  Train: {tr_acc:.4f}  "
            f"Test: {te_acc:.4f}  Best: {best:.4f}  [{elapsed / 60:.1f}m]",
            flush=True,
        )

    print(
        f"\nDone in {elapsed / 60:.1f} min.  Best test accuracy: {best:.4f}",
        flush=True,
    )
    return model, best


def main():
    global tau_m_hab, v_thr_hab, tau_hub, hub_power, hub_w_init
    parser = argparse.ArgumentParser(
        description="MNIST CSDP-SNN with spiking output connected to z1 + z2 (C2 + C3)"
    )
    parser.add_argument("--epochs", type=int, default=num_epochs)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda / cuda:0 / cpu",
    )
    parser.add_argument("--data-dir", default=data_dir)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--dev-batch-size", type=int, default=dev_batch_size)
    parser.add_argument("--tau-m-hab", type=float, default=None)
    parser.add_argument("--v-thr-hab", type=float, default=None)
    parser.add_argument("--tau-hub", type=float, default=None)
    parser.add_argument("--hub-power", type=float, default=None)
    parser.add_argument("--hub-w-init", type=float, default=None)
    parser.add_argument(
        "--amp-fp16",
        action="store_true",
        help="Use FP16 Tensor Cores for matmuls (keeps FP32 weights/Adam; ~5x faster on T4)",
    )
    args = parser.parse_args()

    if args.tau_m_hab is not None:
        tau_m_hab = args.tau_m_hab
    if args.v_thr_hab is not None:
        v_thr_hab = args.v_thr_hab
    if args.tau_hub is not None:
        tau_hub = args.tau_hub
    if args.hub_power is not None:
        hub_power = args.hub_power
    if args.hub_w_init is not None:
        hub_w_init = args.hub_w_init

    print(
        f"MNIST CSDP-SNN Spiking (C2+C3: {hid1}+{hid2} -> {out_dim}) | "
        f"epochs={args.epochs}, device={args.device}, amp_fp16={args.amp_fp16}\n"
        f"Hub params: tau_m_hab={tau_m_hab}, v_thr_hab={v_thr_hab}, "
        f"tau_hub={tau_hub}, hub_power={hub_power}, hub_w_init={hub_w_init}",
        flush=True,
    )

    train_mnist(
        data_dir=args.data_dir,
        epochs=args.epochs,
        device=args.device,
        batch_size=args.batch_size,
        dev_batch_size=args.dev_batch_size,
        amp_fp16=args.amp_fp16,
    )


if __name__ == "__main__":
    main()
