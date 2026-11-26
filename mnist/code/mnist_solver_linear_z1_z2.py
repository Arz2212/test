#!/usr/bin/env python3

from pathlib import Path
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, layer, functional
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────── fixed SNN hyperparameters ───────────────────────
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

# ───────────────────────── linear classifier hyperparameters ─────────────────
lr_cls = 0.001
wd_cls = 1e-3
cls_dropout = 0.15
cls_steps = 1

data_dir = str(Path(__file__).resolve().parents[2] / "data" / "mnist")


def nn_param(*shape):
    p = torch.nn.Parameter(torch.empty(*shape))
    torch.nn.init.uniform_(p, -1, 1)
    return p


def decode_spikes_zscore(spk_cnt: torch.Tensor) -> torch.Tensor:
    """Standardize a layer's spike-count vector per sample (mean=0, std=1)."""
    mu = spk_cnt.mean(dim=1, keepdim=True)
    std = spk_cnt.std(dim=1, keepdim=True, unbiased=False)
    return (spk_cnt - mu) / (std + 1e-5)


class CSDPSNNLinearZ1Z2(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        tau_sj = tau_m / dt

        # Hidden spiking layer 1 (CSDP)
        self.W1 = nn_param(in_dim, hid1)
        self.M1 = nn_param(hid1, hid1)
        self.V2y = nn_param(out_dim, hid1)

        # Hidden spiking layer 2 (CSDP)
        self.W2 = nn_param(hid1, hid2)
        self.M2 = nn_param(hid2, hid2)
        self.V3y = nn_param(out_dim, hid2)

        # Linear classifier attached to both hidden spiking layers (5000 + 1000 -> 10)
        self.classifier = nn.Linear(hid1 + hid2, out_dim, bias=True)
        self.dropout = nn.Dropout(p=cls_dropout)

        # Spiking neurons for hidden layers only
        self.lif1 = neuron.LIFNode(tau=tau_sj, v_threshold=v_thr, v_reset=0.0).to(device)
        self.lif2 = neuron.LIFNode(tau=tau_sj, v_threshold=v_thr, v_reset=0.0).to(device)

        # Hub neurons for CSDP modulation
        self.hub_fc1 = layer.Linear(hid1, 1, bias=False).to(device)
        self.hub_fc2 = layer.Linear(hid2, 1, bias=False).to(device)
        self.hub_lif1 = neuron.LIFNode(
            tau=tau_m_hab, v_threshold=v_thr_hab, v_reset=0.0
        ).to(device)
        self.hub_lif2 = neuron.LIFNode(
            tau=tau_m_hab, v_threshold=v_thr_hab, v_reset=0.0
        ).to(device)

        nn.init.constant_(self.hub_fc1.weight, hub_w_init)
        nn.init.constant_(self.hub_fc2.weight, hub_w_init)
        self.hub_fc1.weight.requires_grad_(False)
        self.hub_fc2.weight.requires_grad_(False)

        self.register_buffer("zmax1", torch.tensor(1.0))
        self.register_buffer("zmax2", torch.tensor(1.0))

        self.to(device)

        # Local CSDP optimizer for hidden spiking layers
        self.opt_snn = torch.optim.Adam(
            [self.W1, self.M1, self.V2y, self.W2, self.M2, self.V3y],
            lr=eta,
            fused=(device.type == "cuda"),
        )
        # Standard gradient optimizer for the linear classifier
        self.opt_cls = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=lr_cls,
            weight_decay=wd_cls,
        )

    def hub_mod(self, z, hub, phase, zmax_buf):
        torch.maximum(zmax_buf, hub.max(), out=zmax_buf)
        factor = (hub / zmax_buf) ** hub_power - phase
        return factor * z

    @torch.no_grad()
    def forward_snn(self, x, y=None, mask=None, learn=True):
        """Simulate spiking layers for T steps and return decoded [z1, z2] features."""
        B = x.shape[0]
        dv = self.device
        functional.reset_net(self.lif1)
        functional.reset_net(self.lif2)
        if learn:
            functional.reset_net(self.hub_lif1)
            functional.reset_net(self.hub_lif2)
            tr1 = torch.zeros(B, hid1, device=dv)
            tr2 = torch.zeros(B, hid2, device=dv)
            hub1 = torch.zeros(B, 1, device=dv)
            hub2 = torch.zeros(B, 1, device=dv)

        spk_cnt1 = torch.zeros(B, hid1, device=dv)
        spk_cnt2 = torch.zeros(B, hid2, device=dv)
        z1_prev = torch.zeros(B, hid1, device=dv)
        z2_prev = torch.zeros(B, hid2, device=dv)

        d = dt / tau_tr
        decay_hub = 1.0 / tau_hub
        has_y = y is not None

        for _ in range(T):
            z0 = (torch.rand(B, in_dim, device=dv) < x).float()

            # Layer 1
            I1 = (z0 @ self.W1) * R_m - (z1_prev @ self.M1) * inh_R
            if has_y:
                I1 = I1 + (y @ self.V2y) * R_m
            z1 = self.lif1(I1)

            if learn:
                tr1 = tr1 + d * (z1 - tr1)
                h1_spk = self.hub_lif1(self.hub_fc1(z1))
                hub1 = hub1 + decay_hub * (-hub1 + h1_spk.float())
                mod1 = self.hub_mod(tr1, hub1, mask, self.zmax1)

            # Layer 2
            I2 = (z1 @ self.W2) * R_m - (z2_prev @ self.M2) * inh_R
            if has_y:
                I2 = I2 + (y @ self.V3y) * R_m
            z2 = self.lif2(I2)

            if learn:
                tr2 = tr2 + d * (z2 - tr2)
                h2_spk = self.hub_lif2(self.hub_fc2(z2))
                hub2 = hub2 + decay_hub * (-hub2 + h2_spk.float())
                mod2 = self.hub_mod(tr2, hub2, mask, self.zmax2)

                mod1_rm = mod1 * R_m
                mod1_inh = mod1 * inh_R
                mod2_rm = mod2 * R_m
                mod2_inh = mod2 * inh_R
                self.W1.grad = z0.T @ mod1_rm
                self.M1.grad = z1_prev.T @ mod1_inh
                self.V2y.grad = y.T @ mod1_rm
                self.W2.grad = z1.T @ mod2_rm
                self.M2.grad = z2_prev.T @ mod2_inh
                self.V3y.grad = y.T @ mod2_rm
                self.opt_snn.step()

                self.W1.data.clamp_(-weight_bound, weight_bound)
                self.M1.data.fill_diagonal_(0.0)
                self.M1.data.clamp_(0, weight_bound)
                self.V2y.data.clamp_(-weight_bound, weight_bound)
                self.W2.data.clamp_(-weight_bound, weight_bound)
                self.M2.data.fill_diagonal_(0.0)
                self.M2.data.clamp_(0, weight_bound)
                self.V3y.data.clamp_(-weight_bound, weight_bound)

            spk_cnt1 = spk_cnt1 + z1
            spk_cnt2 = spk_cnt2 + z2
            z1_prev = z1
            z2_prev = z2

        # Standardize each layer separately so z1 (5000) and z2 (1000) share equal scale
        feat1 = decode_spikes_zscore(spk_cnt1)
        feat2 = decode_spikes_zscore(spk_cnt2)
        return torch.cat([feat1, feat2], dim=1)

    def step_classifier(self, feat, target_idx):
        """Update nn.Linear with dropout + cross-entropy on decoded spike features."""
        with torch.enable_grad():
            logits = None
            loss_val = 0.0
            for _ in range(cls_steps):
                logits = self.classifier(self.dropout(feat))
                loss = F.cross_entropy(logits, target_idx)
                self.opt_cls.zero_grad(set_to_none=True)
                loss.backward()
                self.opt_cls.step()
                loss_val = loss.item()
        return logits.detach(), loss_val

    def forward(self, x, y=None, mask=None, learn=False):
        feat = self.forward_snn(x, y=y, mask=mask, learn=learn)
        with torch.no_grad():
            logits = self.classifier(feat)
            probs = F.softmax(logits, dim=1)
        return probs, logits, feat


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
        feat = model.forward_snn(xb, y=None, mask=None, learn=False)
        logits = model.classifier(feat)
        correct += (logits.argmax(dim=1) == yb.argmax(dim=1).to(device)).sum().item()
        total += xb.shape[0]
    model.train()
    return correct / max(total, 1)


def train_epoch(model, loader, device):
    model.train()
    correct = total = 0
    loss_sum = 0.0
    n_batches = len(loader)

    for i, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.float().to(device)
        B = xb.shape[0]
        target_idx = yb.argmax(dim=1)
        yn = neg_labels(yb, device)

        X_cat = torch.cat([xb, xb], dim=0)
        Y_cat = torch.cat([yb, yn], dim=0)
        mask = torch.cat(
            [torch.ones(B, 1, device=device), torch.zeros(B, 1, device=device)],
            dim=0,
        )

        # 1) Local CSDP training of hidden spiking layers (pos + neg phases)
        model.forward_snn(X_cat, y=Y_cat, mask=mask, learn=True)

        # 2) Extract unclamped (y=0) spikes matching test-time inference & train nn.Linear
        feat_clean = model.forward_snn(xb, y=None, mask=None, learn=False)
        logits, loss_val = model.step_classifier(feat_clean, target_idx)

        batch_correct = (logits.argmax(dim=1) == target_idx).sum().item()
        correct += batch_correct
        loss_sum += loss_val * B
        total += B

        print(
            f"  Batch {i + 1:3d}/{n_batches}  loss: {loss_sum / total:.4f}  "
            f"acc: {batch_correct / B:.4f}  avg: {correct / total:.4f}",
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
        drop_last=False,
    )
    return tr_ld, te_ld


def train_mnist(
    data_dir=data_dir,
    epochs=num_epochs,
    device="cpu",
    batch_size=batch_size,
    dev_batch_size=dev_batch_size,
):
    device = torch.device(device) if isinstance(device, str) else device
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr_ld, te_ld = load_data(data_dir, batch_size, dev_batch_size)
    model = CSDPSNNLinearZ1Z2(device).to(device)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            model.opt_cls, T_max=max(epochs, 1), eta_min=lr_cls * 0.05
        )
        if epochs > 1
        else None
    )

    t0 = time.time()
    print(f"Test accuracy: {evaluate(model, te_ld, device):.4f} ", flush=True)

    best = 0.0
    elapsed = 0.0
    for ep in range(1, epochs + 1):
        tr_acc = train_epoch(model, tr_ld, device)
        te_acc = evaluate(model, te_ld, device)
        if scheduler is not None:
            scheduler.step()
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
    parser = argparse.ArgumentParser(
        description="MNIST CSDP-SNN + Linear Classifier on z1+z2 (6000 -> 10, zscore)"
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
    args = parser.parse_args()

    print(
        f"MNIST CSDP-SNN + Linear (z1+z2: {hid1 + hid2} -> {out_dim}) | "
        f"decode=zscore, lr_cls={lr_cls}, wd_cls={wd_cls}, dropout={cls_dropout} | "
        f"epochs={args.epochs}, device={args.device}",
        flush=True,
    )

    train_mnist(
        data_dir=args.data_dir,
        epochs=args.epochs,
        device=args.device,
        batch_size=args.batch_size,
        dev_batch_size=args.dev_batch_size,
    )


if __name__ == "__main__":
    main()
