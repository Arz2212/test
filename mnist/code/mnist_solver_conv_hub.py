#!/usr/bin/env python3
"""MNIST 100%-local Conv+FC CSDP-SNN with preserved Hub-Neuron modulation.

Architecture (identical for MNIST and KMNIST):
    Input z0 (B, 1, 28, 28) [Bernoulli spike encoding]
    -> Conv1 CSDP-Hub: 64 ch, 5x5 stride=2 pad=2 (14x14) -> 2x2 spike max-pool (7x7x64 = 3136)
    -> Conv2 CSDP-Hub: 128 ch, 3x3 stride=1 pad=1 (7x7x128 = 6272)
    -> FC CSDP-Hub:    6272 -> 1000
    -> Local single-layer softmax-delta readout on multi-scale spike counts
       (3136 + 6272 + 1000 = 10408 -> 10), trained strictly under @torch.no_grad()
       on unclamped (y=0) passes (zero backpropagation, zero autograd).

Hub-neuron logic is preserved identically on every hidden layer:
    hub_fc (fixed constant weights, frozen) -> hub_lif -> hub trace ->
    hub_mod(tr, hub, phase, zmax) = ((hub / zmax) ** hub_power - phase) * tr
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from spikingjelly.activation_based import functional, layer, neuron
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────── fixed SNN hyperparameters ───────────────────────
T = 30
dt = 3.0
tau_m = 100.0
v_thr = 0.055
tau_tr = 13.0
R_m = 1.0
R_y = 1.0
inh_R = 0.05
weight_bound = 1.0
seed = 1234

# Hub-neuron hyperparameters (preserved hub logic)
tau_m_hab = 4.87563
v_thr_hab = 0.249441
tau_hub = 4.39997
hub_power = 1.20313
hub_w_c1 = 0.002
hub_w_c2 = 0.004
hub_w_fc = 0.0503053

# Architecture & optimization
c1_ch, k1, s1, p1 = 64, 5, 2, 2
c2_ch, k2, s2, p2 = 128, 3, 1, 1
hid_fc = 1000
out_dim = 10
conv_init_scale = 0.5
fc_init_scale = 1.0

eta = 0.002
eta_out = 0.002
eta_min_ratio = 0.05
wd_out = 1e-3
out_dropout = 0.15

batch_size = 500
dev_batch_size = 1000
num_epochs = 30

default_data_dir = str(Path(__file__).resolve().parents[2] / "data" / "mnist")


def nn_param(*shape: int, bound: float = 1.0) -> torch.nn.Parameter:
    p = torch.nn.Parameter(torch.empty(*shape))
    torch.nn.init.uniform_(p, -bound, bound)
    return p


def uniform_neg_labels(y_onehot: torch.Tensor) -> torch.Tensor:
    """Strictly uniform sampling over the (out_dim - 1) incorrect classes."""
    B = y_onehot.shape[0]
    correct = y_onehot.argmax(dim=1)
    r = torch.randint(0, out_dim - 1, (B,), device=y_onehot.device)
    neg = r + (r >= correct).long()
    return F.one_hot(neg, out_dim).float()


def decode_zscore(spk_cnt: torch.Tensor) -> torch.Tensor:
    mu = spk_cnt.mean(dim=1, keepdim=True)
    std = spk_cnt.std(dim=1, keepdim=True, unbiased=False)
    return (spk_cnt - mu) / (std + 1e-5)


class ConvCSDPHubLayer(torch.nn.Module):
    """Spiking 2D convolutional layer trained via 100%-local CSDP + Hub Neuron."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        in_hw: int,
        kernel_size: int,
        stride: int,
        padding: int,
        pool_size: int,
        hub_w: float,
        device: torch.device,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool_size = pool_size
        self.out_hw = (in_hw + 2 * padding - kernel_size) // stride + 1
        self.n_loc = self.out_hw * self.out_hw
        self.n_units = self.n_loc * out_ch
        self.pool_hw = self.out_hw // pool_size if pool_size > 1 else self.out_hw
        self.pool_dim = self.pool_hw * self.pool_hw * out_ch

        self.W = nn_param(out_ch, in_ch, kernel_size, kernel_size, bound=conv_init_scale)
        self.M = nn_param(out_ch, out_ch, bound=conv_init_scale)
        self.Vy = nn_param(out_dim, self.n_units, bound=conv_init_scale)

        self.lif = neuron.LIFNode(tau=tau_m / dt, v_threshold=v_thr, v_reset=0.0)

        # Preserved Hub Neuron logic
        self.hub_fc = layer.Linear(self.n_units, 1, bias=False)
        self.hub_lif = neuron.LIFNode(tau=tau_m_hab, v_threshold=v_thr_hab, v_reset=0.0)
        torch.nn.init.constant_(self.hub_fc.weight, hub_w)
        self.hub_fc.weight.requires_grad_(False)
        self.register_buffer("zmax", torch.tensor(1.0))

        self.to(device)

    def hub_mod(
        self, tr: torch.Tensor, hub: torch.Tensor, phase: torch.Tensor, update_zmax: bool = True
    ) -> torch.Tensor:
        if update_zmax:
            torch.maximum(self.zmax, hub.max(), out=self.zmax)
        factor = (hub / self.zmax) ** hub_power - phase
        return factor * tr

    def params(self) -> list[torch.nn.Parameter]:
        return [self.W, self.M, self.Vy]

    def clamp_weights(self) -> None:
        self.W.data.clamp_(-weight_bound, weight_bound)
        self.M.data.fill_diagonal_(0.0)
        self.M.data.clamp_(0.0, weight_bound)
        self.Vy.data.clamp_(-weight_bound, weight_bound)


class FCCSDPHubLayer(torch.nn.Module):
    """Spiking FC layer trained via 100%-local CSDP + Hub Neuron."""

    def __init__(self, in_dim: int, hid_dim: int, hub_w: float, device: torch.device):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim

        self.W = nn_param(in_dim, hid_dim, bound=fc_init_scale)
        self.M = nn_param(hid_dim, hid_dim, bound=fc_init_scale)
        self.Vy = nn_param(out_dim, hid_dim, bound=fc_init_scale)

        self.lif = neuron.LIFNode(tau=tau_m / dt, v_threshold=v_thr, v_reset=0.0)

        # Preserved Hub Neuron logic
        self.hub_fc = layer.Linear(hid_dim, 1, bias=False)
        self.hub_lif = neuron.LIFNode(tau=tau_m_hab, v_threshold=v_thr_hab, v_reset=0.0)
        torch.nn.init.constant_(self.hub_fc.weight, hub_w)
        self.hub_fc.weight.requires_grad_(False)
        self.register_buffer("zmax", torch.tensor(1.0))

        self.to(device)

    def hub_mod(
        self, tr: torch.Tensor, hub: torch.Tensor, phase: torch.Tensor, update_zmax: bool = True
    ) -> torch.Tensor:
        if update_zmax:
            torch.maximum(self.zmax, hub.max(), out=self.zmax)
        factor = (hub / self.zmax) ** hub_power - phase
        return factor * tr

    def params(self) -> list[torch.nn.Parameter]:
        return [self.W, self.M, self.Vy]

    def clamp_weights(self) -> None:
        self.W.data.clamp_(-weight_bound, weight_bound)
        self.M.data.fill_diagonal_(0.0)
        self.M.data.clamp_(0.0, weight_bound)
        self.Vy.data.clamp_(-weight_bound, weight_bound)


class ConvCSDPHubSNN(torch.nn.Module):
    def __init__(self, device: torch.device, amp_bf16: bool = True):
        super().__init__()
        self.device = device
        self.amp_bf16 = amp_bf16 and (device.type == "cuda")

        self.conv1 = ConvCSDPHubLayer(
            in_ch=1, out_ch=c1_ch, in_hw=28, kernel_size=k1, stride=s1,
            padding=k1 // 2, pool_size=p1, hub_w=hub_w_c1, device=device,
        )
        self.conv2 = ConvCSDPHubLayer(
            in_ch=c1_ch, out_ch=c2_ch, in_hw=self.conv1.pool_hw, kernel_size=k2, stride=s2,
            padding=k2 // 2, pool_size=p2, hub_w=hub_w_c2, device=device,
        )
        self.fc = FCCSDPHubLayer(
            in_dim=self.conv2.pool_dim, hid_dim=hid_fc, hub_w=hub_w_fc, device=device,
        )

        # Multi-scale local delta readout weights (p1: 3136, p2: 6272, fc: 1000 -> 10)
        self.C_c1 = torch.nn.Parameter(torch.zeros(self.conv1.pool_dim, out_dim))
        self.C_c2 = torch.nn.Parameter(torch.zeros(self.conv2.pool_dim, out_dim))
        self.C_fc = torch.nn.Parameter(torch.zeros(hid_fc, out_dim))
        self.b_out = torch.nn.Parameter(torch.zeros(out_dim))

        self.to(device)

        self.opt_hidden = torch.optim.Adam(
            self.conv1.params() + self.conv2.params() + self.fc.params(),
            lr=eta,
            fused=(device.type == "cuda"),
        )
        self.opt_out = torch.optim.Adam(
            [self.C_c1, self.C_c2, self.C_fc, self.b_out],
            lr=eta_out,
            fused=(device.type == "cuda"),
        )

    def _conv_step(
        self,
        layer_mod: ConvCSDPHubLayer,
        x_in: torch.Tensor,
        z_prev: torch.Tensor,
        tr: torch.Tensor,
        hub: torch.Tensor | None,
        y: torch.Tensor | None,
        mask: torch.Tensor | None,
        learn: bool,
        collect_diag: bool,
        d: float,
        decay_hub: float,
    ):
        B = x_in.shape[0]
        C = layer_mod.out_ch
        H = layer_mod.out_hw

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp_bf16):
            I = F.conv2d(x_in, layer_mod.W, stride=layer_mod.stride, padding=layer_mod.padding) * R_m
            M_1x1 = layer_mod.M.T.unsqueeze(-1).unsqueeze(-1)
            I = I - F.conv2d(z_prev, M_1x1) * inh_R
            if y is not None:
                I = I + ((y @ layer_mod.Vy) * R_y).view(B, C, H, H)

        z = layer_mod.lif(I.float())
        tr = tr + d * (z - tr)

        h_spk_val = 0.0
        if learn or collect_diag:
            z_flat = z.reshape(B, layer_mod.n_units)
            h_spk = layer_mod.hub_lif(layer_mod.hub_fc(z_flat))
            hub = hub + decay_hub * (-hub + h_spk.float())
            if collect_diag:
                h_spk_val = h_spk.float().mean().item()
            if learn:
                tr_flat = tr.reshape(B, layer_mod.n_units)
                mod_flat = layer_mod.hub_mod(tr_flat, hub, mask, update_zmax=True)
                mod_4d = mod_flat.view(B, C, H, H)
                inv_loc = 1.0 / layer_mod.n_loc
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp_bf16):
                    # Fast cuDNN kernel for local CSDP patch correlation sum_{b,h,w} x_patch * mod
                    gW = torch.nn.grad.conv2d_weight(
                        x_in,
                        layer_mod.W.shape,
                        mod_4d * (R_m * inv_loc),
                        stride=layer_mod.stride,
                        padding=layer_mod.padding,
                    )
                    zp_bcn = z_prev.reshape(B, C, -1)
                    m_bcn = (mod_4d * (inh_R * inv_loc)).reshape(B, C, -1)
                    gM = torch.bmm(zp_bcn, m_bcn.transpose(1, 2)).sum(dim=0)
                    gVy = y.T @ (mod_flat * R_y)
                layer_mod.W.grad = gW.float()
                layer_mod.M.grad = gM.float()
                layer_mod.Vy.grad = gVy.float()

        p_img = F.max_pool2d(z, layer_mod.pool_size, layer_mod.pool_size) if layer_mod.pool_size > 1 else z
        return z, p_img, tr, hub, h_spk_val

    @torch.no_grad()
    def simulate(
        self,
        x_flat: torch.Tensor,
        y: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        learn_hidden: bool = False,
        collect_diag: bool = False,
    ) -> dict[str, torch.Tensor | float]:
        B = x_flat.shape[0]
        dv = self.device
        x_img = x_flat.view(B, 1, 28, 28)

        functional.reset_net(self.conv1.lif)
        functional.reset_net(self.conv2.lif)
        functional.reset_net(self.fc.lif)

        if learn_hidden or collect_diag:
            functional.reset_net(self.conv1.hub_lif)
            functional.reset_net(self.conv2.hub_lif)
            functional.reset_net(self.fc.hub_lif)
            hub1 = torch.zeros(B, 1, device=dv)
            hub2 = torch.zeros(B, 1, device=dv)
            hub_fc = torch.zeros(B, 1, device=dv)
        else:
            hub1 = hub2 = hub_fc = None

        z1_prev = torch.zeros(B, self.conv1.out_ch, self.conv1.out_hw, self.conv1.out_hw, device=dv)
        tr1 = torch.zeros_like(z1_prev)
        z2_prev = torch.zeros(B, self.conv2.out_ch, self.conv2.out_hw, self.conv2.out_hw, device=dv)
        tr2 = torch.zeros_like(z2_prev)
        zfc_prev = torch.zeros(B, hid_fc, device=dv)
        tr_fc = torch.zeros_like(zfc_prev)

        cnt_c1 = torch.zeros(B, self.conv1.pool_dim, device=dv)
        cnt_c2 = torch.zeros(B, self.conv2.pool_dim, device=dv)
        cnt_fc = torch.zeros(B, hid_fc, device=dv)

        d = dt / tau_tr
        decay_hub = 1.0 / tau_hub
        r1_sum = r2_sum = rfc_sum = h1_sum = h2_sum = hfc_sum = 0.0

        for _ in range(T):
            z0_img = (torch.rand_like(x_img) < x_img).float()

            z1, p1_img, tr1, hub1, h1_v = self._conv_step(
                self.conv1, z0_img, z1_prev, tr1, hub1, y, mask,
                learn_hidden, collect_diag, d, decay_hub,
            )
            z2, p2_img, tr2, hub2, h2_v = self._conv_step(
                self.conv2, p1_img, z2_prev, tr2, hub2, y, mask,
                learn_hidden, collect_diag, d, decay_hub,
            )

            c1_flat = p1_img.reshape(B, -1)
            c2_flat = p2_img.reshape(B, -1)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp_bf16):
                Ifc = (c2_flat @ self.fc.W) * R_m - (zfc_prev @ self.fc.M) * inh_R
                if y is not None:
                    Ifc = Ifc + (y @ self.fc.Vy) * R_y
            zfc = self.fc.lif(Ifc.float())
            tr_fc = tr_fc + d * (zfc - tr_fc)

            if collect_diag:
                r1_sum += z1.mean().item()
                r2_sum += z2.mean().item()
                rfc_sum += zfc.mean().item()
                h1_sum += h1_v
                h2_sum += h2_v

            if learn_hidden or collect_diag:
                hfc_spk = self.fc.hub_lif(self.fc.hub_fc(zfc))
                hub_fc = hub_fc + decay_hub * (-hub_fc + hfc_spk.float())
                if collect_diag:
                    hfc_sum += hfc_spk.float().mean().item()
                if learn_hidden:
                    mod_fc = self.fc.hub_mod(tr_fc, hub_fc, mask, update_zmax=True)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp_bf16):
                        self.fc.W.grad = (c2_flat.T @ (mod_fc * R_m)).float()
                        self.fc.M.grad = (zfc_prev.T @ (mod_fc * inh_R)).float()
                        self.fc.Vy.grad = (y.T @ (mod_fc * R_y)).float()

            if learn_hidden:
                self.opt_hidden.step()
                self.conv1.clamp_weights()
                self.conv2.clamp_weights()
                self.fc.clamp_weights()

            cnt_c1 = cnt_c1 + c1_flat
            cnt_c2 = cnt_c2 + c2_flat
            cnt_fc = cnt_fc + zfc
            z1_prev = z1
            z2_prev = z2
            zfc_prev = zfc

        res: dict[str, torch.Tensor | float] = {
            "cnt_c1": cnt_c1,
            "cnt_c2": cnt_c2,
            "cnt_fc": cnt_fc,
        }
        if collect_diag:
            res.update({
                "r1": r1_sum / T,
                "r2": r2_sum / T,
                "rfc": rfc_sum / T,
                "h1": h1_sum / T,
                "h2": h2_sum / T,
                "hfc": hfc_sum / T,
                "zm1": self.conv1.zmax.item(),
                "zm2": self.conv2.zmax.item(),
                "zmfc": self.fc.zmax.item(),
            })
        return res

    @torch.no_grad()
    def local_delta_update(self, res: dict, y_onehot: torch.Tensor) -> torch.Tensor:
        """Purely local single-layer softmax-delta rule (zero autograd/backprop)."""
        B = y_onehot.shape[0]
        f1 = decode_zscore(res["cnt_c1"])
        f2 = decode_zscore(res["cnt_c2"])
        ff = decode_zscore(res["cnt_fc"])
        if out_dropout > 0.0:
            scale = 1.0 / (1.0 - out_dropout)
            f1 = f1 * ((torch.rand_like(f1) > out_dropout).float() * scale)
            f2 = f2 * ((torch.rand_like(f2) > out_dropout).float() * scale)
            ff = ff * ((torch.rand_like(ff) > out_dropout).float() * scale)

        logits = f1 @ self.C_c1 + f2 @ self.C_c2 + ff @ self.C_fc + self.b_out
        probs = F.softmax(logits, dim=1)
        err = (probs - y_onehot) / B

        self.C_c1.grad = f1.T @ err + wd_out * self.C_c1.data
        self.C_c2.grad = f2.T @ err + wd_out * self.C_c2.data
        self.C_fc.grad = ff.T @ err + wd_out * self.C_fc.data
        self.b_out.grad = err.sum(dim=0)
        self.opt_out.step()
        return logits

    @torch.no_grad()
    def predict(self, x_flat: torch.Tensor) -> torch.Tensor:
        res = self.simulate(x_flat, learn_hidden=False)
        f1 = decode_zscore(res["cnt_c1"])
        f2 = decode_zscore(res["cnt_c2"])
        ff = decode_zscore(res["cnt_fc"])
        return f1 @ self.C_c1 + f2 @ self.C_c2 + ff @ self.C_fc + self.b_out


@torch.no_grad()
def evaluate(model: ConvCSDPHubSNN, loader: DataLoader, device: torch.device) -> float:
    correct = total = 0
    for xb, yb in loader:
        scores = model.predict(xb.to(device))
        correct += (scores.argmax(dim=1) == yb.argmax(dim=1).to(device)).sum().item()
        total += xb.shape[0]
    return correct / max(total, 1)


@torch.no_grad()
def train_epoch(model: ConvCSDPHubSNN, loader: DataLoader, device: torch.device) -> tuple[float, dict]:
    correct = total = 0
    diag: dict[str, float] = {}
    n_batches = len(loader)
    for i, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.float().to(device)
        B = xb.shape[0]
        yn = uniform_neg_labels(yb)

        X_cat = torch.cat([xb, xb], dim=0)
        Y_cat = torch.cat([yb, yn], dim=0)
        mask = torch.cat(
            [torch.ones(B, 1, device=device), torch.zeros(B, 1, device=device)], dim=0
        )

        # 1) Local CSDP training of hidden spiking layers (with Hub Neurons)
        is_last = i == n_batches - 1
        pos_res = model.simulate(X_cat, y=Y_cat, mask=mask, learn_hidden=True, collect_diag=is_last)
        if is_last:
            for k in ("r1", "r2", "rfc", "h1", "h2", "hfc", "zm1", "zm2", "zmfc"):
                diag[k] = float(pos_res[k])

        # 2) Clean (y=0) pass + purely local delta update for output readout
        clean_res = model.simulate(xb, learn_hidden=False)
        logits = model.local_delta_update(clean_res, yb)

        correct += (logits.argmax(dim=1) == yb.argmax(dim=1)).sum().item()
        total += B

    return correct / max(total, 1), diag


def load_data(data_dir: str, bs: int, dev_bs: int) -> tuple[DataLoader, DataLoader]:
    p = Path(data_dir).expanduser()
    required = ("trainX.npy", "trainY.npy", "testX.npy", "testY.npy")
    if not all((p / name).exists() for name in required):
        from download_mnist import ensure_mnist

        ensure_mnist(p)
    X_tr = torch.from_numpy(np.load(p / "trainX.npy")).float()
    Y_tr = torch.from_numpy(np.load(p / "trainY.npy")).float()
    X_te = torch.from_numpy(np.load(p / "testX.npy")).float()
    Y_te = torch.from_numpy(np.load(p / "testY.npy")).float()
    tr_ld = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=bs, shuffle=True, drop_last=True)
    te_ld = DataLoader(TensorDataset(X_te, Y_te), batch_size=dev_bs, shuffle=False, drop_last=False)
    return tr_ld, te_ld


def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST 100%-local Conv+FC CSDP-Hub-SNN")
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
        f"MNIST ConvCSDPHubSNN (100% local, preserved hub logic) | "
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
            json.dumps({"dataset": "mnist", "best": best, "best_ep": best_ep, "seed": args.seed, "history": history}, indent=2)
        )


if __name__ == "__main__":
    main()
