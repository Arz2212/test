#!/usr/bin/env python3
"""100% Memristor-Native Multi-Scale Tiled Crossbar CSDP-SNN with Preserved Hub-Neuron Logic (MNIST).

Strict memristive-chip compatibility invariants:
1. ZERO weight sharing: EVERY synaptic conductance is an independent physical memristor
   inside a locally-connected mini-crossbar tile or dense crossbar array:
     - Stage 1 (tile1):       196 independent 5x5 mini-crossbars (25 -> 64) + 2x2 binary OR-pooling (7x7)
     - Stage 2 (tile2):        49 independent 3x3 mini-crossbars (576 -> 80) on 7x7
     - Coarse branch (tc):     25 independent 11x11 mini-crossbars (121 -> 32)
     - Global crossbar (fc):    1 dense crossbar (4720 -> 2000)
     - Readout crossbar:        1 dense crossbar (9856 -> 10), strictly bounded in [-1, 1]
   Total: 17.54M independent physical memristors, ZERO convolutional kernel sharing.
2. 100% LOCAL learning: ALL simulation and weight plasticity execute under @torch.no_grad().
   Zero backpropagation, zero autograd across layers.
3. Preserved Hub-Neuron logic (1-to-1 with original PyTorch CSDP-Hub):
   hub_fc (frozen constant conductances) -> hub_lif -> hub trace ->
   ((hub / zmax) ** hub_power - phase) * tr
4. Memristor-native readout (NO zscore / global-std normalization!):
   Pure crossbar current integration on raw firing rates (cnt_ro / T), mathematically
   equivalent to non-resetting integrate-only output neurons V_out(T) / T, trained locally
   via the 10-output softmax-delta rule with physical conductance clamping [-1, 1].
5. ZERO data augmentation.
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


def nn_param(*shape: int, bound: float = 1.0) -> torch.nn.Parameter:
    p = torch.nn.Parameter(torch.empty(*shape))
    torch.nn.init.uniform_(p, -bound, bound)
    return p


def uniform_neg_labels(y_onehot: torch.Tensor, out_dim: int = 10) -> torch.Tensor:
    B = y_onehot.shape[0]
    correct = y_onehot.argmax(dim=1)
    r = torch.randint(0, out_dim - 1, (B,), device=y_onehot.device)
    neg = r + (r >= correct).long()
    return F.one_hot(neg, out_dim).float()


def hard_neg_labels(
    y_onehot: torch.Tensor, scores: torch.Tensor, hard_prob: float, temp: float = 2.0
) -> torch.Tensor:
    B, out_dim = y_onehot.shape
    unif = uniform_neg_labels(y_onehot, out_dim)
    if hard_prob <= 0.0:
        return unif
    masked = scores.clone()
    masked[y_onehot > 0.5] = -1e9
    probs = F.softmax(masked / temp, dim=1)
    hard_idx = torch.multinomial(probs, 1).squeeze(1)
    hard = F.one_hot(hard_idx, out_dim).float()
    use_hard = (torch.rand(B, 1, device=y_onehot.device) < hard_prob).float()
    return use_hard * hard + (1.0 - use_hard) * unif


class TiledCrossbarCSDPLayer(torch.nn.Module):
    """Array of L independent locally-connected memristive mini-crossbars (ZERO weight sharing)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        in_hw: int,
        kernel_size: int,
        stride: int,
        padding: int,
        pool_size: int,
        out_dim: int,
        hub_w: float,
        args: argparse.Namespace,
        device: torch.device,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool_size = pool_size
        self.args = args

        self.out_hw = (in_hw + 2 * padding - kernel_size) // stride + 1
        self.n_tiles = self.out_hw * self.out_hw
        self.patch_dim = in_ch * kernel_size * kernel_size
        self.n_units = self.n_tiles * out_ch
        self.pool_hw = self.out_hw // pool_size if pool_size > 1 else self.out_hw
        self.pool_dim = self.pool_hw * self.pool_hw * out_ch

        b = args.tile_init_scale
        # Independent physical memristors per spatial tile (ZERO weight sharing!)
        self.W = nn_param(self.n_tiles, self.patch_dim, out_ch, bound=b)
        self.M = nn_param(self.n_tiles, out_ch, out_ch, bound=b)
        self.Vy = nn_param(self.n_tiles, out_dim, out_ch, bound=b)
        self.register_buffer("m_hollow", (1.0 - torch.eye(out_ch, device=device)).unsqueeze(0))

        self.lif = neuron.LIFNode(tau=args.tau_m / args.dt, v_threshold=args.v_thr, v_reset=0.0)

        # Preserved Hub-Neuron logic (1-to-1)
        self.hub_fc = layer.Linear(self.n_units, 1, bias=False)
        self.hub_lif = neuron.LIFNode(
            tau=args.tau_m_hab, v_threshold=args.v_thr_hab, v_reset=0.0
        )
        torch.nn.init.constant_(self.hub_fc.weight, hub_w)
        self.hub_fc.weight.requires_grad_(False)
        self.register_buffer("zmax", torch.tensor(1.0))

        self.to(device)

    def params(self) -> list[torch.nn.Parameter]:
        return [self.W, self.M, self.Vy]

    def clamp_weights(self) -> None:
        wb = self.args.weight_bound
        self.W.data.clamp_(-wb, wb)
        self.M.data.mul_(self.m_hollow)
        self.M.data.clamp_(0.0, wb)
        self.Vy.data.clamp_(-wb, wb)

    def step(
        self,
        x_img: torch.Tensor,
        z_prev: torch.Tensor,
        tr: torch.Tensor,
        hub: torch.Tensor | None,
        y: torch.Tensor | None,
        mask: torch.Tensor | None,
        learn: bool,
        collect_diag: bool,
        d: float,
        decay_hub: float,
        use_bf16: bool,
    ):
        B = x_img.shape[0]
        L = self.n_tiles
        C = self.out_ch

        x_p = F.unfold(
            x_img,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        ).permute(2, 0, 1).contiguous()  # (L, B, P)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            I = torch.bmm(x_p, self.W) * self.args.R_m - torch.bmm(z_prev, self.M) * self.args.inh_R_tile
            if y is not None:
                y_l = y.unsqueeze(0).expand(L, B, -1)
                I = I + torch.bmm(y_l, self.Vy) * self.args.R_y

        z = self.lif(I.float())  # (L, B, C)
        tr = tr + d * (z - tr)

        h_spk_val = 0.0
        if learn or collect_diag:
            z_flat = z.permute(1, 0, 2).reshape(B, self.n_units)
            h_spk = self.hub_lif(self.hub_fc(z_flat))
            hub = hub + decay_hub * (-hub + h_spk.float())
            if collect_diag:
                h_spk_val = h_spk.float().mean().item()
            if learn:
                torch.maximum(self.zmax, hub.max(), out=self.zmax)
                factor = (hub / self.zmax) ** self.args.hub_power - mask  # (B, 1)
                mod = factor.unsqueeze(0) * tr  # (L, B, C)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    # 100% LOCAL per-memristor update (ZERO sharing across tiles!)
                    gW = torch.bmm(x_p.transpose(1, 2), mod * self.args.R_m)
                    gM = torch.bmm(z_prev.transpose(1, 2), mod * self.args.inh_R_tile)
                    y_l = y.unsqueeze(0).expand(L, B, -1)
                    gVy = torch.bmm(y_l.transpose(1, 2), mod * self.args.R_y)
                self.W.grad = gW.float()
                self.M.grad = gM.float()
                self.Vy.grad = gVy.float()

        z_img = z.permute(1, 2, 0).reshape(B, C, self.out_hw, self.out_hw)
        p_img = F.max_pool2d(z_img, self.pool_size, self.pool_size) if self.pool_size > 1 else z_img
        return z, p_img, tr, hub, h_spk_val


class FCCrossbarCSDPLayer(torch.nn.Module):
    """Dense physical memristor crossbar layer with preserved Hub-Neuron logic."""

    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        hub_w: float,
        args: argparse.Namespace,
        device: torch.device,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.args = args

        self.W = nn_param(in_dim, hid_dim, bound=args.fc_init_scale)
        self.M = nn_param(hid_dim, hid_dim, bound=args.fc_init_scale)
        self.Vy = nn_param(out_dim, hid_dim, bound=args.fc_init_scale)

        self.lif = neuron.LIFNode(tau=args.tau_m / args.dt, v_threshold=args.v_thr, v_reset=0.0)

        # Preserved Hub-Neuron logic (1-to-1)
        self.hub_fc = layer.Linear(hid_dim, 1, bias=False)
        self.hub_lif = neuron.LIFNode(
            tau=args.tau_m_hab, v_threshold=args.v_thr_hab, v_reset=0.0
        )
        torch.nn.init.constant_(self.hub_fc.weight, hub_w)
        self.hub_fc.weight.requires_grad_(False)
        self.register_buffer("zmax", torch.tensor(1.0))

        self.to(device)

    def params(self) -> list[torch.nn.Parameter]:
        return [self.W, self.M, self.Vy]

    def clamp_weights(self) -> None:
        wb = self.args.weight_bound
        self.W.data.clamp_(-wb, wb)
        self.M.data.fill_diagonal_(0.0)
        self.M.data.clamp_(0.0, wb)
        self.Vy.data.clamp_(-wb, wb)

    def step(
        self,
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
        use_bf16: bool,
    ):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            I = (x_in @ self.W) * self.args.R_m - (z_prev @ self.M) * self.args.inh_R_fc
            if y is not None:
                I = I + (y @ self.Vy) * self.args.R_y

        z = self.lif(I.float())
        tr = tr + d * (z - tr)

        h_spk_val = 0.0
        if learn or collect_diag:
            h_spk = self.hub_lif(self.hub_fc(z))
            hub = hub + decay_hub * (-hub + h_spk.float())
            if collect_diag:
                h_spk_val = h_spk.float().mean().item()
            if learn:
                torch.maximum(self.zmax, hub.max(), out=self.zmax)
                factor = (hub / self.zmax) ** self.args.hub_power - mask
                mod = factor * tr
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    gW = x_in.T @ (mod * self.args.R_m)
                    gM = z_prev.T @ (mod * self.args.inh_R_fc)
                    gVy = y.T @ (mod * self.args.R_y)
                self.W.grad = gW.float()
                self.M.grad = gM.float()
                self.Vy.grad = gVy.float()

        return z, tr, hub, h_spk_val


class MemristorMultiScaleCSDPHubSNN(torch.nn.Module):
    """Multi-scale 2-stage tiled crossbar + coarse tiles + FC + bounded readout crossbar."""

    def __init__(self, args: argparse.Namespace, device: torch.device):
        super().__init__()
        self.args = args
        self.device = device
        self.out_dim = 10

        # Stage 1: 196 independent 5x5 mini-crossbars (25 -> 64) + 2x2 binary OR-pooling (7x7)
        self.tile1 = TiledCrossbarCSDPLayer(
            in_ch=1,
            out_ch=args.c1,
            in_hw=28,
            kernel_size=args.k1,
            stride=args.s1,
            padding=args.pad1,
            pool_size=args.p1,
            out_dim=self.out_dim,
            hub_w=args.hub_w_t1,
            args=args,
            device=device,
        )

        # Stage 2: 49 independent 3x3 mini-crossbars (576 -> 80) on 7x7
        self.tile2 = TiledCrossbarCSDPLayer(
            in_ch=args.c1,
            out_ch=args.c2,
            in_hw=self.tile1.pool_hw,
            kernel_size=args.k2,
            stride=args.s2,
            padding=args.pad2,
            pool_size=args.p2,
            out_dim=self.out_dim,
            hub_w=args.hub_w_t2,
            args=args,
            device=device,
        )

        # Parallel coarse tiles: 25 independent 11x11 mini-crossbars (121 -> 32)
        self.tile_coarse = TiledCrossbarCSDPLayer(
            in_ch=1,
            out_ch=args.c_coarse,
            in_hw=28,
            kernel_size=11,
            stride=5,
            padding=2,
            pool_size=1,
            out_dim=self.out_dim,
            hub_w=args.hub_w_coarse,
            args=args,
            device=device,
        )

        # Global associative FC crossbar integrating tile2 + coarse tiles
        fc_in = self.tile2.pool_dim + self.tile_coarse.pool_dim
        self.fc = FCCrossbarCSDPLayer(
            in_dim=fc_in,
            hid_dim=args.hid_fc,
            out_dim=self.out_dim,
            hub_w=args.hub_w_fc,
            args=args,
            device=device,
        )

        # Readout crossbar (raw firing rates cnt_ro / T, NO zscore, strictly bounded in [-1, 1])
        self.total_ro_dim = (
            self.tile1.pool_dim + self.tile2.pool_dim + self.tile_coarse.pool_dim + args.hid_fc
        )
        self.C_out = torch.nn.Parameter(torch.zeros(self.total_ro_dim, self.out_dim))
        self.b_out = torch.nn.Parameter(torch.zeros(self.out_dim))

        self.to(device)

        hidden_params = (
            self.tile1.params()
            + self.tile2.params()
            + self.tile_coarse.params()
            + self.fc.params()
        )
        self.opt_hidden = torch.optim.Adam(
            hidden_params, lr=args.eta, fused=(device.type == "cuda")
        )
        self.opt_out = torch.optim.Adam(
            [self.C_out, self.b_out], lr=args.eta_out, fused=(device.type == "cuda")
        )

    @torch.no_grad()
    def simulate(
        self,
        x_flat: torch.Tensor,
        y: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        learn_hidden: bool = False,
        collect_diag: bool = False,
        T_override: int | None = None,
    ) -> dict[str, torch.Tensor | float]:
        args = self.args
        T_steps = T_override if T_override is not None else args.T
        B = x_flat.shape[0]
        dv = self.device
        x_img = x_flat.view(B, 1, 28, 28)
        use_bf16 = args.amp_bf16 and dv.type == "cuda"

        functional.reset_net(self.tile1.lif)
        functional.reset_net(self.tile2.lif)
        functional.reset_net(self.tile_coarse.lif)
        functional.reset_net(self.fc.lif)

        if learn_hidden or collect_diag:
            functional.reset_net(self.tile1.hub_lif)
            functional.reset_net(self.tile2.hub_lif)
            functional.reset_net(self.tile_coarse.hub_lif)
            functional.reset_net(self.fc.hub_lif)
            hub1 = torch.zeros(B, 1, device=dv)
            hub2 = torch.zeros(B, 1, device=dv)
            hub_c = torch.zeros(B, 1, device=dv)
            hub_fc = torch.zeros(B, 1, device=dv)
        else:
            hub1 = hub2 = hub_c = hub_fc = None

        z1_prev = torch.zeros(self.tile1.n_tiles, B, self.tile1.out_ch, device=dv)
        tr1 = torch.zeros_like(z1_prev)
        z2_prev = torch.zeros(self.tile2.n_tiles, B, self.tile2.out_ch, device=dv)
        tr2 = torch.zeros_like(z2_prev)
        zc_prev = torch.zeros(self.tile_coarse.n_tiles, B, self.tile_coarse.out_ch, device=dv)
        trc = torch.zeros_like(zc_prev)
        zfc_prev = torch.zeros(B, self.fc.hid_dim, device=dv)
        tr_fc = torch.zeros_like(zfc_prev)

        cnt_ro = torch.zeros(B, self.total_ro_dim, device=dv)

        d = args.dt / args.tau_tr
        decay_hub = 1.0 / args.tau_hub
        r1_sum = r2_sum = rfc_sum = h1_sum = h2_sum = hfc_sum = 0.0

        for _ in range(T_steps):
            z0_img = (torch.rand_like(x_img) < x_img).float()

            z1, p1_img, tr1, hub1, h1_v = self.tile1.step(
                z0_img, z1_prev, tr1, hub1, y, mask,
                learn_hidden, collect_diag, d, decay_hub, use_bf16,
            )
            p1_flat = p1_img.reshape(B, -1)

            z2, p2_img, tr2, hub2, h2_v = self.tile2.step(
                p1_img, z2_prev, tr2, hub2, y, mask,
                learn_hidden, collect_diag, d, decay_hub, use_bf16,
            )
            p2_flat = p2_img.reshape(B, -1)

            zc, pc_img, trc, hub_c, _ = self.tile_coarse.step(
                z0_img, zc_prev, trc, hub_c, y, mask,
                learn_hidden, collect_diag, d, decay_hub, use_bf16,
            )
            pc_flat = pc_img.reshape(B, -1)

            fc_in = torch.cat([p2_flat, pc_flat], dim=1)
            zfc, tr_fc, hub_fc, hfc_v = self.fc.step(
                fc_in, zfc_prev, tr_fc, hub_fc, y, mask,
                learn_hidden, collect_diag, d, decay_hub, use_bf16,
            )

            if learn_hidden:
                self.opt_hidden.step()
                self.tile1.clamp_weights()
                self.tile2.clamp_weights()
                self.tile_coarse.clamp_weights()
                self.fc.clamp_weights()

            ro_spk = torch.cat([p1_flat, p2_flat, pc_flat, zfc], dim=1)
            cnt_ro = cnt_ro + ro_spk

            if collect_diag:
                r1_sum += z1.mean().item()
                r2_sum += z2.mean().item()
                rfc_sum += zfc.mean().item()
                h1_sum += h1_v
                h2_sum += h2_v
                hfc_sum += hfc_v

            z1_prev, z2_prev, zc_prev, zfc_prev = z1, z2, zc, zfc

        res: dict[str, torch.Tensor | float] = {"cnt_ro": cnt_ro}
        if collect_diag:
            res.update({
                "r1": r1_sum / T_steps,
                "r2": r2_sum / T_steps,
                "rfc": rfc_sum / T_steps,
                "h1": h1_sum / T_steps,
                "h2": h2_sum / T_steps,
                "hfc": hfc_sum / T_steps,
                "zm1": self.tile1.zmax.item(),
                "zm2": self.tile2.zmax.item(),
                "zmfc": self.fc.zmax.item(),
            })
        return res

    @torch.no_grad()
    def crossbar_delta_update(self, cnt_ro: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        """Local 10-column crossbar delta update on raw firing rates (cnt_ro / T), NO zscore."""
        B = y_onehot.shape[0]
        rates = cnt_ro / self.args.T
        logits = rates @ self.C_out + self.b_out
        probs = F.softmax(logits, dim=1)
        err = (probs - y_onehot) / B
        self.C_out.grad = rates.T @ err + self.args.wd_out * self.C_out.data
        self.b_out.grad = err.sum(dim=0)
        self.opt_out.step()
        wb = self.args.weight_bound
        self.C_out.data.clamp_(-wb, wb)
        self.b_out.data.clamp_(-wb, wb)
        return logits

    @torch.no_grad()
    def predict(self, x_flat: torch.Tensor, T_override: int | None = None) -> torch.Tensor:
        T_use = T_override if T_override is not None else self.args.T_eval
        res = self.simulate(x_flat, learn_hidden=False, T_override=T_use)
        return (res["cnt_ro"] / T_use) @ self.C_out + self.b_out


@torch.no_grad()
def evaluate(model: MemristorMultiScaleCSDPHubSNN, loader: DataLoader, device: torch.device) -> float:
    correct = total = 0
    for xb, yb in loader:
        scores = model.predict(xb.to(device))
        correct += (scores.argmax(dim=1) == yb.argmax(dim=1).to(device)).sum().item()
        total += xb.shape[0]
    return correct / max(total, 1)


@torch.no_grad()
def train_epoch(
    model: MemristorMultiScaleCSDPHubSNN,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[float, dict[str, float]]:
    correct = total = 0
    diag: dict[str, float] = {}
    n_batches = len(loader)
    for i, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.float().to(device)
        B = xb.shape[0]

        if args.hard_neg_prob > 0.0:
            clean_scores = model.predict(xb, T_override=min(15, args.T))
            yn = hard_neg_labels(yb, clean_scores, args.hard_neg_prob, temp=args.hard_neg_temp)
        else:
            yn = uniform_neg_labels(yb, model.out_dim)

        X_cat = torch.cat([xb, xb], dim=0)
        Y_cat = torch.cat([yb, yn], dim=0)
        mask = torch.cat(
            [torch.ones(B, 1, device=device), torch.zeros(B, 1, device=device)], dim=0
        )

        # 1) Local CSDP training of hidden memristor crossbars (with preserved Hub Neurons)
        is_last = args.diag and (i == n_batches - 1)
        pos_res = model.simulate(
            X_cat, y=Y_cat, mask=mask, learn_hidden=True, collect_diag=is_last
        )
        if is_last:
            for k in ("r1", "r2", "rfc", "h1", "h2", "hfc", "zm1", "zm2", "zmfc"):
                diag[k] = float(pos_res[k])

        # 2) Clean (y=0) pass + local readout crossbar delta update (ZERO label leakage)
        out_res = model.simulate(xb, learn_hidden=False)
        scores = model.crossbar_delta_update(out_res["cnt_ro"], yb)

        correct += (scores.argmax(dim=1) == yb.argmax(dim=1)).sum().item()
        total += B

    return correct / max(total, 1), diag


def load_dataset(data_dir: str, batch_size: int, dev_batch_size: int) -> tuple[DataLoader, DataLoader]:
    p = Path(data_dir).expanduser()
    X_tr = torch.from_numpy(np.load(p / "trainX.npy")).float()
    Y_tr = torch.from_numpy(np.load(p / "trainY.npy")).float()
    X_te = torch.from_numpy(np.load(p / "testX.npy")).float()
    Y_te = torch.from_numpy(np.load(p / "testY.npy")).float()
    tr_ld = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=batch_size, shuffle=True, drop_last=True)
    te_ld = DataLoader(TensorDataset(X_te, Y_te), batch_size=dev_batch_size, shuffle=False, drop_last=False)
    return tr_ld, te_ld


def parse_args() -> argparse.Namespace:
    default_data = str(Path(__file__).resolve().parent.parent / "data" / "mnist")
    p = argparse.ArgumentParser(description="100% Memristor-Native Multi-Scale CSDP-Hub SNN (MNIST)")
    p.add_argument("--data-dir", default=default_data)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--dev-batch-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Architecture (ZERO weight sharing)
    p.add_argument("--c1", type=int, default=64)
    p.add_argument("--k1", type=int, default=5)
    p.add_argument("--s1", type=int, default=2)
    p.add_argument("--pad1", type=int, default=2)
    p.add_argument("--p1", type=int, default=2)
    p.add_argument("--c2", type=int, default=80)
    p.add_argument("--k2", type=int, default=3)
    p.add_argument("--s2", type=int, default=1)
    p.add_argument("--pad2", type=int, default=1)
    p.add_argument("--p2", type=int, default=1)
    p.add_argument("--c-coarse", type=int, default=32)
    p.add_argument("--hid-fc", type=int, default=2000)

    # SNN dynamics
    p.add_argument("--T", type=int, default=30)
    p.add_argument("--T-eval", type=int, default=60)
    p.add_argument("--dt", type=float, default=3.0)
    p.add_argument("--tau-m", type=float, default=100.0)
    p.add_argument("--v-thr", type=float, default=0.055)
    p.add_argument("--tau-tr", type=float, default=13.0)
    p.add_argument("--R-m", type=float, default=1.0)
    p.add_argument("--R-y", type=float, default=1.0)
    p.add_argument("--inh-R-tile", type=float, default=0.05)
    p.add_argument("--inh-R-fc", type=float, default=0.01)
    p.add_argument("--tile-init-scale", type=float, default=0.5)
    p.add_argument("--fc-init-scale", type=float, default=1.0)

    # Local plasticity & schedules
    p.add_argument("--eta", type=float, default=0.002)
    p.add_argument("--eta-out", type=float, default=0.004)
    p.add_argument("--eta-min-ratio", type=float, default=0.03)
    p.add_argument("--wd-out", type=float, default=1e-4)
    p.add_argument("--weight-bound", type=float, default=1.0)
    p.add_argument("--hard-neg-prob", type=float, default=0.5)
    p.add_argument("--hard-neg-temp", type=float, default=2.0)

    # Preserved Hub-Neuron parameters
    p.add_argument("--tau-m-hab", type=float, default=4.87563)
    p.add_argument("--v-thr-hab", type=float, default=0.249441)
    p.add_argument("--tau-hub", type=float, default=4.39997)
    p.add_argument("--hub-power", type=float, default=1.20313)
    p.add_argument("--hub-w-t1", type=float, default=0.003)
    p.add_argument("--hub-w-t2", type=float, default=0.0065)
    p.add_argument("--hub-w-coarse", type=float, default=0.015)
    p.add_argument("--hub-w-fc", type=float, default=0.038)

    p.add_argument("--amp-bf16", action="store_true", default=True)
    p.add_argument("--no-amp-bf16", dest="amp_bf16", action="store_false")
    p.add_argument("--diag", action="store_true", default=True)
    p.add_argument("--save-json", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tr_ld, te_ld = load_dataset(args.data_dir, args.batch_size, args.dev_batch_size)
    model = MemristorMultiScaleCSDPHubSNN(args, device)
    n_mem = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[MNIST] MemristorMultiScaleCSDPHubSNN (ZERO weight sharing, NO zscore, 100% local) | "
        f"t1={model.tile1.n_tiles}x({model.tile1.patch_dim}->{args.c1})->p{model.tile1.pool_hw} "
        f"t2={model.tile2.n_tiles}x({model.tile2.patch_dim}->{args.c2})->p{model.tile2.pool_hw} "
        f"tc={model.tile_coarse.n_tiles}x({model.tile_coarse.patch_dim}->{args.c_coarse}) "
        f"fc={args.hid_fc} | ro={model.total_ro_dim}->10 | "
        f"T={args.T} T_eval={args.T_eval} hard_neg={args.hard_neg_prob} | "
        f"memristors={n_mem / 1e6:.2f}M (all bounded in [-{args.weight_bound}, {args.weight_bound}])",
        flush=True,
    )

    t0 = time.time()
    best = 0.0
    best_ep = 0
    history = []
    for ep in range(1, args.epochs + 1):
        frac = 0.5 * (1.0 + math.cos(math.pi * (ep - 1) / max(args.epochs - 1, 1)))
        scale = args.eta_min_ratio + (1.0 - args.eta_min_ratio) * frac
        for pg in model.opt_hidden.param_groups:
            pg["lr"] = args.eta * scale
        for pg in model.opt_out.param_groups:
            pg["lr"] = args.eta_out * scale

        tr_acc, diag = train_epoch(model, tr_ld, device, args)
        te_acc = evaluate(model, te_ld, device)
        if te_acc > best:
            best = te_acc
            best_ep = ep
        elapsed = time.time() - t0
        diag_str = (
            f" | r=({diag['r1']:.3f},{diag['r2']:.3f},{diag['rfc']:.3f})"
            f" h=({diag['h1']:.2f},{diag['h2']:.2f},{diag['hfc']:.2f})"
            f" zm=({diag['zm1']:.2f},{diag['zm2']:.2f},{diag['zmfc']:.2f})"
            if diag
            else ""
        )
        print(
            f"Epoch {ep:2d}/{args.epochs}  Train: {tr_acc:.4f}  Test: {te_acc:.4f}  "
            f"Best: {best:.4f} (ep {best_ep})  [{elapsed / 60:.2f}m]{diag_str}",
            flush=True,
        )
        history.append({"epoch": ep, "train": tr_acc, "test": te_acc, **diag})
        if args.save_json:
            Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.save_json).write_text(
                json.dumps({"config": vars(args), "best": best, "best_ep": best_ep, "history": history}, indent=2)
            )

    print(f"FINAL Best: {best:.4f} at epoch {best_ep} in {(time.time() - t0) / 60:.2f}m", flush=True)


if __name__ == "__main__":
    main()
