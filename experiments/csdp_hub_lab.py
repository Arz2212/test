#!/usr/bin/env python3
"""Experimental lab for 100%-local CSDP-SNN with hub-neuron modulation.

Strict invariants:
1. Hub neuron logic is preserved identically:
     hub_fc (fixed constant weights hub_w_init) -> hub_lif -> hub trace ->
     hub_mod(tr, hub, phase, zmax_buf) = ((hub / zmax_buf) ** hub_power - phase) * tr
2. Learning is 100% local (all training inside @torch.no_grad(), zero autograd/backprop).
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


def nn_param(*shape: int) -> torch.nn.Parameter:
    p = torch.nn.Parameter(torch.empty(*shape))
    torch.nn.init.uniform_(p, -1.0, 1.0)
    return p


class AdaptiveLIFLayer(torch.nn.Module):
    """Exact PyTorch equivalent of ngc-learn SLIFCell (with optional adaptive rho_b & thr_jitter).

    When rho_b == 0 and thr_jitter == 0, reduces to standard SpikingJelly LIFNode dynamics:
        v <- (1 - dt/tau_m) * v + (dt/tau_m) * I
        s = (v > thr).float()
        v <- (1 - s) * v
    """

    def __init__(
        self,
        n_units: int,
        tau_m: float,
        dt: float,
        v_thr: float,
        rho_b: float = 0.0,
        thr_jitter: float = 0.0,
        thr_min: float = 0.025,
        thr_gain: float = 0.0,
        thr_leak: float = 0.0,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.n_units = n_units
        self.alpha = dt / tau_m
        self.rho_b = rho_b
        self.thr_min = thr_min
        self.thr_gain = thr_gain
        self.thr_leak = thr_leak
        if thr_jitter > 0.0:
            thr0 = v_thr + torch.empty(1, n_units, device=device).uniform_(-thr_jitter, thr_jitter)
        else:
            thr0 = torch.full((1, n_units), v_thr, device=device)
        self.register_buffer("thr0", thr0)
        self.v: torch.Tensor | None = None
        self.thr: torch.Tensor | None = None

    def reset_state(self, batch_size: int, device: torch.device) -> None:
        self.v = torch.zeros(batch_size, self.n_units, device=device)
        if self.rho_b > 0.0 or self.thr_gain > 0.0:
            self.thr = self.thr0.expand(batch_size, -1).clone()
        else:
            self.thr = self.thr0

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        self.v = self.v + self.alpha * (current - self.v)
        spikes = (self.v > self.thr).to(current.dtype)
        self.v = self.v * (1.0 - spikes)
        if self.rho_b > 0.0:
            dthr = spikes.sum(dim=1, keepdim=True) - 1.0
            self.thr = torch.clamp(self.thr + dthr * self.rho_b, min=self.thr_min)
        elif self.thr_gain > 0.0:
            self.thr = torch.clamp(
                self.thr + spikes * self.thr_gain - (self.thr - self.thr0) * self.thr_leak,
                min=self.thr_min,
            )
        return spikes


def uniform_neg_labels(y_onehot: torch.Tensor, out_dim: int) -> torch.Tensor:
    """Strictly uniform sampling over the (out_dim - 1) incorrect classes."""
    B = y_onehot.shape[0]
    correct = y_onehot.argmax(dim=1)
    r = torch.randint(0, out_dim - 1, (B,), device=y_onehot.device)
    neg = r + (r >= correct).long()
    return F.one_hot(neg, out_dim).float()


def hard_neg_labels(
    y_onehot: torch.Tensor,
    scores: torch.Tensor,
    hard_prob: float,
    temp: float = 1.0,
) -> torch.Tensor:
    """Mix of hard negatives (sampled from model's wrong-class predictions) and uniform negatives."""
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


def decode_features(spk_cnt: torch.Tensor, mode: str, T: int) -> torch.Tensor:
    if mode == "rate":
        return spk_cnt / T
    if mode == "count":
        return spk_cnt
    if mode == "zscore":
        mu = spk_cnt.mean(dim=1, keepdim=True)
        std = spk_cnt.std(dim=1, keepdim=True, unbiased=False)
        return (spk_cnt - mu) / (std + 1e-5)
    if mode == "l2_sqrt":
        s = torch.sqrt(torch.clamp(spk_cnt, min=0.0))
        return s / (s.norm(dim=1, keepdim=True) + 1e-6)
    if mode == "l2_rate":
        return spk_cnt / (spk_cnt.norm(dim=1, keepdim=True) + 1e-6)
    raise ValueError(f"Unknown decode mode: {mode}")


class CSDPHubSNN(torch.nn.Module):
    def __init__(self, args: argparse.Namespace, device: torch.device):
        super().__init__()
        self.args = args
        self.device = device
        self.in_dim = 784
        self.out_dim = 10
        self.hid1 = args.hid1
        self.hid2 = args.hid2

        # Hidden layer 1 synapses
        self.W1 = nn_param(self.in_dim, self.hid1)
        self.M1 = nn_param(self.hid1, self.hid1)
        self.V2y = nn_param(self.out_dim, self.hid1)
        if args.use_v2:
            self.V2 = nn_param(self.hid2, self.hid1)
        else:
            self.V2 = None

        # Hidden layer 2 synapses
        self.W2 = nn_param(self.hid1, self.hid2)
        self.M2 = nn_param(self.hid2, self.hid2)
        self.V3y = nn_param(self.out_dim, self.hid2)

        # Output layer synapses (connected to z1 and z2)
        self.C2 = nn_param(self.hid1, self.out_dim)
        self.C3 = nn_param(self.hid2, self.out_dim)
        if args.out_mode == "local_delta":
            # Zero/small init works best for local softmax-delta readout
            torch.nn.init.zeros_(self.C2)
            torch.nn.init.zeros_(self.C3)
            self.b_out = torch.nn.Parameter(torch.zeros(self.out_dim))
        else:
            self.b_out = None

        # Spiking neurons
        self.lif1 = AdaptiveLIFLayer(
            self.hid1,
            args.tau_m,
            args.dt,
            args.v_thr,
            rho_b=args.rho_b,
            thr_jitter=args.thr_jitter,
            thr_gain=args.thr_gain,
            thr_leak=args.thr_leak,
            device=device,
        )
        self.lif2 = AdaptiveLIFLayer(
            self.hid2,
            args.tau_m,
            args.dt,
            args.v_thr,
            rho_b=args.rho_b,
            thr_jitter=args.thr_jitter,
            thr_gain=args.thr_gain,
            thr_leak=args.thr_leak,
            device=device,
        )
        self.lif_out = AdaptiveLIFLayer(
            self.out_dim,
            args.tau_m,
            args.dt,
            args.v_thr_out,
            rho_b=args.rho_b_out,
            thr_jitter=args.thr_jitter if args.rho_b_out > 0 else 0.0,
            device=device,
        )

        # ── Hub neurons (EXACT preserved hub logic) ──
        self.hub_fc1 = layer.Linear(self.hid1, 1, bias=False).to(device)
        self.hub_fc2 = layer.Linear(self.hid2, 1, bias=False).to(device)
        self.hub_lif1 = neuron.LIFNode(
            tau=args.tau_m_hab, v_threshold=args.v_thr_hab, v_reset=0.0
        ).to(device)
        self.hub_lif2 = neuron.LIFNode(
            tau=args.tau_m_hab, v_threshold=args.v_thr_hab, v_reset=0.0
        ).to(device)
        torch.nn.init.constant_(self.hub_fc1.weight, args.hub_w_init)
        torch.nn.init.constant_(self.hub_fc2.weight, args.hub_w_init)
        self.hub_fc1.weight.requires_grad_(False)
        self.hub_fc2.weight.requires_grad_(False)
        self.register_buffer("zmax1", torch.tensor(1.0))
        self.register_buffer("zmax2", torch.tensor(1.0))

        self.to(device)

        hidden_params = [self.W1, self.M1, self.V2y, self.W2, self.M2, self.V3y]
        if self.V2 is not None:
            hidden_params.append(self.V2)
        self.opt_hidden = torch.optim.Adam(
            hidden_params,
            lr=args.eta,
            fused=(device.type == "cuda"),
        )
        out_params = [self.C2, self.C3] + ([self.b_out] if self.b_out is not None else [])
        self.opt_out = torch.optim.Adam(
            out_params,
            lr=args.eta_out,
            fused=(device.type == "cuda"),
        )

    def hub_mod(
        self,
        z: torch.Tensor,
        hub: torch.Tensor,
        phase: torch.Tensor,
        zmax_buf: torch.Tensor,
        update_zmax: bool = True,
    ) -> torch.Tensor:
        if update_zmax:
            torch.maximum(zmax_buf, hub.max(), out=zmax_buf)
        factor = (hub / zmax_buf) ** self.args.hub_power - phase
        return factor * z

    def _update_trace(self, tr: torch.Tensor, spk: torch.Tensor, d: float) -> torch.Tensor:
        if self.args.trace_mode == "clamp":
            return tr * (1.0 - d) * (1.0 - spk) + spk
        return tr + d * (spk - tr)

    def _clamp_hidden(self) -> None:
        wb = self.args.weight_bound
        self.W1.data.clamp_(-wb, wb)
        self.M1.data.fill_diagonal_(0.0)
        self.M1.data.clamp_(0.0, wb)
        self.V2y.data.clamp_(-wb, wb)
        if self.V2 is not None:
            self.V2.data.clamp_(-wb, wb)
        self.W2.data.clamp_(-wb, wb)
        self.M2.data.fill_diagonal_(0.0)
        self.M2.data.clamp_(0.0, wb)
        self.V3y.data.clamp_(-wb, wb)

    def _apply_csdp_decay(self, pre: torch.Tensor, post: torch.Tensor, W: torch.Tensor) -> None:
        if self.args.w_decay > 0.0:
            # Exact JAX CSDPSynapse activity-dependent LTD decay
            W.data.addmm_(1.0 - pre.T, post, alpha=-self.args.w_decay)

    @torch.no_grad()
    def simulate(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None,
        mask: torch.Tensor | None,
        learn_hidden: bool = False,
        learn_out_online: bool = False,
        collect_diag: bool = False,
    ) -> dict[str, torch.Tensor | float]:
        args = self.args
        B = x.shape[0]
        dv = self.device

        self.lif1.reset_state(B, dv)
        self.lif2.reset_state(B, dv)
        self.lif_out.reset_state(B, dv)
        if learn_hidden or collect_diag:
            functional.reset_net(self.hub_lif1)
            functional.reset_net(self.hub_lif2)
            hub1 = torch.zeros(B, 1, device=dv)
            hub2 = torch.zeros(B, 1, device=dv)

        tr1 = torch.zeros(B, self.hid1, device=dv)
        tr2 = torch.zeros(B, self.hid2, device=dv)
        tr_out = torch.zeros(B, self.out_dim, device=dv)

        spk_cnt1 = torch.zeros(B, self.hid1, device=dv)
        spk_cnt2 = torch.zeros(B, self.hid2, device=dv)
        spk_cnt_out = torch.zeros(B, self.out_dim, device=dv)

        z0_prev = torch.zeros(B, self.in_dim, device=dv)
        z1_prev = torch.zeros(B, self.hid1, device=dv)
        z2_prev = torch.zeros(B, self.hid2, device=dv)

        d = args.dt / args.tau_tr
        decay_hub = 1.0 / args.tau_hub
        has_y = y is not None

        h1_spk_sum = 0.0
        h2_spk_sum = 0.0
        hub1_mean_sum = 0.0
        hub2_mean_sum = 0.0

        R_m = args.R_m
        R_y = args.R_y
        inh_R = args.inh_R
        R_v2 = args.R_v2
        grad_scale = (1.0 / B) if args.norm_grad else 1.0
        use_bf16 = args.amp_bf16 and dv.type == "cuda"

        for t in range(args.T):
            z0 = (torch.rand(B, self.in_dim, device=dv) < x).float()
            in0 = z0_prev if args.sync_delay else z0

            # Layer 1
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                I1 = (in0 @ self.W1) * R_m - (z1_prev @ self.M1) * inh_R
                if self.V2 is not None:
                    I1 = I1 + (z2_prev @ self.V2) * R_v2
                if has_y:
                    I1 = I1 + (y @ self.V2y) * R_y
            z1 = self.lif1(I1.float())
            tr1 = self._update_trace(tr1, z1, d)

            if learn_hidden or collect_diag:
                h1_spk = self.hub_lif1(self.hub_fc1(z1))
                hub1 = hub1 + decay_hub * (-hub1 + h1_spk.float())
                if collect_diag:
                    h1_spk_sum += h1_spk.float().mean().item()
                    hub1_mean_sum += hub1.mean().item()
                if learn_hidden:
                    mod1 = self.hub_mod(tr1, hub1, mask, self.zmax1, update_zmax=True)

            # Layer 2
            in1 = z1_prev if args.sync_delay else z1
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                I2 = (in1 @ self.W2) * R_m - (z2_prev @ self.M2) * inh_R
                if has_y:
                    I2 = I2 + (y @ self.V3y) * R_y
            z2 = self.lif2(I2.float())
            tr2 = self._update_trace(tr2, z2, d)

            if learn_hidden or collect_diag:
                h2_spk = self.hub_lif2(self.hub_fc2(z2))
                hub2 = hub2 + decay_hub * (-hub2 + h2_spk.float())
                if collect_diag:
                    h2_spk_sum += h2_spk.float().mean().item()
                    hub2_mean_sum += hub2.mean().item()
                if learn_hidden:
                    mod2 = self.hub_mod(tr2, hub2, mask, self.zmax2, update_zmax=True)

            # Output spiking layer (if using spiking readout)
            if args.out_mode != "local_delta":
                out_in1 = z1_prev if args.sync_delay else z1
                out_in2 = z2_prev if args.sync_delay else z2
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    I_out = out_in1 @ self.C2 + out_in2 @ self.C3
                zy = self.lif_out(I_out.float())
                tr_out = self._update_trace(tr_out, zy, d)
                spk_cnt_out = spk_cnt_out + zy

            # Local CSDP hidden updates
            if learn_hidden:
                m1_rm = mod1 * (R_m * grad_scale)
                m1_inh = mod1 * (inh_R * grad_scale)
                m1_ry = mod1 * (R_y * grad_scale)
                m2_rm = mod2 * (R_m * grad_scale)
                m2_inh = mod2 * (inh_R * grad_scale)
                m2_ry = mod2 * (R_y * grad_scale)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    self.W1.grad = (in0.T @ m1_rm).float()
                    self.M1.grad = (z1_prev.T @ m1_inh).float()
                    self.V2y.grad = (y.T @ m1_ry).float()
                    if self.V2 is not None:
                        self.V2.grad = (z2_prev.T @ (mod1 * (R_v2 * grad_scale))).float()

                    self.W2.grad = (in1.T @ m2_rm).float()
                    self.M2.grad = (z2_prev.T @ m2_inh).float()
                    self.V3y.grad = (y.T @ m2_ry).float()

                self.opt_hidden.step()

                if args.w_decay > 0.0:
                    self._apply_csdp_decay(in0, z1, self.W1)
                    self._apply_csdp_decay(z1_prev, z1, self.M1)
                    self._apply_csdp_decay(y, z1, self.V2y)
                    if self.V2 is not None:
                        self._apply_csdp_decay(z2_prev, z1, self.V2)
                    self._apply_csdp_decay(in1, z2, self.W2)
                    self._apply_csdp_decay(z2_prev, z2, self.M2)
                    self._apply_csdp_decay(y, z2, self.V3y)

                if args.l2_decay > 0.0:
                    fac = 1.0 - args.l2_decay
                    self.W1.data.mul_(fac)
                    self.W2.data.mul_(fac)
                    self.V2y.data.mul_(fac)
                    self.V3y.data.mul_(fac)
                    if self.V2 is not None:
                        self.V2.data.mul_(fac)

                self._clamp_hidden()

            # Local online spiking readout update (if enabled)
            if learn_out_online and t >= args.out_warmup:
                # y_target is passed via mask-selected rows or clean target
                y_tgt = y
                m_out = mask if mask is not None else torch.ones(B, 1, device=dv)
                N_pos = m_out.sum().clamp(min=1.0) if args.norm_grad else 1.0
                dmu = (2.0 * (y_tgt - tr_out) * m_out) * (args.post_wght_out / N_pos)
                gC2 = -(out_in1.T @ dmu)
                gC3 = -(out_in2.T @ dmu)
                if args.soft_bound_out:
                    gC2 = gC2 * (args.weight_bound - self.C2.data.abs())
                    gC3 = gC3 * (args.weight_bound - self.C3.data.abs())
                if args.wd_out > 0.0:
                    gC2 = gC2 + args.wd_out * self.C2.data
                    gC3 = gC3 + args.wd_out * self.C3.data
                self.C2.grad = gC2
                self.C3.grad = gC3
                self.opt_out.step()
                self.C2.data.clamp_(-args.weight_bound, args.weight_bound)
                self.C3.data.clamp_(-args.weight_bound, args.weight_bound)

            spk_cnt1 = spk_cnt1 + z1
            spk_cnt2 = spk_cnt2 + z2
            z0_prev = z0
            z1_prev = z1
            z2_prev = z2

        res: dict[str, torch.Tensor | float] = {
            "spk_cnt1": spk_cnt1,
            "spk_cnt2": spk_cnt2,
            "spk_cnt_out": spk_cnt_out,
            "tr_out": tr_out,
        }
        if collect_diag:
            res["z1_rate"] = (spk_cnt1 / args.T).mean().item()
            res["z2_rate"] = (spk_cnt2 / args.T).mean().item()
            res["zy_rate"] = (spk_cnt_out / args.T).mean().item()
            res["h1_rate"] = h1_spk_sum / args.T
            res["h2_rate"] = h2_spk_sum / args.T
            res["hub1_mean"] = hub1_mean_sum / args.T
            res["hub2_mean"] = hub2_mean_sum / args.T
            res["zmax1"] = self.zmax1.item()
            res["zmax2"] = self.zmax2.item()
        return res

    @torch.no_grad()
    def local_delta_update(
        self,
        spk_cnt1: torch.Tensor,
        spk_cnt2: torch.Tensor,
        y_onehot: torch.Tensor,
    ) -> torch.Tensor:
        """Purely local single-layer softmax-delta rule (zero autograd/backprop)."""
        args = self.args
        B = y_onehot.shape[0]
        f1 = decode_features(spk_cnt1, args.decode_mode, args.T)
        f2 = decode_features(spk_cnt2, args.decode_mode, args.T)
        if args.out_dropout > 0.0:
            drop1 = (torch.rand_like(f1) > args.out_dropout).float() / (1.0 - args.out_dropout)
            drop2 = (torch.rand_like(f2) > args.out_dropout).float() / (1.0 - args.out_dropout)
            f1_d, f2_d = f1 * drop1, f2 * drop2
        else:
            f1_d, f2_d = f1, f2

        logits = f1_d @ self.C2 + f2_d @ self.C3 + self.b_out
        probs = F.softmax(logits, dim=1)
        err = (probs - y_onehot) / B  # local post-synaptic error vector

        self.C2.grad = f1_d.T @ err + args.wd_out * self.C2.data
        self.C3.grad = f2_d.T @ err + args.wd_out * self.C3.data
        self.b_out.grad = err.sum(dim=0)
        self.opt_out.step()
        return logits

    @torch.no_grad()
    def spiking_clean_ce_update(
        self,
        spk_cnt1: torch.Tensor,
        spk_cnt2: torch.Tensor,
        spk_cnt_out: torch.Tensor,
        y_onehot: torch.Tensor,
    ) -> torch.Tensor:
        """Local rate-level delta update for spiking output C2, C3 (zero autograd)."""
        args = self.args
        B = y_onehot.shape[0]
        r1 = spk_cnt1 / args.T
        r2 = spk_cnt2 / args.T
        probs = F.softmax(spk_cnt_out / args.out_temp, dim=1)
        err = (probs - y_onehot) / B
        gC2 = r1.T @ err
        gC3 = r2.T @ err
        if args.soft_bound_out:
            gC2 = gC2 * (args.weight_bound - self.C2.data.abs())
            gC3 = gC3 * (args.weight_bound - self.C3.data.abs())
        if args.wd_out > 0.0:
            gC2 = gC2 + args.wd_out * self.C2.data
            gC3 = gC3 + args.wd_out * self.C3.data
        self.C2.grad = gC2
        self.C3.grad = gC3
        self.opt_out.step()
        self.C2.data.clamp_(-args.weight_bound, args.weight_bound)
        self.C3.data.clamp_(-args.weight_bound, args.weight_bound)
        return spk_cnt_out

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        res = self.simulate(x, y=None, mask=None, learn_hidden=False, learn_out_online=False)
        if self.args.out_mode == "local_delta":
            f1 = decode_features(res["spk_cnt1"], self.args.decode_mode, self.args.T)
            f2 = decode_features(res["spk_cnt2"], self.args.decode_mode, self.args.T)
            scores = f1 @ self.C2 + f2 @ self.C3 + self.b_out
        else:
            scores = res["spk_cnt_out"]
        return scores, res


@torch.no_grad()
def evaluate(model: CSDPHubSNN, loader: DataLoader, device: torch.device) -> float:
    correct = total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        scores, _ = model.predict(xb)
        correct += (scores.argmax(dim=1) == yb.argmax(dim=1).to(device)).sum().item()
        total += xb.shape[0]
    return correct / max(total, 1)


@torch.no_grad()
def train_epoch(
    model: CSDPHubSNN,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    collect_diag: bool = False,
) -> tuple[float, dict[str, float]]:
    correct = total = 0
    diag: dict[str, float] = {}
    for i, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.float().to(device)
        B = xb.shape[0]

        # Optional clean pass BEFORE hidden update ONLY if needed for hard negatives
        clean_res = None
        if args.hard_neg_prob > 0.0:
            clean_res = model.simulate(
                xb, y=None, mask=None, learn_hidden=False, learn_out_online=False
            )

        if args.hard_neg_prob > 0.0 and clean_res is not None:
            if args.out_mode == "local_delta":
                f1 = decode_features(clean_res["spk_cnt1"], args.decode_mode, args.T)
                f2 = decode_features(clean_res["spk_cnt2"], args.decode_mode, args.T)
                clean_scores = f1 @ model.C2 + f2 @ model.C3 + model.b_out
            else:
                clean_scores = clean_res["spk_cnt_out"]
            yn = hard_neg_labels(yb, clean_scores, args.hard_neg_prob, temp=args.hard_neg_temp)
        else:
            yn = uniform_neg_labels(yb, model.out_dim)

        X_cat = torch.cat([xb, xb], dim=0)
        Y_cat = torch.cat([yb, yn], dim=0)
        mask = torch.cat(
            [torch.ones(B, 1, device=device), torch.zeros(B, 1, device=device)],
            dim=0,
        )

        # Step 1: Local CSDP hidden update (and online readout if spiking_pos)
        is_last = collect_diag and (i == len(loader) - 1)
        pos_res = model.simulate(
            X_cat,
            y=Y_cat,
            mask=mask,
            learn_hidden=True,
            learn_out_online=(args.out_mode == "spiking_pos"),
            collect_diag=is_last,
        )
        if is_last:
            for k in ("z1_rate", "z2_rate", "zy_rate", "h1_rate", "h2_rate", "hub1_mean", "hub2_mean", "zmax1", "zmax2"):
                diag[k] = float(pos_res[k])

        # Step 2: Clean-pass local readout update (after hidden update!)
        if args.out_mode == "spiking_pos":
            scores = pos_res["spk_cnt_out"][:B]
        elif args.out_mode == "spiking_clean":
            # Run spiking clean pass: hidden layers unclamped (y=None), output updated locally
            out_res = _run_spiking_clean_out(model, xb, yb)
            scores = out_res["spk_cnt_out"]
        elif args.out_mode == "spiking_clean_ce":
            out_res = model.simulate(xb, y=None, mask=None, learn_hidden=False, learn_out_online=False)
            scores = model.spiking_clean_ce_update(
                out_res["spk_cnt1"], out_res["spk_cnt2"], out_res["spk_cnt_out"], yb
            )
        elif args.out_mode == "local_delta":
            out_res = model.simulate(xb, y=None, mask=None, learn_hidden=False, learn_out_online=False)
            scores = model.local_delta_update(out_res["spk_cnt1"], out_res["spk_cnt2"], yb)
        else:
            raise ValueError(f"Unknown out_mode: {args.out_mode}")

        correct += (scores.argmax(dim=1) == yb.argmax(dim=1)).sum().item()
        total += B

    return correct / max(total, 1), diag


@torch.no_grad()
def _run_spiking_clean_out(
    model: CSDPHubSNN, x: torch.Tensor, y_target: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Unclamped hidden pass (y=0 for z1/z2) with online local Hebbian update on spiking zy."""
    args = model.args
    B = x.shape[0]
    dv = model.device
    model.lif1.reset_state(B, dv)
    model.lif2.reset_state(B, dv)
    model.lif_out.reset_state(B, dv)

    tr_out = torch.zeros(B, model.out_dim, device=dv)
    spk_cnt_out = torch.zeros(B, model.out_dim, device=dv)
    z0_prev = torch.zeros(B, model.in_dim, device=dv)
    z1_prev = torch.zeros(B, model.hid1, device=dv)
    z2_prev = torch.zeros(B, model.hid2, device=dv)
    d = args.dt / args.tau_tr
    N_pos = float(B) if args.norm_grad else 1.0
    use_bf16 = args.amp_bf16 and dv.type == "cuda"

    for t in range(args.T):
        z0 = (torch.rand(B, model.in_dim, device=dv) < x).float()
        in0 = z0_prev if args.sync_delay else z0
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            I1 = (in0 @ model.W1) * args.R_m - (z1_prev @ model.M1) * args.inh_R
            if model.V2 is not None:
                I1 = I1 + (z2_prev @ model.V2) * args.R_v2
        z1 = model.lif1(I1.float())

        in1 = z1_prev if args.sync_delay else z1
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            I2 = (in1 @ model.W2) * args.R_m - (z2_prev @ model.M2) * args.inh_R
        z2 = model.lif2(I2.float())

        out_in1 = z1_prev if args.sync_delay else z1
        out_in2 = z2_prev if args.sync_delay else z2
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            I_out = out_in1 @ model.C2 + out_in2 @ model.C3
        zy = model.lif_out(I_out.float())
        tr_out = model._update_trace(tr_out, zy, d)
        spk_cnt_out = spk_cnt_out + zy

        if t >= args.out_warmup:
            dmu = (2.0 * (y_target - tr_out)) * (args.post_wght_out / N_pos)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                gC2 = (-(out_in1.T @ dmu)).float()
                gC3 = (-(out_in2.T @ dmu)).float()
            if args.soft_bound_out:
                gC2 = gC2 * (args.weight_bound - model.C2.data.abs())
                gC3 = gC3 * (args.weight_bound - model.C3.data.abs())
            if args.wd_out > 0.0:
                gC2 = gC2 + args.wd_out * model.C2.data
                gC3 = gC3 + args.wd_out * model.C3.data
            model.C2.grad = gC2
            model.C3.grad = gC3
            model.opt_out.step()
            model.C2.data.clamp_(-args.weight_bound, args.weight_bound)
            model.C3.data.clamp_(-args.weight_bound, args.weight_bound)

        z0_prev = z0
        z1_prev = z1
        z2_prev = z2

    return {"spk_cnt_out": spk_cnt_out}


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
    p = argparse.ArgumentParser(description="CSDP-SNN Hub Lab (100% local learning)")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--dev-batch-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Architecture
    p.add_argument("--hid1", type=int, default=5000)
    p.add_argument("--hid2", type=int, default=1000)
    p.add_argument("--use-v2", action="store_true")

    # SNN dynamics
    p.add_argument("--T", type=int, default=50)
    p.add_argument("--dt", type=float, default=3.0)
    p.add_argument("--tau-m", type=float, default=100.0)
    p.add_argument("--v-thr", type=float, default=0.055)
    p.add_argument("--v-thr-out", type=float, default=0.055)
    p.add_argument("--tau-tr", type=float, default=13.0)
    p.add_argument("--trace-mode", choices=["clamp", "ema"], default="clamp")
    p.add_argument("--rho-b", type=float, default=0.001)
    p.add_argument("--rho-b-out", type=float, default=0.001)
    p.add_argument("--thr-jitter", type=float, default=0.025)
    p.add_argument("--thr-gain", type=float, default=0.0)
    p.add_argument("--thr-leak", type=float, default=0.0)
    p.add_argument("--sync-delay", action="store_true")

    # Currents & plasticity
    p.add_argument("--R-m", type=float, default=0.1)
    p.add_argument("--R-y", type=float, default=0.1)
    p.add_argument("--R-v2", type=float, default=0.1)
    p.add_argument("--inh-R", type=float, default=0.01)
    p.add_argument("--eta", type=float, default=0.002)
    p.add_argument("--eta-out", type=float, default=0.002)
    p.add_argument("--eta-min-ratio", type=float, default=0.1)
    p.add_argument("--lr-schedule", choices=["none", "cosine"], default="none")
    p.add_argument("--weight-bound", type=float, default=1.0)
    p.add_argument("--w-decay", type=float, default=5e-5)
    p.add_argument("--l2-decay", type=float, default=0.0)
    p.add_argument("--norm-grad", action="store_true")

    # Output readout
    p.add_argument(
        "--out-mode",
        choices=["spiking_pos", "spiking_clean", "spiking_clean_ce", "local_delta"],
        default="spiking_pos",
    )
    p.add_argument("--post-wght-out", type=float, default=0.1)
    p.add_argument("--soft-bound-out", action="store_true")
    p.add_argument("--wd-out", type=float, default=0.0)
    p.add_argument("--out-warmup", type=int, default=0)
    p.add_argument("--out-temp", type=float, default=5.0)
    p.add_argument("--out-dropout", type=float, default=0.0)
    p.add_argument("--decode-mode", choices=["rate", "count", "zscore", "l2_sqrt", "l2_rate"], default="zscore")

    # Negative sampling
    p.add_argument("--hard-neg-prob", type=float, default=0.0)
    p.add_argument("--hard-neg-temp", type=float, default=2.0)

    # Hub parameters (preserved hub logic)
    p.add_argument("--tau-m-hab", type=float, default=4.87563)
    p.add_argument("--v-thr-hab", type=float, default=0.249441)
    p.add_argument("--tau-hub", type=float, default=4.39997)
    p.add_argument("--hub-power", type=float, default=1.20313)
    p.add_argument("--hub-w-init", type=float, default=0.0503053)

    p.add_argument("--diag", action="store_true")
    p.add_argument("--amp-bf16", action="store_true")
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
    model = CSDPHubSNN(args, device)

    print("CONFIG:", json.dumps(vars(args), sort_keys=True), flush=True)
    t0 = time.time()
    best = 0.0
    best_ep = 0
    history = []

    for ep in range(1, args.epochs + 1):
        if args.lr_schedule == "cosine":
            frac = 0.5 * (1.0 + math.cos(math.pi * (ep - 1) / max(args.epochs - 1, 1)))
            scale = args.eta_min_ratio + (1.0 - args.eta_min_ratio) * frac
            for pg in model.opt_hidden.param_groups:
                pg["lr"] = args.eta * scale
            for pg in model.opt_out.param_groups:
                pg["lr"] = args.eta_out * scale

        tr_acc, diag = train_epoch(model, tr_ld, device, args, collect_diag=args.diag)
        te_acc = evaluate(model, te_ld, device)
        if te_acc > best:
            best = te_acc
            best_ep = ep
        elapsed = time.time() - t0
        diag_str = ""
        if diag:
            diag_str = (
                f" | r1={diag['z1_rate']:.3f} r2={diag['z2_rate']:.3f} ry={diag['zy_rate']:.3f}"
                f" h1={diag['h1_rate']:.2f} h2={diag['h2_rate']:.2f}"
                f" zm=({diag['zmax1']:.2f},{diag['zmax2']:.2f})"
            )
        print(
            f"Epoch {ep:2d}/{args.epochs}  Train: {tr_acc:.4f}  Test: {te_acc:.4f}  "
            f"Best: {best:.4f} (ep {best_ep})  [{elapsed / 60:.2f}m]{diag_str}",
            flush=True,
        )
        history.append({"epoch": ep, "train": tr_acc, "test": te_acc, **diag})

    print(f"FINAL Best: {best:.4f} at epoch {best_ep} in {(time.time() - t0) / 60:.2f}m", flush=True)
    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_json).write_text(
            json.dumps({"config": vars(args), "best": best, "best_ep": best_ep, "history": history}, indent=2)
        )


if __name__ == "__main__":
    main()
