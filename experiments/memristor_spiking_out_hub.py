#!/usr/bin/env python3
"""100% Memristor-Native, 100% ALL-SPIKING Local CSDP-Hub SNN (SpikingJelly LIFNode Output).

Strict invariants:
1. ZERO convolutional weight sharing: every synaptic weight is an independent physical
   memristor inside a tiled locally-connected mini-crossbar or dense crossbar, ALL bounded in [-1, 1].
2. 100% LOCAL learning under @torch.no_grad() (zero backprop / autograd).
3. Preserved Hub-Neuron logic (1-to-1).
4. ZERO zscore / layer normalization.
5. 100% SPIKING OUTPUT LAYER (SpikingJelly neuron.LIFNode, like JAX SLIFCell zy):
   - No linear classifier! Every single layer from input to output emits discrete binary spikes {0, 1}.
   - Classification uses F.softmax(spk_cnt / T, dim=1) (argmax of output LIF spike counts!).
   - Supported 100% spiking output training rules (all local, on clean y=0 pass):
     * jax_dmu:        online per-step dmu = 2 * (target_rate * y - tr_out) / B, C_out.grad = -(ro_spk.T @ dmu)
     * jax_sm_online:  online per-step dmu = (y - softmax(tr_out / out_temp)) / B, C_out.grad = -(ro_spk.T @ dmu)
     * jax_sm_rate:    per-batch rate softmax on output spikes: err = (softmax(spk_cnt / out_temp) - y) / B,
                       C_out.grad = (cnt_ro / T).T @ err
6. ZERO data augmentation.
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
        ).permute(2, 0, 1).contiguous()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            I = torch.bmm(x_p, self.W) * self.args.R_m - torch.bmm(z_prev, self.M) * self.args.inh_R_tile
            if y is not None:
                y_l = y.unsqueeze(0).expand(L, B, -1)
                I = I + torch.bmm(y_l, self.Vy) * self.args.R_y

        z = self.lif(I.float())
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
                factor = (hub / self.zmax) ** self.args.hub_power - mask
                mod = factor.unsqueeze(0) * tr
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
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


class AllSpikingMemristorCSDPHubSNN(torch.nn.Module):
    """100% Spiking (including SpikingJelly LIFNode output) Memristor-Native CSDP-Hub SNN."""

    def __init__(self, args: argparse.Namespace, device: torch.device):
        super().__init__()
        self.args = args
        self.device = device
        self.out_dim = 10

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

        if args.c2 > 0:
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
            fc_in = self.tile2.pool_dim
        else:
            self.tile2 = None
            fc_in = self.tile1.pool_dim

        if args.c_coarse > 0:
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
            fc_in += self.tile_coarse.pool_dim
        else:
            self.tile_coarse = None

        self.fc = FCCrossbarCSDPLayer(
            in_dim=fc_in,
            hid_dim=args.hid_fc,
            out_dim=self.out_dim,
            hub_w=args.hub_w_fc,
            args=args,
            device=device,
        )

        # Readout connections from all spiking hidden layers into SPIKING lif_out
        self.ro_dims = [("t1", self.tile1.pool_dim)]
        if self.tile2 is not None:
            self.ro_dims.append(("t2", self.tile2.pool_dim))
        if self.tile_coarse is not None:
            self.ro_dims.append(("tc", self.tile_coarse.pool_dim))
        self.ro_dims.append(("fc", args.hid_fc))
        self.total_ro_dim = sum(d for _, d in self.ro_dims)

        # 100% SPIKING OUTPUT LAYER (SpikingJelly LIFNode + memristive crossbar C_out)
        if args.out_init_scale > 0.0:
            self.C_out = nn_param(self.total_ro_dim, self.out_dim, bound=args.out_init_scale)
        else:
            self.C_out = torch.nn.Parameter(torch.zeros(self.total_ro_dim, self.out_dim))
        if args.out_bias:
            self.b_out = torch.nn.Parameter(torch.zeros(self.out_dim))
        else:
            self.b_out = None

        v_reset_out = None if args.soft_reset_out else 0.0
        tau_out = (args.tau_m_out if args.tau_m_out > 0 else args.tau_m) / args.dt
        self.lif_out = neuron.LIFNode(
            tau=tau_out, v_threshold=args.v_thr_out, v_reset=v_reset_out
        )

        self.to(device)

        hidden_params = self.tile1.params() + self.fc.params()
        if self.tile2 is not None:
            hidden_params += self.tile2.params()
        if self.tile_coarse is not None:
            hidden_params += self.tile_coarse.params()
        self.opt_hidden = torch.optim.Adam(
            hidden_params, lr=args.eta, fused=(device.type == "cuda")
        )
        out_params = [self.C_out] + ([self.b_out] if self.b_out is not None else [])
        self.opt_out = torch.optim.Adam(
            out_params, lr=args.eta_out, fused=(device.type == "cuda")
        )

    @torch.no_grad()
    def simulate(
        self,
        x_flat: torch.Tensor,
        y: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        learn_hidden: bool = False,
        learn_out_online: bool = False,
        y_out_target: torch.Tensor | None = None,
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
        functional.reset_net(self.fc.lif)
        functional.reset_net(self.lif_out)
        if self.tile2 is not None:
            functional.reset_net(self.tile2.lif)
        if self.tile_coarse is not None:
            functional.reset_net(self.tile_coarse.lif)

        if learn_hidden or collect_diag:
            functional.reset_net(self.tile1.hub_lif)
            functional.reset_net(self.fc.hub_lif)
            hub1 = torch.zeros(B, 1, device=dv)
            hub_fc = torch.zeros(B, 1, device=dv)
            if self.tile2 is not None:
                functional.reset_net(self.tile2.hub_lif)
                hub2 = torch.zeros(B, 1, device=dv)
            else:
                hub2 = None
            if self.tile_coarse is not None:
                functional.reset_net(self.tile_coarse.hub_lif)
                hub_c = torch.zeros(B, 1, device=dv)
            else:
                hub_c = None
        else:
            hub1 = hub2 = hub_c = hub_fc = None

        z1_prev = torch.zeros(self.tile1.n_tiles, B, self.tile1.out_ch, device=dv)
        tr1 = torch.zeros_like(z1_prev)
        if self.tile2 is not None:
            z2_prev = torch.zeros(self.tile2.n_tiles, B, self.tile2.out_ch, device=dv)
            tr2 = torch.zeros_like(z2_prev)
        else:
            z2_prev = tr2 = None
        if self.tile_coarse is not None:
            zc_prev = torch.zeros(self.tile_coarse.n_tiles, B, self.tile_coarse.out_ch, device=dv)
            trc = torch.zeros_like(zc_prev)
        else:
            zc_prev = trc = None

        zfc_prev = torch.zeros(B, self.fc.hid_dim, device=dv)
        tr_fc = torch.zeros_like(zfc_prev)
        tr_out = torch.zeros(B, self.out_dim, device=dv)

        cnt_ro = torch.zeros(B, self.total_ro_dim, device=dv)
        cnt_out = torch.zeros(B, self.out_dim, device=dv)

        d = args.dt / args.tau_tr
        d_out = args.dt / (args.tau_tr_out if args.tau_tr_out > 0 else args.tau_tr)
        decay_hub = 1.0 / args.tau_hub
        r1_sum = r2_sum = rfc_sum = ry_sum = h1_sum = h2_sum = hfc_sum = 0.0

        for t in range(T_steps):
            z0_img = (torch.rand_like(x_img) < x_img).float()

            z1, p1_img, tr1, hub1, h1_v = self.tile1.step(
                z0_img, z1_prev, tr1, hub1, y, mask,
                learn_hidden, collect_diag, d, decay_hub, use_bf16,
            )
            p1_flat = p1_img.reshape(B, -1)

            if self.tile2 is not None:
                z2, p2_img, tr2, hub2, h2_v = self.tile2.step(
                    p1_img, z2_prev, tr2, hub2, y, mask,
                    learn_hidden, collect_diag, d, decay_hub, use_bf16,
                )
                p2_flat = p2_img.reshape(B, -1)
            else:
                z2 = p2_flat = None
                h2_v = 0.0

            if self.tile_coarse is not None:
                zc, pc_img, trc, hub_c, _ = self.tile_coarse.step(
                    z0_img, zc_prev, trc, hub_c, y, mask,
                    learn_hidden, collect_diag, d, decay_hub, use_bf16,
                )
                pc_flat = pc_img.reshape(B, -1)
            else:
                zc = pc_flat = None

            fc_parts = [p2_flat if p2_flat is not None else p1_flat]
            if pc_flat is not None:
                fc_parts.append(pc_flat)
            fc_in = torch.cat(fc_parts, dim=1) if len(fc_parts) > 1 else fc_parts[0]

            zfc, tr_fc, hub_fc, hfc_v = self.fc.step(
                fc_in, zfc_prev, tr_fc, hub_fc, y, mask,
                learn_hidden, collect_diag, d, decay_hub, use_bf16,
            )

            if learn_hidden:
                self.opt_hidden.step()
                self.tile1.clamp_weights()
                if self.tile2 is not None:
                    self.tile2.clamp_weights()
                if self.tile_coarse is not None:
                    self.tile_coarse.clamp_weights()
                self.fc.clamp_weights()

            # Readout binary spike vector into 100% SPIKING lif_out
            ro_parts = [p1_flat]
            if p2_flat is not None:
                ro_parts.append(p2_flat)
            if pc_flat is not None:
                ro_parts.append(pc_flat)
            ro_parts.append(zfc)
            ro_spk = torch.cat(ro_parts, dim=1)
            cnt_ro = cnt_ro + ro_spk

            # 100% SPIKING OUTPUT LAYER (SpikingJelly LIFNode!)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                I_out = (ro_spk @ self.C_out) * args.R_out
                if self.b_out is not None:
                    I_out = I_out + self.b_out * args.R_out
            if args.out_v_base != 0.0:
                I_out = I_out + args.out_v_base
            zy = self.lif_out(I_out.float())  # binary spikes {0, 1}!
            tr_out = tr_out + d_out * (zy - tr_out)
            cnt_out = cnt_out + zy

            if learn_out_online and y_out_target is not None and t >= args.out_warmup:
                if args.out_rule == "jax_dmu":
                    # Exact JAX MaskedErrorCell local rule: dmu = 2 * (target - tr_out) / B
                    targ = y_out_target * args.target_rate + (1.0 - y_out_target) * args.neg_target_rate
                    dmu = 2.0 * (targ - tr_out) / B
                elif args.out_rule == "jax_sm_online":
                    # Softmax on spiking trace tr_out -> local error (y - probs) / B
                    probs = F.softmax(tr_out / args.out_temp, dim=1)
                    dmu = (y_out_target - probs) / B
                else:
                    raise ValueError(f"Unsupported online rule: {args.out_rule}")

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    gC = (-(ro_spk.T @ dmu)).float()
                if args.wd_out > 0.0:
                    gC = gC + args.wd_out * self.C_out.data
                self.C_out.grad = gC
                if self.b_out is not None:
                    self.b_out.grad = -dmu.sum(dim=0)
                self.opt_out.step()
                self.C_out.data.clamp_(-args.weight_bound, args.weight_bound)
                if self.b_out is not None:
                    self.b_out.data.clamp_(-args.weight_bound, args.weight_bound)

            if collect_diag:
                r1_sum += z1.mean().item()
                if z2 is not None:
                    r2_sum += z2.mean().item()
                rfc_sum += zfc.mean().item()
                ry_sum += zy.mean().item()
                h1_sum += h1_v
                h2_sum += h2_v
                hfc_sum += hfc_v

            z1_prev = z1
            if z2 is not None:
                z2_prev = z2
            if zc is not None:
                zc_prev = zc
            zfc_prev = zfc

        y_hat = F.softmax(cnt_out / T_steps, dim=1)
        res: dict[str, torch.Tensor | float] = {
            "y_hat": y_hat,
            "cnt_out": cnt_out,
            "cnt_ro": cnt_ro,
            "tr_out": tr_out,
        }
        if collect_diag:
            max_cnt = cnt_out.max(dim=1).values
            tie_rate = ((cnt_out == max_cnt.unsqueeze(1)).sum(dim=1) > 1).float().mean().item()
            sat_rate = (cnt_out >= T_steps - 0.5).float().mean().item()
            res.update({
                "r1": r1_sum / T_steps,
                "r2": r2_sum / T_steps,
                "rfc": rfc_sum / T_steps,
                "ry": ry_sum / T_steps,
                "h1": h1_sum / T_steps,
                "h2": h2_sum / T_steps,
                "hfc": hfc_sum / T_steps,
                "tie": tie_rate,
                "sat": sat_rate,
                "cmax": max_cnt.mean().item(),
            })
        return res

    @torch.no_grad()
    def rate_softmax_update(
        self, cnt_ro: torch.Tensor, cnt_out: torch.Tensor, y_onehot: torch.Tensor
    ) -> torch.Tensor:
        """Local rate-level softmax update directly on SPIKING lif_out crossbar C_out."""
        B = y_onehot.shape[0]
        rates_ro = cnt_ro / self.args.T
        rates_out = cnt_out / self.args.T
        probs = F.softmax(rates_out / self.args.out_temp, dim=1)
        err = (probs - y_onehot) / B
        self.C_out.grad = rates_ro.T @ err + self.args.wd_out * self.C_out.data
        if self.b_out is not None:
            self.b_out.grad = err.sum(dim=0)
        self.opt_out.step()
        wb = self.args.weight_bound
        self.C_out.data.clamp_(-wb, wb)
        if self.b_out is not None:
            self.b_out.data.clamp_(-wb, wb)
        return F.softmax(rates_out, dim=1)

    @torch.no_grad()
    def predict(self, x_flat: torch.Tensor, T_override: int | None = None) -> tuple[torch.Tensor, dict[str, float]]:
        T_use = T_override if T_override is not None else (self.args.T_eval if self.args.T_eval > 0 else self.args.T)
        res = self.simulate(x_flat, learn_hidden=False, learn_out_online=False, collect_diag=True, T_override=T_use)
        return res["y_hat"], {"tie": float(res["tie"]), "sat": float(res["sat"]), "cmax": float(res["cmax"]), "ry": float(res["ry"])}


@torch.no_grad()
def evaluate(model: AllSpikingMemristorCSDPHubSNN, loader: DataLoader, device: torch.device) -> tuple[float, dict[str, float]]:
    correct = total = 0
    tie_s = sat_s = cmax_s = ry_s = 0.0
    n_b = 0
    for xb, yb in loader:
        y_hat, ed = model.predict(xb.to(device))
        correct += (y_hat.argmax(dim=1) == yb.argmax(dim=1).to(device)).sum().item()
        total += xb.shape[0]
        tie_s += ed["tie"]
        sat_s += ed["sat"]
        cmax_s += ed["cmax"]
        ry_s += ed["ry"]
        n_b += 1
    return correct / max(total, 1), {
        "ev_tie": tie_s / max(n_b, 1),
        "ev_sat": sat_s / max(n_b, 1),
        "ev_cmax": cmax_s / max(n_b, 1),
        "ev_ry": ry_s / max(n_b, 1),
    }


@torch.no_grad()
def train_epoch(
    model: AllSpikingMemristorCSDPHubSNN, loader: DataLoader, device: torch.device, args: argparse.Namespace
) -> tuple[float, dict[str, float]]:
    correct = total = 0
    diag: dict[str, float] = {}
    n_batches = len(loader)
    for i, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.float().to(device)
        B = xb.shape[0]

        if args.hard_neg_prob > 0.0:
            res_hn = model.simulate(xb, learn_hidden=False, learn_out_online=False, T_override=min(20, args.T))
            yn = hard_neg_labels(yb, res_hn["cnt_out"], args.hard_neg_prob, temp=args.hard_neg_temp)
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
            X_cat, y=Y_cat, mask=mask, learn_hidden=True, learn_out_online=False, collect_diag=is_last
        )
        if is_last:
            for k in ("r1", "r2", "rfc", "h1", "h2", "hfc"):
                diag[k] = float(pos_res[k])

        # 2) Clean (y=0) pass + 100% SPIKING lif_out local update (ZERO label leakage!)
        if args.out_rule in ("jax_dmu", "jax_sm_online"):
            out_res = model.simulate(
                xb, learn_hidden=False, learn_out_online=True, y_out_target=yb
            )
            y_hat = out_res["y_hat"]
        elif args.out_rule == "jax_sm_rate":
            out_res = model.simulate(xb, learn_hidden=False, learn_out_online=False)
            y_hat = model.rate_softmax_update(out_res["cnt_ro"], out_res["cnt_out"], yb)
        else:
            raise ValueError(args.out_rule)

        correct += (y_hat.argmax(dim=1) == yb.argmax(dim=1)).sum().item()
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
    p = argparse.ArgumentParser(description="100% ALL-SPIKING Memristor-Native CSDP-Hub SNN")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--dev-batch-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Architecture (ZERO weight sharing)
    p.add_argument("--c1", type=int, default=48)
    p.add_argument("--k1", type=int, default=5)
    p.add_argument("--s1", type=int, default=2)
    p.add_argument("--pad1", type=int, default=2)
    p.add_argument("--p1", type=int, default=2)
    p.add_argument("--c2", type=int, default=64)
    p.add_argument("--k2", type=int, default=3)
    p.add_argument("--s2", type=int, default=1)
    p.add_argument("--pad2", type=int, default=1)
    p.add_argument("--p2", type=int, default=1)
    p.add_argument("--c-coarse", type=int, default=0)
    p.add_argument("--hid-fc", type=int, default=1500)

    # SNN dynamics
    p.add_argument("--T", type=int, default=30)
    p.add_argument("--T-eval", type=int, default=100)
    p.add_argument("--dt", type=float, default=3.0)
    p.add_argument("--tau-m", type=float, default=100.0)
    p.add_argument("--tau-m-out", type=float, default=0.0)
    p.add_argument("--v-thr", type=float, default=0.055)
    p.add_argument("--v-thr-out", type=float, default=0.055)
    p.add_argument("--tau-tr", type=float, default=13.0)
    p.add_argument("--tau-tr-out", type=float, default=0.0)
    p.add_argument("--R-m", type=float, default=1.0)
    p.add_argument("--R-y", type=float, default=1.0)
    p.add_argument("--R-out", type=float, default=0.01)
    p.add_argument("--out-v-base", type=float, default=0.0)
    p.add_argument("--inh-R-tile", type=float, default=0.05)
    p.add_argument("--inh-R-fc", type=float, default=0.01)
    p.add_argument("--tile-init-scale", type=float, default=0.5)
    p.add_argument("--fc-init-scale", type=float, default=1.0)
    p.add_argument("--out-init-scale", type=float, default=0.05)
    p.add_argument("--out-bias", action="store_true")
    p.add_argument("--soft-reset-out", action="store_true")

    # 100% Spiking Output Rule
    p.add_argument("--out-rule", choices=["jax_dmu", "jax_sm_online", "jax_sm_rate"], default="jax_dmu")
    p.add_argument("--target-rate", type=float, default=0.65)
    p.add_argument("--neg-target-rate", type=float, default=0.0)
    p.add_argument("--out-warmup", type=int, default=5)
    p.add_argument("--out-temp", type=float, default=0.1)

    # Optimization
    p.add_argument("--eta", type=float, default=0.002)
    p.add_argument("--eta-out", type=float, default=0.002)
    p.add_argument("--eta-min-ratio", type=float, default=0.03)
    p.add_argument("--wd-out", type=float, default=1e-4)
    p.add_argument("--weight-bound", type=float, default=1.0)
    p.add_argument("--hard-neg-prob", type=float, default=0.0)
    p.add_argument("--hard-neg-temp", type=float, default=2.0)

    # Preserved Hub-Neuron parameters
    p.add_argument("--tau-m-hab", type=float, default=4.87563)
    p.add_argument("--v-thr-hab", type=float, default=0.249441)
    p.add_argument("--tau-hub", type=float, default=4.39997)
    p.add_argument("--hub-power", type=float, default=1.20313)
    p.add_argument("--hub-w-t1", type=float, default=0.004)
    p.add_argument("--hub-w-t2", type=float, default=0.008)
    p.add_argument("--hub-w-coarse", type=float, default=0.015)
    p.add_argument("--hub-w-fc", type=float, default=0.0503053)

    p.add_argument("--amp-bf16", action="store_true")
    p.add_argument("--diag", action="store_true")
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
    model = AllSpikingMemristorCSDPHubSNN(args, device)
    n_mem = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"AllSpikingMemristorCSDPHubSNN (100% SPIKING lif_out, ZERO weight sharing, NO zscore) | "
        f"t1={model.tile1.n_tiles}x({model.tile1.patch_dim}->{args.c1})->p{model.tile1.pool_hw} "
        + (f"t2={model.tile2.n_tiles}x({model.tile2.patch_dim}->{args.c2})->p{model.tile2.pool_hw} " if model.tile2 else "")
        + (f"tc={model.tile_coarse.n_tiles}x({model.tile_coarse.patch_dim}->{args.c_coarse}) " if model.tile_coarse else "")
        + f"fc={args.hid_fc} | lif_out={model.total_ro_dim}->10 ({args.out_rule}, R_out={args.R_out}, "
        f"v_thr_out={args.v_thr_out}, soft_reset={args.soft_reset_out}) | "
        f"T={args.T} T_eval={args.T_eval} | memristors={n_mem/1e6:.2f}M",
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
        te_acc, ev_diag = evaluate(model, te_ld, device)
        if te_acc > best:
            best = te_acc
            best_ep = ep
        elapsed = time.time() - t0
        diag_str = ""
        if diag:
            diag_str = (
                f" | r=({diag['r1']:.3f},{diag['r2']:.3f},{diag['rfc']:.3f}) "
                f"h=({diag['h1']:.2f},{diag['h2']:.2f},{diag['hfc']:.2f}) "
                f"out(ry={ev_diag['ev_ry']:.3f},cmax={ev_diag['ev_cmax']:.1f},tie={ev_diag['ev_tie']:.3f},sat={ev_diag['ev_sat']:.3f})"
            )
        print(
            f"Epoch {ep:2d}/{args.epochs}  Train: {tr_acc:.4f}  Test: {te_acc:.4f}  "
            f"Best: {best:.4f} (ep {best_ep})  [{elapsed / 60:.2f}m]{diag_str}",
            flush=True,
        )
        history.append({"epoch": ep, "train": tr_acc, "test": te_acc, **diag, **ev_diag})
        if args.save_json:
            Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.save_json).write_text(
                json.dumps({"config": vars(args), "best": best, "best_ep": best_ep, "history": history}, indent=2)
            )

    print(f"FINAL Best: {best:.4f} at epoch {best_ep} in {(time.time() - t0) / 60:.2f}m", flush=True)


if __name__ == "__main__":
    main()
