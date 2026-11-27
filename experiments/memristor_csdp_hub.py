#!/usr/bin/env python3
"""100% Memristor-Native Local CSDP-SNN with Preserved Hub-Neuron Logic.

Strict hardware-compatibility invariants:
1. ZERO weight sharing: EVERY synaptic weight is an independent physical memristor
   on a crossbar array (either a tiled locally-connected mini-crossbar array or a dense FC crossbar).
   No convolution weight averaging across space!
2. 100% LOCAL learning: ALL simulation & plasticity run under @torch.no_grad().
   Zero backpropagation, zero autograd across layers. Every memristor updates strictly
   from its own row spike (pre_i) and its own column modulated trace (mod_j).
3. Hub-neuron logic preserved 1-to-1:
   hub_fc (frozen constant conductances) -> hub_lif -> hub trace ->
   hub_mod(tr, hub, phase, zmax) = ((hub / zmax) ** hub_power - phase) * tr
4. Memristor-native output readout (NO zscore/std normalization across layers!):
   - spiking_clean:       spiking LIF output crossbar trained on clean (y=0) pass via local error
   - spiking_clean_rate:  spiking LIF output crossbar updated locally via rate softmax-delta
   - crossbar_delta:      pure crossbar current readout on raw spike rates (cnt / T) + local delta rule
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
    """Array of L independent locally-connected memristive mini-crossbars (ZERO weight sharing!).

    Physical hardware mapping:
      Each spatial patch l in {0..L-1} connects its P = in_ch * k * k input wires
      into its OWN independent physical crossbar tile:
        W[l]:  (P, C)  feedforward memristor array
        M[l]:  (C, C)  local lateral inhibitory memristor array (hollow, non-negative)
        Vy[l]: (10, C) class-context memristor array
      Every single weight is a distinct physical memristor updated strictly locally:
        dW[l, p, c] = sum_b x_patch[l, b, p] * mod[l, b, c] * R_m
    """

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
        hub_mode: str,
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
        self.hub_mode = hub_mode
        self.args = args

        self.out_hw = (in_hw + 2 * padding - kernel_size) // stride + 1
        self.n_tiles = self.out_hw * self.out_hw
        self.patch_dim = in_ch * kernel_size * kernel_size
        self.n_units = self.n_tiles * out_ch
        self.pool_hw = self.out_hw // pool_size if pool_size > 1 else self.out_hw
        self.pool_dim = self.pool_hw * self.pool_hw * out_ch

        b = args.tile_init_scale
        # INDEPENDENT physical memristors per tile (ZERO weight sharing!)
        self.W = nn_param(self.n_tiles, self.patch_dim, out_ch, bound=b)
        self.M = nn_param(self.n_tiles, out_ch, out_ch, bound=b)
        self.Vy = nn_param(self.n_tiles, out_dim, out_ch, bound=b)
        self.register_buffer(
            "m_hollow",
            (1.0 - torch.eye(out_ch, device=device)).unsqueeze(0),
        )

        self.lif = neuron.LIFNode(tau=args.tau_m / args.dt, v_threshold=args.v_thr, v_reset=0.0)

        # EXACT preserved Hub-Neuron logic (either 1 hub for the whole layer or 1 local hub per tile)
        if hub_mode == "tile":
            # Each tile has its own independent 1-column constant-weight hub neuron!
            self.hub_w_val = hub_w
            self.hub_lif = neuron.LIFNode(
                tau=args.tau_m_hab, v_threshold=args.v_thr_hab, v_reset=0.0
            )
            self.register_buffer("zmax", torch.ones(self.n_tiles, 1, 1))
            self.hub_fc = None
        else:
            self.hub_w_val = hub_w
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
        """z_prev, tr have shape (L, B, C)."""
        B = x_img.shape[0]
        L = self.n_tiles
        C = self.out_ch

        # Physical local patch routing into each independent tile: (L, B, P)
        x_p = F.unfold(
            x_img,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        ).permute(2, 0, 1).contiguous()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            # L independent crossbar MVMs in parallel!
            I = torch.bmm(x_p, self.W) * self.args.R_m - torch.bmm(z_prev, self.M) * self.args.inh_R_tile
            if y is not None:
                y_l = y.unsqueeze(0).expand(L, B, -1)
                I = I + torch.bmm(y_l, self.Vy) * self.args.R_y

        z = self.lif(I.float())  # (L, B, C)
        tr = tr + d * (z - tr)

        h_spk_val = 0.0
        if learn or collect_diag:
            if self.hub_mode == "tile":
                # Local hub neuron per tile: fixed conductance hub_w * sum_c z[l, b, c]
                h_in = z.sum(dim=2, keepdim=True) * self.hub_w_val  # (L, B, 1)
                h_spk = self.hub_lif(h_in)
                hub = hub + decay_hub * (-hub + h_spk.float())
                if collect_diag:
                    h_spk_val = h_spk.float().mean().item()
                if learn:
                    torch.maximum(self.zmax, hub.amax(dim=1, keepdim=True), out=self.zmax)
                    phase_l = mask.unsqueeze(0)  # (1, B, 1)
                    factor = (hub / self.zmax) ** self.args.hub_power - phase_l
                    mod = factor * tr  # (L, B, C)
            else:
                z_flat = z.permute(1, 0, 2).reshape(B, self.n_units)
                h_spk = self.hub_lif(self.hub_fc(z_flat))
                hub = hub + decay_hub * (-hub + h_spk.float())
                if collect_diag:
                    h_spk_val = h_spk.float().mean().item()
                if learn:
                    torch.maximum(self.zmax, hub.max(), out=self.zmax)
                    factor = (hub / self.zmax) ** self.args.hub_power - mask  # (B, 1)
                    mod = factor.unsqueeze(0) * tr  # (L, B, C)

            if learn:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    # 100% LOCAL per-memristor update (ZERO sharing across tiles!)
                    gW = torch.bmm(x_p.transpose(1, 2), mod * self.args.R_m)
                    gM = torch.bmm(z_prev.transpose(1, 2), mod * self.args.inh_R_tile)
                    y_l = y.unsqueeze(0).expand(L, B, -1)
                    gVy = torch.bmm(y_l.transpose(1, 2), mod * self.args.R_y)
                gW, gM, gVy = gW.float(), gM.float(), gVy.float()
                if self.args.soft_bound_hidden:
                    wb = self.args.weight_bound
                    gW = gW * (wb - self.W.data.abs())
                    gVy = gVy * (wb - self.Vy.data.abs())
                self.W.grad = gW
                self.M.grad = gM
                self.Vy.grad = gVy

        # Output spatial spike map (B, C, H, W) + optional binary OR-pooling (max_pool2d on {0,1})
        z_img = z.permute(1, 2, 0).reshape(B, C, self.out_hw, self.out_hw)
        if self.pool_size > 1:
            p_img = F.max_pool2d(z_img, self.pool_size, self.pool_size)
        else:
            p_img = z_img
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

        # EXACT preserved Hub-Neuron logic
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
                gW, gM, gVy = gW.float(), gM.float(), gVy.float()
                if self.args.soft_bound_hidden:
                    wb = self.args.weight_bound
                    gW = gW * (wb - self.W.data.abs())
                    gVy = gVy * (wb - self.Vy.data.abs())
                self.W.grad = gW
                self.M.grad = gM
                self.Vy.grad = gVy

        return z, tr, hub, h_spk_val


class MemristorCSDPHubSNN(torch.nn.Module):
    def __init__(self, args: argparse.Namespace, device: torch.device):
        super().__init__()
        self.args = args
        self.device = device
        self.out_dim = 10
        in_ch = 2 if args.on_off else 1

        # Stage 1: Tiled locally-connected mini-crossbars (ZERO weight sharing!)
        self.tile1 = TiledCrossbarCSDPLayer(
            in_ch=in_ch,
            out_ch=args.c1,
            in_hw=28,
            kernel_size=args.k1,
            stride=args.s1,
            padding=args.pad1,
            pool_size=args.p1,
            out_dim=self.out_dim,
            hub_w=args.hub_w_t1,
            hub_mode=args.hub_mode,
            args=args,
            device=device,
        )

        # Optional Stage 2: Second tiled locally-connected mini-crossbar array (ZERO weight sharing!)
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
                hub_mode=args.hub_mode,
                args=args,
                device=device,
            )
            fc_in = self.tile2.pool_dim
        else:
            self.tile2 = None
            fc_in = self.tile1.pool_dim

        # Optional parallel coarse tiles (multi-scale V1 receptive fields, ZERO weight sharing!)
        if args.c_coarse > 0:
            self.tile_coarse = TiledCrossbarCSDPLayer(
                in_ch=in_ch,
                out_ch=args.c_coarse,
                in_hw=28,
                kernel_size=11,
                stride=5,
                padding=2,
                pool_size=1,
                out_dim=self.out_dim,
                hub_w=args.hub_w_coarse,
                hub_mode=args.hub_mode,
                args=args,
                device=device,
            )
            fc_in += self.tile_coarse.pool_dim
        else:
            self.tile_coarse = None

        # Global FC crossbar integrating across tiles
        self.fc = FCCrossbarCSDPLayer(
            in_dim=fc_in,
            hid_dim=args.hid_fc,
            out_dim=self.out_dim,
            hub_w=args.hub_w_fc,
            args=args,
            device=device,
        )

        # Readout crossbar dimensions (connecting from all active hidden stages)
        self.ro_dims = []
        if args.readout_all:
            self.ro_dims.append(("t1", self.tile1.pool_dim))
            if self.tile2 is not None:
                self.ro_dims.append(("t2", self.tile2.pool_dim))
            if self.tile_coarse is not None:
                self.ro_dims.append(("tc", self.tile_coarse.pool_dim))
        else:
            if self.tile2 is not None:
                self.ro_dims.append(("t2", self.tile2.pool_dim))
            else:
                self.ro_dims.append(("t1", self.tile1.pool_dim))
        self.ro_dims.append(("fc", args.hid_fc))
        self.total_ro_dim = sum(d for _, d in self.ro_dims)

        # Output crossbar (memristor-native: NO zscore!)
        if args.out_mode == "crossbar_delta":
            self.C_out = torch.nn.Parameter(torch.zeros(self.total_ro_dim, self.out_dim))
            self.b_out = torch.nn.Parameter(torch.zeros(self.out_dim))
            self.lif_out = None
        else:
            # 100% SPIKING output LIF layer!
            self.C_out = nn_param(self.total_ro_dim, self.out_dim, bound=args.out_init_scale)
            self.b_out = None
            v_thr_o = args.v_thr_out if args.v_thr_out > 0 else args.v_thr
            self.lif_out = neuron.LIFNode(
                tau=args.tau_m / args.dt, v_threshold=v_thr_o, v_reset=0.0
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

    def _encode_input(self, x_img: torch.Tensor) -> torch.Tensor:
        if self.args.on_off:
            x_2ch = torch.cat([x_img, 1.0 - x_img], dim=1)
            return (torch.rand_like(x_2ch) < x_2ch).float()
        return (torch.rand_like(x_img) < x_img).float()

    def _init_hub(self, tile_mod: TiledCrossbarCSDPLayer, B: int, dv: torch.device) -> torch.Tensor:
        if tile_mod.hub_mode == "tile":
            return torch.zeros(tile_mod.n_tiles, B, 1, device=dv)
        return torch.zeros(B, 1, device=dv)

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
        if self.tile2 is not None:
            functional.reset_net(self.tile2.lif)
        if self.tile_coarse is not None:
            functional.reset_net(self.tile_coarse.lif)
        if self.lif_out is not None:
            functional.reset_net(self.lif_out)

        if learn_hidden or collect_diag:
            functional.reset_net(self.tile1.hub_lif)
            functional.reset_net(self.fc.hub_lif)
            hub1 = self._init_hub(self.tile1, B, dv)
            hub_fc = torch.zeros(B, 1, device=dv)
            if self.tile2 is not None:
                functional.reset_net(self.tile2.hub_lif)
                hub2 = self._init_hub(self.tile2, B, dv)
            else:
                hub2 = None
            if self.tile_coarse is not None:
                functional.reset_net(self.tile_coarse.hub_lif)
                hub_c = self._init_hub(self.tile_coarse, B, dv)
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
        decay_hub = 1.0 / args.tau_hub
        r1_sum = r2_sum = rfc_sum = ry_sum = h1_sum = h2_sum = hfc_sum = 0.0

        for t in range(T_steps):
            z0_img = self._encode_input(x_img)

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

            # Assemble readout spike vector (pure binary spikes {0, 1}!)
            ro_parts = []
            for key, _ in self.ro_dims:
                if key == "t1":
                    ro_parts.append(p1_flat)
                elif key == "t2":
                    ro_parts.append(p2_flat)
                elif key == "tc":
                    ro_parts.append(pc_flat)
                elif key == "fc":
                    ro_parts.append(zfc)
            ro_spk = torch.cat(ro_parts, dim=1) if len(ro_parts) > 1 else ro_parts[0]
            cnt_ro = cnt_ro + ro_spk

            # 100% Spiking Output Crossbar (if enabled)
            if self.lif_out is not None:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    I_out = (ro_spk @ self.C_out) * args.R_out
                zy = self.lif_out(I_out.float())
                tr_out = tr_out + d * (zy - tr_out)
                cnt_out = cnt_out + zy
                if collect_diag:
                    ry_sum += zy.mean().item()

                if learn_out_online and y_out_target is not None and t >= args.out_warmup:
                    dmu = 2.0 * (y_out_target - tr_out) * (1.0 / B if args.norm_out_grad else 1.0)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                        gC = (-(ro_spk.T @ dmu)).float()
                    if args.soft_bound_out:
                        gC = gC * (args.weight_bound - self.C_out.data.abs())
                    if args.wd_out > 0.0:
                        gC = gC + args.wd_out * self.C_out.data
                    self.C_out.grad = gC
                    self.opt_out.step()
                    self.C_out.data.clamp_(-args.weight_bound, args.weight_bound)

            if collect_diag:
                r1_sum += z1.mean().item()
                if z2 is not None:
                    r2_sum += z2.mean().item()
                rfc_sum += zfc.mean().item()
                h1_sum += h1_v
                h2_sum += h2_v
                hfc_sum += hfc_v

            z1_prev = z1
            if z2 is not None:
                z2_prev = z2
            if zc is not None:
                zc_prev = zc
            zfc_prev = zfc

        res: dict[str, torch.Tensor | float] = {"cnt_ro": cnt_ro, "cnt_out": cnt_out}
        if collect_diag:
            res.update({
                "r1": r1_sum / T_steps,
                "r2": r2_sum / T_steps,
                "rfc": rfc_sum / T_steps,
                "ry": ry_sum / T_steps,
                "h1": h1_sum / T_steps,
                "h2": h2_sum / T_steps,
                "hfc": hfc_sum / T_steps,
                "zm1": self.tile1.zmax.mean().item(),
                "zm2": self.tile2.zmax.mean().item() if self.tile2 is not None else 0.0,
                "zmfc": self.fc.zmax.item(),
            })
        return res

    @torch.no_grad()
    def crossbar_delta_update(self, cnt_ro: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        """Pure memristor-crossbar readout on raw firing rates (cnt / T) — NO zscore!
        Mathematically equivalent to non-resetting integrate-only output neurons V_out(T)/T.
        Optional analog memristor read noise (--ro-noise) regularizes training.
        """
        B = y_onehot.shape[0]
        rates = cnt_ro / self.args.T  # pure firing rate in [0, 1]
        if self.args.ro_noise > 0.0:
            rates = (rates + self.args.ro_noise * torch.randn_like(rates)).clamp_(0.0, 1.0)
        logits = rates @ self.C_out + self.b_out
        probs = F.softmax(logits, dim=1)
        err = (probs - y_onehot) / B
        self.C_out.grad = rates.T @ err + self.args.wd_out * self.C_out.data
        self.b_out.grad = err.sum(dim=0)
        self.opt_out.step()
        if self.args.clamp_out:
            self.C_out.data.clamp_(-self.args.weight_bound, self.args.weight_bound)
            self.b_out.data.clamp_(-self.args.weight_bound, self.args.weight_bound)
        return logits

    @torch.no_grad()
    def spiking_rate_delta_update(
        self, cnt_ro: torch.Tensor, cnt_out: torch.Tensor, y_onehot: torch.Tensor
    ) -> torch.Tensor:
        """Local rate-level softmax-delta update directly on spiking output crossbar C_out."""
        B = y_onehot.shape[0]
        rates = cnt_ro / self.args.T
        probs = F.softmax(cnt_out / self.args.out_temp, dim=1)
        err = (probs - y_onehot) / B
        gC = rates.T @ err
        if self.args.soft_bound_out:
            gC = gC * (self.args.weight_bound - self.C_out.data.abs())
        if self.args.wd_out > 0.0:
            gC = gC + self.args.wd_out * self.C_out.data
        self.C_out.grad = gC
        self.opt_out.step()
        self.C_out.data.clamp_(-self.args.weight_bound, self.args.weight_bound)
        return cnt_out

    @torch.no_grad()
    def predict(self, x_flat: torch.Tensor, T_override: int | None = None) -> torch.Tensor:
        T_use = T_override if T_override is not None else (self.args.T_eval if self.args.T_eval > 0 else self.args.T)
        res = self.simulate(x_flat, learn_hidden=False, learn_out_online=False, T_override=T_use)
        if self.args.out_mode == "crossbar_delta":
            return (res["cnt_ro"] / T_use) @ self.C_out + self.b_out
        return res["cnt_out"]


def apply_microsaccade(x_flat: torch.Tensor, max_shift: int) -> torch.Tensor:
    """Biological retinal microsaccade: random ±max_shift pixel jitter on 28x28 sensor grid."""
    if max_shift <= 0:
        return x_flat
    B = x_flat.shape[0]
    img = x_flat.view(B, 1, 28, 28)
    padded = F.pad(img, (max_shift, max_shift, max_shift, max_shift), mode="constant", value=0.0)
    dx = torch.randint(0, 2 * max_shift + 1, (B,), device=x_flat.device)
    dy = torch.randint(0, 2 * max_shift + 1, (B,), device=x_flat.device)
    # Vectorized crop per sample
    ys = torch.arange(28, device=x_flat.device).view(1, 28, 1) + dy.view(B, 1, 1)
    xs = torch.arange(28, device=x_flat.device).view(1, 1, 28) + dx.view(B, 1, 1)
    b_idx = torch.arange(B, device=x_flat.device).view(B, 1, 1)
    out = padded[b_idx, 0, ys, xs].view(B, 784)
    return out


@torch.no_grad()
def evaluate(model: MemristorCSDPHubSNN, loader: DataLoader, device: torch.device) -> float:
    correct = total = 0
    for xb, yb in loader:
        scores = model.predict(xb.to(device))
        correct += (scores.argmax(dim=1) == yb.argmax(dim=1).to(device)).sum().item()
        total += xb.shape[0]
    return correct / max(total, 1)


@torch.no_grad()
def train_epoch(
    model: MemristorCSDPHubSNN, loader: DataLoader, device: torch.device, args: argparse.Namespace
) -> tuple[float, dict[str, float]]:
    correct = total = 0
    diag: dict[str, float] = {}
    n_batches = len(loader)
    for i, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.float().to(device)
        if args.microsaccade > 0:
            xb = apply_microsaccade(xb, args.microsaccade)
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

        # 1) Local CSDP training of hidden memristor crossbars (with Hub Neurons)
        is_last = args.diag and (i == n_batches - 1)
        pos_res = model.simulate(
            X_cat, y=Y_cat, mask=mask, learn_hidden=True, learn_out_online=False, collect_diag=is_last
        )
        if is_last:
            for k in ("r1", "r2", "rfc", "ry", "h1", "h2", "hfc", "zm1", "zm2", "zmfc"):
                diag[k] = float(pos_res[k])

        # 2) Clean (y=0) pass + local output crossbar update (ZERO label leakage!)
        if args.out_mode == "spiking_clean":
            out_res = model.simulate(
                xb, learn_hidden=False, learn_out_online=True, y_out_target=yb
            )
            scores = out_res["cnt_out"]
        elif args.out_mode == "spiking_clean_rate":
            out_res = model.simulate(xb, learn_hidden=False, learn_out_online=False)
            scores = model.spiking_rate_delta_update(out_res["cnt_ro"], out_res["cnt_out"], yb)
        elif args.out_mode == "crossbar_delta":
            out_res = model.simulate(xb, learn_hidden=False, learn_out_online=False)
            scores = model.crossbar_delta_update(out_res["cnt_ro"], yb)
        else:
            raise ValueError(args.out_mode)

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
    p = argparse.ArgumentParser(description="100% Memristor-Native Local CSDP-Hub-SNN (NO weight sharing)")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--dev-batch-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Tiled locally-connected crossbars (ZERO weight sharing!)
    p.add_argument("--c1", type=int, default=64)
    p.add_argument("--k1", type=int, default=7)
    p.add_argument("--s1", type=int, default=3)
    p.add_argument("--pad1", type=int, default=1)
    p.add_argument("--p1", type=int, default=2)

    p.add_argument("--c2", type=int, default=0)
    p.add_argument("--k2", type=int, default=3)
    p.add_argument("--s2", type=int, default=1)
    p.add_argument("--pad2", type=int, default=1)
    p.add_argument("--p2", type=int, default=1)

    p.add_argument("--c-coarse", type=int, default=0)
    p.add_argument("--hid-fc", type=int, default=1500)
    p.add_argument("--readout-all", action="store_true")
    p.add_argument("--on-off", action="store_true")

    # SNN dynamics
    p.add_argument("--T", type=int, default=30)
    p.add_argument("--T-eval", type=int, default=0)
    p.add_argument("--ro-noise", type=float, default=0.0)
    p.add_argument("--microsaccade", type=int, default=0)
    p.add_argument("--dt", type=float, default=3.0)
    p.add_argument("--tau-m", type=float, default=100.0)
    p.add_argument("--v-thr", type=float, default=0.055)
    p.add_argument("--v-thr-out", type=float, default=0.0)
    p.add_argument("--clamp-out", action="store_true")
    p.add_argument("--tau-tr", type=float, default=13.0)
    p.add_argument("--R-m", type=float, default=1.0)
    p.add_argument("--R-y", type=float, default=1.0)
    p.add_argument("--R-out", type=float, default=0.2)
    p.add_argument("--inh-R-tile", type=float, default=0.05)
    p.add_argument("--inh-R-fc", type=float, default=0.01)
    p.add_argument("--tile-init-scale", type=float, default=0.5)
    p.add_argument("--fc-init-scale", type=float, default=1.0)
    p.add_argument("--out-init-scale", type=float, default=0.2)

    # Optimization & plasticity
    p.add_argument("--eta", type=float, default=0.002)
    p.add_argument("--eta-out", type=float, default=0.005)
    p.add_argument("--eta-min-ratio", type=float, default=0.05)
    p.add_argument("--lr-schedule", choices=["none", "cosine"], default="cosine")
    p.add_argument("--weight-bound", type=float, default=1.0)
    p.add_argument("--soft-bound-hidden", action="store_true")
    p.add_argument("--soft-bound-out", action="store_true")
    p.add_argument("--norm-out-grad", action="store_true")
    p.add_argument("--out-warmup", type=int, default=5)
    p.add_argument("--out-temp", type=float, default=4.0)
    p.add_argument("--wd-out", type=float, default=1e-4)
    p.add_argument(
        "--out-mode",
        choices=["spiking_clean", "spiking_clean_rate", "crossbar_delta"],
        default="spiking_clean",
    )
    p.add_argument("--hard-neg-prob", type=float, default=0.0)
    p.add_argument("--hard-neg-temp", type=float, default=2.0)

    # Preserved Hub-Neuron parameters
    p.add_argument("--hub-mode", choices=["layer", "tile"], default="layer")
    p.add_argument("--tau-m-hab", type=float, default=4.87563)
    p.add_argument("--v-thr-hab", type=float, default=0.249441)
    p.add_argument("--tau-hub", type=float, default=4.39997)
    p.add_argument("--hub-power", type=float, default=1.20313)
    p.add_argument("--hub-w-t1", type=float, default=0.006)
    p.add_argument("--hub-w-t2", type=float, default=0.006)
    p.add_argument("--hub-w-coarse", type=float, default=0.01)
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
    model = MemristorCSDPHubSNN(args, device)
    n_mem = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"MemristorCSDPHubSNN (ZERO weight sharing, 100% local) | "
        f"t1={model.tile1.n_tiles}x({model.tile1.patch_dim}->{args.c1})->p{model.tile1.pool_hw} "
        + (f"t2={model.tile2.n_tiles}x({model.tile2.patch_dim}->{args.c2})->p{model.tile2.pool_hw} " if model.tile2 else "")
        + (f"tc={model.tile_coarse.n_tiles}x({model.tile_coarse.patch_dim}->{args.c_coarse}) " if model.tile_coarse else "")
        + f"fc={args.hid_fc} | ro={model.total_ro_dim}->10 ({args.out_mode}) | "
        f"hub={args.hub_mode} on_off={args.on_off} T={args.T} | memristors={n_mem/1e6:.2f}M",
        flush=True,
    )

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

        tr_acc, diag = train_epoch(model, tr_ld, device, args)
        te_acc = evaluate(model, te_ld, device)
        if te_acc > best:
            best = te_acc
            best_ep = ep
        elapsed = time.time() - t0
        diag_str = ""
        if diag:
            diag_str = (
                f" | r=({diag['r1']:.3f},{diag['r2']:.3f},{diag['rfc']:.3f},y={diag['ry']:.3f})"
                f" h=({diag['h1']:.2f},{diag['h2']:.2f},{diag['hfc']:.2f})"
                f" zm=({diag['zm1']:.2f},{diag['zm2']:.2f},{diag['zmfc']:.2f})"
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
