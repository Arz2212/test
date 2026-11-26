#!/usr/bin/env python3
"""100%-Local Spiking Conv+FC CSDP-SNN with preserved Hub-Neuron Modulation.

Strict invariants:
1. Hub neuron logic is preserved identically for every hidden CSDP layer:
     hub_fc (fixed constant weights hub_w_init, frozen) -> hub_lif -> hub trace ->
     hub_mod(tr, hub, phase, zmax_buf) = ((hub / zmax_buf) ** hub_power - phase) * tr
2. Learning is 100% local: ALL simulation & parameter updates run under @torch.no_grad()
   with ZERO backpropagation and ZERO autograd across layers.
   (torch.nn.grad.conv2d_weight is used purely as a fast cuDNN kernel for the local
    Hebbian/CSDP patch correlation sum_{b,h,w} x_patch(b,h,w) * mod(b,h,w).)
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


def decode_features(spk_cnt: torch.Tensor, mode: str, T: int) -> torch.Tensor:
    if mode == "rate":
        return spk_cnt / T
    if mode == "zscore":
        mu = spk_cnt.mean(dim=1, keepdim=True)
        std = spk_cnt.std(dim=1, keepdim=True, unbiased=False)
        return (spk_cnt - mu) / (std + 1e-5)
    if mode == "l2_sqrt":
        s = torch.sqrt(torch.clamp(spk_cnt, min=0.0))
        return s / (s.norm(dim=1, keepdim=True) + 1e-6)
    if mode == "sqrt_zscore":
        s = torch.sqrt(torch.clamp(spk_cnt, min=0.0))
        mu = s.mean(dim=1, keepdim=True)
        std = s.std(dim=1, keepdim=True, unbiased=False)
        return (s - mu) / (std + 1e-5)
    raise ValueError(f"Unknown decode mode: {mode}")


class ConvCSDPHubLayer(torch.nn.Module):
    """Spiking 2D convolutional layer trained with 100%-local CSDP + exact Hub Neuron logic."""

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
        hub_w_init: float,
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
        self.out_hw = (in_hw + 2 * padding - kernel_size) // stride + 1
        self.n_loc = self.out_hw * self.out_hw
        self.n_units = self.n_loc * out_ch
        self.pool_hw = self.out_hw // pool_size if pool_size > 1 else self.out_hw
        self.pool_dim = self.pool_hw * self.pool_hw * out_ch
        self.args = args

        init_scale = args.conv_init_scale
        # Conv kernel (out_ch, in_ch, k, k), lateral channel inhibition (out_ch, out_ch),
        # and spatially-specific top-down class context Vy (out_dim, out_ch * H * W)
        self.W = nn_param(out_ch, in_ch, kernel_size, kernel_size, bound=init_scale)
        self.M = nn_param(out_ch, out_ch, bound=init_scale)
        self.Vy = nn_param(out_dim, self.n_units, bound=init_scale)

        tau_sj = args.tau_m / args.dt
        self.lif = neuron.LIFNode(tau=tau_sj, v_threshold=args.v_thr, v_reset=0.0)

        # EXACT preserved Hub Neuron logic
        self.hub_fc = layer.Linear(self.n_units, 1, bias=False)
        self.hub_lif = neuron.LIFNode(
            tau=args.tau_m_hab, v_threshold=args.v_thr_hab, v_reset=0.0
        )
        torch.nn.init.constant_(self.hub_fc.weight, hub_w_init)
        self.hub_fc.weight.requires_grad_(False)
        self.register_buffer("zmax", torch.tensor(1.0))

        self.to(device)

    def hub_mod(
        self, tr: torch.Tensor, hub: torch.Tensor, phase: torch.Tensor, update_zmax: bool
    ) -> torch.Tensor:
        if update_zmax:
            torch.maximum(self.zmax, hub.max(), out=self.zmax)
        factor = (hub / self.zmax) ** self.args.hub_power - phase
        return factor * tr

    def params(self) -> list[torch.nn.Parameter]:
        return [self.W, self.M, self.Vy]

    def clamp_weights(self) -> None:
        wb = self.args.weight_bound
        self.W.data.clamp_(-wb, wb)
        self.M.data.fill_diagonal_(0.0)
        self.M.data.clamp_(0.0, wb)
        self.Vy.data.clamp_(-wb, wb)


class FCCSDPHubLayer(torch.nn.Module):
    """Spiking FC layer trained with 100%-local CSDP + exact Hub Neuron logic."""

    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
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

        tau_sj = args.tau_m / args.dt
        self.lif = neuron.LIFNode(tau=tau_sj, v_threshold=args.v_thr, v_reset=0.0)

        # EXACT preserved Hub Neuron logic
        self.hub_fc = layer.Linear(hid_dim, 1, bias=False)
        self.hub_lif = neuron.LIFNode(
            tau=args.tau_m_hab, v_threshold=args.v_thr_hab, v_reset=0.0
        )
        torch.nn.init.constant_(self.hub_fc.weight, args.hub_w_init)
        self.hub_fc.weight.requires_grad_(False)
        self.register_buffer("zmax", torch.tensor(1.0))

        self.to(device)

    def hub_mod(
        self, tr: torch.Tensor, hub: torch.Tensor, phase: torch.Tensor, update_zmax: bool
    ) -> torch.Tensor:
        if update_zmax:
            torch.maximum(self.zmax, hub.max(), out=self.zmax)
        factor = (hub / self.zmax) ** self.args.hub_power - phase
        return factor * tr

    def params(self) -> list[torch.nn.Parameter]:
        return [self.W, self.M, self.Vy]

    def clamp_weights(self) -> None:
        wb = self.args.weight_bound
        self.W.data.clamp_(-wb, wb)
        self.M.data.fill_diagonal_(0.0)
        self.M.data.clamp_(0.0, wb)
        self.Vy.data.clamp_(-wb, wb)


class ConvCSDPHubSNN(torch.nn.Module):
    def __init__(self, args: argparse.Namespace, device: torch.device):
        super().__init__()
        self.args = args
        self.device = device
        self.out_dim = 10
        in_ch = 2 if args.on_off else 1

        # Stage 1: Conv CSDP + Hub 1
        self.conv1 = ConvCSDPHubLayer(
            in_ch=in_ch,
            out_ch=args.c1,
            in_hw=28,
            kernel_size=args.k1,
            stride=args.s1,
            padding=args.k1 // 2,
            pool_size=args.p1,
            out_dim=self.out_dim,
            hub_w_init=args.hub_w_c1 if args.hub_w_c1 is not None else args.hub_w_init,
            args=args,
            device=device,
        )

        # Stage 2: Optional Conv2 CSDP + Hub 2
        if args.c2 > 0:
            self.conv2 = ConvCSDPHubLayer(
                in_ch=args.c1,
                out_ch=args.c2,
                in_hw=self.conv1.pool_hw,
                kernel_size=args.k2,
                stride=args.s2,
                padding=args.k2 // 2,
                pool_size=args.p2,
                out_dim=self.out_dim,
                hub_w_init=args.hub_w_c2 if args.hub_w_c2 is not None else args.hub_w_init,
                args=args,
                device=device,
            )
            fc_in = self.conv2.pool_dim
            self.conv_feat_dim = self.conv2.pool_dim
        else:
            self.conv2 = None
            fc_in = self.conv1.pool_dim
            self.conv_feat_dim = self.conv1.pool_dim

        # Stage 3: FC CSDP + Hub
        self.fc = FCCSDPHubLayer(
            in_dim=fc_in,
            hid_dim=args.hid_fc,
            out_dim=self.out_dim,
            args=args,
            device=device,
        )

        # Output readout connected to multi-scale pooled conv features + fc hidden layer
        self.readout_conv_dim = self.conv_feat_dim
        self.readout_fc_dim = args.hid_fc
        if args.readout_all and self.conv2 is not None:
            self.c1_pool_ro = 2 if self.conv1.pool_hw >= 14 else 1
            ro1_hw = self.conv1.pool_hw // self.c1_pool_ro
            self.readout_c1_dim = ro1_hw * ro1_hw * args.c1
        else:
            self.c1_pool_ro = 0
            self.readout_c1_dim = 0

        if args.out_mode == "local_delta":
            self.C_conv = torch.nn.Parameter(torch.zeros(self.readout_conv_dim, self.out_dim))
            self.C_fc = torch.nn.Parameter(torch.zeros(self.readout_fc_dim, self.out_dim))
            self.C_c1 = (
                torch.nn.Parameter(torch.zeros(self.readout_c1_dim, self.out_dim))
                if self.readout_c1_dim > 0
                else None
            )
            self.b_out = torch.nn.Parameter(torch.zeros(self.out_dim))
        else:
            self.C_conv = nn_param(self.readout_conv_dim, self.out_dim, bound=0.5)
            self.C_fc = nn_param(self.readout_fc_dim, self.out_dim, bound=0.5)
            self.C_c1 = (
                nn_param(self.readout_c1_dim, self.out_dim, bound=0.5)
                if self.readout_c1_dim > 0
                else None
            )
            self.b_out = None
            tau_sj = args.tau_m / args.dt
            self.lif_out = neuron.LIFNode(tau=tau_sj, v_threshold=args.v_thr, v_reset=0.0)

        self.to(device)

        hidden_params = self.conv1.params() + self.fc.params()
        if self.conv2 is not None:
            hidden_params = hidden_params + self.conv2.params()
        self.opt_hidden = torch.optim.Adam(
            hidden_params, lr=args.eta, fused=(device.type == "cuda")
        )

        out_params = [self.C_conv, self.C_fc]
        if self.C_c1 is not None:
            out_params.append(self.C_c1)
        if self.b_out is not None:
            out_params.append(self.b_out)
        self.opt_out = torch.optim.Adam(
            out_params, lr=args.eta_out, fused=(device.type == "cuda")
        )

    def _encode_input(self, x_img: torch.Tensor) -> torch.Tensor:
        if self.args.on_off:
            x_2ch = torch.cat([x_img, 1.0 - x_img], dim=1)
            return (torch.rand_like(x_2ch) < x_2ch).float()
        return (torch.rand_like(x_img) < x_img).float()

    def _conv_step(
        self,
        layer_mod: ConvCSDPHubLayer,
        x_in: torch.Tensor,
        z_prev: torch.Tensor,
        tr: torch.Tensor,
        hub: torch.Tensor,
        y: torch.Tensor | None,
        mask: torch.Tensor | None,
        learn: bool,
        collect_diag: bool,
        d: float,
        decay_hub: float,
        use_bf16: bool,
    ):
        """z_prev, tr are NCHW tensors of shape (B, Cout, H, W)."""
        B = x_in.shape[0]
        C = layer_mod.out_ch
        H = layer_mod.out_hw

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            # Fast cuDNN local 2D receptive field projection
            I = F.conv2d(x_in, layer_mod.W, stride=layer_mod.stride, padding=layer_mod.padding) * self.args.R_m
            # Channel-wise lateral inhibition via 1x1 conv (exact equivalent of z_prev @ M)
            M_1x1 = layer_mod.M.T.unsqueeze(-1).unsqueeze(-1)
            I = I - F.conv2d(z_prev, M_1x1) * self.args.inh_R
            if y is not None:
                I = I + ((y @ layer_mod.Vy) * self.args.R_y).view(B, C, H, H)

        z = layer_mod.lif(I.float())  # (B, C, H, H)
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
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    # 100% LOCAL CSDP correlation: sum_{b,h,w} x_patch * mod
                    gW = torch.nn.grad.conv2d_weight(
                        x_in,
                        layer_mod.W.shape,
                        mod_4d * (self.args.R_m * inv_loc),
                        stride=layer_mod.stride,
                        padding=layer_mod.padding,
                    )
                    # Zero-copy channel-wise lateral correlation: sum_{b,h,w} z_prev[b,c1,h,w] * mod[b,c2,h,w]
                    zp_bcn = z_prev.reshape(B, C, -1)
                    m_bcn = (mod_4d * (self.args.inh_R * inv_loc)).reshape(B, C, -1)
                    gM = torch.bmm(zp_bcn, m_bcn.transpose(1, 2)).sum(dim=0)
                    gVy = y.T @ (mod_flat * self.args.R_y)
                layer_mod.W.grad = gW.float()
                layer_mod.M.grad = gM.float()
                layer_mod.Vy.grad = gVy.float()

        if layer_mod.pool_size > 1:
            p_img = F.max_pool2d(z, layer_mod.pool_size, layer_mod.pool_size)
        else:
            p_img = z
        return z, p_img, tr, hub, h_spk_val

    @torch.no_grad()
    def simulate(
        self,
        x_flat: torch.Tensor,
        y: torch.Tensor | None,
        mask: torch.Tensor | None,
        learn_hidden: bool = False,
        learn_out_online: bool = False,
        y_out_target: torch.Tensor | None = None,
        collect_diag: bool = False,
    ) -> dict[str, torch.Tensor | float]:
        args = self.args
        B = x_flat.shape[0]
        dv = self.device
        x_img = x_flat.view(B, 1, 28, 28)
        use_bf16 = args.amp_bf16 and dv.type == "cuda"

        functional.reset_net(self.conv1.lif)
        functional.reset_net(self.fc.lif)
        if self.conv2 is not None:
            functional.reset_net(self.conv2.lif)
        if args.out_mode != "local_delta":
            functional.reset_net(self.lif_out)

        if learn_hidden or collect_diag:
            functional.reset_net(self.conv1.hub_lif)
            functional.reset_net(self.fc.hub_lif)
            hub1 = torch.zeros(B, 1, device=dv)
            hub_fc = torch.zeros(B, 1, device=dv)
            if self.conv2 is not None:
                functional.reset_net(self.conv2.hub_lif)
                hub2 = torch.zeros(B, 1, device=dv)
            else:
                hub2 = None
        else:
            hub1 = hub2 = hub_fc = None

        z1_prev = torch.zeros(B, self.conv1.out_ch, self.conv1.out_hw, self.conv1.out_hw, device=dv)
        tr1 = torch.zeros_like(z1_prev)
        if self.conv2 is not None:
            z2_prev = torch.zeros(
                B, self.conv2.out_ch, self.conv2.out_hw, self.conv2.out_hw, device=dv
            )
            tr2 = torch.zeros_like(z2_prev)
        else:
            z2_prev = tr2 = None

        zfc_prev = torch.zeros(B, self.fc.hid_dim, device=dv)
        tr_fc = torch.zeros_like(zfc_prev)
        tr_out = torch.zeros(B, self.out_dim, device=dv)

        cnt_conv = torch.zeros(B, self.readout_conv_dim, device=dv)
        cnt_fc = torch.zeros(B, self.readout_fc_dim, device=dv)
        cnt_c1 = torch.zeros(B, self.readout_c1_dim, device=dv) if self.readout_c1_dim > 0 else None
        cnt_out = torch.zeros(B, self.out_dim, device=dv)

        d = args.dt / args.tau_tr
        decay_hub = 1.0 / args.tau_hub
        r1_sum = r2_sum = rfc_sum = h1_sum = h2_sum = hfc_sum = 0.0

        for t in range(args.T):
            z0_img = self._encode_input(x_img)

            # Conv 1
            z1, p1_img, tr1, hub1, h1_v = self._conv_step(
                self.conv1, z0_img, z1_prev, tr1, hub1, y, mask,
                learn_hidden, collect_diag, d, decay_hub, use_bf16
            )
            if collect_diag:
                r1_sum += z1.mean().item()
                h1_sum += h1_v

            # Conv 2 (optional)
            if self.conv2 is not None:
                z2, p2_img, tr2, hub2, h2_v = self._conv_step(
                    self.conv2, p1_img, z2_prev, tr2, hub2, y, mask,
                    learn_hidden, collect_diag, d, decay_hub, use_bf16
                )
                if collect_diag:
                    r2_sum += z2.mean().item()
                    h2_sum += h2_v
                conv_ro = p2_img.reshape(B, -1)
                if cnt_c1 is not None:
                    c1_ro = (
                        F.max_pool2d(p1_img, self.c1_pool_ro, self.c1_pool_ro).reshape(B, -1)
                        if self.c1_pool_ro > 1
                        else p1_img.reshape(B, -1)
                    )
                    cnt_c1 = cnt_c1 + c1_ro
                else:
                    c1_ro = None
            else:
                conv_ro = p1_img.reshape(B, -1)
                c1_ro = None

            # FC hidden layer
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                Ifc = (conv_ro @ self.fc.W) * args.R_m - (zfc_prev @ self.fc.M) * args.inh_R
                if y is not None:
                    Ifc = Ifc + (y @ self.fc.Vy) * args.R_y
            zfc = self.fc.lif(Ifc.float())
            tr_fc = tr_fc + d * (zfc - tr_fc)
            if collect_diag:
                rfc_sum += zfc.mean().item()

            if learn_hidden or collect_diag:
                hfc_spk = self.fc.hub_lif(self.fc.hub_fc(zfc))
                hub_fc = hub_fc + decay_hub * (-hub_fc + hfc_spk.float())
                if collect_diag:
                    hfc_sum += hfc_spk.float().mean().item()
                if learn_hidden:
                    mod_fc = self.fc.hub_mod(tr_fc, hub_fc, mask, update_zmax=True)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                        self.fc.W.grad = (conv_ro.T @ (mod_fc * args.R_m)).float()
                        self.fc.M.grad = (zfc_prev.T @ (mod_fc * args.inh_R)).float()
                        self.fc.Vy.grad = (y.T @ (mod_fc * args.R_y)).float()

            if learn_hidden:
                self.opt_hidden.step()
                self.conv1.clamp_weights()
                if self.conv2 is not None:
                    self.conv2.clamp_weights()
                self.fc.clamp_weights()

            # Spiking output layer (if spiking mode)
            if args.out_mode != "local_delta":
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    I_out = conv_ro @ self.C_conv + zfc @ self.C_fc
                    if self.C_c1 is not None and c1_ro is not None:
                        I_out = I_out + c1_ro @ self.C_c1
                zy = self.lif_out(I_out.float())
                tr_out = tr_out + d * (zy - tr_out)
                cnt_out = cnt_out + zy

                if learn_out_online and y_out_target is not None:
                    dmu = 2.0 * (y_out_target - tr_out)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                        gC_conv = (-(conv_ro.T @ dmu)).float()
                        gC_fc = (-(zfc.T @ dmu)).float()
                        if self.C_c1 is not None and c1_ro is not None:
                            gC_c1 = (-(c1_ro.T @ dmu)).float()
                    if args.wd_out > 0.0:
                        gC_conv = gC_conv + args.wd_out * self.C_conv.data
                        gC_fc = gC_fc + args.wd_out * self.C_fc.data
                        if self.C_c1 is not None:
                            gC_c1 = gC_c1 + args.wd_out * self.C_c1.data
                    self.C_conv.grad = gC_conv
                    self.C_fc.grad = gC_fc
                    if self.C_c1 is not None:
                        self.C_c1.grad = gC_c1
                    self.opt_out.step()
                    wb = args.weight_bound
                    self.C_conv.data.clamp_(-wb, wb)
                    self.C_fc.data.clamp_(-wb, wb)
                    if self.C_c1 is not None:
                        self.C_c1.data.clamp_(-wb, wb)

            cnt_conv = cnt_conv + conv_ro
            cnt_fc = cnt_fc + zfc
            z1_prev = z1
            if self.conv2 is not None:
                z2_prev = z2
            zfc_prev = zfc

        res: dict[str, torch.Tensor | float] = {
            "cnt_conv": cnt_conv,
            "cnt_fc": cnt_fc,
            "cnt_c1": cnt_c1,
            "cnt_out": cnt_out,
        }
        if collect_diag:
            res["r1"] = r1_sum / args.T
            res["r2"] = r2_sum / args.T
            res["rfc"] = rfc_sum / args.T
            res["h1"] = h1_sum / args.T
            res["h2"] = h2_sum / args.T
            res["hfc"] = hfc_sum / args.T
            res["zm1"] = self.conv1.zmax.item()
            res["zm2"] = self.conv2.zmax.item() if self.conv2 is not None else 0.0
            res["zmfc"] = self.fc.zmax.item()
        return res

    @torch.no_grad()
    def local_delta_update(self, res: dict, y_onehot: torch.Tensor) -> torch.Tensor:
        """Purely local single-layer softmax-delta rule (zero autograd/backprop)."""
        args = self.args
        B = y_onehot.shape[0]
        fc_conv = decode_features(res["cnt_conv"], args.decode_mode, args.T)
        fc_fc = decode_features(res["cnt_fc"], args.decode_mode, args.T)
        fc_c1 = (
            decode_features(res["cnt_c1"], args.decode_mode, args.T)
            if self.C_c1 is not None
            else None
        )
        if args.out_dropout > 0.0:
            p = args.out_dropout
            scale = 1.0 / (1.0 - p)
            fc_conv = fc_conv * ((torch.rand_like(fc_conv) > p).float() * scale)
            fc_fc = fc_fc * ((torch.rand_like(fc_fc) > p).float() * scale)
            if fc_c1 is not None:
                fc_c1 = fc_c1 * ((torch.rand_like(fc_c1) > p).float() * scale)

        logits = fc_conv @ self.C_conv + fc_fc @ self.C_fc + self.b_out
        if fc_c1 is not None:
            logits = logits + fc_c1 @ self.C_c1
        probs = F.softmax(logits, dim=1)
        err = (probs - y_onehot) / B

        self.C_conv.grad = fc_conv.T @ err + args.wd_out * self.C_conv.data
        self.C_fc.grad = fc_fc.T @ err + args.wd_out * self.C_fc.data
        if fc_c1 is not None:
            self.C_c1.grad = fc_c1.T @ err + args.wd_out * self.C_c1.data
        self.b_out.grad = err.sum(dim=0)
        self.opt_out.step()
        return logits

    @torch.no_grad()
    def predict(self, x_flat: torch.Tensor) -> torch.Tensor:
        res = self.simulate(x_flat, y=None, mask=None, learn_hidden=False, learn_out_online=False)
        if self.args.out_mode == "local_delta":
            fc_conv = decode_features(res["cnt_conv"], self.args.decode_mode, self.args.T)
            fc_fc = decode_features(res["cnt_fc"], self.args.decode_mode, self.args.T)
            scores = fc_conv @ self.C_conv + fc_fc @ self.C_fc + self.b_out
            if self.C_c1 is not None:
                fc_c1 = decode_features(res["cnt_c1"], self.args.decode_mode, self.args.T)
                scores = scores + fc_c1 @ self.C_c1
            return scores
        return res["cnt_out"]


def augment_shift(x_flat: torch.Tensor, max_shift: int) -> torch.Tensor:
    if max_shift <= 0:
        return x_flat
    B = x_flat.shape[0]
    imgs = x_flat.view(B, 1, 28, 28)
    padded = F.pad(imgs, (max_shift, max_shift, max_shift, max_shift), mode="constant", value=0.0)
    dx = torch.randint(0, 2 * max_shift + 1, (B,), device=x_flat.device)
    dy = torch.randint(0, 2 * max_shift + 1, (B,), device=x_flat.device)
    out = torch.empty_like(imgs)
    for oy in range(2 * max_shift + 1):
        for ox in range(2 * max_shift + 1):
            sel = (dy == oy) & (dx == ox)
            if sel.any():
                out[sel] = padded[sel, :, oy : oy + 28, ox : ox + 28]
    return out.view(B, 784)


@torch.no_grad()
def evaluate(model: ConvCSDPHubSNN, loader: DataLoader, device: torch.device) -> float:
    correct = total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        scores = model.predict(xb)
        correct += (scores.argmax(dim=1) == yb.argmax(dim=1).to(device)).sum().item()
        total += xb.shape[0]
    return correct / max(total, 1)


@torch.no_grad()
def train_epoch(
    model: ConvCSDPHubSNN, loader: DataLoader, device: torch.device, args: argparse.Namespace
) -> tuple[float, dict[str, float]]:
    correct = total = 0
    diag: dict[str, float] = {}
    for i, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.float().to(device)
        if args.shift_aug > 0:
            xb = augment_shift(xb, args.shift_aug)
        B = xb.shape[0]

        if args.hard_neg_prob > 0.0:
            clean_scores = model.predict(xb)
            yn = hard_neg_labels(yb, clean_scores, args.hard_neg_prob, temp=args.hard_neg_temp)
        else:
            yn = uniform_neg_labels(yb, model.out_dim)

        X_cat = torch.cat([xb, xb], dim=0)
        Y_cat = torch.cat([yb, yn], dim=0)
        mask = torch.cat(
            [torch.ones(B, 1, device=device), torch.zeros(B, 1, device=device)], dim=0
        )

        # 1) Local CSDP hidden training with preserved Hub Neurons
        is_last = args.diag and (i == len(loader) - 1)
        pos_res = model.simulate(
            X_cat, y=Y_cat, mask=mask, learn_hidden=True, learn_out_online=False, collect_diag=is_last
        )
        if is_last:
            for k in ("r1", "r2", "rfc", "h1", "h2", "hfc", "zm1", "zm2", "zmfc"):
                diag[k] = float(pos_res[k])

        # 2) Clean-pass local output training (y=0 in hidden layers -> zero label leakage)
        if args.out_mode == "spiking_clean":
            out_res = model.simulate(
                xb, y=None, mask=None, learn_hidden=False, learn_out_online=True, y_out_target=yb
            )
            scores = out_res["cnt_out"]
        elif args.out_mode == "local_delta":
            out_res = model.simulate(
                xb, y=None, mask=None, learn_hidden=False, learn_out_online=False
            )
            scores = model.local_delta_update(out_res, yb)
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
    p = argparse.ArgumentParser(description="100%-Local Conv+FC CSDP-Hub-SNN")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--dev-batch-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Conv + FC architecture
    p.add_argument("--c1", type=int, default=32)
    p.add_argument("--k1", type=int, default=5)
    p.add_argument("--s1", type=int, default=1)
    p.add_argument("--p1", type=int, default=2)
    p.add_argument("--c2", type=int, default=64)
    p.add_argument("--k2", type=int, default=3)
    p.add_argument("--s2", type=int, default=1)
    p.add_argument("--p2", type=int, default=2)
    p.add_argument("--hid-fc", type=int, default=1000)
    p.add_argument("--readout-all", action="store_true")
    p.add_argument("--on-off", action="store_true")
    p.add_argument("--shift-aug", type=int, default=0)

    # SNN dynamics
    p.add_argument("--T", type=int, default=30)
    p.add_argument("--dt", type=float, default=3.0)
    p.add_argument("--tau-m", type=float, default=100.0)
    p.add_argument("--v-thr", type=float, default=0.055)
    p.add_argument("--tau-tr", type=float, default=13.0)
    p.add_argument("--R-m", type=float, default=1.0)
    p.add_argument("--R-y", type=float, default=1.0)
    p.add_argument("--inh-R", type=float, default=0.05)
    p.add_argument("--conv-init-scale", type=float, default=0.5)
    p.add_argument("--fc-init-scale", type=float, default=1.0)

    # Optimization
    p.add_argument("--eta", type=float, default=0.002)
    p.add_argument("--eta-out", type=float, default=0.002)
    p.add_argument("--eta-min-ratio", type=float, default=0.05)
    p.add_argument("--lr-schedule", choices=["none", "cosine"], default="cosine")
    p.add_argument("--weight-bound", type=float, default=1.0)
    p.add_argument("--out-mode", choices=["spiking_clean", "local_delta"], default="local_delta")
    p.add_argument("--decode-mode", choices=["rate", "zscore", "l2_sqrt", "sqrt_zscore"], default="zscore")
    p.add_argument("--wd-out", type=float, default=1e-3)
    p.add_argument("--out-dropout", type=float, default=0.15)
    p.add_argument("--hard-neg-prob", type=float, default=0.0)
    p.add_argument("--hard-neg-temp", type=float, default=2.0)

    # Preserved Hub Neuron hyperparameters
    p.add_argument("--tau-m-hab", type=float, default=4.87563)
    p.add_argument("--v-thr-hab", type=float, default=0.249441)
    p.add_argument("--tau-hub", type=float, default=4.39997)
    p.add_argument("--hub-power", type=float, default=1.20313)
    p.add_argument("--hub-w-init", type=float, default=0.0503053)
    p.add_argument("--hub-w-c1", type=float, default=None)
    p.add_argument("--hub-w-c2", type=float, default=None)

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
    model = ConvCSDPHubSNN(args, device)
    print(
        f"ConvCSDPHubSNN | c1={args.c1}({model.conv1.out_hw}x{model.conv1.out_hw}->p{model.conv1.pool_hw})"
        f" c2={args.c2}"
        + (f"({model.conv2.out_hw}x{model.conv2.out_hw}->p{model.conv2.pool_hw})" if model.conv2 else "")
        + f" fc={args.hid_fc} | ro=({model.readout_c1_dim}+{model.readout_conv_dim}+{model.readout_fc_dim})->10"
        f" | out={args.out_mode} T={args.T}",
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
                f" | r=({diag['r1']:.3f},{diag['r2']:.3f},{diag['rfc']:.3f})"
                f" h=({diag['h1']:.2f},{diag['h2']:.2f},{diag['hfc']:.2f})"
                f" zm=({diag['zm1']:.2f},{diag['zm2']:.2f},{diag['zmfc']:.2f})"
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
