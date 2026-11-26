#!/usr/bin/env python3
"""Comprehensive sweep of spike-to-signal conversions & linear classifier hyperparameters.

Since the spiking hidden layers (W1, M1, V2y, W2, M2, V3y) are trained locally
via CSDP without feedback from the readout head, we can attach a bank of
independent nn.Linear classifiers to the exact same SNN trajectory and evaluate
dozens of (spike_mode, cls_train_mode, warmup_steps, lr_cls, wd_cls, dropout,
label_smoothing, cls_steps, cls_bias, feat_source) configurations simultaneously.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import mnist_solver_linear as msl


@dataclass
class HeadSpec:
    name: str
    feat_source: str = "z2"          # z2 | z1_z2
    spike_mode: str = "count"        # count | rate | l2_count | zscore | log1p | trace | mean_trace
    warmup_steps: int = 0            # 0 | 5 | 10
    cls_train_mode: str = "clean"    # clean | pos | both
    lr_cls: float = 0.002
    wd_cls: float = 1e-4
    cls_dropout: float = 0.0
    label_smoothing: float = 0.0
    cls_bias: bool = True
    cls_steps: int = 1
    cosine_lr: bool = True


class ProbeHead:
    def __init__(self, spec: HeadSpec, hid1: int, hid2: int, out_dim: int, epochs: int, device: torch.device):
        self.spec = spec
        in_dim = hid2 if spec.feat_source == "z2" else (hid1 + hid2)
        torch.manual_seed(msl.seed + 77)
        self.linear = nn.Linear(in_dim, out_dim, bias=spec.cls_bias).to(device)
        self.dropout = nn.Dropout(p=spec.cls_dropout) if spec.cls_dropout > 0.0 else nn.Identity()
        self.opt = torch.optim.AdamW(
            self.linear.parameters(), lr=spec.lr_cls, weight_decay=spec.wd_cls
        )
        self.sched = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt, T_max=max(epochs, 1), eta_min=spec.lr_cls * 0.05
            )
            if spec.cosine_lr and epochs > 1
            else None
        )
        self.best_test = 0.0
        self.best_epoch = 0
        self.ep1_test = 0.0
        self.last_test = 0.0
        self.last_train = 0.0
        self.history: list[dict] = []

    def extract_feat(self, stats: dict) -> torch.Tensor:
        w = self.spec.warmup_steps
        eff_steps = max(msl.T - w, 1)
        f2 = msl.decode_spikes(
            stats[f"spk2_w{w}"],
            stats["tr2_last"],
            stats[f"tr2_sum_w{w}"],
            eff_steps,
            mode=self.spec.spike_mode,
        )
        if self.spec.feat_source == "z1_z2":
            f1 = msl.decode_spikes(
                stats[f"spk1_w{w}"],
                stats["tr1_last"],
                stats[f"tr1_sum_w{w}"],
                eff_steps,
                mode=self.spec.spike_mode,
            )
            return torch.cat([f1, f2], dim=1)
        return f2

    def train_batch(self, pos_stats: dict, clean_stats: dict, target_idx: torch.Tensor) -> int:
        self.linear.train()
        if self.spec.cls_train_mode == "pos":
            feat = self.extract_feat(pos_stats)
            tgt = target_idx
        elif self.spec.cls_train_mode == "clean":
            feat = self.extract_feat(clean_stats)
            tgt = target_idx
        elif self.spec.cls_train_mode == "both":
            f_pos = self.extract_feat(pos_stats)
            f_cln = self.extract_feat(clean_stats)
            feat = torch.cat([f_pos, f_cln], dim=0)
            tgt = torch.cat([target_idx, target_idx], dim=0)
        else:
            raise ValueError(self.spec.cls_train_mode)

        with torch.enable_grad():
            logits = None
            for _ in range(max(self.spec.cls_steps, 1)):
                x_in = self.dropout(feat) if self.spec.cls_dropout > 0.0 else feat
                logits = self.linear(x_in)
                loss = F.cross_entropy(
                    logits, tgt, label_smoothing=self.spec.label_smoothing
                )
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()

        B = target_idx.shape[0]
        preds = logits[-B:].detach().argmax(dim=1)
        return int((preds == target_idx).sum().item())

    @torch.no_grad()
    def eval_batch(self, clean_stats: dict, target_idx: torch.Tensor) -> int:
        self.linear.eval()
        feat = self.extract_feat(clean_stats)
        logits = self.linear(feat)
        return int((logits.argmax(dim=1) == target_idx).sum().item())


def build_specs() -> list[HeadSpec]:
    specs: list[HeadSpec] = []

    # Group 1: Spike-to-signal conversion modes × train_mode (pos vs clean vs both)
    for mode in msl.SPIKE_MODES:
        for tmode in ("pos", "clean", "both"):
            lr = 0.001 if mode in ("count", "zscore", "log1p") else 0.005
            specs.append(
                HeadSpec(
                    name=f"mode_{mode}_{tmode}_lr{lr}",
                    spike_mode=mode,
                    cls_train_mode=tmode,
                    lr_cls=lr,
                    wd_cls=1e-4,
                )
            )

    # Group 2: Learning rate sweep for key spike modes (count, rate, l2_count, zscore, log1p)
    for mode in ("count", "rate", "l2_count", "zscore", "log1p"):
        for lr in (3e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2):
            specs.append(
                HeadSpec(
                    name=f"lr_{mode}_clean_{lr:g}",
                    spike_mode=mode,
                    cls_train_mode="clean",
                    lr_cls=lr,
                    wd_cls=1e-4,
                )
            )

    # Group 3: Weight decay & dropout & label smoothing & cls_steps & warmup_steps
    for wd in (0.0, 1e-3, 1e-2, 5e-2):
        for mode, lr in (("count", 1e-3), ("l2_count", 5e-3), ("rate", 5e-3)):
            specs.append(
                HeadSpec(
                    name=f"wd_{mode}_{wd:g}",
                    spike_mode=mode,
                    cls_train_mode="clean",
                    lr_cls=lr,
                    wd_cls=wd,
                )
            )

    for drop in (0.1, 0.2, 0.3):
        for tmode in ("pos", "clean", "both"):
            for mode, lr in (("count", 1e-3), ("l2_count", 5e-3)):
                specs.append(
                    HeadSpec(
                        name=f"drop_{mode}_{tmode}_d{drop}",
                        spike_mode=mode,
                        cls_train_mode=tmode,
                        lr_cls=lr,
                        wd_cls=1e-3,
                        cls_dropout=drop,
                    )
                )

    for smooth in (0.05, 0.1):
        for mode, lr in (("count", 1e-3), ("l2_count", 5e-3), ("log1p", 2e-3)):
            specs.append(
                HeadSpec(
                    name=f"smooth_{mode}_s{smooth}",
                    spike_mode=mode,
                    cls_train_mode="clean",
                    lr_cls=lr,
                    wd_cls=1e-3,
                    label_smoothing=smooth,
                )
            )

    for steps in (3, 5):
        for mode, lr in (("count", 1e-3), ("l2_count", 5e-3), ("rate", 5e-3)):
            specs.append(
                HeadSpec(
                    name=f"steps_{mode}_k{steps}",
                    spike_mode=mode,
                    cls_train_mode="clean",
                    lr_cls=lr,
                    wd_cls=1e-3,
                    cls_steps=steps,
                )
            )

    for w in (5, 10):
        for mode, lr in (("count", 1e-3), ("l2_count", 5e-3), ("rate", 5e-3)):
            specs.append(
                HeadSpec(
                    name=f"warmup_{mode}_w{w}",
                    spike_mode=mode,
                    warmup_steps=w,
                    cls_train_mode="clean",
                    lr_cls=lr,
                    wd_cls=1e-4,
                )
            )

    # Bias=False & No-Cosine ablations
    for mode, lr in (("count", 1e-3), ("l2_count", 5e-3)):
        specs.append(
            HeadSpec(
                name=f"nobias_{mode}",
                spike_mode=mode,
                cls_train_mode="clean",
                lr_cls=lr,
                cls_bias=False,
            )
        )
        specs.append(
            HeadSpec(
                name=f"constlr_{mode}",
                spike_mode=mode,
                cls_train_mode="clean",
                lr_cls=lr,
                cosine_lr=False,
            )
        )

    # Group 4: Multi-layer readout ablation (z1 + z2 = 6000 -> 10, like C2+C3 in CSDP paper)
    for mode, lr in (("count", 5e-4), ("rate", 3e-3), ("l2_count", 5e-3), ("log1p", 1e-3)):
        for tmode in ("pos", "clean"):
            specs.append(
                HeadSpec(
                    name=f"z1z2_{mode}_{tmode}",
                    feat_source="z1_z2",
                    spike_mode=mode,
                    cls_train_mode=tmode,
                    lr_cls=lr,
                    wd_cls=1e-3,
                    cls_dropout=0.1,
                )
            )

    # Deduplicate by name
    seen = set()
    uniq = []
    for s in specs:
        if s.name not in seen:
            seen.add(s.name)
            uniq.append(s)
    return uniq


@torch.no_grad()
def forward_snn_collect(model: msl.CSDPSNNLinear, x, y=None, mask=None, learn=True, slice_b: int | None = None):
    """Run SNN for T steps (updating CSDP if learn=True) and collect multi-warmup stats."""
    from spikingjelly.activation_based import functional

    B = x.shape[0]
    dv = model.device
    functional.reset_net(model.lif1)
    functional.reset_net(model.lif2)
    if learn:
        functional.reset_net(model.hub_lif1)
        functional.reset_net(model.hub_lif2)

    tr1 = torch.zeros(B, msl.hid1, device=dv)
    tr2 = torch.zeros(B, msl.hid2, device=dv)
    z1_prev = torch.zeros(B, msl.hid1, device=dv)
    z2_prev = torch.zeros(B, msl.hid2, device=dv)
    if learn:
        hub1 = torch.zeros(B, 1, device=dv)
        hub2 = torch.zeros(B, 1, device=dv)

    sb = slice_b if slice_b is not None else B
    warmups = (0, 5, 10)
    spk1 = {w: torch.zeros(sb, msl.hid1, device=dv) for w in warmups}
    spk2 = {w: torch.zeros(sb, msl.hid2, device=dv) for w in warmups}
    tr1_sum = {w: torch.zeros(sb, msl.hid1, device=dv) for w in warmups}
    tr2_sum = {w: torch.zeros(sb, msl.hid2, device=dv) for w in warmups}

    d = msl.dt / msl.tau_tr
    decay_hub = 1.0 / msl.tau_hub
    has_y = y is not None

    for t in range(msl.T):
        z0 = (torch.rand(B, msl.in_dim, device=dv) < x).float()

        I1 = (z0 @ model.W1) * msl.R_m - (z1_prev @ model.M1) * msl.inh_R
        if has_y:
            I1 = I1 + (y @ model.V2y) * msl.R_m
        z1 = model.lif1(I1)
        tr1 = tr1 + d * (z1 - tr1)

        if learn:
            h1_spk = model.hub_lif1(model.hub_fc1(z1))
            hub1 = hub1 + decay_hub * (-hub1 + h1_spk.float())
            mod1 = model.hub_mod(tr1, hub1, mask, model.zmax1)

        I2 = (z1 @ model.W2) * msl.R_m - (z2_prev @ model.M2) * msl.inh_R
        if has_y:
            I2 = I2 + (y @ model.V3y) * msl.R_m
        z2 = model.lif2(I2)
        tr2 = tr2 + d * (z2 - tr2)

        if learn:
            h2_spk = model.hub_lif2(model.hub_fc2(z2))
            hub2 = hub2 + decay_hub * (-hub2 + h2_spk.float())
            mod2 = model.hub_mod(tr2, hub2, mask, model.zmax2)

            mod1_rm = mod1 * msl.R_m
            mod1_inh = mod1 * msl.inh_R
            mod2_rm = mod2 * msl.R_m
            mod2_inh = mod2 * msl.inh_R
            model.W1.grad = z0.T @ mod1_rm
            model.M1.grad = z1_prev.T @ mod1_inh
            model.V2y.grad = y.T @ mod1_rm
            model.W2.grad = z1.T @ mod2_rm
            model.M2.grad = z2_prev.T @ mod2_inh
            model.V3y.grad = y.T @ mod2_rm
            model.opt_snn.step()

            model.W1.data.clamp_(-msl.weight_bound, msl.weight_bound)
            model.M1.data.fill_diagonal_(0.0)
            model.M1.data.clamp_(0, msl.weight_bound)
            model.V2y.data.clamp_(-msl.weight_bound, msl.weight_bound)
            model.W2.data.clamp_(-msl.weight_bound, msl.weight_bound)
            model.M2.data.fill_diagonal_(0.0)
            model.M2.data.clamp_(0, msl.weight_bound)
            model.V3y.data.clamp_(-msl.weight_bound, msl.weight_bound)

        z1_s = z1[:sb]
        z2_s = z2[:sb]
        tr1_s = tr1[:sb]
        tr2_s = tr2[:sb]
        for w in warmups:
            if t >= w:
                spk1[w].add_(z1_s)
                spk2[w].add_(z2_s)
                tr1_sum[w].add_(tr1_s)
                tr2_sum[w].add_(tr2_s)

        z1_prev = z1
        z2_prev = z2

    out = {
        "tr1_last": tr1[:sb],
        "tr2_last": tr2[:sb],
    }
    for w in warmups:
        out[f"spk1_w{w}"] = spk1[w]
        out[f"spk2_w{w}"] = spk2[w]
        out[f"tr1_sum_w{w}"] = tr1_sum[w]
        out[f"tr2_sum_w{w}"] = tr2_sum[w]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-dir", default=msl.data_dir)
    parser.add_argument("--out-json", default="mnist_linear_sweep_results.json")
    parser.add_argument("--hub-preset", choices=("starter", "tuned"), default="starter")
    args = parser.parse_args()

    if args.hub_preset == "tuned":
        msl.tau_m_hab = 7.254096236354176
        msl.v_thr_hab = 0.2925372944910688
        msl.tau_hub = 2.600392098647199
        msl.hub_power = 1.7025608243682144
        msl.hub_w_init = 0.02389830425680141

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.cuda.manual_seed_all(msl.seed)
    torch.manual_seed(msl.seed)
    np.random.seed(msl.seed)

    tr_ld, te_ld = msl.load_data(args.data_dir, msl.batch_size, msl.dev_batch_size)
    specs = build_specs()
    heads = [
        ProbeHead(s, msl.hid1, msl.hid2, msl.out_dim, args.epochs, device)
        for s in specs
    ]

    # Reset seed right before SNN creation so SNN init matches standalone exactly
    if device.type == "cuda":
        torch.cuda.manual_seed_all(msl.seed)
    torch.manual_seed(msl.seed)
    np.random.seed(msl.seed)
    snn = msl.CSDPSNNLinear(device).to(device)

    print(
        f"Sweep started: {len(heads)} linear classifier heads | "
        f"epochs={args.epochs} | hub_preset={args.hub_preset} | device={device}",
        flush=True,
    )

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tr_correct = [0] * len(heads)
        tr_total = 0

        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.float().to(device)
            B = xb.shape[0]
            tgt = yb.argmax(dim=1)
            yn = msl.neg_labels(yb, device)

            X_cat = torch.cat([xb, xb], dim=0)
            Y_cat = torch.cat([yb, yn], dim=0)
            mask = torch.cat(
                [torch.ones(B, 1, device=device), torch.zeros(B, 1, device=device)],
                dim=0,
            )

            pos_stats = forward_snn_collect(snn, X_cat, y=Y_cat, mask=mask, learn=True, slice_b=B)
            clean_stats = forward_snn_collect(snn, xb, y=None, mask=None, learn=False, slice_b=B)

            for idx, h in enumerate(heads):
                tr_correct[idx] += h.train_batch(pos_stats, clean_stats, tgt)
            tr_total += B

        # Evaluate all heads on test set
        te_correct = [0] * len(heads)
        te_total = 0
        for xb, yb in te_ld:
            xb = xb.to(device)
            tgt = yb.argmax(dim=1).to(device)
            clean_stats = forward_snn_collect(snn, xb, y=None, mask=None, learn=False)
            for idx, h in enumerate(heads):
                te_correct[idx] += h.eval_batch(clean_stats, tgt)
            te_total += xb.shape[0]

        for idx, h in enumerate(heads):
            if h.sched is not None:
                h.sched.step()
            tr_acc = tr_correct[idx] / max(tr_total, 1)
            te_acc = te_correct[idx] / max(te_total, 1)
            h.last_train = tr_acc
            h.last_test = te_acc
            if ep == 1:
                h.ep1_test = te_acc
            if te_acc > h.best_test:
                h.best_test = te_acc
                h.best_epoch = ep
            h.history.append({"epoch": ep, "train": tr_acc, "test": te_acc})

        z2_heads = [h for h in heads if h.spec.feat_source == "z2"]
        top_z2 = max(z2_heads, key=lambda h: h.best_test)
        top_all = max(heads, key=lambda h: h.best_test)
        elapsed_m = (time.time() - t0) / 60.0
        print(
            f"Epoch {ep:2d}/{args.epochs} [{elapsed_m:.1f}m] | "
            f"Best z2: {top_z2.best_test:.4f} (cur {top_z2.last_test:.4f}, {top_z2.spec.name}) | "
            f"Best all: {top_all.best_test:.4f} (cur {top_all.last_test:.4f}, {top_all.spec.name})",
            flush=True,
        )

    # Save full ranked results
    ranked = sorted(heads, key=lambda h: (h.best_test, h.last_test), reverse=True)
    payload = {
        "epochs": args.epochs,
        "hub_preset": args.hub_preset,
        "elapsed_min": (time.time() - t0) / 60.0,
        "num_heads": len(heads),
        "results": [
            {
                "rank": i + 1,
                "best_test": h.best_test,
                "best_epoch": h.best_epoch,
                "ep1_test": h.ep1_test,
                "last_test": h.last_test,
                "last_train": h.last_train,
                "spec": asdict(h.spec),
                "history": h.history,
            }
            for i, h in enumerate(ranked)
        ],
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== TOP 20 STRICT z2 (1000 -> 10) HEADS ===", flush=True)
    z2_ranked = [h for h in ranked if h.spec.feat_source == "z2"]
    for i, h in enumerate(z2_ranked[:20], 1):
        s = h.spec
        print(
            f"{i:2d}. best={h.best_test:.4f} (ep {h.best_epoch:2d}) | ep1={h.ep1_test:.4f} | "
            f"last={h.last_test:.4f} | {s.name} [mode={s.spike_mode}, train={s.cls_train_mode}, "
            f"w={s.warmup_steps}, lr={s.lr_cls:g}, wd={s.wd_cls:g}, drop={s.cls_dropout}, "
            f"sm={s.label_smoothing}, k={s.cls_steps}]",
            flush=True,
        )

    print("\n=== TOP 10 ALL HEADS (INCLUDING z1_z2) ===", flush=True)
    for i, h in enumerate(ranked[:10], 1):
        s = h.spec
        print(
            f"{i:2d}. best={h.best_test:.4f} (ep {h.best_epoch:2d}) | ep1={h.ep1_test:.4f} | "
            f"last={h.last_test:.4f} | {s.name} [feat={s.feat_source}, mode={s.spike_mode}, "
            f"train={s.cls_train_mode}]",
            flush=True,
        )

    # Analytical Ridge-Regression check on final SNN representations (clean y=0)
    print("\n=== ANALYTICAL LINEAR UPPER BOUND (RIDGE ON CLEAN SPIKES) ===", flush=True)
    tr_spk1, tr_spk2, tr_y = [], [], []
    for xb, yb in tr_ld:
        xb = xb.to(device)
        st = forward_snn_collect(snn, xb, y=None, mask=None, learn=False)
        tr_spk1.append(st["spk1_w0"])
        tr_spk2.append(st["spk2_w0"])
        tr_y.append(yb.float().to(device))
    te_spk1, te_spk2, te_y = [], [], []
    for xb, yb in te_ld:
        xb = xb.to(device)
        st = forward_snn_collect(snn, xb, y=None, mask=None, learn=False)
        te_spk1.append(st["spk1_w0"])
        te_spk2.append(st["spk2_w0"])
        te_y.append(yb.float().to(device))

    Xtr1, Xtr2, Ytr = torch.cat(tr_spk1), torch.cat(tr_spk2), torch.cat(tr_y)
    Xte1, Xte2, Yte = torch.cat(te_spk1), torch.cat(te_spk2), torch.cat(te_y)
    ridge_summary = []
    for feat_name, Ftr_raw, Fte_raw in (
        ("z2", Xtr2, Xte2),
        ("z1_z2", torch.cat([Xtr1, Xtr2], dim=1), torch.cat([Xte1, Xte2], dim=1)),
    ):
        for mode in ("count", "l2_count", "zscore", "log1p"):
            Ftr = msl.decode_spikes(Ftr_raw, Ftr_raw, Ftr_raw, msl.T, mode=mode)
            Fte = msl.decode_spikes(Fte_raw, Fte_raw, Fte_raw, msl.T, mode=mode)
            ones_tr = torch.ones(Ftr.shape[0], 1, device=device)
            ones_te = torch.ones(Fte.shape[0], 1, device=device)
            Atr = torch.cat([Ftr, ones_tr], dim=1)
            Ate = torch.cat([Fte, ones_te], dim=1)
            G = Atr.T @ Atr
            rhs = Atr.T @ Ytr
            eye = torch.eye(G.shape[0], device=device)
            eye[-1, -1] = 0.0
            best_r = (0.0, 0.0)
            for lam in (1e-2, 1e-1, 1.0, 10.0, 100.0, 500.0):
                W = torch.linalg.solve(G + lam * eye, rhs)
                acc = float(((Ate @ W).argmax(dim=1) == Yte.argmax(dim=1)).float().mean().item())
                if acc > best_r[0]:
                    best_r = (acc, lam)
            ridge_summary.append({"feat": feat_name, "mode": mode, "acc": best_r[0], "lam": best_r[1]})
            print(f"  Ridge [{feat_name:5s} | {mode:8s}]: test_acc={best_r[0]:.4f} (lambda={best_r[1]:g})", flush=True)

    payload["ridge_final"] = ridge_summary
    Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
