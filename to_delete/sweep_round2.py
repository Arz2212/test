#!/usr/bin/env python3
"""Round-2 focused sweep:
1. Evaluates new variance-stabilized spike encodings (`sqrt`, `l2_sqrt`) alongside
   top encodings from Round 1 (`l2_count`, `zscore`, `log1p`, `count`).
2. Fine-tunes learning rate, cls_steps, weight_decay, dropout for both strict `z2`
   (1000 -> 10) and `z1_z2` (6000 -> 10).
3. Compares `starter` vs `tuned` (Trial 440) hub parameters.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import mnist_solver_linear as msl
from sweep_mnist_linear import HeadSpec, ProbeHead, forward_snn_collect


def build_round2_specs() -> list[HeadSpec]:
    specs: list[HeadSpec] = []

    # Strict z2 (1000 -> 10): fine grid around top encodings + sqrt / l2_sqrt
    for mode in ("l2_count", "l2_sqrt", "sqrt", "zscore", "log1p", "count"):
        lrs = (
            (0.005, 0.01, 0.02, 0.03)
            if mode in ("l2_count", "l2_sqrt")
            else (0.001, 0.002, 0.005, 0.01)
        )
        for lr in lrs:
            for k in (1, 3, 5):
                for wd in (1e-4, 1e-3):
                    specs.append(
                        HeadSpec(
                            name=f"z2_{mode}_lr{lr:g}_k{k}_wd{wd:g}",
                            feat_source="z2",
                            spike_mode=mode,
                            cls_train_mode="clean",
                            lr_cls=lr,
                            wd_cls=wd,
                            cls_steps=k,
                        )
                    )

    # Also test sqrt and l2_sqrt in pos & both modes for completeness
    for mode, lr in (("sqrt", 0.002), ("l2_sqrt", 0.01)):
        for tmode in ("pos", "both"):
            specs.append(
                HeadSpec(
                    name=f"mode_{mode}_{tmode}",
                    feat_source="z2",
                    spike_mode=mode,
                    cls_train_mode=tmode,
                    lr_cls=lr,
                    wd_cls=1e-4,
                )
            )

    # Multi-layer z1_z2 (6000 -> 10): fine grid to approach 97.3% Ridge bound
    for mode, lrs in (
        ("l2_count", (0.005, 0.01, 0.02)),
        ("l2_sqrt", (0.005, 0.01, 0.02)),
        ("zscore", (0.0005, 0.001, 0.002)),
        ("log1p", (0.001, 0.002, 0.005)),
        ("sqrt", (0.001, 0.002, 0.005)),
        ("count", (0.0005, 0.001, 0.002)),
    ):
        for lr in lrs:
            for k in (1, 3):
                for drop in (0.0, 0.15):
                    specs.append(
                        HeadSpec(
                            name=f"z1z2_{mode}_lr{lr:g}_k{k}_d{drop:g}",
                            feat_source="z1_z2",
                            spike_mode=mode,
                            cls_train_mode="clean",
                            lr_cls=lr,
                            wd_cls=1e-3,
                            cls_dropout=drop,
                            cls_steps=k,
                        )
                    )

    return specs


def run_preset(preset: str, epochs: int, device: torch.device, data_dir: str) -> dict:
    if preset == "starter":
        msl.tau_m_hab = 4.87563
        msl.v_thr_hab = 0.249441
        msl.tau_hub = 4.39997
        msl.hub_power = 1.20313
        msl.hub_w_init = 0.0503053
    elif preset == "tuned":
        msl.tau_m_hab = 7.254096236354176
        msl.v_thr_hab = 0.2925372944910688
        msl.tau_hub = 2.600392098647199
        msl.hub_power = 1.7025608243682144
        msl.hub_w_init = 0.02389830425680141

    torch.cuda.manual_seed_all(msl.seed)
    torch.manual_seed(msl.seed)
    np.random.seed(msl.seed)
    tr_ld, te_ld = msl.load_data(data_dir, msl.batch_size, msl.dev_batch_size)

    specs = build_round2_specs()
    heads = [ProbeHead(s, msl.hid1, msl.hid2, msl.out_dim, epochs, device) for s in specs]

    torch.cuda.manual_seed_all(msl.seed)
    torch.manual_seed(msl.seed)
    np.random.seed(msl.seed)
    snn = msl.CSDPSNNLinear(device).to(device)

    print(f"\n=== ROUND 2 SWEEP: preset={preset}, epochs={epochs}, heads={len(heads)} ===", flush=True)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr_correct = [0] * len(heads)
        tr_total = 0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.float().to(device)
            B = xb.shape[0]
            tgt = yb.argmax(dim=1)
            yn = msl.neg_labels(yb, device)
            X_cat = torch.cat([xb, xb], dim=0)
            Y_cat = torch.cat([yb, yn], dim=0)
            mask = torch.cat([torch.ones(B, 1, device=device), torch.zeros(B, 1, device=device)], dim=0)

            pos_stats = forward_snn_collect(snn, X_cat, y=Y_cat, mask=mask, learn=True, slice_b=B)
            clean_stats = forward_snn_collect(snn, xb, y=None, mask=None, learn=False, slice_b=B)
            for idx, h in enumerate(heads):
                tr_correct[idx] += h.train_batch(pos_stats, clean_stats, tgt)
            tr_total += B

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

        z2_top = max([h for h in heads if h.spec.feat_source == "z2"], key=lambda h: h.best_test)
        all_top = max(heads, key=lambda h: h.best_test)
        print(
            f"[{preset}] Ep {ep:2d}/{epochs} [{(time.time()-t0)/60:.1f}m] | "
            f"z2 best={z2_top.best_test:.4f} (cur {z2_top.last_test:.4f}, {z2_top.spec.name}) | "
            f"z1z2 best={all_top.best_test:.4f} (cur {all_top.last_test:.4f}, {all_top.spec.name})",
            flush=True,
        )

    ranked = sorted(heads, key=lambda h: (h.best_test, h.last_test), reverse=True)
    z2_ranked = [h for h in ranked if h.spec.feat_source == "z2"]
    z1z2_ranked = [h for h in ranked if h.spec.feat_source == "z1_z2"]
    print(f"\n--- [{preset}] TOP 10 STRICT z2 (1000 -> 10) ---", flush=True)
    for i, h in enumerate(z2_ranked[:10], 1):
        print(f" {i:2d}. best={h.best_test:.4f} (ep {h.best_epoch}) | ep1={h.ep1_test:.4f} | last={h.last_test:.4f} | {h.spec.name}", flush=True)
    print(f"\n--- [{preset}] TOP 10 z1_z2 (6000 -> 10) ---", flush=True)
    for i, h in enumerate(z1z2_ranked[:10], 1):
        print(f" {i:2d}. best={h.best_test:.4f} (ep {h.best_epoch}) | ep1={h.ep1_test:.4f} | last={h.last_test:.4f} | {h.spec.name}", flush=True)

    return {
        "preset": preset,
        "epochs": epochs,
        "top_z2": [{"best_test": h.best_test, "best_epoch": h.best_epoch, "ep1_test": h.ep1_test, "last_test": h.last_test, "spec": asdict(h.spec)} for h in z2_ranked[:25]],
        "top_z1z2": [{"best_test": h.best_test, "best_epoch": h.best_epoch, "ep1_test": h.ep1_test, "last_test": h.last_test, "spec": asdict(h.spec)} for h in z1z2_ranked[:25]],
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    out = {
        "starter": run_preset("starter", epochs=12, device=device, data_dir=msl.data_dir),
        "tuned": run_preset("tuned", epochs=6, device=device, data_dir=msl.data_dir),
    }
    Path("sweep_round2.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
