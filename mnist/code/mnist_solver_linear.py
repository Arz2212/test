#!/usr/bin/env python3
"""MNIST CSDP-SNN solver with an ANN Linear Classifier output layer.

Replaces the spiking output layer (C3 + lif_out + local delta-trace rule)
from mnist_solver.py with a standard linear classifier (torch.nn.Linear)
operating on continuous signals decoded from the hidden spiking layer(s).
"""

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

# ───────────────── spike-to-signal & linear classifier params ────────────────
spike_mode = "l2_sqrt"      # count | rate | l2_count | sqrt | l2_sqrt | zscore | log1p | trace | mean_trace
feat_source = "z2"          # z2 (hid2 -> 10, strict last-layer replacement) | z1_z2 (hid1+hid2 -> 10)
cls_train_mode = "clean"    # clean (y=0, matches test) | pos (single-pass y=yb) | both
warmup_steps = 0            # skip initial transient steps [0 .. warmup_steps-1]
lr_cls = 0.02               # learning rate for nn.Linear optimizer
wd_cls = 1e-4               # weight decay for nn.Linear optimizer
cls_dropout = 0.0           # dropout on decoded spike signal before nn.Linear
label_smoothing = 0.0       # cross-entropy label smoothing
cls_bias = True             # whether nn.Linear uses bias
cls_steps = 5               # gradient steps on linear head per batch
cosine_lr = True            # cosine annealing schedule for linear head

data_dir = str(Path(__file__).resolve().parents[2] / "data" / "mnist")

SPIKE_MODES = (
    "count",
    "rate",
    "l2_count",
    "sqrt",
    "l2_sqrt",
    "zscore",
    "log1p",
    "trace",
    "mean_trace",
)


def nn_param(*shape):
    p = torch.nn.Parameter(torch.empty(*shape))
    torch.nn.init.uniform_(p, -1, 1)
    return p


def decode_spikes(spk_cnt, tr_last, tr_sum, eff_steps, mode="count"):
    """Convert accumulated spike statistics into a continuous ANN vector."""
    if mode == "count":
        # Raw spike count: "количество спайков подавать в классификатор"
        return spk_cnt
    if mode == "rate":
        # Mean firing rate in [0, 1] over effective simulation window
        return spk_cnt / float(max(eff_steps, 1))
    if mode == "l2_count":
        # Unit L2-norm spike-count direction (invariant to global firing-rate shift)
        return F.normalize(spk_cnt, p=2, dim=1, eps=1e-6)
    if mode == "sqrt":
        # Anscombe-style square-root variance stabilization for Poisson spike counts
        return torch.sqrt(spk_cnt + 1e-8)
    if mode == "l2_sqrt":
        # Hellinger / square-root L2-normalized spike distribution
        return F.normalize(torch.sqrt(spk_cnt + 1e-8), p=2, dim=1, eps=1e-6)
    if mode == "zscore":
        # Per-sample standardized spike counts
        mu = spk_cnt.mean(dim=1, keepdim=True)
        std = spk_cnt.std(dim=1, keepdim=True, unbiased=False)
        return (spk_cnt - mu) / (std + 1e-5)
    if mode == "log1p":
        # Sublinear (Weber-Fechner) log spike count
        return torch.log1p(spk_cnt)
    if mode == "trace":
        # Final low-pass filtered synaptic trace at t = T
        return tr_last
    if mode == "mean_trace":
        # Time-averaged synaptic trace over effective window
        return tr_sum / float(max(eff_steps, 1))
    raise ValueError(f"Unknown spike_mode: {mode}")


class CSDPSNNLinear(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        tau_sj = tau_m / dt

        # Spiking hidden layer 1 (CSDP)
        self.W1 = nn_param(in_dim, hid1)
        self.M1 = nn_param(hid1, hid1)
        self.V2y = nn_param(out_dim, hid1)

        # Spiking hidden layer 2 (CSDP)
        self.W2 = nn_param(hid1, hid2)
        self.M2 = nn_param(hid2, hid2)
        self.V3y = nn_param(out_dim, hid2)

        # ANN Linear Classifier replacing spiking C3 + lif_out
        cls_in_dim = hid2 if feat_source == "z2" else (hid1 + hid2)
        self.classifier = nn.Linear(cls_in_dim, out_dim, bias=cls_bias)
        self.dropout = nn.Dropout(p=cls_dropout) if cls_dropout > 0.0 else nn.Identity()

        # Spiking neurons for hidden layers only (no lif_out!)
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

        # Local CSDP optimizer for spiking hidden layers
        self.opt_snn = torch.optim.Adam(
            [self.W1, self.M1, self.V2y, self.W2, self.M2, self.V3y],
            lr=eta,
            fused=(device.type == "cuda"),
        )
        # Standard gradient-based optimizer for the ANN linear classifier
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
        """Simulate the spiking hidden layers for T steps and decode continuous features."""
        B = x.shape[0]
        dv = self.device
        functional.reset_net(self.lif1)
        functional.reset_net(self.lif2)
        if learn:
            functional.reset_net(self.hub_lif1)
            functional.reset_net(self.hub_lif2)

        need_traces = learn or (spike_mode in ("trace", "mean_trace"))
        need_z1_feat = feat_source == "z1_z2"

        if need_traces:
            tr1 = torch.zeros(B, hid1, device=dv)
            tr2 = torch.zeros(B, hid2, device=dv)
        else:
            tr1 = tr2 = None

        spk_cnt2 = torch.zeros(B, hid2, device=dv)
        spk_cnt1 = torch.zeros(B, hid1, device=dv) if need_z1_feat else None
        tr2_sum = (
            torch.zeros(B, hid2, device=dv) if spike_mode == "mean_trace" else None
        )
        tr1_sum = (
            torch.zeros(B, hid1, device=dv)
            if (spike_mode == "mean_trace" and need_z1_feat)
            else None
        )

        z1_prev = torch.zeros(B, hid1, device=dv)
        z2_prev = torch.zeros(B, hid2, device=dv)
        if learn:
            hub1 = torch.zeros(B, 1, device=dv)
            hub2 = torch.zeros(B, 1, device=dv)

        d = dt / tau_tr
        decay_hub = 1.0 / tau_hub
        has_y = y is not None

        for t in range(T):
            z0 = (torch.rand(B, in_dim, device=dv) < x).float()

            # Layer 1
            I1 = (z0 @ self.W1) * R_m - (z1_prev @ self.M1) * inh_R
            if has_y:
                I1 = I1 + (y @ self.V2y) * R_m
            z1 = self.lif1(I1)
            if need_traces:
                tr1 = tr1 + d * (z1 - tr1)

            if learn:
                h1_spk = self.hub_lif1(self.hub_fc1(z1))
                hub1 = hub1 + decay_hub * (-hub1 + h1_spk.float())
                mod1 = self.hub_mod(tr1, hub1, mask, self.zmax1)

            # Layer 2
            I2 = (z1 @ self.W2) * R_m - (z2_prev @ self.M2) * inh_R
            if has_y:
                I2 = I2 + (y @ self.V3y) * R_m
            z2 = self.lif2(I2)
            if need_traces:
                tr2 = tr2 + d * (z2 - tr2)

            if learn:
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

            if t >= warmup_steps:
                spk_cnt2 = spk_cnt2 + z2
                if need_z1_feat:
                    spk_cnt1 = spk_cnt1 + z1
                if tr2_sum is not None:
                    tr2_sum = tr2_sum + tr2
                if tr1_sum is not None:
                    tr1_sum = tr1_sum + tr1

            z1_prev = z1
            z2_prev = z2

        eff_steps = max(T - warmup_steps, 1)
        feat2 = decode_spikes(spk_cnt2, tr2, tr2_sum, eff_steps, mode=spike_mode)
        if need_z1_feat:
            feat1 = decode_spikes(spk_cnt1, tr1, tr1_sum, eff_steps, mode=spike_mode)
            return torch.cat([feat1, feat2], dim=1)
        return feat2

    def classify(self, feat, apply_dropout=False):
        if apply_dropout and cls_dropout > 0.0:
            feat = self.dropout(feat)
        return self.classifier(feat)

    def step_classifier(self, feat, target_idx):
        """Perform gradient descent step(s) on the linear classifier head."""
        with torch.enable_grad():
            logits = None
            loss_val = 0.0
            for _ in range(max(cls_steps, 1)):
                logits = self.classify(feat, apply_dropout=True)
                loss = F.cross_entropy(
                    logits, target_idx, label_smoothing=label_smoothing
                )
                self.opt_cls.zero_grad(set_to_none=True)
                loss.backward()
                self.opt_cls.step()
                loss_val = loss.item()
        return logits.detach(), loss_val

    def forward(self, x, y=None, mask=None, learn=False):
        feat = self.forward_snn(x, y=y, mask=mask, learn=learn)
        with torch.no_grad():
            logits = self.classify(feat, apply_dropout=False)
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
        logits = model.classify(feat, apply_dropout=False)
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

        # 1) Local CSDP update on spiking hidden layers
        feat_cat = model.forward_snn(X_cat, y=Y_cat, mask=mask, learn=True)

        # 2) Train ANN linear classifier on decoded spike signal
        if cls_train_mode == "pos":
            feat_cls = feat_cat[:B]
            logits, loss_val = model.step_classifier(feat_cls, target_idx)
            preds = logits.argmax(dim=1)
        elif cls_train_mode == "clean":
            # Extract unclamped (y=0) spike signal matching test-time inference
            feat_clean = model.forward_snn(xb, y=None, mask=None, learn=False)
            logits, loss_val = model.step_classifier(feat_clean, target_idx)
            preds = logits.argmax(dim=1)
        elif cls_train_mode == "both":
            feat_clean = model.forward_snn(xb, y=None, mask=None, learn=False)
            feat_both = torch.cat([feat_cat[:B], feat_clean], dim=0)
            target_both = torch.cat([target_idx, target_idx], dim=0)
            logits_both, loss_val = model.step_classifier(feat_both, target_both)
            preds = logits_both[B:].argmax(dim=1)
        else:
            raise ValueError(f"Unknown cls_train_mode: {cls_train_mode}")

        batch_correct = (preds == target_idx).sum().item()
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
    model = CSDPSNNLinear(device).to(device)

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            model.opt_cls, T_max=max(epochs, 1), eta_min=lr_cls * 0.05
        )
        if cosine_lr and epochs > 1
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
    global tau_m_hab, v_thr_hab, tau_hub, hub_power, hub_w_init
    global spike_mode, feat_source, cls_train_mode, warmup_steps
    global lr_cls, wd_cls, cls_dropout, label_smoothing, cls_bias, cls_steps, cosine_lr
    global hid1, hid2

    parser = argparse.ArgumentParser(
        description="MNIST CSDP-SNN solver with ANN Linear Classifier head"
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
    parser.add_argument("--hid1", type=int, default=hid1)
    parser.add_argument("--hid2", type=int, default=hid2)

    # Spike-to-signal & linear classifier options
    parser.add_argument("--spike-mode", choices=SPIKE_MODES, default=spike_mode)
    parser.add_argument(
        "--feat-source", choices=("z2", "z1_z2"), default=feat_source
    )
    parser.add_argument(
        "--cls-train-mode", choices=("clean", "pos", "both"), default=cls_train_mode
    )
    parser.add_argument("--warmup-steps", type=int, default=warmup_steps)
    parser.add_argument("--lr-cls", type=float, default=lr_cls)
    parser.add_argument("--wd-cls", type=float, default=wd_cls)
    parser.add_argument("--cls-dropout", type=float, default=cls_dropout)
    parser.add_argument("--label-smoothing", type=float, default=label_smoothing)
    parser.add_argument("--no-cls-bias", action="store_true")
    parser.add_argument("--cls-steps", type=int, default=cls_steps)
    parser.add_argument("--no-cosine-lr", action="store_true")

    # Hub neuron overrides
    parser.add_argument("--tau-m-hab", type=float, default=None)
    parser.add_argument("--v-thr-hab", type=float, default=None)
    parser.add_argument("--tau-hub", type=float, default=None)
    parser.add_argument("--hub-power", type=float, default=None)
    parser.add_argument("--hub-w-init", type=float, default=None)
    args = parser.parse_args()

    hid1 = args.hid1
    hid2 = args.hid2
    spike_mode = args.spike_mode
    feat_source = args.feat_source
    cls_train_mode = args.cls_train_mode
    warmup_steps = args.warmup_steps
    lr_cls = args.lr_cls
    wd_cls = args.wd_cls
    cls_dropout = args.cls_dropout
    label_smoothing = args.label_smoothing
    cls_bias = not args.no_cls_bias
    cls_steps = args.cls_steps
    cosine_lr = not args.no_cosine_lr

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
        f"MNIST CSDP-SNN + Linear Classifier | hid1={hid1}, hid2={hid2}, "
        f"epochs={args.epochs}, device={args.device}\n"
        f"Classifier: feat={feat_source}, spike_mode={spike_mode}, "
        f"train_mode={cls_train_mode}, warmup={warmup_steps}, lr_cls={lr_cls}, "
        f"wd_cls={wd_cls}, dropout={cls_dropout}, smooth={label_smoothing}, "
        f"steps={cls_steps}, bias={cls_bias}, cosine={cosine_lr}\n"
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
    )


if __name__ == "__main__":
    main()
