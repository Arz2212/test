#!/usr/bin/env python3
"""
CSDP-SNN (SpikingJelly) — гиперпараметрический поиск через Ray Tune + HyperOpt (TPE).

Перебираются три параметра хабитуации/хаба:
    tau_m_hab   (база 1.8)
    v_thr_hab   (база 0.07)
    tau_hub     (база 3.0)

Оценка одной конфигурации (objective):
    1 эпоха обучения = 100 батчей (trainX = 50000, batch_size = 500),
    затем accuracy на MNIST test-сете (10000 примеров).
    Эта accuracy и есть метрика для HyperOpt (max).

CPU-only: каждый trial занимает 1 ядро (torch.set_num_threads(1)),
Ray параллелит trials по ядрам.

Запуск:
    python main.py                     # 1000 семплов
    python main.py --samples 200       # своё число семплов
    RAY_NUM_CPUS=16 python main.py     # ограничить число ядер
"""

import os
import math
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from spikingjelly.activation_based import neuron, layer, functional

import ray
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp

# ───────────────────────────── фиксированные гиперпараметры ─────────────────
# Опциональные env-переопределения для быстрого smoke-теста:
#   CSDP_T, CSDP_HID1, CSDP_HID2, CSDP_BATCH
def _env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v else default

T            = _env_int("CSDP_T", 50)
dt           = 3.0
tau_m        = 100.0
v_thr        = 0.055
tau_tr       = 13.0
R_m          = 0.1
inh_R        = 0.01
eta          = 0.002
weight_bound = 1.0
SEED         = 5464

in_dim  = 784
out_dim = 10
hid1    = _env_int("CSDP_HID1", 1024)
hid2    = _env_int("CSDP_HID2", 128)

batch_size     = _env_int("CSDP_BATCH", 500)
dev_batch_size = 1000

# 1 эпоха = 100 батчей (trainX = 50000 / batch_size = 500)
TRAIN_EPOCHS = 1

data_dir = "/home/arz/SNN/jax_version/cSNN/data/mnist/mnist"

# Базовые значения (точка, вокруг которой идёт поиск)
DEFAULT_TAU_M_HAB = 1.8
DEFAULT_V_THR_HAB = 0.07
DEFAULT_TAU_HUB   = 3.0

if not torch.cuda.is_available():
    # 1 поток на trial — иначе torch сам захватит все ядра и Ray не сможет параллелить
    torch.set_num_threads(1)

# Пространство поиска (диапазоны включают базовые значения; легко поменять)
search_space = {
    "tau_m_hab": hp.uniform("tau_m_hab", 0.5, 5.0),
    "v_thr_hab": hp.loguniform("v_thr_hab", math.log(0.01), math.log(0.30)),
    "tau_hub":   hp.uniform("tau_hub", 1.0, 8.0),
}


# CSDP-SNN (4 слоя)
class CSDPSNN(torch.nn.Module):
    def __init__(self, device, tau_m_hab=DEFAULT_TAU_M_HAB,
                 v_thr_hab=DEFAULT_V_THR_HAB, tau_hub=DEFAULT_TAU_HUB):
        super().__init__()
        self.device = device
        self.tau_hub = tau_hub
        tau_sj = tau_m / dt

        self.W1  = nn_param(in_dim, hid1)
        self.M1  = nn_param(hid1,   hid1)
        self.V2y = nn_param(out_dim, hid1)
        self.m1_mask = (1.0 - torch.eye(hid1, device=device))

        self.W2  = nn_param(hid1,   hid2)
        self.M2  = nn_param(hid2,   hid2)
        self.V3y = nn_param(out_dim, hid2)
        self.m2_mask = (1.0 - torch.eye(hid2, device=device))

        self.C3  = nn_param(hid2,   out_dim)

        self.lif1 = neuron.LIFNode(tau=tau_sj, v_threshold=v_thr, v_reset=0.0).to(device)
        self.lif2 = neuron.LIFNode(tau=tau_sj, v_threshold=v_thr, v_reset=0.0).to(device)
        self.lif_out = neuron.LIFNode(tau=tau_sj, v_threshold=v_thr, v_reset=0.0).to(device)

        self.hub_fc1 = layer.Linear(hid1, 1, bias=False).to(device)
        self.hub_fc2 = layer.Linear(hid2, 1, bias=False).to(device)
        self.hub_lif1 = neuron.LIFNode(tau=tau_m_hab, v_threshold=v_thr_hab, v_reset=0.0).to(device)
        self.hub_lif2 = neuron.LIFNode(tau=tau_m_hab, v_threshold=v_thr_hab, v_reset=0.0).to(device)

        torch.nn.init.constant_(self.hub_fc1.weight, 0.2)
        torch.nn.init.constant_(self.hub_fc2.weight, 0.2)

        self.hub_fc1.weight.requires_grad_(False)
        self.hub_fc2.weight.requires_grad_(False)
        self.register_buffer('zmax1', torch.tensor(0.0))
        self.register_buffer('zmax2', torch.tensor(0.0))

        self.opt_W1  = torch.optim.Adam([self.W1],  lr=eta)
        self.opt_M1  = torch.optim.Adam([self.M1],  lr=eta)
        self.opt_V2y = torch.optim.Adam([self.V2y], lr=eta)
        self.opt_W2  = torch.optim.Adam([self.W2],  lr=eta)
        self.opt_M2  = torch.optim.Adam([self.M2],  lr=eta)
        self.opt_V3y = torch.optim.Adam([self.V3y], lr=eta)
        self.opt_C3  = torch.optim.Adam([self.C3],  lr=eta)

    def hub_mod(self, z, hub, phase, zmax_buf):
        cur_max = hub.max().item()
        if cur_max > zmax_buf.item():
            zmax_buf.fill_(cur_max)
        factor = (hub / zmax_buf) ** 2 - phase
        return factor * z

    def forward(self, x, y, mask, learn=True):
        B = x.shape[0]
        dv = self.device
        functional.reset_net(self.lif1)
        functional.reset_net(self.lif2)
        functional.reset_net(self.lif_out)
        functional.reset_net(self.hub_lif1)
        functional.reset_net(self.hub_lif2)

        tr1 = torch.zeros(B, hid1, device=dv)
        tr2 = torch.zeros(B, hid2, device=dv)
        tr_out = torch.zeros(B, out_dim, device=dv)
        spk_cnt = torch.zeros(B, out_dim, device=dv)
        z1_prev = torch.zeros(B, hid1, device=dv)
        z2_prev = torch.zeros(B, hid2, device=dv)
        hub1 = torch.zeros(B, 1, device=dv)
        hub2 = torch.zeros(B, 1, device=dv)
        d = dt / tau_tr
        decay_hub = 1.0 / self.tau_hub

        for _ in range(T):
            z0 = (torch.rand(B, in_dim, device=dv) < x).float()

            # слой 2
            I1 = (z0 @ self.W1) * R_m - (z1_prev @ self.M1) * inh_R + (y @ self.V2y) * R_m
            z1 = self.lif1(I1)
            tr1 = tr1 + d * (z1 - tr1)

            h1_spk = self.hub_lif1(self.hub_fc1(z1))
            hub1 = hub1 + decay_hub * (-hub1 + h1_spk.float())
            mod1 = self.hub_mod(tr1, hub1, mask, self.zmax1)

            # слой 3
            I2 = (z1 @ self.W2) * R_m - (z2_prev @ self.M2) * inh_R + (y @ self.V3y) * R_m
            z2 = self.lif2(I2)
            tr2 = tr2 + d * (z2 - tr2)

            h2_spk = self.hub_lif2(self.hub_fc2(z2))
            hub2 = hub2 + decay_hub * (-hub2 + h2_spk.float())
            mod2 = self.hub_mod(tr2, hub2, mask, self.zmax2)

            # выходной слой
            zy = self.lif_out(z2 @ self.C3)
            tr_out = tr_out + d * (zy - tr_out)
            spk_cnt = spk_cnt + zy
            dmu = 2.0 * (y - tr_out) * mask

            if learn:
                csdp_step(self.opt_W1,  self.W1,  z0.T      @ (mod1 * R_m),   (-weight_bound, weight_bound))
                csdp_step(self.opt_M1,  self.M1,  z1_prev.T @ (mod1 * inh_R), (0, weight_bound), self.m1_mask)
                csdp_step(self.opt_V2y, self.V2y, y.T       @ (mod1 * R_m),   (-weight_bound, weight_bound))

                csdp_step(self.opt_W2,  self.W2,  z1.T      @ (mod2 * R_m),   (-weight_bound, weight_bound))
                csdp_step(self.opt_M2,  self.M2,  z2_prev.T @ (mod2 * inh_R), (0, weight_bound), self.m2_mask)
                csdp_step(self.opt_V3y, self.V3y, y.T       @ (mod2 * R_m),   (-weight_bound, weight_bound))

                hebb_step(self.opt_C3, self.C3, z2.T @ dmu, (-weight_bound, weight_bound))

            z1_prev = z1
            z2_prev = z2

        return F.softmax(spk_cnt / T, dim=1), spk_cnt, tr_out


def nn_param(*shape):
    p = torch.nn.Parameter(torch.empty(*shape))
    torch.nn.init.uniform_(p, -1, 1)
    return p


def csdp_step(opt, param, grad, bounds, mask=None):
    opt.zero_grad()
    param.grad = grad
    opt.step()
    if mask is not None:
        param.data.mul_(mask)
    param.data.clamp_(*bounds)


def hebb_step(opt, param, grad, bounds):
    opt.zero_grad()
    param.grad = -grad
    opt.step()
    param.data.clamp_(*bounds)


def neg_labels(y_onehot, device):
    B = y_onehot.shape[0]
    correct = y_onehot.argmax(dim=1)
    neg = torch.randint(0, out_dim, (B,), device=device)
    neg[neg == correct] = (neg[neg == correct] + 1) % out_dim
    return F.one_hot(neg, out_dim).float()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    c = t = 0
    for xb, yb in loader:
        xb = xb.to(device)
        B = xb.shape[0]
        yh, _, _ = model(xb, torch.zeros(B, out_dim, device=device),
                         torch.ones(B, 1, device=device), learn=False)
        c += (yh.argmax(dim=1) == yb.argmax(dim=1).to(device)).sum().item()
        t += B
    model.train()
    return c / t


def train_epoch(model, loader, device, verbose=False):
    model.train()
    c = t = 0
    n_batches = len(loader)
    for i, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.float().to(device)
        B = xb.shape[0]
        yn = neg_labels(yb, device)

        X_cat = torch.cat([xb, xb], dim=0)
        Y_cat = torch.cat([yb, yn], dim=0)
        m = torch.cat([torch.ones(B, 1, device=device),
                       torch.zeros(B, 1, device=device)], dim=0)

        yh, _, _ = model(X_cat, Y_cat, m, learn=True)
        batch_correct = (yh[:B].argmax(dim=1) == yb.argmax(dim=1).to(device)).sum().item()
        c += batch_correct
        t += B
        if verbose:
            print(f"  Batch {i+1:3d}/{n_batches}  acc: {batch_correct/B:.4f}  "
                  f"avg: {c/t:.4f}", end="\r", flush=True)
    if verbose:
        print()
    return c / t


def objective(config, data_refs):
    """Один trial: 1 эпоха обучения → accuracy на тесте."""
    tau_m_hab = config["tau_m_hab"]
    v_thr_hab = config["v_thr_hab"]
    tau_hub   = config["tau_hub"]

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # .copy() — ray.get отдаёт read-only массив из shared memory;
    # копия делается writable, чтобы torch не ругался (и не было UB при записи).
    X_tr = torch.from_numpy(ray.get(data_refs["X_tr"]).copy())
    Y_tr = torch.from_numpy(ray.get(data_refs["Y_tr"]).copy())
    X_te = torch.from_numpy(ray.get(data_refs["X_te"]).copy())
    Y_te = torch.from_numpy(ray.get(data_refs["Y_te"]).copy())

    device = torch.device("cpu")

    tr_ld = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=batch_size,
                       shuffle=True, drop_last=True)
    te_ld = DataLoader(TensorDataset(X_te, Y_te), batch_size=dev_batch_size,
                       shuffle=False, drop_last=True)

    model = CSDPSNN(device, tau_m_hab, v_thr_hab, tau_hub).to(device)

    for _ in range(TRAIN_EPOCHS):
        train_epoch(model, tr_ld, device, verbose=False)

    acc = evaluate(model, te_ld, device)

    tune.report({"accuracy": acc})
    return {"accuracy": acc}


def main():
    p = argparse.ArgumentParser(description="CSDP-SNN hyperopt search (Ray Tune + HyperOpt, CPU)")
    p.add_argument("--samples", type=int, default=1000, help="число семплов (trials)")
    p.add_argument("--max-concurrent", type=int, default=None,
                   help="максимум параллельных trials (по умолчанию N_CPUS-1)")
    p.add_argument("--initial-points", type=int, default=100,
                   help="число случайных стартовых точек для TPE")
    args = p.parse_args()

    N_CPUS = int(os.environ.get("RAY_NUM_CPUS", os.cpu_count() or 1))
    max_concurrent = args.max_concurrent or max(1, N_CPUS - 1)

    ray.init(num_cpus=N_CPUS, num_gpus=0, ignore_reinit_error=True)

    # Данные грузим один раз и кладём в object store (shared memory, zero-copy для воркеров)
    X_tr = np.load(f"{data_dir}/trainX.npy").astype(np.float32)
    Y_tr = np.load(f"{data_dir}/trainY.npy").astype(np.float32)
    X_te = np.load(f"{data_dir}/testX.npy").astype(np.float32)
    Y_te = np.load(f"{data_dir}/testY.npy").astype(np.float32)

    data_refs = {
        "X_tr": ray.put(X_tr),
        "Y_tr": ray.put(Y_tr),
        "X_te": ray.put(X_te),
        "Y_te": ray.put(Y_te),
    }

    algo = HyperOptSearch(
        search_space,
        metric="accuracy",
        mode="max",
        n_initial_points=args.initial_points,
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(objective, data_refs=data_refs),
            {"cpu": 1},
        ),
        tune_config=tune.TuneConfig(
            search_alg=algo,
            num_samples=args.samples,
            max_concurrent_trials=max_concurrent,
        ),
        run_config=tune.RunConfig(
            name="mnist_csdp_hyperopt",
            storage_path=os.path.abspath("./ray_results"),
            verbose=1,
        ),
    )

    t0 = time.time()
    results = tuner.fit()
    elapsed = time.time() - t0

    best = results.get_best_result(metric="accuracy", mode="max")
    print("\n" + "=" * 60)
    print(f"Поиск завершён за {elapsed/60:.1f} мин")
    print(f"Всего trials: {args.samples}, ядер: {N_CPUS}, concurrent: {max_concurrent}")
    print("Лучшая конфигурация:")
    for k, v in best.config.items():
        print(f"  {k:12s} = {v:.6g}")
    print(f"Лучшая test-accuracy (1 эпоха): {best.metrics['accuracy']:.4f}")
    print("=" * 60)

    ray.shutdown()


if __name__ == "__main__":
    main()
