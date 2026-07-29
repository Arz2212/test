#!/usr/bin/env python3
"""
XOR на аналоговой схеме с per-нейронными конденсаторами.

Архитектура:  2(in) → 5(h1) → 3(h2) → 2(out)
Кодирование:  1 → rate=1.0,  0 → rate=0.1

Фазы обучения:
  good: правильная метка → все нейроны заряжают C_good → LTP активных синапсов
  bad:  неправильная метка → все нейроны заряжают C_bad → LTD активных синапсов

Контраст создаётся за счёт контекстного входа (label→layer), который
меняет паттерн активности между фазами.
"""

import torch
import torch.nn.functional as F
import numpy as np

torch.manual_seed(42); np.random.seed(42)

# ── Гиперпараметры ──────────────────────────────────────────────
T         = 100          # временных шагов на фазу
dt        = 1.0
tau_m     = 5.0          # постоянная мембраны
tau_tr    = 8.0          # постоянная следа (z)
tau_cap   = 8.0          # постоянная конденсатора
v_thr     = 0.5          # порог спайка
g_cap     = 0.3          # сила конденсатора на мембрану
alpha_w   = 0.005        # скорость обучения feedforward весов
alpha_v   = 0.01         # скорость обучения контекстных весов
w_clip    = 3.0
rate_1    = 1.0
rate_0    = 0.15
ctx_scale = 8.0          # СИЛЬНЫЙ контекст
N_EPOCHS  = 500
BATCH     = 64

N_IN   = 2
N_H1   = 5
N_H2   = 3
N_OUT  = 2


class CapacitorLIF:
    """LIF-нейрон с двумя конденсаторами. Все состояния [B, N]."""
    def __init__(self, n):
        self.n = n
        self.v = self.c_good = self.c_bad = self.z = None

    def _init(self, B):
        self.v       = torch.zeros(B, self.n)
        self.c_good  = torch.zeros(B, self.n)
        self.c_bad   = torch.zeros(B, self.n)
        self.z       = torch.zeros(B, self.n)

    def step(self, current, phase):
        B = current.shape[0]
        if self.v is None: self._init(B)

        # ── модуляция мембраны конденсатором ──
        if phase == 'good':
            cap_mod = +g_cap * self.c_good
        elif phase == 'bad':
            cap_mod = -g_cap * self.c_bad
        else:
            cap_mod = torch.zeros(B, self.n)

        # ── мембрана ──
        d_m = dt / tau_m
        self.v = self.v + d_m * (-self.v + current + cap_mod)

        # ── спайк ──
        spike = (self.v >= v_thr).float()
        self.v[spike > 0] = 0.0

        # ── след ──
        d_tr = dt / tau_tr
        self.z = self.z + d_tr * (-self.z + spike)

        # ── заряд конденсатора (без гейта: все нейроны) ──
        d_cap = dt / tau_cap
        if phase == 'good':
            self.c_good = self.c_good + d_cap * (-self.c_good + spike)
        elif phase == 'bad':
            self.c_bad  = self.c_bad  + d_cap * (-self.c_bad  + spike)

        return spike

    def reset(self):
        self.v = self.c_good = self.c_bad = self.z = None


class CapacitorSNN:
    def __init__(self):
        self.W1 = torch.nn.Parameter(torch.empty(N_IN,  N_H1))
        self.W2 = torch.nn.Parameter(torch.empty(N_H1,  N_H2))
        self.W3 = torch.nn.Parameter(torch.empty(N_H2,  N_OUT))
        self.V1 = torch.nn.Parameter(torch.empty(N_OUT, N_H1))
        self.V2 = torch.nn.Parameter(torch.empty(N_OUT, N_H2))
        for w in [self.W1, self.W2, self.W3, self.V1, self.V2]:
            torch.nn.init.uniform_(w, -1.0, 1.0)

        self.lif1 = CapacitorLIF(N_H1)
        self.lif2 = CapacitorLIF(N_H2)
        self.lif3 = CapacitorLIF(N_OUT)

    def forward(self, x_rates, y_onehot, phase):
        """Один проход фазы. y_onehot используется только для ctx."""
        B = x_rates.shape[0]
        ctx1 = (y_onehot @ self.V1) * ctx_scale
        ctx2 = (y_onehot @ self.V2) * ctx_scale

        for lif in [self.lif1, self.lif2, self.lif3]:
            lif.reset()

        sc3 = torch.zeros(B, N_OUT)
        z1_list, z2_list, z3_list = [], [], []
        s0_list, s1_list = [], []

        for _ in range(T):
            s0 = (torch.rand(B, N_IN) < x_rates).float()
            s1 = self.lif1.step(s0 @ self.W1 + ctx1, phase)
            s2 = self.lif2.step(s1 @ self.W2 + ctx2, phase)
            s3 = self.lif3.step(s2 @ self.W3, phase)
            sc3 += s3
            z1_list.append(self.lif1.z.clone())
            z2_list.append(self.lif2.z.clone())
            z3_list.append(self.lif3.z.clone())
            s0_list.append(s0)
            s1_list.append(s1)

        return {
            'z1': torch.stack(z1_list).mean(0),
            'z2': torch.stack(z2_list).mean(0),
            'z3': torch.stack(z3_list).mean(0),
            's0': torch.stack(s0_list).mean(0),
            's1': torch.stack(s1_list).mean(0),
            'cg1': self.lif1.c_good, 'cb1': self.lif1.c_bad,
            'cg2': self.lif2.c_good, 'cb2': self.lif2.c_bad,
            'cg3': self.lif3.c_good, 'cb3': self.lif3.c_bad,
            'sc3': sc3,
        }

    def cap_update(self, z_pre, z_post, c_good, c_bad, sign):
        """ΔW = sign · α · mean(z_pre ⊗ z_post ⊙ (c_good − c_bad))"""
        B = z_pre.shape[0]
        cap = c_good - c_bad
        return sign * alpha_w * (z_pre.t() @ (z_post * cap)) / B

    def apply_updates(self, good, bad):
        B = good['z1'].shape[0]

        # W1, W2, W3 — через конденсаторы
        for i, (w, pre_g, post_g, pre_b, post_b, cg, cb_g, cb_b) in enumerate([
            (self.W1, good['s0'], good['z1'], bad['s0'], bad['z1'],
             good['cg1'], good['cb1'], bad['cb1']),
            (self.W2, good['s1'], good['z2'], bad['s1'], bad['z2'],
             good['cg2'], good['cb2'], bad['cb2']),
            (self.W3, good['z2'], good['z3'], bad['z2'], bad['z3'],
             good['cg3'], good['cb3'], bad['cb3']),
        ]):
            dw = (self.cap_update(pre_g, post_g, cg, torch.zeros_like(cg), +1.0) +
                  self.cap_update(pre_b, post_b, torch.zeros_like(cb_b), cb_b, -1.0))
            w.data += dw
            w.data.clamp_(-w_clip, w_clip)

        # V1, V2 — тоже трёхфакторное, pre = onehot метка
        y_g = torch.eye(N_OUT).repeat_interleave(B // N_OUT, dim=0)[:B]
        y_b = y_g[torch.randperm(B)]

        # V1
        cap_g1 = good['cg1']  # C_good (bad phase C_bad для V не используется, т.к. pre разный)
        cap_b1 = bad['cb1']
        self.V1.data += alpha_v * (y_g.t() @ (good['z1'] * cap_g1)) / B
        self.V1.data -= alpha_v * (y_b.t() @ (bad['z1']  * cap_b1)) / B
        self.V1.data.clamp_(-w_clip, w_clip)

        # V2
        cap_g2 = good['cg2']
        cap_b2 = bad['cb2']
        self.V2.data += alpha_v * (y_g.t() @ (good['z2'] * cap_g2)) / B
        self.V2.data -= alpha_v * (y_b.t() @ (bad['z2']  * cap_b2)) / B
        self.V2.data.clamp_(-w_clip, w_clip)


def make_xor(n=256):
    X, Y = [], []
    for (a, b), l in [([0,0],0), ([0,1],1), ([1,0],1), ([1,1],0)]:
        for _ in range(n // 4):
            X.append([rate_1 if a else rate_0, rate_1 if b else rate_0])
            Y.append(l)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.long)


@torch.no_grad()
def accuracy(model, X, Y):
    correct = 0
    for i in range(0, X.shape[0], BATCH):
        xb, yb = X[i:i+BATCH], Y[i:i+BATCH]
        y_oh = F.one_hot(yb, N_OUT).float()
        res = model.forward(xb, y_oh, phase=None)
        correct += (res['sc3'].argmax(1) == yb).sum().item()
    return correct / X.shape[0]


def train():
    X_tr, Y_tr = make_xor(256)
    X_te, Y_te = make_xor(64)
    model = CapacitorSNN()

    print(f"{'Epoch':>5s}  {'Train':>8s}  {'Test':>8s}")
    print("-" * 30)

    for ep in range(1, N_EPOCHS + 1):
        perm = torch.randperm(X_tr.shape[0])
        X_s, Y_s = X_tr[perm], Y_tr[perm]

        for b in range(0, X_tr.shape[0], BATCH):
            xb = X_s[b:b+BATCH]
            yb = Y_s[b:b+BATCH]
            y_g = F.one_hot(yb, N_OUT).float()
            y_b = F.one_hot((yb + 1) % N_OUT, N_OUT).float()

            good = model.forward(xb, y_g, phase='good')
            bad  = model.forward(xb, y_b, phase='bad')
            model.apply_updates(good, bad)

        if ep % 25 == 0 or ep == 1:
            ta = accuracy(model, X_tr, Y_tr)
            te = accuracy(model, X_te, Y_te)
            print(f"{ep:5d}  {ta:8.4f}  {te:8.4f}")

    final = accuracy(model, X_te, Y_te)
    print(f"\n{'='*30}\nFinal: {final:.4f}\n{'='*30}")
    return model, final


if __name__ == "__main__":
    train()
