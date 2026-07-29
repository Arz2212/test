import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


from brudforse import XOR_SNN, CSDPLearner, SupervisedLearner, make_xor_dataset, DEVICE


T                  = 100
TRAIN_ONLY_EPOCHS  = 100    
TRAIN_EVAL_EPOCHS  = 30     
VERBOSE            = True   
VERBOSE_EVERY      = 1    
BATCH              = 64


CFG = {
    "alpha_csdp_log": -2.3805873732083946,
    "alpha_ctx_log": -1.5078153917994113,
    "alpha_sup_log": -0.8523012632022566,
    "tau_hub": 3.5653850256485,
    "tau_m": 3.453130166014691,
    "tau_m_hab": 4.078025835656965,
    "tau_z": 4.218990109063842,
    "v_thr": 0.4994297455144787,
    "v_thr_hab": 0.19482415609785836,
    "w_clip": 2.2211192793854346,
    "w_decay_log": -2.7153454040577487,
}



@torch.no_grad()
def accuracy(net, loader, T):
    net.eval()
    ok, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = net(x, T, labels=y)
        ok += (out.sum(0).argmax(1).cpu() == y.cpu()).sum().item()
        total += y.size(0)
    return ok / total


def run(cfg):
    tau_m     = cfg["tau_m"]
    v_thr     = cfg["v_thr"]
    tau_hub   = cfg["tau_hub"]
    v_thr_hab = cfg["v_thr_hab"]
    tau_m_hab = cfg["tau_m_hab"]
    tau_z     = cfg["tau_z"]
    w_clip    = cfg["w_clip"]

    alpha_csdp = 10 ** cfg["alpha_csdp_log"]
    alpha_ctx  = 10 ** cfg["alpha_ctx_log"]
    alpha_sup  = 10 ** cfg["alpha_sup_log"]
    w_decay    = 10 ** cfg["w_decay_log"]

    X_train, y_train = make_xor_dataset(n_samples=200)
    X_test,  y_test  = make_xor_dataset(n_samples=50)
    train_ldr = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH, shuffle=True, drop_last=True)
    test_ldr  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=BATCH, shuffle=False)

    net = XOR_SNN(
        n_in=3, n_h1=5, n_h2=3, n_out=2,
        tau_m=tau_m, v_thr=v_thr,
        tau_m_hab=tau_m_hab, v_thr_hab=v_thr_hab, tau_hub=tau_hub,
    )
    net.to(DEVICE)

    csdp = [
        CSDPLearner(net.fc1,  alpha=alpha_csdp, tau_z=tau_z, w_decay=w_decay, w_clip=w_clip),
        CSDPLearner(net.fc2,  alpha=alpha_csdp, tau_z=tau_z, w_decay=w_decay, w_clip=w_clip),
        CSDPLearner(net.ctx1, alpha=alpha_ctx,  tau_z=tau_z, w_decay=w_decay, w_clip=w_clip),
        CSDPLearner(net.ctx2, alpha=alpha_ctx,  tau_z=tau_z, w_decay=w_decay, w_clip=w_clip),
    ]
    sup = SupervisedLearner(net.fc3, alpha=alpha_sup, w_clip=w_clip)

    for epoch in range(TRAIN_ONLY_EPOCHS):
        net.train()
        for xb, yb in train_ldr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            net(xb, T, csdp_learners=csdp, sup_learner=sup, phase=1.0, labels=yb)
            for lrn in csdp:
                lrn.reset()
            wrong = 1 - yb
            net(xb, T, csdp_learners=csdp, sup_learner=None, phase=0.0, labels=wrong.to(DEVICE))
            for lrn in csdp:
                lrn.reset()
        if VERBOSE_EVERY and (epoch + 1) % VERBOSE_EVERY == 0:
            acc = accuracy(net, test_ldr, T)
            print(f"  epoch {epoch + 1:3d}/{TRAIN_ONLY_EPOCHS}  acc={acc:.4f}")

    eval_accs = []
    total_epochs = TRAIN_ONLY_EPOCHS + TRAIN_EVAL_EPOCHS
    for epoch in range(TRAIN_EVAL_EPOCHS):
        net.train()
        for xb, yb in train_ldr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            net(xb, T, csdp_learners=csdp, sup_learner=sup, phase=1.0, labels=yb)
            for lrn in csdp:
                lrn.reset()
            wrong = 1 - yb
            net(xb, T, csdp_learners=csdp, sup_learner=None, phase=0.0, labels=wrong.to(DEVICE))
            for lrn in csdp:
                lrn.reset()
        acc = accuracy(net, test_ldr, T)
        eval_accs.append(acc)
        if VERBOSE:
            print(f"  epoch {TRAIN_ONLY_EPOCHS + epoch + 1:3d}/{total_epochs}  acc={acc:.4f}")

    return float(np.mean(eval_accs))


if __name__ == "__main__":
    final_acc = run(CFG)
    print(f"\n{'='*40}\nФинальная точность: {final_acc:.4f}\n{'='*40}")
