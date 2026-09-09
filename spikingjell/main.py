#!/usr/bin/env python3


import torch, torch.nn.functional as F, numpy as np, argparse, time
from torch.utils.data import DataLoader, TensorDataset
from spikingjelly.activation_based import neuron, layer, functional


T           = 50
dt          = 3.0
tau_m       = 100.0
v_thr       = 0.055
tau_tr      = 13.0
R_m         = 0.1
inh_R       = 0.01
goodness_thr = 10.0
eta         = 0.002
weight_bound = 1.0
seed        = 5464

tau_m_hab   = 1.8
v_thr_hab   = 0.07
tau_hub     = 3

in_dim      = 784
out_dim     = 10
hid1        = 1024
hid2        = 128

batch_size      = 500
dev_batch_size  = 1000
num_epochs      = 30

data_dir = "/home/arz/SNN/jax_version/cSNN/data/mnist/mnist"



# CSDP-SNN (4 слоя)
class CSDPSNN(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
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
        factor = (hub / zmax_buf)**2 - phase
        return factor * z


    def forward(self, x, y, mask, learn=True):
        B = x.shape[0]; dv = self.device
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
        decay_hub = 1.0 / tau_hub 

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

            z1_prev = z1; z2_prev = z2

        return F.softmax(spk_cnt / T, dim=1), spk_cnt, tr_out

def nn_param(*shape):
    p = torch.nn.Parameter(torch.empty(*shape))
    torch.nn.init.uniform_(p, -1, 1)
    return p

def csdp_step(opt, param, grad, bounds, mask=None):
    opt.zero_grad(); param.grad = grad; opt.step()
    if mask is not None: param.data.mul_(mask)
    param.data.clamp_(*bounds)

def hebb_step(opt, param, grad, bounds):
    opt.zero_grad(); param.grad = -grad; opt.step()  
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
        xb = xb.to(device); B = xb.shape[0]
        yh, _, _ = model(xb, torch.zeros(B, out_dim, device=device),
                         torch.ones(B, 1, device=device), learn=False)
        c += (yh.argmax(dim=1) == yb.argmax(dim=1).to(device)).sum().item()
        t += B
    model.train()
    return c / t

def train_epoch(model, loader, device, epoch):
    model.train()
    c = t = 0
    n_batches = len(loader)
    for i, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.float().to(device)
        B = xb.shape[0]
        yn = neg_labels(yb, device)

        X_cat = torch.cat([xb, xb], dim=0)
        Y_cat = torch.cat([yb, yn], dim=0)
        m  = torch.cat([torch.ones(B, 1, device=device),
                        torch.zeros(B, 1, device=device)], dim=0)

        yh, _, _ = model(X_cat, Y_cat, m, learn=True)
        batch_correct = (yh[:B].argmax(dim=1) == yb.argmax(dim=1).to(device)).sum().item()
        c += batch_correct
        t += B
        print(f"  Batch {i+1:3d}/{n_batches}  acc: {batch_correct/B:.4f}  "
              f"avg: {c/t:.4f}", end="\r", flush=True)
    print()
    return c / t


def main():
    p = argparse.ArgumentParser(description="CSDP-SNN (SpikingJelly)")
    p.add_argument("--epochs", type=int, default=num_epochs)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)


    torch.manual_seed(seed); np.random.seed(seed)

    X_tr = torch.from_numpy(np.load(f"{data_dir}/trainX.npy")).float()
    Y_tr = torch.from_numpy(np.load(f"{data_dir}/trainY.npy")).float()
    X_te = torch.from_numpy(np.load(f"{data_dir}/testX.npy")).float()
    Y_te = torch.from_numpy(np.load(f"{data_dir}/testY.npy")).float()

    tr_ld = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=batch_size, shuffle=True, drop_last=True)
    te_ld = DataLoader(TensorDataset(X_te, Y_te), batch_size=dev_batch_size, shuffle=False, drop_last=True)

    model = CSDPSNN(device).to(device)

    t0 = time.time()
    print(f"Test accuracy: {evaluate(model, te_ld, device):.4f} ")

    best = 0.0
    for ep in range(1, args.epochs + 1):
        tr_acc = train_epoch(model, tr_ld, device, ep)
        te_acc = evaluate(model, te_ld, device)
        best   = max(best, te_acc)
        elapsed = time.time() - t0
        print(f"Epoch {ep:2d}/{args.epochs}  Train: {tr_acc:.4f}  Test: {te_acc:.4f}  Best: {best:.4f}  [{elapsed/60:.1f}m]")

    print(f"\nDone in {elapsed/60:.1f} min.  Best test accuracy: {best:.4f}")


if __name__ == "__main__":
    main()
