import numpy as np
import pandas as pd
import torch
import torch.nn as nn

df = pd.read_csv("sweep_dataset_smooth.csv")
X = df[["Vin", "Iload"]].values.astype(np.float32)
Y = df[["Kp_v", "Ki_v", "K_vin_rate", "w_min", "w_max"]].values.astype(np.float32)

x_mean, x_std = X.mean(0), X.std(0)
Xn = (X - x_mean) / x_std
g_lo, g_hi = Y.min(0), Y.max(0)
scale = g_hi - g_lo
N = len(Xn)

class GainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 5), nn.Sigmoid(),
        )
        self.register_buffer("lo", torch.tensor(g_lo))
        self.register_buffer("hi", torch.tensor(g_hi))
    def forward(self, x):
        return self.lo + (self.hi - self.lo) * self.net(x)

errors = np.zeros((N, 5))
Xt = torch.tensor(Xn)
Yt = torch.tensor(Y)
scale_t = torch.tensor(scale)

print(f"Running leave-one-out over {N} points...")
for i in range(N):
    mask = np.ones(N, dtype=bool)
    mask[i] = False
    Xtr, Ytr = Xt[mask], Yt[mask]
    torch.manual_seed(42)
    model = GainNet()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2000)
    for ep in range(2000):
        opt.zero_grad()
        loss = ((model(Xtr) - Ytr) / scale_t).pow(2).mean()
        loss.backward()
        opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xt[i:i+1]).numpy()[0]
    errors[i] = np.abs(pred - Y[i]) / scale * 100
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{N} done")

print("\n=== LEAVE-ONE-OUT RESULTS ===")
print(f"{'gain':10s} {'mean %':>10s} {'worst %':>10s}")
for j, name in enumerate(["Kp_v", "Ki_v", "K_vin", "w_min", "w_max"]):
    print(f"{name:10s} {errors[:,j].mean():>10.2f} {errors[:,j].max():>10.2f}")
print(f"{'overall':10s} {errors.mean():>10.2f} {errors.max():>10.2f}")

worst_idx = errors.mean(1).argmax()
print(f"\nHardest operating point: Vin={X[worst_idx,0]:.1f} V, Iload={X[worst_idx,1]:.2f} A")
print(f"  errors at that point: {errors[worst_idx]}")

np.savez("loo_errors.npz", errors=errors, X=X)
print("\nsaved: loo_errors.npz")
