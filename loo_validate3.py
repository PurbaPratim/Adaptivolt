import numpy as np
import pandas as pd
import torch
import torch.nn as nn

df = pd.read_csv("sweep_dataset_smooth.csv")
X = df[["Vin", "Iload"]].values.astype(np.float32)
OUT_GAINS = ["Kp_v", "Ki_v", "K_vin_rate"]
Y = df[OUT_GAINS].values.astype(np.float32)

x_mean, x_std = X.mean(0), X.std(0)
Xn = (X - x_mean) / x_std
g_lo, g_hi = Y.min(0), Y.max(0)
scale = g_hi - g_lo
N = len(Xn)

class GainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 24), nn.ReLU(),
            nn.Linear(24, 24), nn.ReLU(),
            nn.Linear(24, 3), nn.Sigmoid(),
        )
        self.register_buffer("lo", torch.tensor(g_lo))
        self.register_buffer("hi", torch.tensor(g_hi))
    def forward(self, x):
        return self.lo + (self.hi - self.lo) * self.net(x)

Xt, Yt = torch.tensor(Xn), torch.tensor(Y)
scale_t = torch.tensor(scale)
errors = np.zeros((N, 3))

print(f"Running leave-one-out over {N} points...")
for i in range(N):
    mask = np.ones(N, dtype=bool); mask[i] = False
    torch.manual_seed(42)
    model = GainNet()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2000)
    for ep in range(2000):
        opt.zero_grad()
        loss = ((model(Xt[mask]) - Yt[mask]) / scale_t).pow(2).mean()
        loss.backward(); opt.step(); sched.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xt[i:i+1]).numpy()[0]
    errors[i] = np.abs(pred - Y[i]) / scale * 100
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{N} done")

print("\n=== LEAVE-ONE-OUT RESULTS (3-gain) ===")
print(f"{'gain':12s} {'mean %':>10s} {'worst %':>10s}")
for j, name in enumerate(OUT_GAINS):
    print(f"{name:12s} {errors[:,j].mean():>10.2f} {errors[:,j].max():>10.2f}")
print(f"{'overall':12s} {errors.mean():>10.2f} {errors.max():>10.2f}")

wi = errors.mean(1).argmax()
print(f"\nHardest point: Vin={X[wi,0]:.1f} V, Iload={X[wi,1]:.2f} A -> {errors[wi]}")
