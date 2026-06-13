import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json

df = pd.read_csv("sweep_dataset_smooth.csv")
X = df[["Vin", "Iload"]].values.astype(np.float32)

OUT_GAINS = ["Kp_v", "Ki_v", "K_vin_rate"]
Y = df[OUT_GAINS].values.astype(np.float32)

w_min_fixed = 1.8
w_max_fixed = 3.0
print(f"Fixed gains (held constant): w_min={w_min_fixed:.3f}, w_max={w_max_fixed:.3f}")

x_mean, x_std = X.mean(0), X.std(0)
Xn = (X - x_mean) / x_std
g_lo, g_hi = Y.min(0), Y.max(0)
scale = g_hi - g_lo

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

model = GainNet()
print(sum(p.numel() for p in model.parameters()), "parameters")

Xt, Yt = torch.tensor(Xn), torch.tensor(Y)
scale_t = torch.tensor(scale)
torch.manual_seed(42)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3000)

for epoch in range(3000):
    model.train(); opt.zero_grad()
    loss = ((model(Xt) - Yt) / scale_t).pow(2).mean()
    loss.backward(); opt.step(); sched.step()
    if epoch % 500 == 0:
        print(f"epoch {epoch:5d}  train {loss.item():.5f}")

model.eval()
with torch.no_grad():
    pred = model(Xt).numpy()
err = np.abs(pred - Y) / scale * 100
print("\nper-gain mean error (% of range):")
for name, e in zip(OUT_GAINS, err.mean(0)):
    print(f"  {name:10s} {e:.2f}%")

# Export weights to JSON for MATLAB (no toolbox needed)
sd = model.state_dict()
weights = {
    "x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
    "g_lo": g_lo.tolist(), "g_hi": g_hi.tolist(),
    "w_min_fixed": w_min_fixed, "w_max_fixed": w_max_fixed,
    "W0": sd["net.0.weight"].numpy().tolist(), "b0": sd["net.0.bias"].numpy().tolist(),
    "W1": sd["net.2.weight"].numpy().tolist(), "b1": sd["net.2.bias"].numpy().tolist(),
    "W2": sd["net.4.weight"].numpy().tolist(), "b2": sd["net.4.bias"].numpy().tolist(),
    "out_gains": OUT_GAINS,
}
with open("gain_net_weights.json", "w") as f:
    json.dump(weights, f, indent=2)

torch.save(model.state_dict(), "gain_net3.pt")
np.savez("norm_params3.npz", x_mean=x_mean, x_std=x_std, g_lo=g_lo, g_hi=g_hi,
         w_min_fixed=w_min_fixed, w_max_fixed=w_max_fixed)
print("\nsaved: gain_net3.pt, norm_params3.npz, gain_net_weights.json")
