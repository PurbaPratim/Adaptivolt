import numpy as np
import pandas as pd
import torch
import torch.nn as nn

df = pd.read_csv("sweep_dataset_smooth.csv")
X = df[["Vin", "Iload"]].values.astype(np.float32)
Y = df[["Kp_v", "Ki_v", "K_vin_rate", "w_min", "w_max"]].values.astype(np.float32)

x_mean, x_std = X.mean(0), X.std(0)
Xn = (X - x_mean) / x_std

g_lo = Y.min(0)
g_hi = Y.max(0)

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

model = GainNet()
print(sum(p.numel() for p in model.parameters()), "parameters")

Yt = torch.tensor(Y)
Xt = torch.tensor(Xn)
scale = torch.tensor(g_hi - g_lo)

idx = np.random.permutation(len(Xt))
n_val = 8
tr, va = idx[n_val:], idx[:n_val]

opt = torch.optim.Adam(model.parameters(), lr=3e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3000)
lossfn = nn.MSELoss()

for epoch in range(3000):
    model.train()
    opt.zero_grad()
    pred = model(Xt[tr])
    loss = lossfn(pred / scale, Yt[tr] / scale)
    loss.backward()
    opt.step()
    sched.step()
    if epoch % 500 == 0:
        model.eval()
        with torch.no_grad():
            vloss = lossfn(model(Xt[va]) / scale, Yt[va] / scale)
        print(f"epoch {epoch:5d}  train {loss.item():.5f}  val {vloss.item():.5f}")

model.eval()
with torch.no_grad():
    pred_all = model(Xt).numpy()
err = np.abs(pred_all - Y) / (g_hi - g_lo) * 100
print("\nper-gain mean error (% of range):")
for name, e in zip(["Kp_v", "Ki_v", "K_vin", "w_min", "w_max"], err.mean(0)):
    print(f"  {name:8s} {e:.2f}%")

np.savez("norm_params.npz", x_mean=x_mean, x_std=x_std, g_lo=g_lo, g_hi=g_hi)
torch.save(model.state_dict(), "gain_net.pt")

dummy = torch.zeros(1, 2)
torch.onnx.export(model, dummy, "gain_net.onnx",
                  input_names=["op_point"], output_names=["gains"],
                  opset_version=13)
print("\nsaved: gain_net.pt, gain_net.onnx, norm_params.npz")
