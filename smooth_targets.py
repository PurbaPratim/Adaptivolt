import pandas as pd
import numpy as np

K = 5
df = pd.read_csv("stability_map_full.csv")
df = df[df["stable"] == 1].copy()

rows = []
for (vin, iload), g in df.groupby(["Vin", "Iload"]):
    topk = g.nsmallest(K, "J")
    rows.append({
        "Vin": vin, "Iload": iload,
        "Kp_v": topk["Kp_v"].mean(),
        "Ki_v": topk["Ki_v"].mean(),
        "K_vin_rate": topk["K_vin_rate"].mean(),
        "w_min": topk["w_min"].mean(),
        "w_max": topk["w_max"].mean(),
        "J_mean": topk["J"].mean(),
        "J_best": topk["J"].min(),
    })
out = pd.DataFrame(rows)
out.to_csv("sweep_dataset_smooth.csv", index=False)
print(f"wrote sweep_dataset_smooth.csv ({len(out)} rows, top-{K} averaged)")
print("\ngain ranges in smoothed targets:")
for c in ["Kp_v", "Ki_v", "K_vin_rate", "w_min", "w_max"]:
    print(f"  {c:12s} [{out[c].min():.4g}, {out[c].max():.4g}]")
print(f"\nmean J penalty vs argmin: {(out['J_mean']/out['J_best']).mean():.3f}x")
