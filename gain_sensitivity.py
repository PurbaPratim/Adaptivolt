import pandas as pd
import numpy as np

df = pd.read_csv("stability_map_full.csv")
df = df[df["stable"] == 1].copy()

gains = ["Kp_v", "Ki_v", "K_vin_rate", "w_min", "w_max"]

print("How much does J vary when ONLY each gain changes,")
print("holding the operating point and other gains fixed?\n")
print(f"{'gain':12s} {'median J swing':>16s} {'as % of best J':>16s}")

swings = {g: [] for g in gains}
pct = {g: [] for g in gains}

for (vin, iload), grp in df.groupby(["Vin", "Iload"]):
    bestJ = grp["J"].min()
    for g in gains:
        others = [x for x in gains if x != g]
        for _, sub in grp.groupby(others):
            if len(sub) > 1:
                swing = sub["J"].max() - sub["J"].min()
                swings[g].append(swing)
                pct[g].append(swing / bestJ * 100)

for g in gains:
    s = np.median(swings[g])
    p = np.median(pct[g])
    print(f"{g:12s} {s:>16.5f} {p:>15.2f}%")

print("\nInterpretation: gains with tiny J swing barely affect performance.")
print("Those are the ones the NN struggles to predict (and shouldn't need to).")
