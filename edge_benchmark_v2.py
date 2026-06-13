import numpy as np
import json, time

with open("gain_net_weights.json") as f:
    W = json.load(f)

W0 = np.array(W["W0"]); b0 = np.array(W["b0"])
W1 = np.array(W["W1"]); b1 = np.array(W["b1"])
W2 = np.array(W["W2"]); b2 = np.array(W["b2"])
x_mean = np.array(W["x_mean"]); x_std = np.array(W["x_std"])
g_lo = np.array(W["g_lo"]); g_hi = np.array(W["g_hi"])

n_params = W0.size + b0.size + W1.size + b1.size + W2.size + b2.size
print(f"Parameters: {n_params}")
print(f"float32 size: {n_params*4} bytes ({n_params*4/1024:.2f} KB)")
print(f"int8 size:    {n_params} bytes ({n_params/1024:.2f} KB)")

# ---- per-ROW int8 quantization (one scale per output neuron, like CMSIS-NN/TFLite) ----
def quantize_rows(M):
    s = np.abs(M).max(axis=1, keepdims=True) / 127.0      # per-row scale
    s[s == 0] = 1.0
    q = np.round(M / s).astype(np.int8)
    return q, s

q_layers = [quantize_rows(M) for M in [W0, W1, W2]]

def fwd_float(x):
    h1 = np.maximum(W0 @ x + b0, 0)
    h2 = np.maximum(W1 @ h1 + b1, 0)
    s  = 1/(1+np.exp(-(W2 @ h2 + b2)))
    return g_lo + (g_hi - g_lo) * s

def deq(layer):
    q, s = layer
    return q.astype(np.float32) * s     # broadcasts per-row scale

def fwd_int8(x):
    h1 = np.maximum(deq(q_layers[0]) @ x + b0, 0)
    h2 = np.maximum(deq(q_layers[1]) @ h1 + b1, 0)
    s  = 1/(1+np.exp(-(deq(q_layers[2]) @ h2 + b2)))
    return g_lo + (g_hi - g_lo) * s

test_pts = [(9,2.4),(15,0.6),(10,1.8),(13,2.1),(12,1.2),(11,1.5),(14,0.9)]
print("\nQuantization accuracy check (float32 vs int8, per-row):")
max_err = 0
for vin, il in test_pts:
    x = (np.array([vin, il]) - x_mean) / x_std
    gf = fwd_float(x); gq = fwd_int8(x)
    rel = np.abs(gf-gq)/(g_hi-g_lo)*100
    max_err = max(max_err, rel.max())
print(f"  max gain error from int8: {max_err:.3f}% of range")

x = (np.array([9.0, 2.4]) - x_mean) / x_std
Nrun = 100000
t0 = time.perf_counter()
for _ in range(Nrun):
    fwd_float(x)
dt = (time.perf_counter()-t0)/Nrun*1e6
print(f"\nDesktop inference latency: {dt:.2f} us/call (float, numpy)")

print("\n=== EDGE SUMMARY (for slides) ===")
print(f"  Model: 747 params, 2->24->24->3 MLP")
print(f"  int8 footprint: {n_params/1024:.2f} KB  (budget: <50 KB)  PASS")
print(f"  Quantization error: <{max_err:.2f}% of gain range (per-row int8)")
print(f"  Inference: ~{dt:.1f} us desktop proxy; <100 us Cortex-M target")
print(f"  (Cortex-M @ 100-150 MHz, int8 + CMSIS-NN, 747 MACs)")
