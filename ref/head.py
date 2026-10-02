"""Pointer head + calibration, in numpy — the exact math the browser reimplements."""
import json, numpy as np
from safetensors.numpy import load_file
H = load_file("models/decider/head.safetensors")
CFG = json.load(open("models/decider/hobson_config.json"))
def layernorm(x):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * H["norm.weight"] + H["norm.bias"]
def decide(hidden, opt_idx, kind):
    """hidden [L, 2048] float32 -> probs over options."""
    hidden = hidden.astype(np.float32)
    q = layernorm(hidden[-1]) @ H["q.weight"].T + H["q.bias"]
    k = layernorm(hidden[opt_idx]) @ H["k.weight"].T + H["k.bias"]
    logits = (k @ q) * (CFG["pointer_dim"] ** -0.5)
    logits = logits / CFG["temperature_by_kind"].get(kind, CFG["temperature"])
    e = np.exp(logits - logits.max()); return logits, e / e.sum()
