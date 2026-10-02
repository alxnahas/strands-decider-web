"""Shared-prefix path in native ORT: encode state once, then question suffixes from its cache."""
import json, sys, time, numpy as np, onnxruntime as ort
sys.path.insert(0, "ref"); from head import decide
s = ort.InferenceSession(sys.argv[1], providers=["CPUExecutionProvider"])
outs = [o.name for o in s.get_outputs()]
def past_name(present): return present.replace("present.", "past_key_values.") if present.endswith((".key", ".value")) else present.replace("present.", "past.")
def run(ids, past=None, start=0):
    L = len(ids); f = {"input_ids": np.array([ids], np.int64), "attention_mask": np.ones((1, start + L), np.int64),
                       "position_ids": np.tile(np.arange(start, start + L, dtype=np.int64), (3, 1, 1))}
    for i in s.get_inputs():
        if i.name in f: continue
        if past is not None: f[i.name] = past[i.name]
        else: f[i.name] = np.zeros([1 if d == "batch_size" else 0 if d == "past_sequence_length" else 256 if d == "kv_cache_dim" else d for d in i.shape], np.float16)
    r = dict(zip(outs, s.run(outs, f)))
    return r["hidden_states"][0], {past_name(k): v for k, v in r.items() if k != "hidden_states"}
ref = json.load(open("ref/reference_cpu.json"))["results"]
from tokenizers import Tokenizer
tok = Tokenizer.from_file("models/decider/tokenizer.json")
state_ids = tok.encode("<state>\nHelp! My payouts have been failing for 3 days!\n</state>\n").ids
t = time.time(); _, cache = run(state_ids); print(f"prefix {len(state_ids)} tokens {1000*(time.time()-t):.0f}ms")
for r in ref[:4]:
    q = r["input_ids"][len(state_ids):]; assert r["input_ids"][:len(state_ids)] == state_ids
    t = time.time(); h, _ = run(q, cache, len(state_ids)); dt = time.time() - t
    _, p = decide(h, [i - len(state_ids) for i in r["opt_idx"]], r["kind"])
    hf, _ = run(r["input_ids"]); _, pf = decide(hf, r["opt_idx"], r["kind"])
    print(r["name"], "prefix", np.round(p, 4), "full", np.round(pf, 4), f"{1000*dt:.0f}ms")
