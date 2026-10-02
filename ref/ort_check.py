"""Run an exported ONNX torso in native ORT on the reference fixtures; compare with PyTorch."""
import json, sys, time, numpy as np, onnxruntime as ort
sys.path.insert(0, "ref"); from head import decide
path = sys.argv[1]
so = ort.SessionOptions()
t = time.time(); s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"]); print(f"session {time.time()-t:.1f}s")
TYPES = {"tensor(float16)": np.float16, "tensor(float)": np.float32, "tensor(int64)": np.int64, "tensor(int32)": np.int32}
def feeds(ids):
    L = len(ids); f = {}
    for i in s.get_inputs():
        if i.name == "input_ids": f[i.name] = np.array([ids], np.int64)
        elif i.name == "attention_mask": f[i.name] = np.ones((1, L), np.int64)
        elif i.name == "position_ids": f[i.name] = np.tile(np.arange(L, dtype=np.int64), (3, 1, 1))
        else:
            shp = [1 if d == "batch_size" else 0 if isinstance(d, str) and "past" in d else (256 if d == "kv_cache_dim" else d) for d in i.shape]
            f[i.name] = np.zeros(shp, TYPES[i.type])
    return f
if "--inputs" in sys.argv:
    for i in s.get_inputs(): print(i.name, i.type, i.shape)
ref = json.load(open(next((a for a in sys.argv[2:] if a.endswith(".json")), "ref/reference_cpu.json"))); worst = 0
for r in ref["results"]:
    t = time.time(); h = s.run(["hidden_states"], feeds(r["input_ids"]))[0][0]; dt = time.time() - t
    _, p = decide(h, r["opt_idx"], r["kind"])
    a = r["answer"]; rp = np.array(list(a["probabilities"].values())) if "probabilities" in a else np.array([1 - a["noul"], a["noul"]])
    if r["kind"] == "noul": rp = rp  # slots are (false, true)
    err = float(np.abs(p - rp).max()); worst = max(worst, err)
    print(f"{r['name']:20s} ort={np.round(p,4)} ref={np.round(rp,4)} maxerr={err:.4f} top_ok={p.argmax()==rp.argmax()} {dt*1000:.0f}ms")
print("worst", worst)
