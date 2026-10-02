"""Export the merged Strands Decider torso for the hand-written WebGPU engine.

int4 symmetric RTN, block 32, MatMulNBits layout ([N][K/32][16 bytes], element 2i in the low nibble, zero point 8,
fp16 scale per block). Projections sharing an input are fused by concatenating output rows:
  linear layers: in_proj_qkv | in_proj_z | in_proj_a | in_proj_b (+ zero rows to a multiple of 64)
  attention:     q_proj (per head [q | gate]) | k_proj | v_proj
  mlp:           gate_proj | up_proj
Also writes a fake-quantised fp32 HF reference (per-layer hidden states) for the payouts prompt.
"""
import json, os, sys, numpy as np, torch
from safetensors import safe_open
OUT = "engine"; SRC = "models/merged/model.safetensors"
f = safe_open(SRC, "pt"); P = "model.language_model."
cfg = json.load(open("models/merged/config.json"))["text_config"]
get = lambda k: f.get_tensor(P + k).float()

def quant(W):
    N, K = W.shape; assert K % 32 == 0
    b = W.reshape(N, K // 32, 32)
    idx = b.abs().argmax(-1, keepdim=True)
    scale = torch.gather(b, -1, idx) / -8.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale).to(torch.float16).float()
    q = torch.clamp(torch.round(b / scale) + 8, 0, 15).to(torch.uint8)
    packed = (q[..., 0::2] | (q[..., 1::2] << 4)).contiguous()
    deq = ((q.float() - 8) * scale).reshape(N, K)
    return packed.numpy(), scale.squeeze(-1).to(torch.float16).numpy(), deq

blobs, index, offset = [], {}, 0
def add(name, arr, **meta):
    global offset
    arr = np.ascontiguousarray(arr); pad = (-offset) % 256
    if pad: blobs.append(b"\0" * pad); offset += pad
    index[name] = {"offset": offset, "bytes": arr.nbytes, "dtype": str(arr.dtype), "shape": list(arr.shape), **meta}
    blobs.append(arr.tobytes()); offset += arr.nbytes
deq_cache = {}
def add_q(name, W, keys=None):
    N = W.shape[0]; Np = (N + 63) // 64 * 64
    if Np != N: W = torch.cat([W, torch.zeros(Np - N, W.shape[1])])
    pk, sc, dq = quant(W)
    add(name + ".q", pk, N=Np, K=W.shape[1]); add(name + ".s", sc)
    if keys:  # remember dequantised slices for the reference model
        o = 0
        for k, n in keys: deq_cache[k] = dq[o:o + n]; o += n

pk, sc, dq = quant(get("embed_tokens.weight")); add("embed.q", pk, N=pk.shape[0], K=dq.shape[1]); add("embed.s", sc); deq_cache["embed_tokens.weight"] = dq
for i, kind in enumerate(cfg["layer_types"]):
    L = f"layers.{i}."
    add(L + "in_norm", 1 + get(L + "input_layernorm.weight").numpy().astype(np.float32))
    add(L + "post_norm", 1 + get(L + "post_attention_layernorm.weight").numpy().astype(np.float32))
    if kind == "linear_attention":
        a = L + "linear_attn."
        ws = [(a + n + ".weight", get(a + n + ".weight")) for n in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b")]
        add_q(L + "in_proj", torch.cat([w for _, w in ws]), [(k, w.shape[0]) for k, w in ws])
        add_q(L + "out_proj", get(a + "out_proj.weight"), [(a + "out_proj.weight", 2048)])
        add(L + "conv", get(a + "conv1d.weight").squeeze(1).numpy().astype(np.float32))   # [6144, 4]
        add(L + "neg_exp_A", (-get(a + "A_log").exp()).numpy().astype(np.float32))
        add(L + "dt_bias", get(a + "dt_bias").numpy().astype(np.float32))
        add(L + "gnorm", get(a + "norm.weight").numpy().astype(np.float32))
    else:
        a = L + "self_attn."
        ws = [(a + n + ".weight", get(a + n + ".weight")) for n in ("q_proj", "k_proj", "v_proj")]
        add_q(L + "qkv", torch.cat([w for _, w in ws]), [(k, w.shape[0]) for k, w in ws])
        add_q(L + "o_proj", get(a + "o_proj.weight"), [(a + "o_proj.weight", 2048)])
        add(L + "q_norm", 1 + get(a + "q_norm.weight").numpy().astype(np.float32))
        add(L + "k_norm", 1 + get(a + "k_norm.weight").numpy().astype(np.float32))
    m = L + "mlp."
    ws = [(m + n + ".weight", get(m + n + ".weight")) for n in ("gate_proj", "up_proj")]
    add_q(L + "gate_up", torch.cat([w for _, w in ws]), [(k, w.shape[0]) for k, w in ws])
    add_q(L + "down", get(m + "down_proj.weight"), [(m + "down_proj.weight", 2048)])
add("final_norm", 1 + get("norm.weight").numpy().astype(np.float32))

# shards of <= 512 MB, split only between tensors
os.makedirs(OUT, exist_ok=True)
shards, cur, cur_start, pos = [], [], 0, 0
for b in blobs:
    if cur and pos + len(b) - cur_start > (512 << 20):
        shards.append((cur_start, cur)); cur, cur_start = [], pos
    cur.append(b); pos += len(b)
shards.append((cur_start, cur))
for n, (start, bs) in enumerate(shards):
    with open(f"{OUT}/weights.{n}.bin", "wb") as fh:
        for b in bs: fh.write(b)
manifest = {"config": {k: cfg[k] for k in ("hidden_size", "intermediate_size", "num_hidden_layers", "layer_types", "num_attention_heads", "num_key_value_heads", "head_dim", "linear_num_key_heads", "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim", "rms_norm_eps", "vocab_size")} | {"rope_theta": cfg["rope_parameters"]["rope_theta"], "rotary_dim": int(cfg["head_dim"] * cfg["rope_parameters"]["partial_rotary_factor"])},
            "shards": [{"path": f"weights.{n}.bin", "start": s, "bytes": sum(len(b) for b in bs)} for n, (s, bs) in enumerate(shards)], "tensors": index}
json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=0)
print("shards", [(s["path"], s["bytes"] >> 20) for s in manifest["shards"]])

# Fake-quantised fp32 reference: same int4 weights, dequantised, run through the HF implementation.
import transformers
tcfg = transformers.AutoConfig.from_pretrained("models/merged").get_text_config()
lm = transformers.Qwen3_5ForCausalLM.from_pretrained("models/merged", config=tcfg, dtype=torch.float32).eval()
sd = lm.model.state_dict()
with torch.no_grad():
    for k, v in deq_cache.items():
        sd[k.removeprefix("") if k in sd else k].copy_(v) if k in sd else None
    for k, v in deq_cache.items():
        if k not in sd: raise KeyError(k)
ids = json.load(open("ref/reference_cpu.json"))["results"][0]["input_ids"]
outs = {}
hooks = [layer.register_forward_hook(lambda mod, inp, out, i=i: outs.__setitem__(f"layer{i}", (out[0] if isinstance(out, tuple) else out)[0].numpy())) for i, layer in enumerate(lm.model.layers)]
hooks.append(lm.model.embed_tokens.register_forward_hook(lambda m, i, o: outs.__setitem__("embed", o[0].numpy())))
with torch.inference_mode():
    final = lm.model(input_ids=torch.tensor([ids])).last_hidden_state[0].numpy()
outs["final"] = final
np.savez(f"{OUT}/ref_payouts.npz", ids=np.array(ids), **outs)
sys.path.insert(0, "ref"); from head import decide
print("fake-quant reference probs", decide(final, json.load(open("ref/reference_cpu.json"))["results"][0]["opt_idx"], "choice")[1])
