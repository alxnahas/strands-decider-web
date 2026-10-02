"""Fold the v19 LoRA adapter into Qwen3.5-2B-Base: W' = W + (alpha/r) * B @ A.

Writes a Qwen3_5ForConditionalGeneration-shaped checkpoint (text weights only; vision
tower and MTP head dropped) in fp16 so the ORT GenAI builder can consume it.
"""
import json, os, shutil, sys, torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE, ADAPTER, OUT = "models/qwen-base", "models/decider/lora", sys.argv[1] if len(sys.argv) > 1 else "models/merged"
cfg = json.load(open(f"{ADAPTER}/adapter_config.json"))
scale = cfg["lora_alpha"] / cfg["r"]
ad = safe_open(f"{ADAPTER}/adapter_model.safetensors", "pt")
lora = {}
for k in ad.keys():
    # base_model.model.layers.0.linear_attn.in_proj_a.lora_A.weight -> model.language_model.layers.0.linear_attn.in_proj_a.weight
    mod, ab = k.removeprefix("base_model.model.").rsplit(".lora_", 1)
    lora.setdefault("model.language_model." + mod + ".weight", {})[ab[0]] = ad.get_tensor(k).float()
base = safe_open(f"{BASE}/model.safetensors-00001-of-00001.safetensors", "pt")
out, merged = {}, 0
for k in base.keys():
    if k.startswith(("mtp.", "model.visual.")):
        continue
    w = base.get_tensor(k)
    if k in lora:
        w = w.float() + scale * (lora[k]["B"] @ lora[k]["A"]); merged += 1
    out[k] = w.to(torch.float16) if w.is_floating_point() else w
assert merged == len(lora), (merged, len(lora))
os.makedirs(OUT, exist_ok=True)
save_file(out, f"{OUT}/model.safetensors", metadata={"format": "pt"})
for f in ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "preprocessor_config.json"]:
    shutil.copy(f"{BASE}/{f}", OUT)
c = json.load(open(f"{OUT}/config.json")); c["torch_dtype"] = c["dtype"] = "float16"
c.get("text_config", {})["dtype"] = "float16"
json.dump(c, open(f"{OUT}/config.json", "w"), indent=2)
print(f"merged {merged} LoRA modules, wrote {len(out)} tensors to {OUT}")
