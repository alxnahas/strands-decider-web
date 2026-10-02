import json, sys, torch, numpy as np, transformers
sys.path.insert(0, "ref"); from head import decide
cfg = transformers.AutoConfig.from_pretrained("models/merged").get_text_config()
lm = transformers.Qwen3_5ForCausalLM.from_pretrained("models/merged", config=cfg, dtype=torch.float32).eval()
ref = json.load(open("ref/reference_cpu.json"))
for r in ref["results"]:
    with torch.inference_mode():
        h = lm.model(input_ids=torch.tensor([r["input_ids"]])).last_hidden_state[0].numpy()
    _, p = decide(h, r["opt_idx"], r["kind"])
    print(r["name"], np.round(p, 4), list(r["answer"].get("probabilities", {}).values()) or r["answer"])
