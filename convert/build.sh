#!/usr/bin/env bash
# Reproduce the browser model artifacts from the public checkpoints. Run from the repository root.
set -euo pipefail
source .venv/bin/activate
# 1. Pinned public checkpoints (adapter + head + tokenizer; base revision recorded in the adapter's provenance.json).
hf download StrandsAgents/strands-decider-2B-hobson-v19 --revision bb282d786bc251fd4e3068de3ada9ddbb38127cd --local-dir models/decider
hf download Qwen/Qwen3.5-2B-Base --revision b1485b2fa6dfa1287294f269f5fb618e03d52d7c --local-dir models/qwen-base
# 2. Fold LoRA (r=16, alpha=32) into the text torso; drop vision tower + MTP head.
python convert/merge_lora.py models/merged
# 3. ORT GenAI model builder: Qwen3.5 hybrid torso -> ONNX with LinearAttention/CausalConvWithState contrib ops,
#    no LM head (outputs hidden_states), int4 (or int8) weight-only quantisation incl. the embedding table.
for p in int4 int8; do
  out=onnx/q${p#int}f16
  python -m onnxruntime_genai.models.builder -i models/merged -o $out -p $p -e webgpu -c onnx/cache \
    --extra_options exclude_lm_head=true exclude_mtp=true op_types_to_quantize=MatMul/Gather
  # 4. Split external data into <=512 MB shards (browser ArrayBuffer limits) + manifest.json.
  python convert/shard.py $out $out-web && rm $out/model.onnx.data
done
# 5. Portable int4 graph: GatherBlockQuantized -> standard ops so the WASM fallback can load it.
mkdir -p onnx/q4f16p-web && python convert/portable_embed.py onnx/q4f16-web/model.onnx onnx/q4f16p-web/model.onnx
for f in onnx/q4f16-web/model.onnx.data.* onnx/q4f16-web/manifest.json; do ln -f "$f" onnx/q4f16p-web/; done
rm -rf onnx/cache
# 6. Hand-written WebGPU engine weights (int4 RTN, fused projections) -> engine/, plus a fake-quant reference.
python convert/export_engine.py
