"""Diagnostic only: replace one op type with a pass-through to measure its share of latency."""
import os, sys, onnx
from onnx import helper as h
src_dir, op = sys.argv[1], sys.argv[2]
dst = f"{src_dir.rstrip('/').removesuffix('-web')}-no{op}-web"
m = onnx.load(f"{src_dir}/model.onnx", load_external_data=False)
PASS = {"LinearAttention": (2, {1: 3}), "CausalConvWithState": (0, {1: 3}), "GroupQueryAttention": (0, {1: 3, 2: 4}),
        "MatMulNBitsMlp": (0, {}), "GatedRMSNorm": (0, {})}
data_in, state_map = PASS[op]
nodes = []
for n in m.graph.node:
    if n.op_type != op: nodes.append(n); continue
    nodes.append(h.make_node("Identity", [n.input[data_in]], [n.output[0]], name=n.name + "/ablated"))
    for o, i in state_map.items(): nodes.append(h.make_node("Identity", [n.input[i]], [n.output[o]], name=n.name + f"/ablated{o}"))
del m.graph.node[:]; m.graph.node.extend(nodes)
os.makedirs(dst, exist_ok=True); onnx.save(m, f"{dst}/model.onnx")
for f in os.listdir(src_dir):
    if f != "model.onnx" and not os.path.exists(f"{dst}/{f}"): os.link(f"{src_dir}/{f}", f"{dst}/{f}")
print(dst)
