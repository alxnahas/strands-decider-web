"""Variant with a MatMulNBits attribute changed (graph only; weight shards hard-linked)."""
import os, sys, onnx
src_dir, attr, val, dst = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
m = onnx.load(f"{src_dir}/model.onnx", load_external_data=False)
n_set = 0
for n in m.graph.node:
    if n.op_type == "MatMulNBits":
        for a in n.attribute:
            if a.name == attr: a.i = val; n_set += 1
os.makedirs(dst, exist_ok=True); onnx.save(m, f"{dst}/model.onnx")
for f in os.listdir(src_dir):
    if f != "model.onnx" and not os.path.exists(f"{dst}/{f}"): os.link(f"{src_dir}/{f}", f"{dst}/{f}")
print(dst, n_set)
