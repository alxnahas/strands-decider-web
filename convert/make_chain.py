"""Chain of 8 (2048->6144, 6144->2048) MatMulNBits pairs with distinct weights (~100 MB > GPU cache)."""
import json, numpy as np, onnx
from onnx import helper as h, TensorProto as T
rng = np.random.default_rng(0)
P = 8; nodes, inits, layers = [], [], []
x = "A"
for i in range(2 * P):
    K, N = (2048, 6144) if i % 2 == 0 else (6144, 2048)
    q = rng.integers(0, 16, size=(N, K // 32, 32), dtype=np.uint8)
    packed = (q[..., 0::2] | (q[..., 1::2] << 4)).astype(np.uint8)
    scales = (rng.standard_normal((N, K // 32)) * (1.0 / (np.sqrt(K) * 4.6))).astype(np.float16)
    packed.tofile(f"onnx/gemm/chain_B{i}.bin"); scales.tofile(f"onnx/gemm/chain_S{i}.bin")
    inits += [h.make_tensor(f"B{i}", T.UINT8, packed.shape, packed.tobytes(), raw=True), h.make_tensor(f"S{i}", T.FLOAT16, scales.shape, scales.tobytes(), raw=True)]
    y = f"Y{i}"
    nodes.append(h.make_node("MatMulNBits", [x, f"B{i}", f"S{i}"], [y], domain="com.microsoft", K=K, N=N, bits=4, block_size=32, accuracy_level=4))
    layers.append({"K": K, "N": N}); x = y
g = h.make_graph(nodes, "chain", [h.make_tensor_value_info("A", T.FLOAT16, ["M", 2048])], [h.make_tensor_value_info(x, T.FLOAT16, ["M", 2048])], initializer=inits)
m = h.make_model(g, opset_imports=[h.make_opsetid("", 21), h.make_opsetid("com.microsoft", 1)]); m.ir_version = 10
onnx.save(m, "onnx/gemm/chain.onnx", save_as_external_data=True, location="chain.onnx.data", size_threshold=0)
json.dump({"layers": layers, "out": x}, open("onnx/gemm/chain.json", "w"))
print(len(nodes), "nodes")
