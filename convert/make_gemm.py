"""Single-node MatMulNBits graphs (int4, block 32, fp16 io) + raw weights for the WGSL kernel bench."""
import json, numpy as np, onnx
from onnx import helper as h, TensorProto as T
rng = np.random.default_rng(0)
shapes = [(2048, 6144), (2048, 2048), (6144, 2048), (2048, 12288)]
for K, N in shapes:
    blocks = K // 32
    q = rng.integers(0, 16, size=(N, blocks, 32), dtype=np.uint8)          # uint4 with zero point 8 (symmetric int4)
    packed = (q[..., 0::2] | (q[..., 1::2] << 4)).astype(np.uint8)          # [N, blocks, 16], element 2i low nibble
    scales = (rng.standard_normal((N, blocks)) * 0.02).astype(np.float16)
    node = h.make_node("MatMulNBits", ["A", "B", "S"], ["Y"], domain="com.microsoft", K=K, N=N, bits=4, block_size=32, accuracy_level=4)
    g = h.make_graph([node], "mmnb", [h.make_tensor_value_info("A", T.FLOAT16, ["M", K])], [h.make_tensor_value_info("Y", T.FLOAT16, ["M", N])],
                     initializer=[h.make_tensor("B", T.UINT8, packed.shape, packed.tobytes(), raw=True), h.make_tensor("S", T.FLOAT16, scales.shape, scales.tobytes(), raw=True)])
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 21), h.make_opsetid("com.microsoft", 1)]); m.ir_version = 10
    onnx.save(m, f"onnx/gemm/mmnb_K{K}_N{N}.onnx")
    packed.tofile(f"onnx/gemm/B_K{K}_N{N}.bin"); scales.tofile(f"onnx/gemm/S_K{K}_N{N}.bin")
json.dump([{"K": K, "N": N} for K, N in shapes], open("onnx/gemm/shapes.json", "w"))
print("ok")
