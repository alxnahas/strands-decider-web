"""Replace com.microsoft.GatherBlockQuantized (int4 embedding) with standard ONNX ops.

ORT Web's WASM build has no CPU kernel for GatherBlockQuantized, so the WASM fallback
cannot load the builder's graph. The int4 table's raw bytes are reinterpreted as uint8
[V, D/2] (element 2i in the low nibble), so the shards are reused untouched.
"""
import sys, onnx
from onnx import helper as h, TensorProto as T
src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src, load_external_data=False)
g = m.graph
node = next(n for n in g.node if n.op_type == "GatherBlockQuantized")
qname, ids, sname = node.input
block = next(a.i for a in node.attribute if a.name == "block_size")
init = next(i for i in g.initializer if i.name == qname)
assert init.data_type == T.INT4, init.data_type
V, D = init.dims
init.data_type = T.UINT8; del init.dims[:]; init.dims.extend([V, D // 2])
io = next(i for i in g.initializer if i.name == sname).data_type  # fp16 scales
c = lambda name, dt, dims, vals: h.make_tensor(name, dt, dims, vals)
P = "/model/embed_tokens/portable/"
consts = [c(P + "15", T.UINT8, [], [15]), c(P + "4", T.UINT8, [], [4]), c(P + "8u", T.UINT8, [], [8]),
          c(P + "8f", io, [], [8.0]), c(P + "ax", T.INT64, [1], [-1]),
          c(P + "s4", T.INT64, [4], [0, 0, D // block, block]), c(P + "s3", T.INT64, [3], [0, 0, D])]
for t in consts: g.initializer.append(t)
out = node.output[0]
new = [
    h.make_node("Gather", [qname, ids], [P + "g"], axis=0),
    h.make_node("BitwiseAnd", [P + "g", P + "15"], [P + "lo"]),
    h.make_node("BitShift", [P + "g", P + "4"], [P + "hi"], direction="RIGHT"),
    h.make_node("Unsqueeze", [P + "lo", P + "ax"], [P + "lo1"]),
    h.make_node("Unsqueeze", [P + "hi", P + "ax"], [P + "hi1"]),
    h.make_node("Concat", [P + "lo1", P + "hi1"], [P + "pair"], axis=-1),
    h.make_node("BitwiseXor", [P + "pair", P + "8u"], [P + "x"]),
    h.make_node("Cast", [P + "x"], [P + "xf"], to=io),
    h.make_node("Sub", [P + "xf", P + "8f"], [P + "q"]),          # signed int4 in [-8, 7]
    h.make_node("Reshape", [P + "q", P + "s4"], [P + "qb"]),
    h.make_node("Gather", [sname, ids], [P + "sc"], axis=0),
    h.make_node("Unsqueeze", [P + "sc", P + "ax"], [P + "sc1"]),
    h.make_node("Mul", [P + "qb", P + "sc1"], [P + "deq"]),
    h.make_node("Reshape", [P + "deq", P + "s3"], [out]),
]
i = list(g.node).index(node); g.node.remove(node)
for k, n in enumerate(new): g.node.insert(i + k, n)
onnx.save(m, dst)
print("rewrote", node.name, "->", len(new), "standard ops")
