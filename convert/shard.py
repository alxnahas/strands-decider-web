"""Rewrite an ONNX model's external data into <= MAX-byte shards + manifest.json for the browser.

Browsers cap single ArrayBuffer allocations (~2 GB in Chrome workers), so a 2.5 GB
weights file cannot be handed to ORT Web in one piece.
"""
import json, os, sys, onnx
from onnx.external_data_helper import ExternalDataInfo
src, dst = sys.argv[1], sys.argv[2]
MAX = int(sys.argv[3]) if len(sys.argv) > 3 else 512 << 20
m = onnx.load(f"{src}/model.onnx", load_external_data=False)
os.makedirs(dst, exist_ok=True)
fin = open(f"{src}/model.onnx.data", "rb")
shards, cur, size = [], None, 0
for t in m.graph.initializer:
    if t.data_location != onnx.TensorProto.EXTERNAL: continue
    info = ExternalDataInfo(t)
    if cur is None or size + info.length > MAX:
        if cur: cur.close()
        shards.append(f"model.onnx.data.{len(shards)}"); cur = open(f"{dst}/{shards[-1]}", "wb"); size = 0
    pad = (-size) % 64; cur.write(b"\0" * pad); size += pad
    fin.seek(info.offset); cur.write(fin.read(info.length))
    del t.external_data[:]
    for k, v in (("location", shards[-1]), ("offset", str(size)), ("length", str(info.length))):
        e = t.external_data.add(); e.key, e.value = k, v
    size += info.length
cur.close()
onnx.save(m, f"{dst}/model.onnx")
manifest = {"graph": "model.onnx", "shards": [{"path": s, "bytes": os.path.getsize(f"{dst}/{s}")} for s in shards]}
json.dump(manifest, open(f"{dst}/manifest.json", "w"), indent=1)
print(json.dumps(manifest))
