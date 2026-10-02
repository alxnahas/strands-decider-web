// Static server for the local demo: COOP/COEP (for threaded WASM), Range support, no external calls.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
const ROOT = path.resolve(import.meta.dirname);
const MODEL_DIR = path.resolve(ROOT, "../onnx");
const ROUTES = [
  ["/ort/", path.join(ROOT, "node_modules/onnxruntime-web/dist")],
  ["/ort-local/", path.join(ROOT, "ort-local")],
  ["/tokenizers/", path.join(ROOT, "node_modules/@huggingface/tokenizers/dist")],
  ["/model/onnx/", MODEL_DIR],
  ["/gemm/", path.resolve(ROOT, "../onnx/gemm")],
  ["/engine-weights/", path.resolve(ROOT, "../engine")],
  ["/model/", path.resolve(ROOT, "../models/decider")],
  ["/", path.join(ROOT, "public")],
];
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".wasm": "application/wasm", ".css": "text/css" };
const port = Number(process.env.PORT || 8787);
http.createServer((req, res) => {
  const url = decodeURIComponent(new URL(req.url, "http://x").pathname);
  const [prefix, dir] = ROUTES.find(([p]) => url.startsWith(p));
  let file = path.join(dir, url.slice(prefix.length) || "index.html");
  if (fs.existsSync(file) && fs.statSync(file).isDirectory()) file = path.join(file, "index.html");
  if (!file.startsWith(dir) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404).end("not found"); return; }
  const size = fs.statSync(file).size;
  const headers = {
    "Content-Type": TYPES[path.extname(file)] || "application/octet-stream",
    "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp",
    "Accept-Ranges": "bytes", "Cache-Control": "no-cache",
  };
  const m = /bytes=(\d+)-(\d*)/.exec(req.headers.range || "");
  if (m) {
    const start = +m[1], end = m[2] ? +m[2] : size - 1;
    res.writeHead(206, { ...headers, "Content-Range": `bytes ${start}-${end}/${size}`, "Content-Length": end - start + 1 });
    fs.createReadStream(file, { start, end }).pipe(res);
  } else {
    res.writeHead(200, { ...headers, "Content-Length": size });
    if (req.method === "HEAD") res.end(); else fs.createReadStream(file).pipe(res);
  }
}).listen(port, "127.0.0.1", () => console.log(`http://127.0.0.1:${port}/  model=${MODEL_DIR}`));
