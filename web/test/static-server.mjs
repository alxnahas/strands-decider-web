// Plain static file server for tests that mimic production hosting: GitHub Pages (subpath, no custom headers)
// and Hugging Face (another origin, CORS). Usage: startStatic({ root, port, prefix = "/", cors = false }).
import http from "node:http";
import fs from "node:fs";
import path from "node:path";

const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".wasm": "application/wasm", ".css": "text/css" };

export function startStatic({ root, port, prefix = "/", cors = false }) {
  const server = http.createServer((req, res) => {
    const url = decodeURIComponent(new URL(req.url, "http://x").pathname);
    if (!url.startsWith(prefix)) { res.writeHead(404).end(); return; }
    let file = path.join(root, url.slice(prefix.length));
    if (fs.existsSync(file) && fs.statSync(file).isDirectory()) file = path.join(file, "index.html");
    if (!file.startsWith(root) || !fs.existsSync(file)) { res.writeHead(404).end(); return; }
    const headers = { "Content-Type": TYPES[path.extname(file)] || "application/octet-stream", "Content-Length": fs.statSync(file).size };
    if (cors) headers["Access-Control-Allow-Origin"] = "*";
    res.writeHead(200, headers);
    fs.createReadStream(file).pipe(res);
  });
  return new Promise((resolve) => server.listen(port, "127.0.0.1", () => resolve(server)));
}
