// Plain static file server for tests that mimic production hosting: GitHub Pages (subpath, no custom headers)
// and Hugging Face (another origin, CORS, byte ranges). Usage: startStatic({ root, port, prefix = "/", cors = false }).
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
    const size = fs.statSync(file).size, headers = { "Content-Type": TYPES[path.extname(file)] || "application/octet-stream", "Content-Length": size, "Accept-Ranges": "bytes" };
    if (cors) Object.assign(headers, { "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Headers": "range", "Access-Control-Expose-Headers": "content-range" });
    if (req.method === "OPTIONS") { res.writeHead(204, headers).end(); return; }
    const m = /^bytes=(\d+)-(\d*)$/.exec(req.headers.range || "");
    if (m) {  // single ranges, as Hugging Face serves them (on-demand embedding rows)
      const start = +m[1], end = Math.min(m[2] ? +m[2] : size - 1, size - 1);
      res.writeHead(206, { ...headers, "Content-Range": `bytes ${start}-${end}/${size}`, "Content-Length": end - start + 1 });
      fs.createReadStream(file, { start, end }).pipe(res);
      return;
    }
    res.writeHead(200, headers);
    fs.createReadStream(file).pipe(res);
  });
  return new Promise((resolve) => server.listen(port, "127.0.0.1", () => resolve(server)));
}
