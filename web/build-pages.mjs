// Static build for GitHub Pages: the demo with the hand-written engine only (ORT Web and its 1.1 GB graph are not
// shipped). Writes <out>/site (small, goes to Pages) and <out>/assets (model files + engine weights, ~1.1 GB, goes to
// a Hugging Face repo). Usage: node build-pages.mjs --out dist --assets https://huggingface.co/<user>/<repo>/resolve/main/ [--site-only]
import fs from "node:fs";
import path from "node:path";

const arg = (k) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : null; };
const out = path.resolve(arg("out") || "dist"), assetsUrl = arg("assets"), siteOnly = process.argv.includes("--site-only");
if (!assetsUrl) throw new Error("--assets <base url of the model files> is required");
const here = path.dirname(new URL(import.meta.url).pathname), pub = path.join(here, "public");

fs.rmSync(out, { recursive: true, force: true });
const site = path.join(out, "site"), assets = path.join(out, "assets");
const copy = (from, to) => { fs.mkdirSync(path.dirname(to), { recursive: true }); fs.cpSync(from, to, { recursive: true }); };
// Weights are large: hard-link instead of copying (same volume), falling back to a copy.
const link = (from, to) => { fs.mkdirSync(path.dirname(to), { recursive: true }); try { fs.linkSync(from, to); } catch { fs.copyFileSync(from, to); } };

// site/
let html = fs.readFileSync(path.join(pub, "demo.html"), "utf8");
html = html.replace('<meta name="decider-assets" content="">', `<meta name="decider-assets" content="${assetsUrl}">`)
  .replace('<meta name="decider-backends" content="engine ort">', '<meta name="decider-backends" content="engine">');
if (!html.includes(assetsUrl)) throw new Error("demo.html assets meta tag not found");
fs.mkdirSync(site, { recursive: true });
fs.writeFileSync(path.join(site, "index.html"), html);
for (const f of ["decider-client.js", "worker.js", "prompt.js", "head.js"]) copy(path.join(pub, "js", f), path.join(site, "js", f));
for (const f of ["engine.js", "wgsl.js"]) copy(path.join(pub, "engine", f), path.join(site, "engine", f));
copy(path.join(pub, "gemmbench", "kernels.js"), path.join(site, "gemmbench", "kernels.js"));
copy(path.join(here, "node_modules/@huggingface/tokenizers/dist"), path.join(site, "tokenizers"));
fs.writeFileSync(path.join(site, ".nojekyll"), "");

// assets/ (skipped in CI, where the weights are not available; they are uploaded once from a local build)
if (!siteOnly) {
  const model = path.resolve(here, "../models/decider"), engine = path.resolve(here, "../engine");
  for (const f of ["tokenizer.json", "tokenizer_config.json", "hobson_config.json", "head.safetensors", "LICENSE.md"]) link(path.join(model, f), path.join(assets, "model", f));
  const manifest = JSON.parse(fs.readFileSync(path.join(engine, "manifest.json"), "utf8"));
  link(path.join(engine, "manifest.json"), path.join(assets, "engine-weights", "manifest.json"));
  for (const s of manifest.shards) link(path.join(engine, s.path), path.join(assets, "engine-weights", s.path));
}

const size = (d) => fs.readdirSync(d, { recursive: true }).reduce((n, f) => n + (fs.statSync(path.join(d, f)).isFile() ? fs.statSync(path.join(d, f)).size : 0), 0);
console.log(`site   ${(size(site) / 2 ** 20).toFixed(1)} MiB  ${site}`);
if (!siteOnly) console.log(`assets ${(size(assets) / 2 ** 20).toFixed(0)} MiB  ${assets}  (served from ${assetsUrl})`);
