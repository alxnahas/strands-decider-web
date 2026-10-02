// Inference host: owns the ORT session, tokenizer and pointer head; answers {state, question} requests.
// Paths resolve against the site root (one level above js/), so the app also works under a subpath (GitHub Pages).
// ?assets=<url> loads the model files and engine weights from another origin (e.g. a Hugging Face repo).
const SITE = new URL("../", self.location);
const ORT_DIR = new URL(new URL(self.location).searchParams.get("ort") === "local" ? "ort-local/" : "ort/", SITE).href;
let ort;  // loaded in init(): a top-level await would delay onmessage and drop the init message
import { Tokenizer } from "../tokenizers/tokenizers.min.mjs";
import { buildInputs, readAnswer } from "./prompt.js";
import { loadHead, pointerLogits } from "./head.js";

const QS = new URL(self.location).searchParams;
const VARIANT = QS.get("model") || "q4f16p";
const BACKEND = QS.get("backend") || "ort";
const ASSETS = new URL(QS.get("assets") || "./", SITE);
const asset = (p) => new URL(p, ASSETS).href;   // "ort" (ONNX Runtime Web) or "engine" (hand-written WebGPU)
let engine = null;
const stateCache = new Map();   // engine backend: state-token key -> prefix snapshot (LRU, 4 entries)
function cachedState(sIds) {
  const key = sIds.join(","); const hit = stateCache.get(key);
  if (hit) { stateCache.delete(key); stateCache.set(key, hit); }
  return { key, hit };
}
function storeState(key, cache) {
  stateCache.set(key, cache);
  while (stateCache.size > 4) { const [k, c] = stateCache.entries().next().value; c.release(); stateCache.delete(k); }
}
const MODEL_REV = `v19-bb282d7-${VARIANT}-ortgenai0.17.1`;  // bump to invalidate the asset cache
let session, tokenizer, head, cfg, runtime;
const post = (type, data) => self.postMessage({ type, ...data });

async function fetchCached(url, label) {
  // OPFS: streamed to disk once, read back on later loads. Cache Storage refused a 1 GB put.
  const name = `${MODEL_REV}__${url.replace(ASSETS.href, "").replaceAll("/", "_")}`;  // keyed by asset path, not host
  let root = null;
  if (!QS.get("nocache")) try { root = await self.navigator.storage.getDirectory(); } catch {}
  if (root) {
    try {
      const fh = await root.getFileHandle(name);
      const file = await fh.getFile();
      if (file.size > 0) {
        const buf = new Uint8Array(await file.arrayBuffer());
        post("progress", { label, got: buf.length, total: buf.length, cached: true });
        return { buf, hit: true };
      }
    } catch {}
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = +res.headers.get("Content-Length") || 0;
  const buf = new Uint8Array(total); let got = 0, last = 0;
  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read(); if (done) break;
    buf.set(value, got); got += value.length;
    if (got - last > 64 << 20) { last = got; post("progress", { label, got, total }); }
  }
  if (root) {
    try {
      const fh = await root.getFileHandle(name + ".tmp", { create: true });
      const w = await fh.createWritable(); await w.write(buf); await w.close();
      await fh.move(name);
    } catch (e) { post("status", { msg: `OPFS write failed for ${label}: ${e}` }); }
  }
  post("progress", { label, got, total, cached: false });
  return { buf, hit: false };
}

async function init({ device = "webgpu" }) {
  const t0 = performance.now(), timings = {};
  const gpu = self.navigator.gpu ? await self.navigator.gpu.requestAdapter() : null;
  const adapterInfo = gpu ? { vendor: gpu.info?.vendor, architecture: gpu.info?.architecture, maxBufferSize: gpu.limits.maxBufferSize, maxStorageBufferBindingSize: gpu.limits.maxStorageBufferBindingSize, shaderF16: gpu.features.has("shader-f16") } : null;
  runtime = device === "webgpu" && gpu ? "webgpu" : "wasm";
  post("status", { msg: `runtime: ${runtime}${gpu ? "" : " (navigator.gpu unavailable)"}`, runtime, adapterInfo });

  let t = performance.now();
  const [tj, tc, cj, hb] = await Promise.all([
    fetch(asset("model/tokenizer.json")).then((r) => r.json()), fetch(asset("model/tokenizer_config.json")).then((r) => r.json()),
    fetch(asset("model/hobson_config.json")).then((r) => r.json()), fetchCached(asset("model/head.safetensors"), "head"),
  ]);
  tokenizer = new Tokenizer(tj, tc); cfg = cj; head = loadHead(hb.buf.buffer);
  timings.tokenizer_head_ms = performance.now() - t;

  if (BACKEND === "engine") {
    if (!gpu) throw new Error("engine backend needs WebGPU");
    const { Engine } = await import("../engine/engine.js");
    t = performance.now(); let hit = true;
    engine = await Engine.create(asset("engine-weights"), { fetchShard: async (url, label) => { const r = await fetchCached(url, label); hit &&= r.hit; return r.buf; } });
    timings.fetch_ms = performance.now() - t;
    t = performance.now(); await engine.forward([27, 2374, 29], [2]); timings.session_ms = performance.now() - t;  // compile + warm
    engine.precompile();  // remaining matmul tile pipelines, in the background
    timings.total_load_ms = performance.now() - t0; runtime = "webgpu-engine";
    post("ready", { variant: "engine-int4", runtime, timings, cached: hit, adapterInfo, kernels: engine.sgm ? "subgroup-matrix" : "portable" });
    return;
  }
  ort = await import(`${ORT_DIR}ort.webgpu.min.mjs`);
  ort.env.wasm.wasmPaths = ORT_DIR;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(8, self.navigator.hardwareConcurrency || 4) : 1;
  ort.env.logLevel = "warning";
  t = performance.now();
  const base = asset(`model/onnx/${VARIANT}-web`);
  const manifest = await fetch(`${base}/manifest.json`).then((r) => r.json());
  const graph = await fetchCached(`${base}/${manifest.graph}`, "graph");
  const shards = [];
  for (const s of manifest.shards) shards.push({ path: s.path, ...(await fetchCached(`${base}/${s.path}`, s.path)) });
  const data = { hit: shards.every((s) => s.hit), buf: { length: shards.reduce((n, s) => n + s.buf.length, 0) } };
  timings.fetch_ms = performance.now() - t;
  post("status", { msg: `assets ${data.hit ? "from OPFS cache" : "downloaded from localhost"} (${(data.buf.length / 2 ** 20).toFixed(0)} MiB)`, cached: data.hit });

  t = performance.now();
  session = await ort.InferenceSession.create(graph.buf, {
    executionProviders: [runtime === "webgpu" ? { name: "webgpu", ...JSON.parse(QS.get("ep") || "{}") } : runtime],
    externalData: shards.map((s) => ({ path: s.path, data: s.buf })),
    graphOptimizationLevel: "all",
    logSeverityLevel: QS.get("verbose") ? 0 : 2, logVerbosityLevel: QS.get("verbose") ? 1 : 0,
    enableProfiling: !!QS.get("profile"),
    // Recurrent/KV state stays on the GPU so a shared state prefix can feed later question suffixes.
    preferredOutputLocation: runtime === "webgpu" ? "gpu-buffer" : "cpu",
  });
  timings.session_ms = performance.now() - t;
  timings.total_load_ms = performance.now() - t0;
  post("ready", { variant: VARIANT, runtime, timings, cached: data.hit, adapterInfo, inputs: session.inputNames.length });
}

function zerosFor(name, L) {
  // Fresh (prefix-free) forward: empty KV cache, zero conv/recurrent state.
  if (name.includes(".conv")) return new ort.Tensor("float16", new Uint16Array(6144 * 3), [1, 6144, 3]);
  if (name.includes(".recurrent")) return new ort.Tensor("float16", new Uint16Array(16 * 128 * 128), [1, 16, 128, 128]);
  if (name.startsWith("past_key_values")) return new ort.Tensor("float16", new Uint16Array(0), [1, 2, 0, 256]);
  throw new Error(`unexpected input ${name}`);
}

const pastName = (o) => (o.endsWith(".key") || o.endsWith(".value") ? o.replace("present.", "past_key_values.") : o.replace("present.", "past."));
const i64 = (arr, dims) => new ort.Tensor("int64", BigInt64Array.from(arr, BigInt), dims);

/** One forward over `ids`, continuing from `past` (a prefix cache) or from empty state. */
async function forward(ids, past = null, start = 0) {
  const L = ids.length, feeds = {};
  for (const name of session.inputNames) {
    if (name === "input_ids") feeds[name] = i64(ids, [1, L]);
    else if (name === "attention_mask") feeds[name] = new ort.Tensor("int64", new BigInt64Array(start + L).fill(1n), [1, start + L]);
    else if (name === "position_ids") feeds[name] = i64(Array.from({ length: 3 * L }, (_, i) => start + (i % L)), [3, 1, L]);
    else feeds[name] = past ? past[name] : zerosFor(name, L);
  }
  const out = await session.run(feeds);
  const hs = out.hidden_states;
  const hidden = hs.location === "cpu" ? hs.data : await hs.getData(true);
  const cache = {};
  for (const [k, v] of Object.entries(out)) if (k !== "hidden_states") cache[pastName(k)] = v;
  return { hidden, d: hs.dims[2], cache };
}
const release = (cache) => Object.values(cache || {}).forEach((t) => t.dispose?.());

function answerFrom(rq, hidden, d, answerPos, optIdx) {
  const temp = cfg.temperature_by_kind[rq.kind] ?? cfg.temperature;
  const { logits, probs } = pointerLogits(head, hidden, d, answerPos, optIdx, temp);
  return { answer: readAnswer(rq, probs, cfg.ordinal_smoothing), logits };
}

async function decide({ id, state, question }) {
  const t0 = performance.now();
  const { ids, optIdx, rq } = buildInputs(tokenizer, state, question, cfg.max_length);
  const t1 = performance.now();
  if (engine) {
    const sLen = buildInputs(tokenizer, state, question, cfg.max_length).stateLen;
    const { hit } = cachedState(ids.slice(0, sLen));
    // State seen before: run only the question suffix from its snapshot.
    const r = hit
      ? await engine.forward(ids.slice(sLen), [...optIdx.map((i) => i - sLen), ids.length - sLen - 1], { cache: hit })
      : await engine.forward(ids, [...optIdx, ids.length - 1]);
    const t2e = performance.now();
    const { answer, logits } = answerFrom(rq, r.hidden, r.d, optIdx.length, optIdx.map((_, i) => i));
    post("result", { id, answer, logits, tokens: ids.length, cached_state_tokens: hit ? sLen : 0, timings: { prep_ms: t1 - t0, forward_ms: t2e - t1, total_ms: performance.now() - t0 }, runtime, kernel: r.kernel });
    return;
  }
  const { hidden, d, cache } = await forward(ids);
  release(cache);
  const t2 = performance.now();
  const { answer, logits } = answerFrom(rq, hidden, d, ids.length - 1, optIdx);
  if (QS.get("profile")) session.endProfiling();
  post("result", { id, answer, logits, tokens: ids.length, timings: { prep_ms: t1 - t0, forward_ms: t2 - t1, total_ms: performance.now() - t0 }, runtime });
}

/** Several questions about one state: encode the state once, then each question suffix from its cache. */
async function decideMany({ id, state, questions }) {
  const t0 = performance.now();
  const built = Object.entries(questions).map(([name, q]) => [name, buildInputs(tokenizer, state, q, cfg.max_length)]);
  if (engine) {
    const sLen = built[0][1].stateLen, sIds = built[0][1].ids.slice(0, sLen);
    let { key, hit } = cachedState(sIds);
    if (!hit) { hit = (await engine.forward(sIds, [], { keepCache: true })).cache; storeState(key, hit); }
    const t1 = performance.now(), answers = {}, per = {};
    for (const [name, b] of built) {
      const ts = performance.now(), suffix = b.ids.slice(sLen);
      const r = await engine.forward(suffix, [...b.optIdx.map((i) => i - sLen), suffix.length - 1], { cache: hit });
      answers[name] = answerFrom(b.rq, r.hidden, r.d, b.optIdx.length, b.optIdx.map((_, i) => i)).answer;
      per[name] = performance.now() - ts;
    }
    post("result", { id, answers, tokens: sLen + built.reduce((n, [, b]) => n + b.ids.length - sLen, 0), timings: { prefix_ms: t1 - t0, per_question_ms: per, total_ms: performance.now() - t0 }, runtime });
    return;
  }
  const sLen = built[0][1].stateLen;
  const prefix = await forward(built[0][1].ids.slice(0, sLen));
  const t1 = performance.now();
  const answers = {}, per = {};
  for (const [name, b] of built) {
    const ts = performance.now();
    const suffix = b.ids.slice(sLen);
    const { hidden, d, cache } = await forward(suffix, prefix.cache, sLen);
    release(cache);
    answers[name] = answerFrom(b.rq, hidden, d, suffix.length - 1, b.optIdx.map((i) => i - sLen)).answer;
    per[name] = performance.now() - ts;
  }
  release(prefix.cache);
  post("result", { id, answers, tokens: sLen + built.reduce((n, [, b]) => n + b.ids.length - sLen, 0), timings: { prefix_ms: t1 - t0, per_question_ms: per, total_ms: performance.now() - t0 }, runtime });
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "init") await init(data);
    else if (data.type === "decide") await decide(data);
    else if (data.type === "decideMany") await decideMany(data);
    else if (data.type === "bench") {  // raw forward latency for n tokens (diagnostics)
      const r = {};
      for (const n of data.lengths) { const ms = []; for (let i = 0; i < 4; i++) { const t = performance.now(); release((await forward(Array(n).fill(198))).cache); ms.push(performance.now() - t); } r[n] = ms.slice(1).map((x) => +x.toFixed(1)); }
      post("result", { id: data.id, bench: r });
    }
  } catch (e) { post("error", { id: data.id, error: String(e?.stack || e) }); }
};
