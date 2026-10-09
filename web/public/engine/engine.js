// Hand-written WebGPU prefill engine for the Strands Decider torso (Qwen3.5-2B hybrid, 2-4 bit weights).
import * as W from "./wgsl.js";
import { decodeScales, unpackShard } from "./wire.js";

const U = GPUBufferUsage;

/**
 * Matmul kernel choice by M (measured on M4 Pro; see gemmbench/chain.html). Without subgroup-matrix support
 * (stock Chrome exposes it only behind --enable-unsafe-webgpu) every M uses the portable multi-row GEMV.
 */
function matmulKernel(M, force = null, sgm = true) {
  const v1 = { key: "v1", code: (fmt) => W.kernelV1({ fmt }), groups: (M, N) => [N / 16, Math.ceil(M / 8), 1], SK: 1 };
  if (!sgm) return v1;
  if (force) { const [TM, SK] = force; if (TM === 0) return v1;
    return { key: `v4_${TM}_${SK}`, code: (fmt) => W.kernelV4({ TM, SK, KB: 2, fmt }), groups: (M, N) => [N / 64, Math.ceil(M / TM), SK], SK }; }
  if (M <= 8) return v1;
  // Tuned on M4 Pro (engine/tune.html): one exact-height tile (multiple of 16) up to 96 rows, then 64-row tiles.
  let TM, SK;
  if (M <= 96) [TM, SK] = [Math.ceil(M / 16) * 16, 4];
  else if (M <= 128) [TM, SK] = [64, 4];
  else if (M <= 256) { TM = Math.ceil(M / 32) * 32 - M < Math.ceil(M / 64) * 64 - M - 16 ? 32 : 64; SK = TM === 32 ? 2 : 1; }
  else [TM, SK] = [64, 1];
  return { key: `v4_${TM}_${SK}`, code: (fmt) => W.kernelV4({ TM, SK, KB: 2, fmt }), groups: (M, N) => [N / 64, Math.ceil(M / TM), SK], SK };
}

/** A matmul weight's format (see gemmbench/kernels.js); manifests without one are int4, block 32, symmetric. */
const fmtOf = (t) => ({ bits: t.bits ?? 4, group: t.group ?? 32, asym: !!t.asym });

/** Bytes of a file; a .gz one decompressed (Hugging Face serves it as is, without Content-Encoding). */
async function download(url) {
  const r = await fetch(url); if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return new Uint8Array(await (url.endsWith(".gz") ? new Response(r.body.pipeThrough(new DecompressionStream("gzip"))) : r).arrayBuffer());
}

/** In-place fast Walsh-Hadamard transform (Sylvester order) of x[o .. o + n). */
function fwht(x, o, n) {
  for (let h = 1; h < n; h *= 2) for (let i = o; i < o + n; i += 2 * h) for (let j = i; j < i + h; j++) { const a = x[j], b = x[j + h]; x[j] = a + b; x[j + h] = a - b; }
}

/**
 * Why `adapter` cannot run the engine (empty if it can). The WGSL assumes 32-wide subgroups (one lane per 4 key dims
 * in the gated-delta kernel, 8 subgroups per 256-thread norm), so an adapter that may pick another size is refused.
 */
export function engineProblems(adapter) {
  const p = [];
  for (const f of ["shader-f16", "subgroups"]) if (!adapter.features.has(f)) p.push(`no ${f}`);
  const { subgroupMinSize: lo, subgroupMaxSize: hi } = adapter.info ?? {};
  if (lo !== 32 || hi !== 32) p.push(`subgroup size ${lo ?? "?"}-${hi ?? "?"} (needs exactly 32)`);
  return p;
}

export class Engine {
  static async create(base = "/engine-weights", { onProgress = () => {}, fetchShard } = {}) {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    const problems = engineProblems(adapter);
    if (problems.length) throw new Error(`this GPU can't run the WebGPU engine: ${problems.join(", ")}`);
    const want = ["shader-f16", "subgroups", "chromium-experimental-subgroup-matrix", "timestamp-query"];
    const device = await adapter.requestDevice({
      requiredFeatures: want.filter((f) => adapter.features.has(f)),
      requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize, maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup },
    });
    const manifest = await fetch(`${base}/manifest.json`).then((r) => r.json());
    const shards = [];
    for (const s of manifest.shards) {
      // a packed shard ("wire", pack_wire.py) is rebuilt into the GPU layout here
      const data = fetchShard ? await fetchShard(`${base}/${s.path}`, s.path, s.wire) : await download(`${base}/${s.path}`);
      const buf = device.createBuffer({ size: Math.ceil((s.wire ? s.bytes : data.byteLength) / 4) * 4, usage: U.STORAGE | U.COPY_DST });
      const put = (off, b) => { if (b.byteLength % 4) { const p = new Uint8Array(Math.ceil(b.byteLength / 4) * 4); p.set(b); b = p; }
        device.queue.writeBuffer(buf, off, b.buffer, b.byteOffset, b.byteLength); };
      if (s.wire) unpackShard(data, s, manifest.tensors, put); else put(0, data);
      shards.push({ ...s, buf }); onProgress(s.path);
    }
    const e = manifest.embed, bundle = e && (fetchShard ? await fetchShard(`${base}/${e.bundle}`, e.bundle) : await download(`${base}/${e.bundle}`));
    return new Engine(device, manifest, shards, bundle && { url: new URL(`${base}/${e.rows}`, self.location.href).href, bundle });
  }

  constructor(device, manifest, shards, lazy = null) {
    this.d = device; this.cfg = manifest.config; this.sgm = device.features.has("chromium-experimental-subgroup-matrix"); this.tensors = manifest.tensors; this.shards = shards;
    const c = this.cfg; this.LT = 4096;
    this.linear = c.layer_types.map((k, i) => (k === "linear_attention" ? i : -1)).filter((i) => i >= 0);
    this.full = c.layer_types.map((k, i) => (k === "full_attention" ? i : -1)).filter((i) => i >= 0);
    const C = 2 * c.linear_num_key_heads * c.linear_key_head_dim + c.linear_num_key_heads * c.linear_value_head_dim;
    this.sizes = { state: c.linear_num_key_heads * c.linear_key_head_dim * c.linear_value_head_dim * 4, hist: 3 * C * 2, kvRow: c.head_dim * 2, HK: c.num_key_value_heads };
    const SU = U.STORAGE | U.COPY_SRC | U.COPY_DST;
    this.work = {
      state: Object.fromEntries(this.linear.map((i) => [i, this.buf(this.sizes.state, SU)])),
      hist: Object.fromEntries(this.linear.map((i) => [i, this.buf(this.sizes.hist, SU)])),
      K: Object.fromEntries(this.full.map((i) => [i, this.buf(this.sizes.HK * this.LT * this.sizes.kvRow, SU)])),
      V: Object.fromEntries(this.full.map((i) => [i, this.buf(this.sizes.HK * this.LT * this.sizes.kvRow, SU)])),
      Lu: this.buf(16, U.UNIFORM | U.COPY_DST),
    };
    const RD = c.rotary_dim, rope = new Float32Array(this.LT * RD);
    for (let t = 0; t < this.LT; t++) for (let i = 0; i < RD / 2; i++) { const a = t / c.rope_theta ** ((2 * i) / RD); rope[t * RD + 2 * i] = Math.cos(a); rope[t * RD + 2 * i + 1] = Math.sin(a); }
    this.rope = this.buf(rope.byteLength, U.STORAGE | U.COPY_DST); device.queue.writeBuffer(this.rope, 0, rope);
    this.pipes = new Map(); this.plans = new Map();
    this.errors = []; device.addEventListener("uncapturederror", (e) => this.errors.push(e.error.message));
    if (lazy) this.initEmbed(manifest.embed, lazy);
  }

  /**
   * Lazy embedding (manifest.embed, export_variant.py --lazy-embed): the bundled rows sit in slots 0..R-1 of
   * per-part buffers with room for `slots` more; any other token's row is fetched by byte range from the full table
   * on first use. Token ids are mapped to slots before each forward, so the EMBED kernel is unchanged.
   */
  initEmbed(e, { url, bundle }) {
    const R = e.bundled, ids = new Uint32Array(bundle.buffer, bundle.byteOffset, R);
    const E = this.emb = { url, cdn: null, e, R, next: R, dyn: [], slot: new Int32Array(e.vocab).fill(-1), bufs: {}, fetched: 0, requests: 0 };
    for (const [name, bytes] of e.parts) E.bufs[name] = { buf: this.buf((R + e.slots) * bytes, U.STORAGE | U.COPY_DST), bytes };
    this.putRows(bundle.subarray(R * 4), 0, R);
    ids.forEach((t, i) => { E.slot[t] = i; });
  }

  /** n downloaded rows (e.wire_row_bytes each with 8-bit scales if e.u8, else e.row_bytes) into slots s0.. */
  putRows(data, s0, n) {
    const e = this.emb.e, RB = e.wire_row_bytes ?? e.row_bytes; let off = 0;
    for (const [name, bytes] of e.parts) {
      const part = new Uint8Array(n * bytes);
      if (e.u8 && name === "embed.s") {
        const G = bytes / 2, d = new Uint16Array(part.buffer);
        for (let r = 0; r < n; r++) decodeScales(data, r * RB + off, 1, G, d, r * G);
        off += 4 + G;
      } else {
        for (let r = 0; r < n; r++) part.set(data.subarray(r * RB + off, r * RB + off + bytes), r * bytes);
        off += bytes;
      }
      this.d.queue.writeBuffer(this.emb.bufs[name].buf, s0 * bytes, part);
    }
  }

  /** Engine row indices for token ids: identity without a lazy embedding, else slots (fetching missing rows). */
  async embedRows(ids) {
    const E = this.emb; if (!E) return ids;
    let need = [...new Set(ids)].filter((t) => E.slot[t] < 0);
    if (E.next + need.length > E.R + E.e.slots) { for (const t of E.dyn) E.slot[t] = -1; E.dyn = []; E.next = E.R; need = [...new Set(ids)].filter((t) => E.slot[t] < 0); }
    need.sort((a, b) => a - b);
    const runs = [];  // ranges of ids, merged across gaps of up to 8 rows (cheaper than another request)
    for (const t of need) { if (runs.length && t - runs.at(-1)[1] <= 8) runs.at(-1)[1] = t; else runs.push([t, t]); }
    const want = new Set(need), RB = E.e.wire_row_bytes ?? E.e.row_bytes;
    await Promise.all(runs.map(async ([a, b]) => {
      let res;
      for (let tries = 0; ; tries++) {
        // Hugging Face redirects each request to a signed CDN URL; reusing it saves a round trip (until it expires).
        const url = (tries === 0 && E.cdn) || E.url;
        try {
          res = await fetch(url, { headers: { Range: `bytes=${a * RB}-${(b + 1) * RB - 1}` } });
          if (res.status === 206) { if (res.redirected) E.cdn = res.url; break; }
          throw new Error(`HTTP ${res.status}`);
        } catch (err) { if (url === E.cdn) E.cdn = null; if (tries === 2) throw new Error(`embedding rows ${a}-${b}: ${err.message}`); }
      }
      const data = new Uint8Array(await res.arrayBuffer()), keep = [];
      for (let t = a; t <= b; t++) if (want.has(t)) keep.push(t);
      const rows = new Uint8Array(keep.length * RB), s0 = E.next; E.next += keep.length;
      keep.forEach((t, i) => { rows.set(data.subarray((t - a) * RB, (t - a + 1) * RB), i * RB); E.slot[t] = s0 + i; E.dyn.push(t); });
      this.putRows(rows, s0, keep.length);
      E.requests++;
    }));
    E.fetched += need.length;
    return ids.map((t) => E.slot[t]);
  }

  /** Binding resource for a named weight tensor (a range of one shard buffer). */
  w(name) {
    const lz = this.emb?.bufs[name]; if (lz) return { buffer: lz.buf, offset: 0, size: lz.buf.size };
    const t = this.tensors[name]; if (!t) throw new Error(`no tensor ${name}`);
    const s = this.shards.find((s) => t.offset >= s.start && t.offset < s.start + s.bytes);
    return { buffer: s.buf, offset: t.offset - s.start, size: Math.ceil(t.bytes / 4) * 4 };
  }
  pipe(key, code) {
    if (!this.pipes.has(key)) this.pipes.set(key, this.d.createComputePipeline({ layout: "auto", compute: { module: this.d.createShaderModule({ code: code() }), entryPoint: "main" } }));
    return this.pipes.get(key);
  }
  buf(size, usage = U.STORAGE) { return this.d.createBuffer({ size: Math.max(16, Math.ceil(size / 16) * 16), usage }); }
  uniform(words) { const b = this.buf(16, U.UNIFORM | U.COPY_DST); this.d.queue.writeBuffer(b, 0, new Uint32Array(words)); return b; }

  /** Activation buffers for up to `cap` tokens, created once per capacity bucket. */
  pool(cap, make, dims) {
    this.pools ??= new Map();
    if (!this.pools.has(cap)) {
      const L = cap;
      const B = make.call(this, L);
      // split-K partials: SK <= 4 up to 128 rows, <= 2 up to 256, none beyond (see matmulKernel)
      B.part = this.buf(cap <= 128 ? 4 * cap * dims.maxN * 2 : cap <= 256 ? 2 * cap * dims.maxN * 2 : 16); this.pools.set(cap, B);
    }
    return this.pools.get(cap);
  }

  /** Compile the matmul pipelines the tuned table can pick, without blocking the first request. */
  async precompile() {
    const keys = new Map(), fmts = new Map(Object.entries(this.tensors).filter(([n]) => n.endsWith(".q") && n !== "embed.q").map(([, t]) => [W.fmtKey(fmtOf(t)), fmtOf(t)]));
    for (const M of [8, 16, 32, 48, 64, 80, 96, 128, 160, 256, 512]) for (const [fk, f] of fmts) { const k = matmulKernel(M, null, this.sgm); keys.set(k.key + fk, () => k.code(f)); }
    await Promise.all([...keys].map(async ([key, code]) => {
      if (this.pipes.has(key)) return;
      this.pipes.set(key, await this.d.createComputePipelineAsync({ layout: "auto", compute: { module: this.d.createShaderModule({ code: code() }), entryPoint: "main" } }));
    }));
  }

  /** Build (and cache) the dispatch list for sequence length L. */
  plan(L, force = null) {
    const pk = `${L}/${force}`;
    if (this.plans.has(pk)) return this.plans.get(pk);
    const c = this.cfg, D = c.hidden_size, I = c.intermediate_size, H = c.linear_num_key_heads, DK = c.linear_key_head_dim, DV = c.linear_value_head_dim;
    const HQ = c.num_attention_heads, HK = c.num_key_value_heads, HD = c.head_dim, RD = c.rotary_dim, eps = c.rms_norm_eps;
    const inN = this.tensors[`layers.${this.linear[0]}.in_proj.q`].N, qkvN = this.tensors[`layers.${this.full[0]}.qkv.q`].N;
    const maxN = Math.max(2 * I, inN, qkvN);
    const cap = L <= 128 ? 128 : 2 ** Math.ceil(Math.log2(L));   // activations sized for a bucket, shared by all L in it
    const B = this.pool(cap, (L) => ({
      ids: this.buf(L * 4, U.STORAGE | U.COPY_DST), x: this.buf(L * D * 4, U.STORAGE | U.COPY_SRC), h: this.buf(L * D * 2),
      big: this.buf(L * maxN * 2), mid: this.buf(L * I * 2), qkvc: this.buf(L * 2 * H * DK * 4 + L * H * DV * 4), of: this.buf(L * H * DV * 4),
      o: this.buf(L * D * 2), y: this.buf(L * D * 2), Q: this.buf(HQ * L * HD * 2), K: this.buf(HK * L * HD * 2),
      rowsAll: this.buf(L * 4, U.STORAGE | U.COPY_DST),
      rowsOut: this.buf(L * 4, U.STORAGE | U.COPY_DST), out: this.buf(L * D * 4, U.STORAGE | U.COPY_SRC),
    }), { maxN, D, I, H, DK, DV, HQ, HK, HD, RD });
    const mk = matmulKernel(L, force, this.sgm);
    if (mk.SK > 1 && B.part.size < mk.SK * L * maxN * 2) throw new Error(`split-K scratch too small for L=${L}`);
    this.d.queue.writeBuffer(B.rowsAll, 0, Uint32Array.from({ length: L }, (_, i) => i));
    const Lu = this.work.Lu;

    const steps = [];
    const bind = (p, entries) => this.d.createBindGroup({ layout: p.getBindGroupLayout(0), entries: Object.entries(entries).map(([binding, r]) => ({ binding: +binding, resource: r.buffer ? r : { buffer: r } })) });
    const step = (key, code, entries, groups, label) => { const p = this.pipe(key, code); steps.push({ p, bg: bind(p, entries), groups, label }); };
    const slice = (buf, bytes) => ({ buffer: buf, offset: 0, size: Math.ceil(bytes / 4) * 4 });
    const matmul = (A, wname, Y, label) => {
      const q = this.tensors[wname + ".q"], N = q.N, K = q.K, P = this.uniform([L, N, K, K / 32]), f = fmtOf(q);
      if (q.had) step(`fwht${q.had}`, () => W.FWHT(q.had), { 0: slice(A, L * K * 2) }, [L * K / q.had, 1, 1], label + "/had");  // A is not read again
      const e = { 0: slice(A, L * K * 2), 1: this.w(wname + ".q"), 2: this.w(wname + ".s"), 3: slice(Y, L * N * 2), 4: P };
      if (mk.SK > 1) e[5] = B.part;
      if (f.asym) e[6] = this.w(wname + ".b");
      step(mk.key + W.fmtKey(f), () => mk.code(f), e, mk.groups(L, N), label);
      if (mk.SK > 1) step(`reduce${mk.SK}`, () => W.REDUCE(mk.SK), { 3: slice(Y, L * N * 2), 4: P, 5: B.part }, [Math.ceil((L * N) / 256), 1, 1], label + "/reduce");
      return N;
    };
    const addNorm = (add, wname, label) => step(`addnorm${add}`, () => W.ADD_NORM(D, add), { 0: B.x, 1: B.y, 2: this.w(wname), 3: B.h, 4: B.rowsAll }, [L, 1, 1], label);
    void eps;

    const ef = fmtOf(this.tensors["embed.q"]), ee = { 0: B.ids, 1: this.w("embed.q"), 2: this.w("embed.s"), 3: B.x };
    if (ef.asym) ee[4] = this.w("embed.b");
    step("embed" + W.fmtKey(ef), () => W.EMBED(D, ef), ee, [L, 1, 1], "embed");
    c.layer_types.forEach((kind, i) => {
      const P = `layers.${i}.`;
      addNorm(i > 0, P + "in_norm", `L${i}/in_norm`);
      if (kind === "adapter") {  // linear stand-in for removed layers: x += proj(rms_norm(x)) + bias
        matmul(B.h, P + "proj", B.y, `L${i}/proj`);
        const n = L * D, gx = Math.min(Math.ceil(n / 256), 32768);
        step(`bias${D}`, () => W.ADD_BIAS(D), { 0: slice(B.y, n * 2), 1: this.w(P + "bias") }, [gx, Math.ceil(Math.ceil(n / 256) / gx), 1], `L${i}/bias`);
        return;
      }
      if (kind === "linear_attention") {
        const N = matmul(B.h, P + "in_proj", B.big, `L${i}/in_proj`);
        const CC = 2 * H * DK + H * DV;
        step(`conv${N}`, () => W.CONV(CC, N, L), { 0: B.big, 1: this.w(P + "conv"), 2: B.qkvc, 3: this.work.hist[i] }, [Math.ceil(CC / 256), L, 1], `L${i}/conv`);
        step(`hist${N}`, () => W.HIST_UPDATE(CC, N), { 0: B.big, 1: this.work.hist[i], 2: Lu }, [Math.ceil(CC / 256), 1, 1], `L${i}/hist`);
        step(`gdn${N}`, () => W.GDN({ H, DK, DV, L: 0, stride: N, aOff: 2 * H * DK + H * DV + H * DV, bOff: 2 * H * DK + H * DV + H * DV + H }),
          { 0: B.qkvc, 1: B.big, 2: this.w(P + "neg_exp_A"), 3: this.w(P + "dt_bias"), 4: B.of, 5: Lu, 6: this.work.state[i] }, [H * DV / 32, 1, 1], `L${i}/gdn`);
        step(`gnorm${N}`, () => W.GNORM({ H, DV, stride: N, zOff: 2 * H * DK + H * DV }), { 0: B.of, 1: B.big, 2: this.w(P + "gnorm"), 3: B.o }, [L * H / 4, 1, 1], `L${i}/gnorm`);
        matmul(B.o, P + "out_proj", B.y, `L${i}/out_proj`);
      } else {
        const N = matmul(B.h, P + "qkv", B.big, `L${i}/qkv`);
        step(`qkprep${N}`, () => W.QK_PREP({ HQ, HK, HD, RD, stride: N }), { 0: B.big, 1: this.w(P + "q_norm"), 2: this.w(P + "k_norm"), 3: this.rope, 4: B.Q, 5: this.work.K[i], 6: Lu, 7: this.work.V[i] }, [L, HQ + HK, 1], `L${i}/qk`);
        step(`attn2_${N}`, () => W.ATTN2({ HQ, HK, HD, stride: N, vOff: HQ * HD * 2 + HK * HD }), { 0: B.Q, 1: this.work.K[i], 2: B.big, 3: B.o, 4: Lu, 5: this.work.V[i] }, [Math.ceil(L / 16), HQ, 1], `L${i}/attn`);
        matmul(B.o, P + "o_proj", B.y, `L${i}/o_proj`);
      }
      addNorm(true, P + "post_norm", `L${i}/post_norm`);
      matmul(B.h, P + "gate_up", B.big, `L${i}/gate_up`);
      const n = L * I, gx = Math.min(Math.ceil(n / 256), 32768);
      step("silu", () => W.SILU_MUL(I), { 0: slice(B.big, L * 2 * I * 2), 1: slice(B.mid, n * 2) }, [gx, Math.ceil(Math.ceil(n / 256) / gx), 1], `L${i}/silu`);
      matmul(B.mid, P + "down", B.y, `L${i}/down`);
    });
    const finalStep = { p: this.pipe("finalnorm", () => W.ADD_NORM(D, true, true)) };
    finalStep.bg = bind(finalStep.p, { 0: B.x, 1: B.y, 2: this.w("final_norm"), 3: B.out, 4: B.rowsOut });
    const plan = { L, B, steps, finalStep, kernel: mk.key };
    this.plans.set(pk, plan);
    return plan;
  }

  /** Run the torso on ids; returns final-normed hidden states (f32) for `rows` (default: all). */
  /** Per-step GPU time (one compute pass per dispatch, timestamped). Diagnostics only. */
  async profile(ids, force = null) {
    const L = ids.length, plan = this.plan(L, force), { B } = plan;
    this.d.queue.writeBuffer(B.ids, 0, Uint32Array.from(await this.embedRows(ids)));
    const n = plan.steps.length, qs = this.d.createQuerySet({ type: "timestamp", count: 2 * n });
    const enc = this.d.createCommandEncoder();
    plan.steps.forEach((s, i) => {
      const pass = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } });
      pass.setPipeline(s.p); pass.setBindGroup(0, s.bg); pass.dispatchWorkgroups(...s.groups); pass.end();
    });
    const r = this.buf(16 * n, U.QUERY_RESOLVE | U.COPY_SRC), m = this.buf(16 * n, U.COPY_DST | U.MAP_READ);
    enc.resolveQuerySet(qs, 0, 2 * n, r, 0); enc.copyBufferToBuffer(r, 0, m, 0, 16 * n); this.d.queue.submit([enc.finish()]);
    await m.mapAsync(GPUMapMode.READ); const t = new BigUint64Array(m.getMappedRange().slice(0)); m.unmap();
    const by = {};
    plan.steps.forEach((s, i) => { const k = s.label.replace(/^L\d+\//, ""); by[k] = (by[k] || 0) + Number(t[2 * i + 1] - t[2 * i]) / 1e6; });
    return Object.fromEntries(Object.entries(by).sort((a, b) => b[1] - a[1]).map(([k, v]) => [k, +v.toFixed(2)]));
  }

  /**
   * Run the torso on `ids`, continuing from `cache` (a prefix snapshot) when given. Returns final-normed hidden
   * states (f32) for `rows` (indices into `ids`), and with keepCache a snapshot covering cache.ids ++ ids.
   */
  async forward(ids, rows = null, { debugLayers = false, timestamps = false, force = null, cache = null, keepCache = false } = {}) {
    const L = ids.length, plan = this.plan(L, force), { B } = plan, D = this.cfg.hidden_size;
    const P = cache ? cache.P : 0;
    if (P + L > this.LT) throw new Error(`context ${P + L} exceeds ${this.LT}`);
    rows ??= Array.from({ length: L }, (_, i) => i);
    const tf = performance.now();
    this.d.queue.writeBuffer(B.ids, 0, Uint32Array.from(await this.embedRows(ids)));
    const fetchMs = performance.now() - tf;
    this.d.queue.writeBuffer(B.rowsOut, 0, Uint32Array.from(rows.length ? rows : [0]));
    this.d.queue.writeBuffer(this.work.Lu, 0, new Uint32Array([L, P, this.LT, cache ? 1 : 0]));
    const enc = this.d.createCommandEncoder();
    const kvBytes = (n) => n * this.sizes.kvRow, headStride = this.LT * this.sizes.kvRow;
    if (cache) {
      for (const i of this.linear) { enc.copyBufferToBuffer(cache.state[i], 0, this.work.state[i], 0, this.sizes.state); enc.copyBufferToBuffer(cache.hist[i], 0, this.work.hist[i], 0, this.sizes.hist); }
      for (const i of this.full) for (const kv of ["K", "V"]) for (let h = 0; h < this.sizes.HK; h++) enc.copyBufferToBuffer(cache[kv][i], h * kvBytes(P), this.work[kv][i], h * headStride, kvBytes(P));
    } else for (const i of this.linear) enc.clearBuffer(this.work.hist[i]);
    const dbg = [];
    let qs = null;
    if (timestamps && this.d.features.has("timestamp-query")) qs = this.d.createQuerySet({ type: "timestamp", count: 2 });
    let pass = enc.beginComputePass(qs ? { timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0 } } : {});
    for (const s of plan.steps) {
      if (debugLayers && s.label.endsWith("/in_norm") && s.label !== "L0/in_norm") {
        // x after the fused residual add at the start of this layer = output of the previous layer.
        pass.setPipeline(s.p); pass.setBindGroup(0, s.bg); pass.dispatchWorkgroups(...s.groups); pass.end();
        const b = this.buf(L * D * 4, U.COPY_DST | U.MAP_READ); enc.copyBufferToBuffer(B.x, 0, b, 0, L * D * 4); dbg.push(b);
        pass = enc.beginComputePass(); continue;
      }
      pass.setPipeline(s.p); pass.setBindGroup(0, s.bg); pass.dispatchWorkgroups(...s.groups);
    }
    if (rows.length) { pass.setPipeline(plan.finalStep.p); pass.setBindGroup(0, plan.finalStep.bg); pass.dispatchWorkgroups(rows.length, 1, 1); }
    pass.end();
    if (qs) { const p2 = enc.beginComputePass({ timestampWrites: { querySet: qs, endOfPassWriteIndex: 1 } }); p2.end(); }
    const outBytes = rows.length * D * 4;
    const rb = this.buf(outBytes, U.COPY_DST | U.MAP_READ); enc.copyBufferToBuffer(B.out, 0, rb, 0, outBytes);
    let tsb = null;
    if (qs) { const r = this.buf(16, U.QUERY_RESOLVE | U.COPY_SRC); tsb = this.buf(16, U.COPY_DST | U.MAP_READ); enc.resolveQuerySet(qs, 0, 2, r, 0); enc.copyBufferToBuffer(r, 0, tsb, 0, 16); }
    let snap = null;
    if (keepCache) {
      const T = P + L, SU = U.COPY_DST | U.COPY_SRC;
      snap = { P: T, ids: [...(cache ? cache.ids : []), ...ids], state: {}, hist: {}, K: {}, V: {} };
      for (const i of this.linear) {
        snap.state[i] = this.buf(this.sizes.state, SU); enc.copyBufferToBuffer(this.work.state[i], 0, snap.state[i], 0, this.sizes.state);
        snap.hist[i] = this.buf(this.sizes.hist, SU); enc.copyBufferToBuffer(this.work.hist[i], 0, snap.hist[i], 0, this.sizes.hist);
      }
      for (const i of this.full) for (const kv of ["K", "V"]) {
        snap[kv][i] = this.buf(this.sizes.HK * kvBytes(T), SU);
        for (let h = 0; h < this.sizes.HK; h++) enc.copyBufferToBuffer(this.work[kv][i], h * headStride, snap[kv][i], h * kvBytes(T), kvBytes(T));
      }
      snap.release = () => [snap.state, snap.hist, snap.K, snap.V].forEach((m) => Object.values(m).forEach((b) => b.destroy()));
    }
    this.d.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ); const out = new Float32Array(rb.getMappedRange().slice(0, outBytes)); rb.unmap(); rb.destroy();
    const c = this.cfg.unrotate;  // rotated torso: final_norm is 1, and the original basis is c * FWHT(row)
    if (c) for (let r = 0; r < rows.length; r++) { fwht(out, r * D, D); for (let j = 0; j < D; j++) out[r * D + j] *= c[j]; }
    const res = { hidden: out, d: D, kernel: plan.kernel, cache: snap, fetch_ms: fetchMs };
    if (tsb) { await tsb.mapAsync(GPUMapMode.READ); const [a, b] = new BigUint64Array(tsb.getMappedRange()); tsb.unmap(); res.gpu_ms = Number(b - a) / 1e6; }
    if (debugLayers) {
      res.layers = [];
      for (const b of dbg) { await b.mapAsync(GPUMapMode.READ); res.layers.push(new Float32Array(b.getMappedRange().slice(0))); b.unmap(); b.destroy(); }
    }
    if (this.errors.length) throw new Error(this.errors.join("\n"));
    return res;
  }
}
