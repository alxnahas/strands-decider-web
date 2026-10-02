// Fair bench: 16 chained int4 projections (8 x [2048->6144, 6144->2048], distinct weights).
// ORT: one session.run per chain (GPU output). Ours: 16 dispatches in one pass, GPU timestamps.
import * as ort from "/ort/ort.webgpu.min.mjs";
import { KERNELS } from "./kernels.js";
const QS = new URLSearchParams(location.search);
const Ms = (QS.get("M") || "1,8,16,32,64,128").split(",").map(Number);
const names = (QS.get("k") || Object.keys(KERNELS).join(",")).split(",");
const out = document.getElementById("out"); const log = (s) => { out.textContent += "\n" + s; console.log(s); };
const h2f = (h) => { const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023; return e === 0 ? s * m * 2 ** -24 : s * (1 + m / 1024) * 2 ** (e - 15); };
const f2h = (() => { const f = new Float32Array(1), u = new Uint32Array(f.buffer); return (x) => { f[0] = x; const b = u[0], s = (b >> 16) & 0x8000, e = ((b >> 23) & 0xff) - 112, m = b & 0x7fffff; if (e <= 0) return s; if (e >= 31) return s | 0x7c00; return s | (e << 10) | ((m + 0x1000) >> 13); }; })();
ort.env.wasm.wasmPaths = "/ort/";
const relErr = (a, b) => { let e = 0, m = 0; for (let i = 0; i < a.length; i++) { e = Math.max(e, Math.abs(h2f(a[i]) - h2f(b[i]))); m = Math.max(m, Math.abs(h2f(b[i]))); } return e / m; };

(async () => {
  const adapter = await navigator.gpu.requestAdapter();
  const features = ["shader-f16", "timestamp-query", "subgroups", "chromium-experimental-subgroup-matrix"].filter((f) => adapter.features.has(f));
  const dev = await adapter.requestDevice({ requiredFeatures: features, requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize, maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize, maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup } });
  dev.addEventListener("uncapturederror", (e) => log("GPU ERROR " + e.error.message));
  const meta = await fetch("/gemm/chain.json").then((r) => r.json());
  const weights = [];
  for (const [i, L] of meta.layers.entries()) {
    const B = new Uint8Array(await fetch(`/gemm/chain_B${i}.bin`).then((r) => r.arrayBuffer()));
    const S = new Uint8Array(await fetch(`/gemm/chain_S${i}.bin`).then((r) => r.arrayBuffer()));
    const mk = (d) => { const b = dev.createBuffer({ size: d.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); dev.queue.writeBuffer(b, 0, d); return b; };
    weights.push({ ...L, B: mk(B), S: mk(S) });
  }
  const sess = await ort.InferenceSession.create("/gemm/chain.onnx", { executionProviders: ["webgpu"], preferredOutputLocation: "gpu-buffer", externalData: ["chain.onnx.data"].map((p) => ({ path: p, data: `/gemm/${p}` })) });
  const pipes = {};
  for (const n of names) {
    const mk = (code) => dev.createComputePipeline({ layout: "auto", compute: { module: dev.createShaderModule({ code }), entryPoint: "main" } });
    try { pipes[n] = mk(KERNELS[n].code); if (KERNELS[n].post) pipes[n + "/post"] = mk(KERNELS[n].post.code); }
    catch (e) { log(`kernel ${n} failed to compile: ${e}`); }
  }
  const rows = [];
  for (const M of Ms) {
    const A16 = new Uint16Array(M * 2048); for (let i = 0; i < A16.length; i++) A16[i] = f2h(Math.sin(i * 12.9898) * 0.5);
    // ORT
    const A = new ort.Tensor("float16", A16, [M, 2048]);
    (await sess.run({ A }))[meta.out].dispose();
    const it = 20; let t = performance.now(), last;
    for (let i = 0; i < it; i++) { last?.dispose(); last = (await sess.run({ A }))[meta.out]; }
    let ortY = await last.getData(); if (!(ortY instanceof Uint16Array)) ortY = new Uint16Array(ortY.buffer, ortY.byteOffset, ortY.length); const ortMs = (performance.now() - t) / it; last.dispose();
    const row = { M, ort_ms: +ortMs.toFixed(3) };
    // ours
    const actBytes = (n) => Math.ceil(M * n * 2 / 4) * 4;
    const bufA = dev.createBuffer({ size: actBytes(6144), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const bufB = dev.createBuffer({ size: actBytes(6144), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const scratch = dev.createBuffer({ size: 64 << 20, usage: GPUBufferUsage.STORAGE });
    let refY = null;
    for (const n of names) {
      if (!pipes[n]) continue;
      const k = KERNELS[n];
      const binds = weights.map((w, i) => {
        const U = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); dev.queue.writeBuffer(U, 0, new Uint32Array([M, w.N, w.K, w.K / 32]));
        const [src, dst] = i % 2 === 0 ? [bufA, bufB] : [bufB, bufA];
        const entries = [src, w.B, w.S, dst, U].map((buffer, binding) => ({ binding, resource: { buffer } }));
        if (k.scratch) entries.push({ binding: 5, resource: { buffer: scratch } });
        const post = k.post && { bg: dev.createBindGroup({ layout: pipes[n + "/post"].getBindGroupLayout(0), entries: [{ binding: 3, resource: { buffer: dst } }, { binding: 4, resource: { buffer: U } }, { binding: 5, resource: { buffer: scratch } }] }), groups: k.post.groups(M, w.N) };
        return { bg: dev.createBindGroup({ layout: pipes[n].getBindGroupLayout(0), entries }), w, groups: k.groups(M, w.N, w.K), post };
      });
      const runOnce = async (timed) => {
        dev.queue.writeBuffer(bufA, 0, A16);
        const qs = timed ? dev.createQuerySet({ type: "timestamp", count: 2 }) : null;
        const e = dev.createCommandEncoder();
        const pass = e.beginComputePass(timed ? { timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
        for (const b of binds) {
          pass.setPipeline(pipes[n]); pass.setBindGroup(0, b.bg); pass.dispatchWorkgroups(...b.groups);
          if (b.post) { pass.setPipeline(pipes[n + "/post"]); pass.setBindGroup(0, b.post.bg); pass.dispatchWorkgroups(...b.post.groups); }
        }
        pass.end();
        const rb = dev.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
        const st = dev.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        if (timed) { e.resolveQuerySet(qs, 0, 2, rb, 0); e.copyBufferToBuffer(rb, 0, st, 0, 16); }
        const yst = dev.createBuffer({ size: actBytes(2048), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        e.copyBufferToBuffer(bufA, 0, yst, 0, actBytes(2048));
        dev.queue.submit([e.finish()]);
        await yst.mapAsync(GPUMapMode.READ); const y = new Uint16Array(yst.getMappedRange().slice(0, M * 2048 * 2)); yst.unmap();
        if (!timed) return { y };
        await st.mapAsync(GPUMapMode.READ); const [a, b] = new BigUint64Array(st.getMappedRange()); st.unmap();
        return { y, ms: Number(b - a) / 1e6 };
      };
      await runOnce(false);
      const ms = []; let y;
      for (let i = 0; i < 10; i++) { const r = await runOnce(true); ms.push(r.ms); y = r.y; }
      ms.sort((a, b) => a - b);
      refY ??= y;
      row[`${n}_ms`] = +ms[4].toFixed(3); row[`${n}_err`] = +relErr(y, refY).toExponential(1);
    }
    row.ort_err_vs_ref = +relErr(ortY, refY).toExponential(1);
    const mx = (a) => a.reduce((m, v) => Math.max(m, Math.abs(h2f(v))), 0); row.max_ort = mx(ortY); row.max_ref = mx(refY); row.ort_len = ortY.length; row.ref_len = refY.length;
    rows.push(row); log(JSON.stringify(row));
  }
  window.rows = rows; document.body.dataset.done = "1";
})().catch((e) => { log("ERROR " + (e.stack || e)); document.body.dataset.done = "error"; });
