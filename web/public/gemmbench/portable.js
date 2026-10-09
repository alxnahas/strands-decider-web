// Portable matmul kernels (no subgroup-matrix) on the model's int4 shapes, random weights, GPU timestamps.
// ?M=68,512,2048&k=v1,v6_64x64_4x4 ; error is max |y - y_v1| / max |y_v1| plus a CPU check of a few outputs.
// A v6 config whose TN does not divide N is skipped on that shape (the kernel has no column guard).
import { kernelV1, kernelV6 } from "./kernels.js";
const QS = new URLSearchParams(location.search);
const Ms = (QS.get("M") || "68,128,256,512,2048").split(",").map(Number);
const KS = {
  v1: { code: kernelV1(), groups: (M, N) => [N / 16, Math.ceil(M / 8), 1] },
  ...Object.fromEntries([[64, 64, 4, 4], [64, 128, 8, 4], [128, 64, 8, 4], [32, 64, 4, 2], [64, 64, 8, 2], [128, 128, 8, 8], [64, 128, 4, 8]].map(([TM, TN, RM, RN]) =>
    [`v6_${TM}x${TN}_${RM}x${RN}`, { code: kernelV6({ TM, TN, RM, RN }), TN, groups: (M, N) => [N / TN, Math.ceil(M / TM), 1] }])),
};
const names = (QS.get("k") || Object.keys(KS).join(",")).split(",");
for (const n of names) {  // any v6_<TM>x<TN>_<RM>x<RN>
  const g = n.match(/^v6_(\d+)x(\d+)_(\d+)x(\d+)$/); if (!g || KS[n]) continue;
  const [TM, TN, RM, RN] = g.slice(1).map(Number);
  KS[n] = { code: kernelV6({ TM, TN, RM, RN }), TN, groups: (M, N) => [N / TN, Math.ceil(M / TM), 1] };
}
const shapes = [[2048, 12288], [6144, 2048], [2048, 8256], [2048, 5120], [2048, 2048]];
const out = document.getElementById("out"); const log = (s) => { out.textContent += "\n" + s; console.log(s); };
const h2f = (h) => { const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023; return e === 0 ? s * m * 2 ** -24 : s * (1 + m / 1024) * 2 ** (e - 15); };
const f2h = (() => { const f = new Float32Array(1), u = new Uint32Array(f.buffer); return (x) => { f[0] = x; const b = u[0], s = (b >> 16) & 0x8000, e = ((b >> 23) & 0xff) - 112, m = b & 0x7fffff; if (e <= 0) return s; if (e >= 31) return s | 0x7c00; return s | (e << 10) | ((m + 0x1000) >> 13); }; })();
let seed = 1; const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 2 ** 32);

(async () => {
  const adapter = await navigator.gpu.requestAdapter();
  const dev = await adapter.requestDevice({ requiredFeatures: ["shader-f16", "timestamp-query"].filter((f) => adapter.features.has(f)), requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize, maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize, maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup } });
  dev.addEventListener("uncapturederror", (e) => log("GPU ERROR " + e.error.message));
  log(`features ${[...adapter.features].join(" ")}`);
  const pipes = {};
  for (const n of names) try { pipes[n] = dev.createComputePipeline({ layout: "auto", compute: { module: dev.createShaderModule({ code: KS[n].code }), entryPoint: "main" } }); } catch (e) { log(`${n} failed: ${e}`); }
  const mk = (d, usage) => { const b = dev.createBuffer({ size: Math.ceil(d.byteLength / 4) * 4, usage: usage | GPUBufferUsage.COPY_DST }); dev.queue.writeBuffer(b, 0, d); return b; };
  const rows = [], tot = {};
  for (const [K, N] of shapes) {
    const Bw = new Uint32Array(N * K / 8); for (let i = 0; i < Bw.length; i++) Bw[i] = (rnd() * 2 ** 32) >>> 0;
    const Sc = new Uint16Array(N * K / 32); for (let i = 0; i < Sc.length; i++) Sc[i] = f2h(0.005 + 0.01 * rnd());
    const bufB = mk(Bw, GPUBufferUsage.STORAGE), bufS = mk(Sc, GPUBufferUsage.STORAGE);
    for (const M of Ms) {
      const A16 = new Uint16Array(M * K); for (let i = 0; i < A16.length; i++) A16[i] = f2h((rnd() - 0.5) * 2);
      const bufA = mk(A16, GPUBufferUsage.STORAGE), bufY = dev.createBuffer({ size: Math.ceil(M * N * 2 / 4) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const U = mk(new Uint32Array([M, N, K, K / 32]), GPUBufferUsage.UNIFORM);
      const row = { K, N, M }; let ref = null;
      for (const n of names) {
        if (!pipes[n] || N % (KS[n].TN || 16)) continue;
        const bg = dev.createBindGroup({ layout: pipes[n].getBindGroupLayout(0), entries: [bufA, bufB, bufS, bufY, U].map((buffer, binding) => ({ binding, resource: { buffer } })) });
        const run = async (iters) => {
          const qs = dev.createQuerySet({ type: "timestamp", count: 2 });
          const rb = dev.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }), st = dev.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
          const ys = dev.createBuffer({ size: bufY.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
          const e = dev.createCommandEncoder(); e.clearBuffer(bufY); const pass = e.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } });
          for (let i = 0; i < iters; i++) { pass.setPipeline(pipes[n]); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(...KS[n].groups(M, N)); }
          pass.end(); e.resolveQuerySet(qs, 0, 2, rb, 0); e.copyBufferToBuffer(rb, 0, st, 0, 16); e.copyBufferToBuffer(bufY, 0, ys, 0, bufY.size); dev.queue.submit([e.finish()]);
          await st.mapAsync(GPUMapMode.READ); const [a, b] = new BigUint64Array(st.getMappedRange()); st.unmap();
          await ys.mapAsync(GPUMapMode.READ); const y = new Uint16Array(ys.getMappedRange().slice(0, M * N * 2)); ys.unmap();
          return { ms: Number(b - a) / 1e6 / iters, y };
        };
        await run(1); const t = []; let y;
        for (let i = 0; i < 5; i++) { const r = await run(M >= 1024 ? 3 : 10); t.push(r.ms); y = r.y; }
        t.sort((a, b) => a - b); const ms = t[2];
        if (!ref) {  // CPU check of v1 on a few outputs
          let e = 0, mg = 0;
          for (const nn of [0, 1, N >> 1, N - 1]) for (const m of [0, M >> 1, M - 1]) {
            let s = 0; for (let bk = 0; bk < K / 32; bk++) { const sc = h2f(Sc[nn * K / 32 + bk]); for (let el = 0; el < 32; el++) s += h2f(A16[m * K + bk * 32 + el]) * ((((Bw[(nn * K / 32 + bk) * 4 + (el >> 3)] >>> (4 * (el & 7))) & 15) - 8) * sc); }
            e = Math.max(e, Math.abs(h2f(y[m * N + nn]) - s)); mg = Math.max(mg, Math.abs(s));
          }
          row.cpu_err = +(e / mg).toExponential(1); ref = y;
        }
        let e = 0, mg = 0; for (let i = 0; i < y.length; i++) { e = Math.max(e, Math.abs(h2f(y[i]) - h2f(ref[i]))); mg = Math.max(mg, Math.abs(h2f(ref[i]))); }
        row[n] = +ms.toFixed(3); if (n !== names[0]) row[n + "_err"] = +(e / mg).toExponential(1);
        tot[`${M}/${n}`] = (tot[`${M}/${n}`] || 0) + ms * { 12288: 15, 2048: K === 6144 ? 15 : 16, 8256: 11, 5120: 4 }[N];
      }
      rows.push(row); log(JSON.stringify(row));
    }
  }
  log("per forward (shape x count), ms: " + JSON.stringify(Object.fromEntries(Object.entries(tot).map(([k, v]) => [k, +v.toFixed(1)]))));
  window.rows = rows; document.body.dataset.done = "1";
})().catch((e) => { log("ERROR " + (e.stack || e)); document.body.dataset.done = "error"; });
