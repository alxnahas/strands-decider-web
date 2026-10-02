// Hand-written int4 (MatMulNBits layout) GEMM kernels for small M.
// B: u32[N][K/32][4] (8 nibbles/word, element e at bits 4*(e%8), zero point 8), S: f16[N][K/32], A: f16[M][K], Y: f16[M][N].

export function kernelV1({ C = 16, S = 8, MT = 8 } = {}) {
  return /* wgsl */ `
enable f16;
struct P { M: u32, N: u32, K: u32, blocks: u32 };
@group(0) @binding(0) var<storage, read> A: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> B: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> Sc: array<f16>;
@group(0) @binding(3) var<storage, read_write> Y: array<f16>;
@group(0) @binding(4) var<uniform> p: P;
const C = ${C}u; const S = ${S}u; const MT = ${MT}u;
var<workgroup> part: array<f32, ${C * S * MT}>;
@compute @workgroup_size(${C * S})
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let c = li / S; let s = li % S;
  let n = wg.x * C + c; let m0 = wg.y * MT;
  var acc: array<f32, MT>;
  if (n < p.N) {
    for (var b = s; b < p.blocks; b += S) {
      let w = B[n * p.blocks + b];
      let sc = f32(Sc[n * p.blocks + b]);
      for (var j = 0u; j < 4u; j++) {
        let bytes = unpack4xU8(w[j]);
        let lo = vec4<f32>(bytes & vec4<u32>(15u)) - 8.0;
        let hi = vec4<f32>(bytes >> vec4<u32>(4u)) - 8.0;
        let w0 = vec4<f32>(lo.x, hi.x, lo.y, hi.y) * sc;
        let w1 = vec4<f32>(lo.z, hi.z, lo.w, hi.w) * sc;
        let k4 = (b * 32u + j * 8u) / 4u;
        for (var m = 0u; m < MT; m++) {
          if (m0 + m < p.M) {
            let base = (m0 + m) * (p.K / 4u) + k4;
            acc[m] += dot(vec4<f32>(A[base]), w0) + dot(vec4<f32>(A[base + 1u]), w1);
          }
        }
      }
    }
  }
  for (var m = 0u; m < MT; m++) { part[li * MT + m] = acc[m]; }
  workgroupBarrier();
  if (s == 0u && n < p.N) {
    for (var m = 0u; m < MT; m++) {
      if (m0 + m < p.M) {
        var t = 0.0;
        for (var i = 0u; i < S; i++) { t += part[(c * S + i) * MT + m]; }
        Y[(m0 + m) * p.N + n] = f16(t);
      }
    }
  }
}`;
}

export class Gemm {
  constructor(device, { K, N, B, Sc }, kernel = kernelV1(), cfg = { C: 16, MT: 8 }) {
    this.d = device; this.K = K; this.N = N; this.cfg = cfg;
    const buf = (data, usage) => { const b = device.createBuffer({ size: Math.ceil(data.byteLength / 4) * 4, usage: usage | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, Math.ceil(data.byteLength / 4) * 4 <= data.buffer.byteLength - data.byteOffset ? Math.ceil(data.byteLength / 4) * 4 : data.byteLength); return b; };
    this.B = buf(B, GPUBufferUsage.STORAGE); this.S = buf(Sc, GPUBufferUsage.STORAGE);
    this.pipe = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: kernel }), entryPoint: "main" } });
  }
  setA(M, A16) {  // A16: Uint16Array fp16 bits [M,K]
    const d = this.d; this.M = M;
    this.A = d.createBuffer({ size: A16.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); d.queue.writeBuffer(this.A, 0, A16);
    this.Y = d.createBuffer({ size: Math.ceil(M * this.N * 2 / 4) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.U = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(this.U, 0, new Uint32Array([M, this.N, this.K, this.K / 32]));
    this.bg = d.createBindGroup({ layout: this.pipe.getBindGroupLayout(0), entries: [this.A, this.B, this.S, this.Y, this.U].map((buffer, binding) => ({ binding, resource: { buffer } })) });
  }
  encode(pass) { pass.setPipeline(this.pipe); pass.setBindGroup(0, this.bg); pass.dispatchWorkgroups(Math.ceil(this.N / this.cfg.C), Math.ceil(this.M / this.cfg.MT)); }
  async readY() {
    const st = this.d.createBuffer({ size: this.Y.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const e = this.d.createCommandEncoder(); e.copyBufferToBuffer(this.Y, 0, st, 0, this.Y.size); this.d.queue.submit([e.finish()]);
    await st.mapAsync(GPUMapMode.READ); const out = new Uint16Array(st.getMappedRange().slice(0)); st.unmap(); return out;
  }
}

/**
 * v2: subgroup-matrix (8x8x8) kernel for small M. Workgroup = NSG subgroups; each owns a contiguous K-slice
 * of the same TM x TN output tile (split-K inside the workgroup for occupancy), stages one 32-wide quant
 * block of A and dequantised B into its own workgroup memory, runs (TM/8)x(TN/8) MMAs per 8-step, then the
 * subgroups reduce through workgroup memory. T = "f32" (fp32 accumulate) or "f16".
 */
export function kernelV2({ T = "f32", TM = 16, TN = 32, NSG = 4 } = {}) {
  const I = TM / 8, J = TN / 8, lanesPerCol = 32 / TN;  // TN in {8,16,32}: lanes cooperating on one column's block
  const accs = [], mmas = [], stores = [];
  for (let i = 0; i < I; i++) for (let j = 0; j < J; j++) {
    accs.push(`var c${i}${j}: subgroup_matrix_result<T, 8, 8>;`);
    mmas.push(`c${i}${j} = subgroupMatrixMultiplyAccumulate(a${i}, b${j}, c${i}${j});`);
    stores.push(`subgroupMatrixStore<row_major>(&red, sg * ${TM * TN}u + ${i * 8 * TN + j * 8}u, c${i}${j}, ${TN}u);`);
  }
  const aLoads = [...Array(I).keys()].map((i) => `let a${i} = subgroupMatrixLoad<subgroup_matrix_left<T, 8, 8>, row_major>(&stA, aBase + ${i * 8 * 32}u + kk, 32u);`).join("\n        ");
  const bLoads = [...Array(J).keys()].map((j) => `let b${j} = subgroupMatrixLoad<subgroup_matrix_right<T, 8, 8>, col_major>(&stB, bBase + ${j * 8 * 32}u + kk, 32u);`).join("\n        ");
  return /* wgsl */ `
enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;
alias T = ${T};
struct P { M: u32, N: u32, K: u32, blocks: u32 };
@group(0) @binding(0) var<storage, read> A: array<f16>;
@group(0) @binding(1) var<storage, read> B: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> Sc: array<f16>;
@group(0) @binding(3) var<storage, read_write> Y: array<f16>;
@group(0) @binding(4) var<uniform> p: P;
const TM = ${TM}u; const TN = ${TN}u; const NSG = ${NSG}u;
var<workgroup> stA: array<T, ${NSG * TM * 32}>;
var<workgroup> stB: array<T, ${NSG * TN * 32}>;
var<workgroup> red: array<T, ${NSG * TM * TN}>;
@compute @workgroup_size(${32 * NSG})
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>,
        @builtin(subgroup_id) sg: u32, @builtin(subgroup_invocation_id) lane: u32) {
  let n0 = wg.x * TN; let m0 = wg.y * TM;
  let per = p.blocks / NSG; let b0 = sg * per;
  let aBase = sg * TM * 32u; let bBase = sg * TN * 32u;
  ${accs.join("\n  ")}
  // B staging: lane -> (column, quarter of the 32-wide block) when TN < 32, else one column per lane.
  let col = lane / ${lanesPerCol}u; let part = lane % ${lanesPerCol}u;
  for (var t = 0u; t < per; t++) {
    let b = b0 + t;
    for (var r = 0u; r < TM; r++) {
      let m = m0 + r;
      stA[aBase + r * 32u + lane] = select(T(0), T(A[m * p.K + b * 32u + lane]), m < p.M);
    }
    let n = n0 + col;
    let w = B[n * p.blocks + b];
    let sc = T(Sc[n * p.blocks + b]);
    for (var j = part; j < 4u; j += ${lanesPerCol}u) {
      let bytes = unpack4xU8(w[j]);
      let lo = (vec4<T>(bytes & vec4<u32>(15u)) - T(8)) * sc;
      let hi = (vec4<T>(bytes >> vec4<u32>(4u)) - T(8)) * sc;
      let o = bBase + col * 32u + j * 8u;
      stB[o] = lo.x; stB[o + 1u] = hi.x; stB[o + 2u] = lo.y; stB[o + 3u] = hi.y;
      stB[o + 4u] = lo.z; stB[o + 5u] = hi.z; stB[o + 6u] = lo.w; stB[o + 7u] = hi.w;
    }
    workgroupBarrier();
    for (var kk = 0u; kk < 32u; kk += 8u) {
        ${aLoads}
        ${bLoads}
        ${mmas.join("\n  ")}
    }
    workgroupBarrier();
  }
  ${stores.join("\n  ")}
  workgroupBarrier();
  for (var e = li; e < TM * TN; e += 32u * NSG) {
    let r = e / TN; let c = e % TN;
    if (m0 + r < p.M) {
      var t = 0.0;
      for (var s = 0u; s < NSG; s++) { t += f32(red[s * TM * TN + e]); }
      Y[(m0 + r) * p.N + n0 + c] = f16(t);
    }
  }
}`;
}

/**
 * v3: llama.cpp-style tiled GEMM on 8x8x8 f16 subgroup matrices. Workgroup tile TM x 64 (TM = 64 or 32), 4 subgroups
 * in a 2x2 grid each owning (TM/2) x 32, double-buffered A/B staging (one barrier per 32-wide K block),
 * optional split-K across workgroups (grid z = SK) into f16 partials reduced by REDUCE.
 */
export function kernelV3({ TM = 64, SK = 1, AG = false } = {}) {
  const TN = 64, I = TM / 16, J = 4;
  const acc = [], mma = [], st = [];
  for (let i = 0; i < I; i++) for (let j = 0; j < J; j++) {
    acc.push(`var c${i}${j}: subgroup_matrix_result<f16, 8, 8>;`);
    mma.push(`c${i}${j} = subgroupMatrixMultiplyAccumulate(a${i}, b${j}, c${i}${j});`);
    st.push(`subgroupMatrixStore<row_major>(&stC, sg * ${TM / 2 * 32}u + ${i * 8 * 32 + j * 8}u, c${i}${j}, 32u);`);
  }
  const la = [...Array(I).keys()].map((i) => AG
    ? `let a${i} = subgroupMatrixLoad<subgroup_matrix_left<f16, 8, 8>, row_major>(&Ah, (m0 + sm * ${TM / 2}u + ${i * 8}u) * p.K + (bs + t) * 32u + kk, p.K);`
    : `let a${i} = subgroupMatrixLoad<subgroup_matrix_left<f16, 8, 8>, row_major>(&stA, ao + ${i * 8 * 32}u + kk, 32u);`).join("\n      ");
  const lb = [...Array(J).keys()].map((j) => `let b${j} = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>, col_major>(&stB, bo + ${j * 8 * 32}u + kk, 32u);`).join("\n      ");
  const outBuf = SK > 1 ? "part" : "Y";
  return /* wgsl */ `
enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;
struct P { M: u32, N: u32, K: u32, blocks: u32 };
${AG ? "@group(0) @binding(0) var<storage, read> Ah: array<f16>;" : "@group(0) @binding(0) var<storage, read> A: array<vec4<f16>>;"}
@group(0) @binding(1) var<storage, read> B: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> Sc: array<f16>;
@group(0) @binding(3) var<storage, read_write> Y: array<f16>;
@group(0) @binding(4) var<uniform> p: P;
${SK > 1 ? "@group(0) @binding(5) var<storage, read_write> part: array<f16>;" : ""}
const TM = ${TM}u; const TN = ${TN}u; const SK = ${SK}u;
var<workgroup> stA: array<f16, ${2 * TM * 32}>;
var<workgroup> stB: array<f16, ${2 * TN * 32}>;
var<workgroup> stC: array<f16, ${TM * TN}>;

fn stage(li: u32, buf: u32, b: u32, m0: u32, n0: u32) {
  for (var t = 0u; t < ${AG ? 0 : TM * 8 / 128}u; t++) {          // A: TM rows x 8 vec4
    let idx = li + 128u * t; let r = idx / 8u; let q = idx % 8u;
    var v = vec4<f16>(0.0);
    ${AG ? "" : "if (m0 + r < p.M) { v = A[(m0 + r) * (p.K / 4u) + b * 8u + q]; }"}
    let o = buf * TM * 32u + r * 32u + q * 4u;
    stA[o] = v.x; stA[o + 1u] = v.y; stA[o + 2u] = v.z; stA[o + 3u] = v.w;
  }
  for (var t = 0u; t < 2u; t++) {                       // B: 64 cols x 4 words (8 nibbles each)
    let idx = li + 128u * t; let c = idx / 4u; let j = idx % 4u;
    let blk = (n0 + c) * p.blocks + b;
    let bytes = unpack4xU8(B[blk][j]);
    let sc = Sc[blk];
    let lo = (vec4<f16>(bytes & vec4<u32>(15u)) - 8.0h) * sc;
    let hi = (vec4<f16>(bytes >> vec4<u32>(4u)) - 8.0h) * sc;
    let o = buf * TN * 32u + c * 32u + j * 8u;
    stB[o] = lo.x; stB[o + 1u] = hi.x; stB[o + 2u] = lo.y; stB[o + 3u] = hi.y;
    stB[o + 4u] = lo.z; stB[o + 5u] = hi.z; stB[o + 6u] = lo.w; stB[o + 7u] = hi.w;
  }
}

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_id) sg: u32) {
  let n0 = wg.x * TN; let m0 = wg.y * TM;
  let per = p.blocks / SK; let bs = wg.z * per;
  if (p.M == 0xffffffffu) { Y[0] = 0.0h; }  // keep binding 3 in the auto layout
  let sm = sg / 2u; let sn = sg % 2u;
  ${acc.join("\n  ")}
  stage(li, 0u, bs, m0, n0);
  workgroupBarrier();
  for (var t = 0u; t < per; t++) {
    let cur = t % 2u;
    if (t + 1u < per) { stage(li, 1u - cur, bs + t + 1u, m0, n0); }
    let ao = cur * TM * 32u + sm * ${TM / 2}u * 32u; let bo = cur * TN * 32u + sn * 32u * 32u;
    for (var kk = 0u; kk < 32u; kk += 8u) {
      ${la}
      ${lb}
      ${mma.join("\n      ")}
    }
    workgroupBarrier();
  }
  ${st.join("\n  ")}
  workgroupBarrier();
  for (var e = li; e < TM * TN; e += 128u) {
    let s = e / ${TM / 2 * 32}u; let l = e % ${TM / 2 * 32}u;
    let r = m0 + (s / 2u) * ${TM / 2}u + l / 32u; let c = n0 + (s % 2u) * 32u + l % 32u;
    if (r < p.M) { ${outBuf}[${SK > 1 ? "wg.z * p.M * p.N + " : ""}r * p.N + c] = stC[e]; }
  }
}`;
}

export const REDUCE = (SK) => /* wgsl */ `
enable f16;
struct P { M: u32, N: u32, K: u32, blocks: u32 };
@group(0) @binding(3) var<storage, read_write> Y: array<f16>;
@group(0) @binding(4) var<uniform> p: P;
@group(0) @binding(5) var<storage, read_write> part: array<f16>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; let n = p.M * p.N;
  if (i >= n) { return; }
  var t = 0.0;
  for (var s = 0u; s < ${SK}u; s++) { t += f32(part[s * n + i]); }
  Y[i] = f16(t);
}`;

/**
 * v4: v3 with coalesced weight fetch. Each stage covers KB quant blocks (32*KB of K): thread reads one whole
 * 16-byte block (vec4<u32>), and KB consecutive threads read one column's KB consecutive blocks (contiguous in
 * the MatMulNBits layout). Single-buffered (two barriers per stage) to fit 32 KB of workgroup memory.
 */
export function kernelV4({ TM = 32, SK = 4, KB = 4 } = {}) {
  const TN = 64, I = TM / 16, J = 4, KS = 32 * KB;
  const acc = [], mma = [], st = [];
  for (let i = 0; i < I; i++) for (let j = 0; j < J; j++) {
    acc.push(`var c${i}${j}: subgroup_matrix_result<f16, 8, 8>;`);
    mma.push(`c${i}${j} = subgroupMatrixMultiplyAccumulate(a${i}, b${j}, c${i}${j});`);
    st.push(`subgroupMatrixStore<row_major>(&stB, sg * ${TM / 2 * 32}u + ${i * 8 * 32 + j * 8}u, c${i}${j}, 32u);`);
  }
  const la = [...Array(I).keys()].map((i) => `let a${i} = subgroupMatrixLoad<subgroup_matrix_left<f16, 8, 8>, row_major>(&stA, ao + ${i * 8 * KS}u + kk, ${KS}u);`).join("\n      ");
  const lb = [...Array(J).keys()].map((j) => `let b${j} = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>, col_major>(&stB, bo + ${j * 8 * KS}u + kk, ${KS}u);`).join("\n      ");
  const outBuf = SK > 1 ? "part" : "Y";
  return /* wgsl */ `
enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;
struct P { M: u32, N: u32, K: u32, blocks: u32 };
@group(0) @binding(0) var<storage, read> A: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> B: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> Sc: array<f16>;
@group(0) @binding(3) var<storage, read_write> Y: array<f16>;
@group(0) @binding(4) var<uniform> p: P;
${SK > 1 ? "@group(0) @binding(5) var<storage, read_write> part: array<f16>;" : ""}
const TM = ${TM}u; const TN = ${TN}u; const KB = ${KB}u; const KS = ${KS}u;
var<workgroup> stA: array<f16, ${TM * KS}>;
var<workgroup> stB: array<f16, ${Math.max(TN * KS, TM * TN)}>;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_id) sg: u32) {
  let n0 = wg.x * TN; let m0 = wg.y * TM;
  let stages = p.blocks / KB / ${SK}u; let s0 = wg.z * stages;
  if (p.M == 0xffffffffu) { Y[0] = 0.0h; }
  let sm = sg / 2u; let sn = sg % 2u;
  let ao = sm * ${TM / 2}u * KS; let bo = sn * 32u * KS;
  ${acc.join("\n  ")}
  for (var t = 0u; t < stages; t++) {
    let b0 = (s0 + t) * KB;
    for (var u = 0u; u < ${TM * KS / 4 / 128}u; u++) {           // A: TM rows x KS/4 vec4
      let idx = li + 128u * u; let r = idx / ${KS / 4}u; let q = idx % ${KS / 4}u;
      var v = vec4<f16>(0.0);
      if (m0 + r < p.M) { v = A[(m0 + r) * (p.K / 4u) + b0 * 8u + q]; }
      let o = r * KS + q * 4u;
      stA[o] = v.x; stA[o + 1u] = v.y; stA[o + 2u] = v.z; stA[o + 3u] = v.w;
    }
    for (var u = 0u; u < ${TN * KB / 128}u; u++) {                // B: 64 cols x KB blocks, one 16 B block per thread
      let idx = li + 128u * u; let c = idx / KB; let kb = idx % KB;
      let blk = (n0 + c) * p.blocks + b0 + kb;
      let w = B[blk]; let sc = Sc[blk];
      for (var j = 0u; j < 4u; j++) {
        let bytes = unpack4xU8(w[j]);
        let lo = (vec4<f16>(bytes & vec4<u32>(15u)) - 8.0h) * sc;
        let hi = (vec4<f16>(bytes >> vec4<u32>(4u)) - 8.0h) * sc;
        let o = c * KS + kb * 32u + j * 8u;
        stB[o] = lo.x; stB[o + 1u] = hi.x; stB[o + 2u] = lo.y; stB[o + 3u] = hi.y;
        stB[o + 4u] = lo.z; stB[o + 5u] = hi.z; stB[o + 6u] = lo.w; stB[o + 7u] = hi.w;
      }
    }
    workgroupBarrier();
    for (var kk = 0u; kk < KS; kk += 8u) {
      ${la}
      ${lb}
      ${mma.join("\n      ")}
    }
    workgroupBarrier();
  }
  ${st.join("\n  ")}
  workgroupBarrier();
  for (var e = li; e < TM * TN; e += 128u) {
    let s = e / ${TM / 2 * 32}u; let l = e % ${TM / 2 * 32}u;
    let r = m0 + (s / 2u) * ${TM / 2}u + l / 32u; let c = n0 + (s % 2u) * 32u + l % 32u;
    if (r < p.M) { ${outBuf}[${SK > 1 ? "wg.z * p.M * p.N + " : ""}r * p.N + c] = stB[e]; }
  }
}`;
}

export function kernelV5({ TM = 32, SK = 4, KB = 4 } = {}) {
  const TN = 64, I = TM / 16, J = 4, KS = 32 * KB;
  const acc = [], mma = [], st = [];
  for (let i = 0; i < I; i++) for (let j = 0; j < J; j++) {
    acc.push(`var c${i}${j}: subgroup_matrix_result<f16, 8, 8>;`);
    mma.push(`c${i}${j} = subgroupMatrixMultiplyAccumulate(a${i}, b${j}, c${i}${j});`);
    st.push(`subgroupMatrixStore<row_major>(&stB, sg * ${TM / 2 * 32}u + ${i * 8 * 32 + j * 8}u, c${i}${j}, 32u);`);
  }
  const la = [...Array(I).keys()].map((i) => `let a${i} = subgroupMatrixLoad<subgroup_matrix_left<f16, 8, 8>, row_major>(&stA, ao + ${i * 8 * KS}u + kk, ${KS}u);`).join("\n      ");
  const lb = [...Array(J).keys()].map((j) => `let b${j} = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>, col_major>(&stB, bo + ${j * 8 * KS}u + kk, ${KS}u);`).join("\n      ");
  const outBuf = SK > 1 ? "part" : "Y";
  return /* wgsl */ `
enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;
struct P { M: u32, N: u32, K: u32, blocks: u32 };
@group(0) @binding(0) var<storage, read> A: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> B: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> Sc: array<f16>;
@group(0) @binding(3) var<storage, read_write> Y: array<f16>;
@group(0) @binding(4) var<uniform> p: P;
${SK > 1 ? "@group(0) @binding(5) var<storage, read_write> part: array<f16>;" : ""}
const TM = ${TM}u; const TN = ${TN}u; const KB = ${KB}u; const KS = ${KS}u;
var<workgroup> stA: array<f16, ${TM * KS}>;
var<workgroup> stB: array<f16, ${Math.max(TN * KS, TM * TN)}>;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_id) sg: u32) {
  let n0 = wg.x * TN; let m0 = wg.y * TM;
  let stages = p.blocks / KB / ${SK}u; let s0 = wg.z * stages;
  if (p.M == 0xffffffffu) { Y[0] = 0.0h; }
  let sm = sg / 2u; let sn = sg % 2u;
  let ao = sm * ${TM / 2}u * KS; let bo = sn * 32u * KS;
  var acc: array<subgroup_matrix_result<f16, 8, 8>, ${I * J}>;
  for (var t = 0u; t < stages; t++) {
    let b0 = (s0 + t) * KB;
    for (var u = 0u; u < ${TM * KS / 4 / 128}u; u++) {           // A: TM rows x KS/4 vec4
      let idx = li + 128u * u; let r = idx / ${KS / 4}u; let q = idx % ${KS / 4}u;
      var v = vec4<f16>(0.0);
      if (m0 + r < p.M) { v = A[(m0 + r) * (p.K / 4u) + b0 * 8u + q]; }
      let o = r * KS + q * 4u;
      stA[o] = v.x; stA[o + 1u] = v.y; stA[o + 2u] = v.z; stA[o + 3u] = v.w;
    }
    for (var u = 0u; u < ${TN * KB / 128}u; u++) {                // B: 64 cols x KB blocks, one 16 B block per thread
      let idx = li + 128u * u; let c = idx / KB; let kb = idx % KB;
      let blk = (n0 + c) * p.blocks + b0 + kb;
      let w = B[blk]; let sc = Sc[blk];
      for (var j = 0u; j < 4u; j++) {
        let bytes = unpack4xU8(w[j]);
        let lo = (vec4<f16>(bytes & vec4<u32>(15u)) - 8.0h) * sc;
        let hi = (vec4<f16>(bytes >> vec4<u32>(4u)) - 8.0h) * sc;
        let o = c * KS + kb * 32u + j * 8u;
        stB[o] = lo.x; stB[o + 1u] = hi.x; stB[o + 2u] = lo.y; stB[o + 3u] = hi.y;
        stB[o + 4u] = lo.z; stB[o + 5u] = hi.z; stB[o + 6u] = lo.w; stB[o + 7u] = hi.w;
      }
    }
    workgroupBarrier();
    for (var kk = 0u; kk < KS; kk += 8u) {
      var bm: array<subgroup_matrix_right<f16, 8, 8>, ${J}>;
      for (var j = 0u; j < ${J}u; j++) { bm[j] = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>, col_major>(&stB, bo + j * ${8 * KS}u + kk, ${KS}u); }
      for (var i = 0u; i < ${I}u; i++) {
        let am = subgroupMatrixLoad<subgroup_matrix_left<f16, 8, 8>, row_major>(&stA, ao + i * ${8 * KS}u + kk, ${KS}u);
        for (var j = 0u; j < ${J}u; j++) { acc[i * ${J}u + j] = subgroupMatrixMultiplyAccumulate(am, bm[j], acc[i * ${J}u + j]); }
      }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < ${I}u; i++) { for (var j = 0u; j < ${J}u; j++) { subgroupMatrixStore<row_major>(&stB, sg * ${TM / 2 * 32}u + i * 256u + j * 8u, acc[i * ${J}u + j], 32u); } }
  workgroupBarrier();
  for (var e = li; e < TM * TN; e += 128u) {
    let s = e / ${TM / 2 * 32}u; let l = e % ${TM / 2 * 32}u;
    let r = m0 + (s / 2u) * ${TM / 2}u + l / 32u; let c = n0 + (s % 2u) * 32u + l % 32u;
    if (r < p.M) { ${outBuf}[${SK > 1 ? "wg.z * p.M * p.N + " : ""}r * p.N + c] = stB[e]; }
  }
}`;
}

// name -> { code, groups(M, N, K) => [x, y, z] }. The first entry is the fp32-accurate reference for error checks.
export const KERNELS = {
  ...Object.fromEntries([[32, 1], [64, 4], [16, 4], [80, 4]].map(([TM, SK]) => [`v5_${TM}_sk${SK}`, {
    code: kernelV5({ TM, SK, KB: 2 }), groups: (M, N) => [N / 64, Math.ceil(M / TM), SK], scratch: SK > 1,
    post: SK > 1 ? { code: REDUCE(SK), groups: (M, N) => [Math.ceil(M * N / 256), 1, 1] } : null }])),
  v1: { code: kernelV1(), groups: (M, N) => [Math.ceil(N / 16), Math.ceil(M / 8), 1] },
  ...Object.fromEntries([[32, 1, 2], [64, 4, 2], [16, 4, 2]].map(([TM, SK, KB]) => [`v4_${TM}_sk${SK}_kb${KB}`, {
    code: kernelV4({ TM, SK, KB }), groups: (M, N) => [N / 64, Math.ceil(M / TM), SK], scratch: SK > 1,
    post: SK > 1 ? { code: REDUCE(SK), groups: (M, N) => [Math.ceil(M * N / 256), 1, 1] } : null }])),
  ...Object.fromEntries([].map(([TM, SK, AG]) => [`v3_${TM}_sk${SK}${AG ? "_ag" : ""}`, {
    code: kernelV3({ TM, SK, AG }), groups: (M, N) => [N / 64, Math.ceil(M / TM), SK], scratch: SK > 1,
    post: SK > 1 ? { code: REDUCE(SK), groups: (M, N) => [Math.ceil(M * N / 256), 1, 1] } : null }])),
  ...Object.fromEntries([].map(([T, TM, TN, NSG]) =>
    [`v2_${T}_${TM}x${TN}_sg${NSG}`, { code: kernelV2({ T, TM, TN, NSG }), groups: (M, N) => [N / TN, Math.ceil(M / TM), 1] }])),
};
