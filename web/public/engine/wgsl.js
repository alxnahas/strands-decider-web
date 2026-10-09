// WGSL for the hand-written Qwen3.5 (Strands Decider torso) prefill engine. Batch 1, L tokens.
// Activations: residual stream f32, matmul inputs/outputs f16, recurrent math f32.
import { kernelV1, kernelV4, kernelV6, REDUCE, fmtKey } from "../gemmbench/kernels.js";
export { kernelV1, kernelV4, kernelV6, REDUCE, fmtKey };

const HDR = `enable f16;\nenable subgroups;\n`;

/**
 * A kernel without subgroup operations, for GPUs whose subgroup size is not fixed at 32: "subgroup" sg / lane become
 * aligned 32-thread groups of local_invocation_index, and subgroupAdd / subgroupMax a sum / max of the group through
 * workgroup memory (two barriers; every call site is in uniform control flow).
 */
export function noSubgroups(code) {
  code = code.replace("enable subgroups;\n", "");
  if (!code.includes("@builtin(subgroup_id)")) return code;
  const li = code.match(/@builtin\(local_invocation_index\) (\w+): u32/)?.[1] ?? "li32";
  code = code.replace(/,\s*@builtin\(subgroup_id\) sg: u32, @builtin\(subgroup_invocation_id\) lane: u32/, li === "li32" ? ", @builtin(local_invocation_index) li32: u32" : "")
    .replace(/(fn main\([\s\S]*?\)\s*\{)/, `$1\n  let sg = ${li} / 32u; let lane = ${li} % 32u;`)
    .replaceAll("subgroupAdd(", `sum32(${li}, `).replaceAll("subgroupMax(", `max32(${li}, `);
  if (/\bsubgroup(Add|Max|Min|Mul|And|Or|Xor|Shuffle\w*|Broadcast\w*|Ballot|Elect|All|Any|_id|_invocation_id|_size)\b/.test(code)) throw new Error("noSubgroups: unhandled subgroup use");
  const NT = +code.match(/@workgroup_size\((\d+)\)/)[1];
  const fold = (name, init, op) => `fn ${name}(li: u32, x: f32) -> f32 {
  workgroupBarrier(); red32[li] = x; workgroupBarrier();
  let b = li & ~31u; var r = ${init}; for (var i = 1u; i < 32u; i++) { r = ${op}; } return r;
}`;
  return code.replace("\n@compute", `\nvar<workgroup> red32: array<f32, ${NT}>;
${fold("sum32", "red32[b]", "r + red32[b + i]")}
${fold("max32", "red32[b]", "max(r, red32[b + i])")}
@compute`);
}

/** ids -> x (f32 [L, D]) from the quantized embedding table (formats as gemmbench/kernels.js). One thread per 8 elements. */
export const EMBED = (D, f = { bits: 4, group: 32, asym: false }) => {
  const g = f.group === 32 ? "blk" : `blk / ${f.group / 32}u`;
  const q = f.bits === 4 ? "(Q[blk * 4u + e / 8u] >> (4u * (e % 8u))) & 15u"
    : `((Q[blk * ${f.bits}u + e / 16u] >> (2u * (e % 16u))) & 3u)${f.bits === 3 ? " | (((Q[blk * 3u + 2u] >> e) & 1u) << 2u)" : ""}`;
  return HDR + `
@group(0) @binding(0) var<storage, read> ids: array<u32>;
@group(0) @binding(1) var<storage, read> Q: array<u32>;
@group(0) @binding(2) var<storage, read> S: array<f16>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;
${f.asym ? "@group(0) @binding(4) var<storage, read> Bi: array<f16>;" : ""}
@compute @workgroup_size(${D / 8})
fn main(@builtin(local_invocation_index) w: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let t = wg.x; let id = ids[t];
  let blk = id * ${D / 32}u + w / 4u; let sc = f32(S[${g}]);${f.asym ? ` let bi = f32(Bi[${g}]);` : ""}
  for (var j = 0u; j < 8u; j++) {
    let e = (w % 4u) * 8u + j;
    x[t * ${D}u + w * 8u + j] = ${f.asym ? `f32(${q}) * sc + bi` : `(f32(${q}) - ${2 ** (f.bits - 1)}.0) * sc`};
  }
}`;
};

/** Optional residual add (x += y), then zero-centred RMSNorm (weight already 1+w) -> h (f16). One WG per row. */
export const ADD_NORM = (D, add, outF32 = false) => HDR + `
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f16>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> h: array<${outF32 ? "f32" : "f16"}>;
@group(0) @binding(4) var<storage, read> rows: array<u32>;   // output row -> source row (final norm gathers)
var<workgroup> red: array<f32, 8>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_id) sg: u32, @builtin(subgroup_invocation_id) lane: u32) {
  let r = rows[wg.x]; let o = r * ${D}u;
  var v: array<f32, ${D / 256}>; var ss = 0.0;
  for (var i = 0u; i < ${D / 256}u; i++) {
    let c = li + 256u * i;
    var a = x[o + c];
    ${add ? "a += f32(y[o + c]); x[o + c] = a;" : "if (false) { a += f32(y[0]); }"}
    v[i] = a; ss += a * a;
  }
  ss = subgroupAdd(ss);
  if (lane == 0u) { red[sg] = ss; }
  workgroupBarrier();
  var tot = 0.0; for (var i = 0u; i < 8u; i++) { tot += red[i]; }
  let inv = inverseSqrt(tot / ${D}.0 + 1e-6);
  for (var i = 0u; i < ${D / 256}u; i++) { let c = li + 256u * i; h[wg.x * ${D}u + c] = ${outF32 ? "" : "f16"}(v[i] * inv * w[c]); }
}`;

/** mid[t, j] = silu(g[t, j]) * u[t, j] where big[t] = [g (I) | u (I)]. */
export const SILU_MUL = (I) => HDR + `
@group(0) @binding(0) var<storage, read> big: array<f16>;
@group(0) @binding(1) var<storage, read_write> mid: array<f16>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let i = g.x + g.y * nw.x * 256u; let t = i / ${I}u; let j = i % ${I}u;
  if (t >= arrayLength(&mid) / ${I}u) { return; }
  let gv = f32(big[t * ${2 * I}u + j]); let uv = f32(big[t * ${2 * I}u + ${I}u + j]);
  mid[i] = f16(gv / (1.0 + exp(-gv)) * uv);
}`;

/** In-place orthonormal Walsh-Hadamard transform of each contiguous n-block of a (one workgroup per block):
 * the online input rotation of a tensor exported with `had: n`. */
export const FWHT = (n) => HDR + `
@group(0) @binding(0) var<storage, read_write> a: array<f16>;
var<workgroup> sh: array<f32, ${n}>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let o = wg.x * ${n}u;
  for (var e = li; e < ${n}u; e += 256u) { sh[e] = f32(a[o + e]); }
  workgroupBarrier();
  for (var h = 1u; h < ${n}u; h = h * 2u) {
    for (var p = li; p < ${n / 2}u; p += 256u) {
      let i = (p / h) * 2u * h + p % h; let x = sh[i]; let y = sh[i + h];
      sh[i] = x + y; sh[i + h] = x - y;
    }
    workgroupBarrier();
  }
  for (var e = li; e < ${n}u; e += 256u) { a[o + e] = f16(sh[e] * ${1 / Math.sqrt(n)}); }
}`;

/** y[t, j] += c[j] (f32 bias of an adapter layer). */
export const ADD_BIAS = (N) => HDR + `
@group(0) @binding(0) var<storage, read_write> y: array<f16>;
@group(0) @binding(1) var<storage, read> c: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let i = g.x + g.y * nw.x * 256u;
  if (i >= arrayLength(&y)) { return; }
  y[i] = f16(f32(y[i]) + c[i % ${N}u]);
}`;

/** Depthwise causal conv (kernel 4) + SiLU over the qkv channels of the fused in_proj output. */
export const CONV = (C, stride, L) => HDR + `
@group(0) @binding(0) var<storage, read> big: array<f16>;
@group(0) @binding(1) var<storage, read> w: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> hist: array<f16>;     // [3, C] pre-conv rows before t = 0 (zeros if none)
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let c = g.x; let t = g.y;
  if (c >= ${C}u) { return; }
  let k = w[c]; var s = 0.0;
  for (var j = 0u; j < 4u; j++) {
    let tt = i32(t) + i32(j) - 3;
    if (tt >= 0) { s += k[j] * f32(big[u32(tt) * ${stride}u + c]); } else { s += k[j] * f32(hist[u32(tt + 3) * ${C}u + c]); }
  }
  out[t * ${C}u + c] = s / (1.0 + exp(-s));
}`;

/**
 * Gated delta rule, sequential over tokens. WG = 4 subgroups for one head and 32 value columns; each subgroup owns
 * 8 value columns, lane l owns key dims 4l..4l+3, so the 128x8 state slice lives in 32 registers per lane and every
 * reduction over key dims is a subgroupAdd (no barriers).
 */
export const GDN = ({ H, DK, DV, L, stride, aOff, bOff }) => HDR + `
@group(0) @binding(0) var<storage, read> qkv: array<f32>;       // [L, 2*H*DK + H*DV] after conv+silu
@group(0) @binding(1) var<storage, read> big: array<f16>;       // fused in_proj output (for a, b)
@group(0) @binding(2) var<storage, read> negA: array<f32>;
@group(0) @binding(3) var<storage, read> dtb: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;   // [L, H*DV]
@group(0) @binding(5) var<uniform> Lu: vec4<u32>;           // x = L, y = P, z = LT, w = 1 if state holds a prefix state
@group(0) @binding(6) var<storage, read_write> state: array<f32>;   // [H][DK][DV]: prefix state in, final state out
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_id) sg: u32, @builtin(subgroup_invocation_id) lane: u32) {
  let h = wg.x / ${DV / 32}u; let dv0 = (wg.x % ${DV / 32}u) * 32u + sg * 8u;
  let C = ${2 * H * DK + H * DV}u;
  var S: array<vec4<f32>, 8>;   // S[c] = state[4*lane .. 4*lane+3][dv0 + c]
  let sb = h * ${DK * DV}u + lane * 4u * ${DV}u + dv0;
  if (Lu.w == 1u) { for (var c = 0u; c < 8u; c++) { S[c] = vec4<f32>(state[sb + c], state[sb + ${DV}u + c], state[sb + ${2 * DV}u + c], state[sb + ${3 * DV}u + c]); } }
  let na = negA[h]; let db = dtb[h];
  let qs = 1.0 / sqrt(${DK}.0);
  for (var t = 0u; t < Lu.x; t++) {
    let base = t * C;
    let q4 = vec4<f32>(qkv[base + h * ${DK}u + lane * 4u], qkv[base + h * ${DK}u + lane * 4u + 1u], qkv[base + h * ${DK}u + lane * 4u + 2u], qkv[base + h * ${DK}u + lane * 4u + 3u]);
    let kb = base + ${H * DK}u + h * ${DK}u + lane * 4u;
    let k4 = vec4<f32>(qkv[kb], qkv[kb + 1u], qkv[kb + 2u], qkv[kb + 3u]);
    let qn = q4 * inverseSqrt(subgroupAdd(dot(q4, q4)) + 1e-6) * qs;
    let kn = k4 * inverseSqrt(subgroupAdd(dot(k4, k4)) + 1e-6);
    let a = f32(big[t * ${stride}u + ${aOff}u + h]); let b = f32(big[t * ${stride}u + ${bOff}u + h]);
    let beta = 1.0 / (1.0 + exp(-b));
    let x = a + db; let sp = select(log(1.0 + exp(x)), x, x > 20.0);
    let decay = exp(na * sp);
    let vb = base + ${2 * H * DK}u + h * ${DV}u + dv0;
    var mine = 0.0;
    for (var c = 0u; c < 8u; c++) {
      S[c] *= decay;
      let kv = subgroupAdd(dot(S[c], kn));
      let delta = (qkv[vb + c] - kv) * beta;
      S[c] += kn * delta;
      let oc = subgroupAdd(dot(S[c], qn));
      if (lane == c) { mine = oc; }
    }
    if (lane < 8u) { o[t * ${H * DV}u + h * ${DV}u + dv0 + lane] = mine; }
  }
  for (var c = 0u; c < 8u; c++) { state[sb + c] = S[c].x; state[sb + ${DV}u + c] = S[c].y; state[sb + ${2 * DV}u + c] = S[c].z; state[sb + ${3 * DV}u + c] = S[c].w; }
}`;

/**
 * GDN without subgroups: the same layout (32-thread groups as the subgroups), with the per-token reductions batched
 * into two rounds through workgroup memory: (S k for the 8 columns, |q|^2, |k|^2), then S q. S kn = (S k) / |k|.
 */
export const GDN_NOSUB = ({ H, DK, DV, L, stride, aOff, bOff }) => HDR + `
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> big: array<f16>;
@group(0) @binding(2) var<storage, read> negA: array<f32>;
@group(0) @binding(3) var<storage, read> dtb: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
@group(0) @binding(5) var<uniform> Lu: vec4<u32>;
@group(0) @binding(6) var<storage, read_write> state: array<f32>;
var<workgroup> part: array<vec4<f32>, 384>;    // [thread][slot]: up to 3 vec4 partial sums per thread
var<workgroup> chunk: array<vec4<f32>, 48>;    // [group][slot][8-lane chunk]
/** Sums over this thread's 32-thread group of up to 3 vec4 slots (ns used); every thread of the group gets them. */
fn sums(li: u32, ns: u32, x0: vec4<f32>, x1: vec4<f32>, x2: vec4<f32>) -> array<vec4<f32>, 3> {
  part[li * 3u] = x0; part[li * 3u + 1u] = x1; part[li * 3u + 2u] = x2;
  workgroupBarrier();
  let g = li / 32u; let l = li % 32u;
  if (l < ns * 4u) {
    let j = l / 4u; let k = l % 4u; var s = vec4<f32>(0.0);
    for (var i = 0u; i < 8u; i++) { s += part[(g * 32u + k * 8u + i) * 3u + j]; }
    chunk[(g * 3u + j) * 4u + k] = s;
  }
  workgroupBarrier();
  var r: array<vec4<f32>, 3>;
  for (var j = 0u; j < ns; j++) { let b = (g * 3u + j) * 4u; r[j] = (chunk[b] + chunk[b + 1u]) + (chunk[b + 2u] + chunk[b + 3u]); }
  return r;
}
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let sg = li / 32u; let lane = li % 32u;
  let h = wg.x / ${DV / 32}u; let dv0 = (wg.x % ${DV / 32}u) * 32u + sg * 8u;
  let C = ${2 * H * DK + H * DV}u;
  var S: array<vec4<f32>, 8>;
  let sb = h * ${DK * DV}u + lane * 4u * ${DV}u + dv0;
  if (Lu.w == 1u) { for (var c = 0u; c < 8u; c++) { S[c] = vec4<f32>(state[sb + c], state[sb + ${DV}u + c], state[sb + ${2 * DV}u + c], state[sb + ${3 * DV}u + c]); } }
  let na = negA[h]; let db = dtb[h];
  let qs = 1.0 / sqrt(${DK}.0);
  for (var t = 0u; t < Lu.x; t++) {
    let base = t * C;
    let q4 = vec4<f32>(qkv[base + h * ${DK}u + lane * 4u], qkv[base + h * ${DK}u + lane * 4u + 1u], qkv[base + h * ${DK}u + lane * 4u + 2u], qkv[base + h * ${DK}u + lane * 4u + 3u]);
    let kb = base + ${H * DK}u + h * ${DK}u + lane * 4u;
    let k4 = vec4<f32>(qkv[kb], qkv[kb + 1u], qkv[kb + 2u], qkv[kb + 3u]);
    let a = f32(big[t * ${stride}u + ${aOff}u + h]); let b = f32(big[t * ${stride}u + ${bOff}u + h]);
    let beta = 1.0 / (1.0 + exp(-b));
    let x = a + db; let sp = select(log(1.0 + exp(x)), x, x > 20.0);
    let decay = exp(na * sp);
    for (var c = 0u; c < 8u; c++) { S[c] *= decay; }
    let A = sums(li, 3u, vec4<f32>(dot(S[0], k4), dot(S[1], k4), dot(S[2], k4), dot(S[3], k4)),
      vec4<f32>(dot(S[4], k4), dot(S[5], k4), dot(S[6], k4), dot(S[7], k4)), vec4<f32>(dot(q4, q4), dot(k4, k4), 0.0, 0.0));
    let rk = inverseSqrt(A[2].y + 1e-6);
    let qn = q4 * inverseSqrt(A[2].x + 1e-6) * qs; let kn = k4 * rk;
    let vb = base + ${2 * H * DK}u + h * ${DV}u + dv0;
    for (var c = 0u; c < 8u; c++) { S[c] += kn * ((qkv[vb + c] - A[c / 4u][c % 4u] * rk) * beta); }
    let B = sums(li, 2u, vec4<f32>(dot(S[0], qn), dot(S[1], qn), dot(S[2], qn), dot(S[3], qn)),
      vec4<f32>(dot(S[4], qn), dot(S[5], qn), dot(S[6], qn), dot(S[7], qn)), vec4<f32>(0.0));
    if (lane < 8u) { o[t * ${H * DV}u + h * ${DV}u + dv0 + lane] = B[lane / 4u][lane % 4u]; }
  }
  for (var c = 0u; c < 8u; c++) { state[sb + c] = S[c].x; state[sb + ${DV}u + c] = S[c].y; state[sb + ${2 * DV}u + c] = S[c].z; state[sb + ${3 * DV}u + c] = S[c].w; }
}`;

/** Per-head gated RMSNorm: out = w * norm(o) * silu(z), z from the fused in_proj output. One subgroup per (t, head). */
export const GNORM = ({ H, DV, stride, zOff }) => HDR + `
@group(0) @binding(0) var<storage, read> o: array<f32>;
@group(0) @binding(1) var<storage, read> big: array<f16>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f16>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_id) sg: u32, @builtin(subgroup_invocation_id) lane: u32) {
  let row = wg.x * 4u + sg; let t = row / ${H}u; let h = row % ${H}u;
  let b = t * ${H * DV}u + h * ${DV}u;
  var v: array<f32, ${DV / 32}>; var ss = 0.0;
  for (var i = 0u; i < ${DV / 32}u; i++) { v[i] = o[b + lane + 32u * i]; ss += v[i] * v[i]; }
  let inv = inverseSqrt(subgroupAdd(ss) / ${DV}.0 + 1e-6);
  for (var i = 0u; i < ${DV / 32}u; i++) {
    let c = lane + 32u * i; let z = f32(big[t * ${stride}u + ${zOff}u + h * ${DV}u + c]);
    out[b + c] = f16(w[c] * v[i] * inv * z / (1.0 + exp(-z)));
  }
}`;

/** q/k RMSNorm (1+w) + partial rotate-half RoPE. WG per (token, head slot); slots 0..HQ-1 = q, HQ.. = k. */
export const QK_PREP = ({ HQ, HK, HD, RD, stride }) => HDR + `
@group(0) @binding(0) var<storage, read> big: array<f16>;
@group(0) @binding(1) var<storage, read> qw: array<f32>;
@group(0) @binding(2) var<storage, read> kw: array<f32>;
@group(0) @binding(3) var<storage, read> rope: array<vec2<f32>>;  // [L][RD/2] (cos, sin)
@group(0) @binding(4) var<storage, read_write> Qo: array<f16>;    // [HQ][L][HD]
@group(0) @binding(5) var<storage, read_write> Ko: array<f16>;    // [HK][LT][HD], rows P.. for this call
@group(0) @binding(6) var<uniform> L: vec4<u32>;                   // x = L (new tokens), y = P (cached), z = LT capacity
@group(0) @binding(7) var<storage, read_write> Vo: array<f16>;    // [HK][LT][HD]
var<workgroup> red: array<f32, 8>;
var<workgroup> nv: array<f32, ${HD}>;
@compute @workgroup_size(${HD})
fn main(@builtin(local_invocation_index) d: u32, @builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_id) sg: u32, @builtin(subgroup_invocation_id) lane: u32) {
  let t = wg.x; let s = wg.y; let isq = s < ${HQ}u; let pos = L.y + t;
  if (s >= ${HQ}u) { let kh = s - ${HQ}u; Vo[(kh * L.z + pos) * ${HD}u + d] = big[t * ${stride}u + ${HQ * HD * 2 + HK * HD}u + kh * ${HD}u + d]; }
  let src = select(t * ${stride}u + ${HQ * HD * 2}u + (s - ${HQ}u) * ${HD}u + d, t * ${stride}u + s * ${2 * HD}u + d, isq);
  let v = f32(big[src]);
  let ss = subgroupAdd(v * v);
  if (lane == 0u) { red[sg] = ss; }
  workgroupBarrier();
  var tot = 0.0; for (var i = 0u; i < ${HD / 32}u; i++) { tot += red[i]; }
  let wv = select(kw[d], qw[d], isq);
  let n = v * inverseSqrt(tot / ${HD}.0 + 1e-6) * wv;
  nv[d] = n;
  workgroupBarrier();
  var r = n;
  if (d < ${RD}u) {
    let i = d % ${RD / 2}u; let cs = rope[pos * ${RD / 2}u + i];
    if (d < ${RD / 2}u) { r = n * cs.x - nv[d + ${RD / 2}u] * cs.y; } else { r = n * cs.x + nv[d - ${RD / 2}u] * cs.y; }
  }
  if (isq) { Qo[(s * L.x + t) * ${HD}u + d] = f16(r); } else { Ko[((s - ${HQ}u) * L.z + pos) * ${HD}u + d] = f16(r); }
}`;

/** Causal GQA attention, one WG per (query token, q head); output gated by sigmoid(gate). */
export const ATTN = ({ HQ, HK, HD, stride, vOff, maxL }) => HDR + `
@group(0) @binding(0) var<storage, read> Q: array<f16>;
@group(0) @binding(1) var<storage, read> K: array<f16>;
@group(0) @binding(2) var<storage, read> big: array<f16>;     // v and gate live here
@group(0) @binding(3) var<storage, read_write> out: array<f16>;
@group(0) @binding(4) var<uniform> L: vec4<u32>;
var<workgroup> q: array<f32, ${HD}>;
var<workgroup> p: array<f32, ${maxL}>;
var<workgroup> red: array<f32, 8>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_id) sg: u32, @builtin(subgroup_invocation_id) lane: u32) {
  let t = wg.x; let h = wg.y; let kh = h / ${HQ / HK}u; let n = L.x;
  for (var d = li; d < ${HD}u; d += 256u) { q[d] = f32(Q[(h * n + t) * ${HD}u + d]); }
  workgroupBarrier();
  var mx = -1e30;
  for (var j = li; j <= t; j += 256u) {
    var s = 0.0; let kb = (kh * n + j) * ${HD}u;
    for (var d = 0u; d < ${HD}u; d++) { s += q[d] * f32(K[kb + d]); }
    s *= ${1 / Math.sqrt(HD)};
    p[j] = s; mx = max(mx, s);
  }
  mx = subgroupMax(mx); if (lane == 0u) { red[sg] = mx; }
  workgroupBarrier();
  mx = red[0]; for (var i = 1u; i < 8u; i++) { mx = max(mx, red[i]); }
  workgroupBarrier();
  var sum = 0.0;
  for (var j = li; j <= t; j += 256u) { let e = exp(p[j] - mx); p[j] = e; sum += e; }
  sum = subgroupAdd(sum); if (lane == 0u) { red[sg] = sum; }
  workgroupBarrier();
  sum = 0.0; for (var i = 0u; i < 8u; i++) { sum += red[i]; }
  for (var d = li; d < ${HD}u; d += 256u) {
    var acc = 0.0;
    for (var j = 0u; j <= t; j++) { acc += p[j] * f32(big[j * ${stride}u + ${vOff}u + kh * ${HD}u + d]); }
    let g = f32(big[t * ${stride}u + h * ${2 * HD}u + ${HD}u + d]);
    out[t * ${HQ * HD}u + h * ${HD}u + d] = f16(acc / sum / (1.0 + exp(-g)));
  }
}`;

/**
 * Flash-style causal GQA attention. WG = (block of BQ query tokens, q head). Keys processed in chunks of 256 with an
 * online softmax: phase 1 thread j scores key j against all BQ queries (K row read once per BQ queries), phase 2
 * thread d accumulates BQ outputs for head dim d (V row read once per BQ queries).
 */
export const ATTN2 = ({ HQ, HK, HD, stride, vOff, BQ = 16 }) => HDR + `
@group(0) @binding(0) var<storage, read> Q: array<f16>;
@group(0) @binding(1) var<storage, read> K: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> big: array<f16>;
@group(0) @binding(3) var<storage, read_write> out: array<f16>;
@group(0) @binding(4) var<uniform> L: vec4<u32>;   // x = new tokens, y = P, z = LT
@group(0) @binding(5) var<storage, read> V: array<f16>;
const BQ = ${BQ}u; const HD = ${HD}u; const CH = 256u;
var<workgroup> q: array<vec4<f16>, ${BQ * HD / 4}>;
var<workgroup> S: array<f32, ${BQ * 256}>;
var<workgroup> mrow: array<f32, ${BQ}>;
var<workgroup> lrow: array<f32, ${BQ}>;
var<workgroup> alpha: array<f32, ${BQ}>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_id) sg: u32, @builtin(subgroup_invocation_id) lane: u32) {
  let n = L.x; let P = L.y; let LT = L.z; let t0 = wg.x * BQ; let h = wg.y; let kh = h / ${HQ / HK}u;
  let nq = min(BQ, n - t0); let tlast = P + t0 + nq - 1u;   // last absolute key position
  for (var e = li; e < BQ * HD / 4u; e += 256u) {
    let i = e / (HD / 4u); let d4 = e % (HD / 4u);
    var v = vec4<f32>(0.0);
    if (i < nq) { let b = ((h * n + t0 + i) * HD) + d4 * 4u; v = vec4<f32>(f32(Q[b]), f32(Q[b + 1u]), f32(Q[b + 2u]), f32(Q[b + 3u])) * ${1 / Math.sqrt(HD)}; }
    q[e] = vec4<f16>(v);
  }
  if (li < BQ) { mrow[li] = -1e30; lrow[li] = 0.0; }
  var acc: array<f32, ${BQ}>;
  workgroupBarrier();
  for (var j0 = 0u; j0 <= tlast; j0 += CH) {
    // phase 1: scores for key j = j0 + li
    let j = j0 + li;
    var s: array<f32, ${BQ}>;
    if (j <= tlast) {
      let kb = (kh * LT + j) * (HD / 4u);
      for (var d4 = 0u; d4 < HD / 4u; d4++) {
        let k = vec4<f32>(K[kb + d4]);
        for (var i = 0u; i < BQ; i++) { s[i] += dot(vec4<f32>(q[i * (HD / 4u) + d4]), k); }
      }
    }
    for (var i = 0u; i < BQ; i++) { S[i * CH + li] = select(-1e30, s[i], j <= P + t0 + i && i < nq); }
    workgroupBarrier();
    // online softmax update: subgroup sg handles queries 2sg, 2sg+1 (BQ = 16, 8 subgroups)
    for (var r = 0u; r < ${BQ / 8}u; r++) {
      let i = sg * ${BQ / 8}u + r;
      var mx = -1e30;
      for (var c = lane; c < CH; c += 32u) { mx = max(mx, S[i * CH + c]); }
      mx = subgroupMax(mx);
      let mnew = max(mrow[i], mx);
      var sum = 0.0;
      for (var c = lane; c < CH; c += 32u) { let p = exp(S[i * CH + c] - mnew); S[i * CH + c] = p; sum += p; }
      sum = subgroupAdd(sum);
      let a = exp(mrow[i] - mnew);
      workgroupBarrier();
      if (lane == 0u) { alpha[i] = a; lrow[i] = lrow[i] * a + sum; mrow[i] = mnew; }
    }
    workgroupBarrier();
    // phase 2: thread d accumulates outputs for all queries
    let d = li;
    for (var i = 0u; i < BQ; i++) { acc[i] *= alpha[i]; }
    let jend = min(j0 + CH, tlast + 1u);
    for (var jj = j0; jj < jend; jj++) {
      let v = f32(V[(kh * LT + jj) * HD + d]);
      for (var i = 0u; i < BQ; i++) { acc[i] += S[i * CH + (jj - j0)] * v; }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < nq; i++) {
    let t = t0 + i;
    let g = f32(big[t * ${stride}u + h * ${2 * HD}u + HD + li]);
    out[t * ${HQ * HD}u + h * HD + li] = f16(acc[i] / lrow[i] / (1.0 + exp(-g)));
  }
}`;

/** After the conv: hist <- the last 3 pre-conv rows of (old hist ++ this call's rows). */
export const HIST_UPDATE = (C, stride) => HDR + `
@group(0) @binding(0) var<storage, read> big: array<f16>;
@group(0) @binding(1) var<storage, read_write> hist: array<f16>;
@group(0) @binding(2) var<uniform> Lu: vec4<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let c = g.x; if (c >= ${C}u) { return; }
  let L = i32(Lu.x);
  var v: array<f16, 3>;
  for (var r = 0; r < 3; r++) {
    let src = L - 3 + r;                       // row index in this call; < 0 means older history
    if (src >= 0) { v[r] = big[u32(src) * ${stride}u + c]; } else { v[r] = hist[u32(src + 3) * ${C}u + c]; }
  }
  for (var r = 0u; r < 3u; r++) { hist[r * ${C}u + c] = v[r]; }
}`;
