// Download format of a packed export (pack_wire.py in the export tooling): a shard's tensors stored back to back,
// fp16 scale tensors as 8-bit log codes. A row of G scales is m (fp16 bits), j (u16), then G codes sign << 7 | k;
// |scale| = fp16(f32(m 2^(-k j / 4096))). These functions rebuild the GPU layout the kernels read.

const fromHalf = (h) => { const e = (h >> 10) & 31, f = h & 1023, v = e ? (1024 + f) * 2 ** (e - 25) : f * 2 ** -24; return h & 0x8000 ? -v : v; };
/** fp16 bits of x > 0 rounded to float32 first, then to fp16 (both nearest even), as pack_wire.py decodes. */
const F = new Float32Array(1), B = new Uint32Array(F.buffer);
function toHalf(x) {
  F[0] = x; const b = B[0], e = ((b >>> 23) & 255) - 112, rest = b & 0x1fff;
  if (e > 0 && e < 31) { const h = (e << 10) | ((b >>> 13) & 1023); return rest > 0x1000 || (rest === 0x1000 && h & 1) ? h + 1 : h; }
  if (e >= 31) return 0x7c00;
  const u = F[0] * 2 ** 24, f = Math.floor(u);  // subnormal: units of 2^-24
  return u - f > 0.5 || (u - f === 0.5 && f & 1) ? f + 1 : f;
}
let pow = new Float64Array(0);  // 2^(-e / 4096)
const powTo = (n) => { if (pow.length <= n) { pow = new Float64Array(n + 1); for (let i = 0; i <= n; i++) pow[i] = 2 ** (-i / 4096); } return pow; };

/** N rows of G scale codes at src[off..] -> fp16 bits in dst (a Uint16Array) from dstOff. */
export function decodeScales(src, off, N, G, dst, dstOff = 0) {
  const dv = new DataView(src.buffer, src.byteOffset + off, 4 * N); let jmax = 0;
  for (let r = 0; r < N; r++) jmax = Math.max(jmax, dv.getUint16(4 * r + 2, true));
  const P = powTo(127 * jmax), codes = src.subarray(off + 4 * N, off + 4 * N + N * G);
  const tab = new Uint16Array(128), row = new Int32Array(128).fill(-1);  // this row's decoded magnitudes
  for (let r = 0; r < N; r++) {
    const m = fromHalf(dv.getUint16(4 * r, true)), j = dv.getUint16(4 * r + 2, true);
    for (let g = 0; g < G; g++) {
      const c = codes[r * G + g], k = c & 127;
      if (row[k] !== r) { row[k] = r; tab[k] = toHalf(m * P[k * j]); }
      dst[dstOff + r * G + g] = tab[k] | ((c & 128) << 8);
    }
  }
}

/**
 * Shard s from its download (tensors at "wire" offsets, "u8" ones as scale codes): put(offset in the GPU layout,
 * bytes) for each tensor. The gaps between tensors are zeros.
 */
export function unpackShard(wire, s, tensors, put) {
  for (const t of Object.values(tensors)) {
    if (t.wire === undefined || t.offset < s.start || t.offset >= s.start + s.bytes) continue;
    if (!t.u8) { put(t.offset - s.start, wire.subarray(t.wire, t.wire + t.bytes)); continue; }
    const [N, G] = t.shape, d = new Uint16Array(N * G);
    decodeScales(wire, t.wire, N, G, d);
    put(t.offset - s.start, new Uint8Array(d.buffer));
  }
}
