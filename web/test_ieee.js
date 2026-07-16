// An IEEE-754 binary32 oracle, written from the standard, in exact integer math.
//
// Why this exists. The kernel epilogue and the JS mirror agree, and I made them
// agree by changing the mirror. Asked what decides which one is wrong when they
// disagree, the honest answer was: me, reasoning about the WGSL spec, with
// nothing enforcing it. That reasoning lived in a paragraph. This file is that
// paragraph made executable.
//
// Nothing here is derived from Verified.epi or from any shader. It is built from
// the binary32 definition: values are sign * mantissa * 2^exp with a 24-bit
// significand, products are computed EXACTLY as BigInt (no float arithmetic
// anywhere in the oracle), and the result is rounded once, round-to-nearest-even,
// per IEEE 754-2019 section 4.3.1. If the mirror and this disagree, the mirror is
// wrong, and no judgement call is involved.

const fs = require("fs");
const path = require("path");
const V = require("./public/verified_core.js");
const L = { mul: new Int16Array(fs.readFileSync(path.join(__dirname, "public", "mul_lut.bin")).buffer.slice(0)) };

// ---- the oracle -------------------------------------------------------------
function bitLength(n) { let b = 0; while (n > 0n) { n >>= 1n; b++; } return b; }

// binary32 -> exact (sign, mantissa, exp2) with value = ±mant * 2^exp2
function decomposeF32(x) {
  const dv = new DataView(new ArrayBuffer(4));
  dv.setFloat32(0, x);
  const bits = dv.getUint32(0);
  const s = bits >>> 31, e = (bits >>> 23) & 0xFF, m = bits & 0x7FFFFF;
  if (e === 0xFF) throw new Error("inf/nan out of scope");
  // subnormal: no implicit leading 1, fixed exponent 2^-149
  if (e === 0) return { neg: s === 1, mant: BigInt(m), exp2: -149 };
  // normal: implicit leading 1; 2^(e-127) * (1 + m/2^23) = (2^23 + m) * 2^(e-150)
  return { neg: s === 1, mant: BigInt(m) | (1n << 23n), exp2: e - 150 };
}

// Round ±mant * 2^exp2 to binary32, round-to-nearest-ties-to-even.
// Two constraints decide how many low bits get discarded:
//   normals    keep 24 significant bits            -> shift = bitLength - 24
//   subnormals snap to the 2^-149 grid             -> shift = -exp2 - 149
// whichever discards more wins, which is exactly the binary32 value set.
function roundToF32(neg, mant, exp2) {
  if (mant === 0n) return neg ? -0 : 0;
  const bl = bitLength(mant);
  const shift = Math.max(bl - 24, -exp2 - 149);
  let r, e;
  if (shift <= 0) {
    r = mant << BigInt(-shift);
    e = exp2 + shift;
  } else {
    const sh = BigInt(shift);
    const keep = mant >> sh;
    const rem = mant - (keep << sh);          // exact remainder, no float involved
    const half = 1n << (sh - 1n);
    r = keep;
    if (rem > half || (rem === half && (keep & 1n) === 1n)) r += 1n;   // ties to even
    e = exp2 + shift;
  }
  if (bitLength(r) > 24) { r >>= 1n; e += 1; } // rounding carried into a 25th bit
  if (e + bitLength(r) - 1 > 127) return neg ? -Infinity : Infinity;
  const val = Number(r) * Math.pow(2, e);     // r <= 2^24 and e in range: exact in f64
  return neg ? -val : val;
}

const i32ToF32Spec = (s) => s === 0 ? 0 : roundToF32(s < 0, BigInt(Math.abs(s)), 0);
// IEEE 754-2019 §6.3: the sign of a product is the XOR of the operand signs,
// even when the product is zero. `a < 0` is false for -0, so the sign test must
// use the sign BIT. (Found by a random sweep here: a subnormal float rounded to
// f32 becomes ±0, and positive*(-0) came back +0 where the spec says -0. The
// epilogue never multiplies by -0 — block scales are clamped >= 1e-8 — but an
// oracle that is wrong about a sign bit is not an oracle.)
const signbit = (x) => x < 0 || Object.is(x, -0);
function mulF32Spec(a, b) {
  if (a === 0 || b === 0) return (signbit(a) !== signbit(b)) ? -0 : 0;
  const A = decomposeF32(a), B = decomposeF32(b);
  return roundToF32(A.neg !== B.neg, A.mant * B.mant, A.exp2 + B.exp2);
}
// what WGSL `f32(s) * a * b` means: left-associative, rounded after every step
const epiSpec = (s, a, b) => mulF32Spec(mulF32Spec(i32ToF32Spec(s), a), b);

// ---- checks -----------------------------------------------------------------
let pass = true;
const ok = (c, msg) => { console.log(`${c ? "  ok  " : "  FAIL"}  ${msg}`); if (!c) pass = false; };
const f32 = Math.fround;

console.log("\nthe oracle reproduces known binary32 behaviour:");
ok(i32ToF32Spec(1 << 24) === 16777216, "2^24 is exact");
ok(i32ToF32Spec((1 << 24) + 1) === 16777216, "2^24+1 ties down to even (not 2^24+2)");
ok(i32ToF32Spec((1 << 24) + 3) === 16777220, "2^24+3 rounds up to the even neighbour");
ok(i32ToF32Spec(-((1 << 24) + 1)) === -16777216, "sign symmetry on the tie");
ok(mulF32Spec(f32(0.1), f32(0.1)) === f32(f32(0.1) * f32(0.1)), "0.1*0.1 matches a correctly-rounded f32 multiply");
ok(mulF32Spec(1.5, 2) === 3, "exact small product");
ok(mulF32Spec(f32(1e-30), f32(1e-15)) === f32(f32(1e-30) * f32(1e-15)), "underflow into subnormals");
ok(i32ToF32Spec(0) === 0 && mulF32Spec(0, 5) === 0, "zeros");
ok(Object.is(mulF32Spec(5, -0), -0) && Object.is(mulF32Spec(-0, -0), 0), "signed zero: sign of a zero product is the XOR of the operand signs");

// broad agreement with Math.fround, which is an independently correctly-rounded
// f32 conversion (spec'd by ECMA-262, implemented by V8 — neither is this file)
console.log("\nthe oracle agrees with an independent correctly-rounded implementation:");
const f32buf = new Float32Array(1);
const asF32 = (x) => { f32buf[0] = x; return f32buf[0]; };
{
  let bad = 0, n = 0;
  const cases = [];
  for (let t = 0; t < 200000; t++) cases.push(((Math.random() * 2 - 1) * 2.1e9) | 0);
  for (const s of cases) { n++; if (!Object.is(i32ToF32Spec(s), f32(s))) bad++; }
  ok(bad === 0, `i32ToF32Spec matches Math.fround on ${n} random int32 (incl. |s| > 2^24)`);
}
{
  // magnitudes chosen to land products in every regime: exact, rounded normal,
  // subnormal underflow, and overflow to infinity. The near-zero magnitudes also
  // generate ±0 operands, which is what caught the signed-zero bug above.
  let bad = 0, n = 0;
  const mags = [1, 1e-3, 1e-8, 1e-20, 1e-38, 1e-40, 1e-44, 3e38, 1e30, 127, 1 / 127];
  for (let t = 0; t < 300000; t++) {
    const a = asF32((Math.random() * 2 - 1) * mags[(Math.random() * mags.length) | 0]);
    const b = asF32((Math.random() * 2 - 1) * mags[(Math.random() * mags.length) | 0]);
    n++;
    if (!Object.is(mulF32Spec(a, b), f32(a * b))) {   // f64 product of f32s is exact, so fround(a*b) is the correctly-rounded f32 product
      if (bad++ < 3) console.log(`      ${a} * ${b}  spec=${mulF32Spec(a, b)}  fround=${f32(a * b)}`);
    }
  }
  ok(bad === 0, `mulF32Spec matches the correctly-rounded product on ${n} draws (subnormal..overflow)`);
}

// ---- the actual point: is Verified.epi the spec, or just agreeable? ---------
console.log("\nVerified.epi vs the oracle, over the range the kernels produce:");
let mism = 0, n = 0;
for (let t = 0; t < 200000; t++) {
  // s spans the int32 accumulator range the LUT path can reach (k up to ~16k,
  // products up to 127*127), scales span plausible block-scale magnitudes
  const s = Math.round((Math.random() * 2 - 1) * 2.6e8);
  const a = f32(Math.random() * 2e-2 + 1e-8);
  const b = f32(Math.random() * 2e-2 + 1e-8);
  const mine = V.epi(s, a, b), spec = epiSpec(s, a, b);
  n++;
  if (mine !== spec && !(Number.isNaN(mine) && Number.isNaN(spec))) {
    if (mism++ < 3) console.log(`      s=${s} a=${a} b=${b}  epi=${mine}  spec=${spec}`);
  }
}
ok(mism === 0, `${n} random triples, ${mism} disagreements`);

// edge magnitudes, where rounding actually decides things
console.log("\nsame check pinned to the awkward values:");
let em = 0;
for (const s of [0, 1, -1, (1 << 24) - 1, 1 << 24, (1 << 24) + 1, (1 << 24) + 3, -(1 << 24) - 1, 2 ** 30, -(2 ** 30)])
  for (const a of [f32(1), f32(0.5), f32(1 / 3), f32(1e-20), f32(3e-8)])
    for (const b of [f32(1), f32(2), f32(1 / 7), f32(1e-20)]) {
      if (V.epi(s, a, b) !== epiSpec(s, a, b)) {
        if (em++ < 3) console.log(`      s=${s} a=${a} b=${b}  epi=${V.epi(s, a, b)}  spec=${epiSpec(s, a, b)}`);
      }
    }
ok(em === 0, `edge grid, ${em} disagreements`);

// the tie ladder: s straddling 2^24 with power-of-two scales keeps every product
// dyadic, so f32(s)*a lands EXACTLY on rounding midpoints — where ties-to-even
// and ties-away disagree, and where a round-once mirror slips a whole ulp
{
  let bad = 0, n = 0;
  for (let k = 0; k <= 20; k++) {
    const a = f32(Math.pow(2, -k)), b = f32(1);
    for (let s = (1 << 24) - 4; s <= (1 << 24) + 4; s++) {
      n += 2;
      if (V.epi(s, a, b) !== epiSpec(s, a, b)) bad++;
      if (V.epi(-s, a, b) !== epiSpec(-s, a, b)) bad++;
    }
  }
  ok(bad === 0, `tie-to-even ladder around 2^24, ${n} cases, ${bad} disagreements`);
}

// end to end: rebuild a live GEMM's outputs from its raw int32 accumulator using
// ONLY the oracle, and compare against the mirror's finished floats. This closes
// the loop from exact integer accumulation to final f32 with nothing but IEEE-754.
{
  const d = { m: 7, k: 40, n: 6, batch: 2 };
  const Xf = Float32Array.from({ length: d.batch * d.m * d.k }, () => Math.random() * 2 - 1);
  const Wf = Float32Array.from({ length: d.batch * d.k * d.n }, () => Math.random() * 2 - 1);
  const x = V.quantizeRows(Xf, d.batch * d.m, d.k);
  const wq = new Int8Array(d.batch * d.k * d.n), ws = new Float32Array(d.batch * d.n);
  for (let bz = 0; bz < d.batch; bz++) {
    const w = V.quantizeCols(Wf.subarray(bz * d.k * d.n, (bz + 1) * d.k * d.n), d.k, d.n);
    wq.set(w.q, bz * d.k * d.n); ws.set(w.s, bz * d.n);
  }
  const raw = V.bgemmJS(x.q, wq, x.s, ws, { ...d, acc: true }, L);   // exact int32
  const fin = V.bgemmJS(x.q, wq, x.s, ws, d, L);                     // mirror epilogue
  let bad = 0;
  for (let bz = 0; bz < d.batch; bz++)
    for (let i = 0; i < d.m; i++)
      for (let j = 0; j < d.n; j++) {
        const idx = (bz * d.m + i) * d.n + j;
        if (!Object.is(epiSpec(raw[idx], x.s[bz * d.m + i], ws[bz * d.n + j]), fin[idx])) bad++;
      }
  ok(bad === 0, `bgemmJS outputs rebuilt from the raw accumulator via the oracle (${d.batch}x${d.m}x${d.n}), ${bad} disagreements`);
}

// ---- prove the oracle bites --------------------------------------------------
// The mirror USED to do the whole chain in f64 and round once on the store. That
// is not what WGSL computes. The oracle must reject it, or it is not an oracle.
console.log("\nthe oracle rejects the mirror I actually shipped before:");
const oldEpi = (s, a, b) => f32(s * a * b);           // one rounding, not three
let oldBad = 0;
for (let t = 0; t < 200000; t++) {
  const s = Math.round((Math.random() * 2 - 1) * 2.6e8);
  const a = f32(Math.random() * 2e-2 + 1e-8);
  const b = f32(Math.random() * 2e-2 + 1e-8);
  if (oldEpi(s, a, b) !== epiSpec(s, a, b)) oldBad++;
}
ok(oldBad > 0, `old single-rounding mirror disagrees on ${oldBad} of 200000 (${(oldBad / 2000).toFixed(2)}% — the bug the 1e-6 tolerance hid)`);
// and a value bug of the kind the corpus uses
const factorBug = (s, a, b) => f32(V.epi(s, a, b) * 2);
ok(factorBug(12345, f32(0.01), f32(0.02)) !== epiSpec(12345, f32(0.01), f32(0.02)),
   "oracle catches a dropped-constant-factor bug (corpus: gelu_triton_buggy)");

console.log(pass ? "\nIEEE ORACLE TEST PASSED — the mirror's epilogue IS the spec, not merely agreeable."
                 : "\nIEEE ORACLE TEST FAILED");
process.exit(pass ? 0 : 1);
