// Reductions and accumulators that used to accept bad input silently (2026-09-30). Each check
// fails on the previous code:
//  1. the GPU MLP chain's row max comes from an atomicMax over bitcast(abs(h)) and never passed a
//     non-finite guard -- scalesFromAbsMax now refuses, and the bit-order argument that makes
//     that ONE check complete is checked here too;
//  2. averageGrads averaged a truncated gradient over a prefix, and fed NaN/Inf into DaisyAdam
//     on every peer at once -- it now refuses both;
//  3. the Int32Array / WGSL i32 accumulators WRAP past K = 131071 (CPU and GPU agree on the wrong
//     sum) -- the GEMMs now refuse, the same bound as the Python kernel's MAX_INT32_K.
"use strict";
const V = require("./public/verified_core.js");
const TC = require("./public/traincore.js");

let fails = 0;
function ck(name, cond) { console.log((cond ? "  ok    " : "  FAIL  ") + name); if (!cond) fails++; }
function throwsName(fn, name) { try { fn(); return false; } catch (e) { return e.name === name; } }

// ---- 1. the GPU row-max readback ------------------------------------------------------------
for (const bad of [NaN, Infinity, -Infinity]) {
  ck("scalesFromAbsMax refuses a row max of " + bad,
     throwsName(() => V.scalesFromAbsMax(Float32Array.from([1.5, bad, 0.25])), "NonFiniteError"));
}
const fin = V.scalesFromAbsMax(Float32Array.from([127, 0, 1e-12]));
ck("finite row maxes give the same scales as before (1, 1e-8 floor)",
   fin.scale[0] === 1 && Math.abs(fin.scale[1] - 1e-8) < 1e-15 && fin.inv[0] === 1);

// What the WGSL kernel computes: atomicMax over the u32 bits of |h|. Emulated exactly with
// typed-array views. The guard above is complete iff "max bits non-finite <=> some h non-finite".
function gpuRowMax(h) {
  const f = new Float32Array(1), u = new Uint32Array(f.buffer);
  let mx = 0;
  for (const v of h) { f[0] = Math.abs(v); if (u[0] > mx) mx = u[0]; }
  u[0] = mx; return f[0];
}
let rng = 12345;
const rnd = () => ((rng = (rng * 1103515245 + 12345) >>> 0) / 4294967296);
let complete = true, exact = true;
for (let trial = 0; trial < 2000; trial++) {
  const h = new Float32Array(16).map(() => (rnd() - 0.5) * Math.pow(2, Math.floor(rnd() * 200) - 100));
  const want = h.reduce((a, v) => Math.max(a, Math.abs(v)), 0);
  if (gpuRowMax(h) !== Math.fround(want)) exact = false;
  if (trial % 2) h[Math.floor(rnd() * 16)] = [NaN, Infinity, -Infinity, -NaN][trial % 4];
  const nonfinite = h.some(v => !Number.isFinite(v));
  if (Number.isFinite(gpuRowMax(h)) === nonfinite) complete = false;
}
ck("bit-order row max equals the true |max| on finite rows (2000 rows, wide exponents)", exact);
ck("bit-order row max is non-finite exactly when some element is (NaN/Inf/-Inf injected)", complete);

// ---- 2. averageGrads ------------------------------------------------------------------------
const g1 = Float32Array.from([1, 2, 3, 4]), g2 = Float32Array.from([3, 2, 1, 0]);
const avg = TC.averageGrads([g1, g2]);
ck("equal-length finite gradients average as before", Array.from(avg).join() === "2,2,2,2");
ck("a truncated gradient is refused (was averaged over a prefix)",
   throwsName(() => TC.averageGrads([g1, Float32Array.from([1, 2])]), "GradientError"));
ck("a longer gradient is refused (its tail was silently dropped)",
   throwsName(() => TC.averageGrads([g1, Float32Array.from([1, 2, 3, 4, 5])]), "GradientError"));
for (const bad of [NaN, Infinity]) {
  ck("a gradient containing " + bad + " is refused before DaisyAdam sees it",
     throwsName(() => TC.averageGrads([g1, Float32Array.from([0, bad, 0, 0])]), "GradientError"));
}
// why refusing matters: one NaN step poisons Adam's moments for good
const adam = TC.makeAdam(1, { lr: 0.1 });
adam.step(Float32Array.from([NaN]));
ck("(context) one NaN step leaves Adam's moments NaN on a finite gradient afterwards",
   Number.isNaN(adam.step(Float32Array.from([1]))[0]));

// ---- 3. int32 accumulator headroom ----------------------------------------------------------
ck("MAX_INT32_K matches the Python kernel: floor((2^31-1) / 16384) = 131071", V.MAX_INT32_K === 131071);
const L = { mul: new Int32Array(65536) };
for (let a = 0; a < 256; a++) for (let b = 0; b < 256; b++) L.mul[a * 256 + b] = (a << 24 >> 24) * (b << 24 >> 24);
const K = V.MAX_INT32_K + 1;
const worst = new Int8Array(K).fill(-128);
ck("lutMatmulJS refuses K = 131072 (all -128*-128: the sum 2^31 wraps to -2^31)",
   throwsName(() => V.lutMatmulJS(worst, worst, 1, K, 1, L), "AccumulatorBoundError"));
const acc = new Int32Array(1); acc[0] = 2 ** 31;
ck("(context) Int32Array really wraps 2^31 to -2^31 -- the silent failure being refused", acc[0] === -(2 ** 31));
ck("bgemmJS refuses the same K",
   throwsName(() => V.bgemmJS(worst, worst, new Float32Array(1), new Float32Array(1),
                              { m: 1, k: K, n: 1 }, L), "AccumulatorBoundError"));
const okK = worst.subarray(0, V.MAX_INT32_K);
ck("K = 131071 still runs and is exact (131071 * 16384)",
   V.lutMatmulJS(okK, okK, 1, V.MAX_INT32_K, 1, L)[0] === 131071 * 16384);

if (fails) { console.log("REDUCTIONS TEST FAILED: " + fails); process.exit(1); }
console.log("REDUCTIONS TEST PASSED — non-finite row maxes, bad gradients and int32 overflow are refused.");
