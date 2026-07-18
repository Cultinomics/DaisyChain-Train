// Mutation test for the kernel admission gate.
//
// A gate that has never rejected anything is decoration. This injects deliberately
// buggy "kernels" and asserts the gate rejects each one — and, separately, shows
// two things the OLD gate (single tiny shape, allclose) let through.
//
// Runs in Node against the CPU mirrors, so the buggy kernels stand in for what a
// wrong GPU shader would produce.
const fs = require("fs");
const path = require("path");
const V = require("./public/verified_core.js");
const TC = require("./public/traincore.js");

const mul = new Int16Array(fs.readFileSync(path.join(__dirname, "public", "mul_lut.bin")).buffer.slice(0));
const L = { mul };

// ---- the gate under test (mirrors webgpu.js gateBgemm) ----------------------
const SWEEP = [
  { m: 5, k: 9, n: 6, batch: 3, relu: true },
  { m: 32, k: 64, n: 32, batch: 1, relu: false },
  { m: 7, k: 253, n: 5, batch: 2, relu: true },
  { m: 1, k: 4, n: 1, batch: 1, relu: false },
  { m: 17, k: 33, n: 9, batch: 1, relu: true },
];
const OLD_SHAPE = [{ m: 5, k: 9, n: 6, batch: 3, relu: true }];   // what we used to test
const rnd = (n, f) => { const a = new Int8Array(n); for (let i = 0; i < n; i++) a[i] = f(); return a; };

function inputs(d) {
  const Xq = rnd(d.batch * d.m * d.k, () => (Math.random() * 256 - 128) | 0);
  const Wq = rnd(d.batch * d.k * d.n, () => (Math.random() * 256 - 128) | 0);
  const rs = Float32Array.from({ length: d.batch * d.m }, () => Math.random() + 0.5);
  const cs = Float32Array.from({ length: d.batch * d.n }, () => Math.random() + 0.5);
  return { Xq, Wq, rs, cs };
}

// exact gate: int32 accumulator with !==, then f32 epilogue compared AT THE
// BIT LEVEL (V.bitDiff) — `!==` can't see -0 vs +0, and the fleet's replica
// checks hash raw bytes, so the gate must compare what the hash sees.
function gateExact(kernel, shapes) {
  for (const d of shapes) {
    const { Xq, Wq, rs, cs } = inputs(d);
    const accHw = kernel(Xq, Wq, rs, cs, { ...d, acc: true }, L);
    const accRef = V.bgemmJS(Xq, Wq, rs, cs, { ...d, acc: true }, L);
    for (let i = 0; i < accRef.length; i++) if (accHw[i] !== accRef[i]) return `acc @${i}`;
    const hw = kernel(Xq, Wq, rs, cs, d, L);
    const ref = V.bgemmJS(Xq, Wq, rs, cs, d, L);
    for (let i = 0; i < ref.length; i++) if (V.bitDiff(hw[i], ref[i])) return `epilogue @${i}`;
  }
  return null;
}
// the old gate: one shape, allclose on the fused f32 output only
function gateOld(kernel) {
  for (const d of OLD_SHAPE) {
    const { Xq, Wq, rs, cs } = inputs(d);
    const hw = kernel(Xq, Wq, rs, cs, d, L);
    const ref = V.bgemmJS(Xq, Wq, rs, cs, d, L);
    for (let i = 0; i < ref.length; i++)
      if (Math.abs(hw[i] - ref[i]) > Math.abs(ref[i]) * 1e-6 + 1e-6) return `allclose @${i}`;
  }
  return null;
}

// ---- buggy kernels ----------------------------------------------------------
// Each is bgemmJS with exactly one realistic defect injected.
function mutant(bug) {
  return function (Xq, Wq, rs, cs, d, LL) {
    const { m, k, n } = d, batch = d.batch || 1, relu = !!d.relu, mulT = LL.mul;
    const raw = !!d.acc;
    const out = raw ? new Int32Array(batch * m * n) : new Float32Array(batch * m * n);
    const acc = new Int32Array(n);
    for (let bz = 0; bz < batch; bz++) {
      const xo = bz * m * k, oo = bz * m * n, co = bz * n;
      // BUG: batch stride dropped on W — invisible unless batch > 1
      const wo = bug === "batchStride" ? 0 : bz * k * n;
      for (let i = 0; i < m; i++) {
        acc.fill(0);
        const xrow = xo + i * k;
        // BUG: last element of the dot product skipped
        const kEnd = bug === "shortK" ? k - 1 : k;
        for (let p = 0; p < kEnd; p++) {
          const au = (Xq[xrow + p] & 0xFF) * 256, wrow = wo + p * n;
          for (let j = 0; j < n; j++) acc[j] += mulT[au + (Wq[wrow + j] & 0xFF)];
        }
        const orow = oo + i * n;
        if (raw) { for (let j = 0; j < n; j++) out[orow + j] = acc[j]; continue; }
        const rscale = rs[bz * m + i];
        for (let j = 0; j < n; j++) {
          // BUG: column scale ignored / ReLU dropped / f64 rounding (the real one)
          let v;
          if (bug === "missingColScale") v = V.epi(acc[j], rscale, 1);
          else if (bug === "f64Rounding") v = acc[j] * rscale * cs[co + j];   // rounds once, not thrice
          else v = V.epi(acc[j], rscale, cs[co + j]);
          out[orow + j] = (relu && bug !== "noRelu" && v < 0) ? 0 : v;
        }
      }
    }
    return out;
  };
}

let pass = true;
const ok = (cond, msg) => { console.log(`${cond ? "  ok  " : "  FAIL"}  ${msg}`); if (!cond) pass = false; };

console.log("\nthe gate must reject every injected bug:");
for (const bug of ["batchStride", "shortK", "missingColScale", "noRelu", "f64Rounding"])
  ok(gateExact(mutant(bug), SWEEP) !== null, `rejects ${bug}  (${gateExact(mutant(bug), SWEEP) || "NOT CAUGHT"})`);

console.log("\nthe gate must accept the correct kernel:");
ok(gateExact(V.bgemmJS, SWEEP) === null, "accepts the real mirror across the shape sweep");

console.log("\nwhat the OLD gate (one tiny shape, allclose) let through:");
const oldMissedRounding = gateOld(mutant("f64Rounding")) === null;
const newCatchesRounding = gateExact(mutant("f64Rounding"), SWEEP) !== null;
ok(oldMissedRounding && newCatchesRounding,
   "f64-vs-f32 epilogue rounding: allclose PASSES it, exact gate catches it");
const oldMissedStride = gateOld(mutant("batchStride")) === null;
console.log(`  note   batchStride under the old gate: ${oldMissedStride ? "MISSED" : "caught (batch>1 was in the shape)"}`);

// ---- the sign of zero -------------------------------------------------------
// Real ISAs have non-IEEE modes that flush -0 to +0 (RDNA2 output modifiers /
// DX9-legacy multiplies). JS `!==` treats -0 === 0, so a value-level gate is
// BLIND to that flush — yet the sync guard hashes raw bits, so it would fork
// the fleet. The audit must therefore compare bit patterns.
console.log("\nthe audit must see the sign of zero (`!==` cannot):");
{
  const d = { m: 2, k: 4, n: 2 };
  const Xq = new Int8Array(d.m * d.k);                        // all zero -> acc = 0 -> epi = +0
  const Wq = rnd(d.k * d.n, () => (Math.random() * 256 - 128) | 0);
  const rs = Float32Array.from({ length: d.m }, () => Math.random() + 0.5);
  const cs = Float32Array.from({ length: d.n }, () => Math.random() + 0.5);
  const got = V.bgemmJS(Xq, Wq, rs, cs, { ...d, batch: 1 }, L);
  ok(got.every((v) => !V.bitDiff(v, 0)), "sanity: the units produce +0 here");
  got[1] = -0;                                                // what a -0-flushing device's INVERSE would look like; either direction diverges the hash
  ok(got.every((v) => v === 0), "sanity: `!==` cannot tell the corrupted output apart");
  const bad = V.auditTile(Xq, Wq, rs, cs, d, got, L, 400);    // 400 samples over 4 cells: hits the bad cell w.p. 1-(3/4)^400
  ok(bad !== null, `auditTile flags the -0 (${bad || "NOT CAUGHT"})`);
}

// ---- the audit's sampling strategy ------------------------------------------
// A bounds-guard off-by-one or a pack-tail padding bug corrupts the LAST
// row/column only. At the live logits shape (n = 16512) that is 1 cell in
// 16512, so uniformly random sampling essentially never lands on it. The audit
// now spends its first cells on the structural danger points deliberately.
// This measures both strategies against that named bug class.
console.log("\nthe audit must catch a LAST-COLUMN bug (uniform sampling cannot):");
{
  const d = { m: 64, k: 16, n: 512 };                 // n wide, like the real unembed
  const Xq = rnd(d.m * d.k, () => (Math.random() * 256 - 128) | 0);
  const Wq = rnd(d.k * d.n, () => (Math.random() * 256 - 128) | 0);
  const rs = Float32Array.from({ length: d.m }, () => Math.random() + 0.5);
  const cs = Float32Array.from({ length: d.n }, () => Math.random() + 0.5);
  const good = V.bgemmJS(Xq, Wq, rs, cs, { ...d, batch: 1 }, L);

  const corrupt = () => {                              // last column only
    const g = Float32Array.from(good);
    for (let i = 0; i < d.m; i++) g[i * d.n + (d.n - 1)] = 0;
    return g;
  };
  // the OLD strategy: every cell uniform-random
  const uniformAudit = (got, nCells) => {
    for (let t = 0; t < nCells; t++) {
      const i = (Math.random() * d.m) | 0, j = (Math.random() * d.n) | 0;
      let acc = 0;
      for (let p = 0; p < d.k; p++) acc += mul[(Xq[i * d.k + p] & 0xFF) * 256 + (Wq[p * d.n + j] & 0xFF)];
      if (V.bitDiff(got[i * d.n + j], V.epi(acc, rs[i], cs[j]))) return true;
    }
    return false;
  };
  const TRIALS = 300;
  let uniHits = 0, stratHits = 0;
  for (let t = 0; t < TRIALS; t++) {
    if (uniformAudit(corrupt(), 12)) uniHits++;
    if (V.auditTile(Xq, Wq, rs, cs, d, corrupt(), L, 12) !== null) stratHits++;
  }
  console.log(`  uniform 12 cells : caught ${uniHits}/${TRIALS} audits  (expected ~${(100*(1-Math.pow(1-1/d.n,12))).toFixed(1)}%)`);
  console.log(`  stratified 12    : caught ${stratHits}/${TRIALS} audits`);
  ok(stratHits === TRIALS, "stratified sampling catches the last-column bug on EVERY audit");
  ok(uniHits < TRIALS * 0.25, "uniform sampling misses it almost always — the strategy, not the cell count, is what changed");
  // and it must not cry wolf on a clean kernel
  let fp = 0;
  for (let t = 0; t < TRIALS; t++) if (V.auditTile(Xq, Wq, rs, cs, d, good, L, 12) !== null) fp++;
  ok(fp === 0, `no false positives on the clean kernel (${TRIALS} audits)`);
}

// ---- the attention kernels now get a LIVE audit too --------------------------
// They had exact init gates but nothing at live shapes — the same gap the GEMM
// audit exists to close. These must reject a corrupted attention output and
// accept a clean one, at a shape the init gate never sees.
console.log("\nthe attention audits must reject corrupted output:");
{
  const d = { B: 2, T: 24, heads: 3, hd: 8 };          // not one of the gate's shapes
  const C = d.heads * d.hd, BH = d.B * d.heads;
  const qq = rnd(d.B * d.T * C, () => (Math.random() * 256 - 128) | 0);
  const kq = rnd(d.B * d.T * C, () => (Math.random() * 256 - 128) | 0);
  const qs = Float32Array.from({ length: d.B * d.T * d.heads }, () => Math.random() + 0.5);
  const ks = Float32Array.from({ length: d.B * d.T * d.heads }, () => Math.random() + 0.5);
  const good = V.attScoresJS(qq, kq, qs, ks, d, L);
  ok(V.auditAttScores(qq, kq, qs, ks, d, good, L, 12) === null, "att.scores audit accepts the clean kernel");
  // corrupt the LAST head's last token pair — the classic head-stride bug
  const badS = Float32Array.from(good);
  badS[((d.B * d.heads - 1) * d.T + (d.T - 1)) * d.T + (d.T - 1)] = 0;
  ok(V.auditAttScores(qq, kq, qs, ks, d, badS, L, 12) !== null, "att.scores audit rejects a corrupted last head/token");

  const aq = rnd(BH * d.T * d.T, () => (Math.random() * 127) | 0);
  const vq = rnd(d.B * d.T * C, () => (Math.random() * 256 - 128) | 0);
  const as = Float32Array.from({ length: BH * d.T }, () => Math.random() + 0.5);
  const vs = Float32Array.from({ length: BH * d.hd }, () => Math.random() + 0.5);
  const goodC = V.attCtxJS(aq, vq, as, vs, d, L);
  ok(V.auditAttCtx(aq, vq, as, vs, d, goodC, L, 12) === null, "att.ctx audit accepts the clean kernel");
  const badC = Float32Array.from(goodC);               // last channel of the last head
  badC[((d.B - 1) * d.T + (d.T - 1)) * C + (d.heads - 1) * d.hd + (d.hd - 1)] = 0;
  ok(V.auditAttCtx(aq, vq, as, vs, d, badC, L, 12) !== null, "att.ctx audit rejects a corrupted last channel (scatter write-back)");
}

// ---- the f32 backward GEMM is no longer gated by a tolerance -----------------
// fgemmMirror reproduces split-K's partition and accumulation order, so the
// gate can compare with `!==`. This checks the mirror is self-consistent and
// that the two rounding schedules it offers are genuinely different (i.e. the
// gate is choosing between real alternatives, not two names for one thing).
console.log("\nthe split-K mirror is exact and discriminating:");
{
  const m0 = 6, k0 = 5000, n0 = 4;                     // k > 4096 exercises split-K
  const A = Float32Array.from({ length: m0 * k0 }, () => Math.random() - 0.5);
  const Bm = Float32Array.from({ length: k0 * n0 }, () => Math.random() - 0.5);
  const dG = { m: m0, k: k0, n: n0 };
  const stepped = V.fgemmMirror(A, Bm, dG, false), fused = V.fgemmMirror(A, Bm, dG, true);
  ok(V.fgemmMirror(A, Bm, dG, false).every((v, i) => !V.bitDiff(v, stepped[i])), "mirror is deterministic");
  let diff = 0;
  for (let i = 0; i < stepped.length; i++) if (V.bitDiff(stepped[i], fused[i])) diff++;
  ok(diff > 0, `the two rounding schedules really differ (${diff}/${stepped.length} cells) — the gate picks between real alternatives`);
  const naive = TC.matmul(A, Bm, m0, k0, n0);
  let vsNaive = 0;
  for (let i = 0; i < naive.length; i++) if (V.bitDiff(naive[i], stepped[i])) vsNaive++;
  console.log(`  note   split-K order differs from a naive matmul on ${vsNaive}/${naive.length} cells — which is why the old gate had to use a tolerance, and why the mirror had to be written`);
}

console.log(pass ? "\nGATE TEST PASSED — the gate rejects real bugs and accepts the real kernel."
                 : "\nGATE TEST FAILED");
process.exit(pass ? 0 : 1);
