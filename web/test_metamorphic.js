// Metamorphic tests: correctness properties that need NO reference implementation.
//
// Differential testing (kernel vs mirror) is only as strong as the independence
// of the two implementations, and every time you tune the oracle to agree with
// the thing it's checking, you spend some of that independence. It is also
// structurally blind to a bug in SHARED source: both replicas compile the same
// WGSL string, so a mutation there propagates to every device and every mirror
// tuned to match it.
//
// These properties come from the DEFINITION of a block-scaled batched GEMM, not
// from any implementation of one. When one fails, there is no referee question:
// the behaviour is wrong, regardless of which side is "right". They hold exactly
// (not approximately) because per-row/per-column quantization commutes with
// permutation, and because int32 accumulation is exactly associative.
const fs = require("fs");
const path = require("path");
const V = require("./public/verified_core.js");
const L = { mul: new Int16Array(fs.readFileSync(path.join(__dirname, "public", "mul_lut.bin")).buffer.slice(0)) };

const randf = (n, f) => Float32Array.from({ length: n }, f || (() => Math.random() * 2 - 1));

// The kernel under test: float in, float out, block-scaled through the units.
// Swap in a mutant to prove the properties actually bite.
function makeKernel(bug) {
  return async (Xf, Wf, d) => {
    const { m, k, n } = d, batch = d.batch || 1;
    if (bug === "accOverwrite") {                 // corpus: matmul_triton_buggy
      const q = V.quantizeRows(Xf, batch * m, k), w = V.quantizeCols(Wf, k, n);
      const out = d.acc ? new Int32Array(batch * m * n) : new Float32Array(batch * m * n);
      for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) {
        let a = 0;
        for (let p = 0; p < k; p++) a = L.mul[(q.q[i * k + p] & 0xFF) * 256 + (w.q[p * n + j] & 0xFF)];
        out[i * n + j] = d.acc ? a : V.epi(a, q.s[i], w.s[j]);
      }
      return out;
    }
    const x = V.quantizeRows(Xf, batch * m, k);
    let wq, ws;
    if (batch === 1) { const w = V.quantizeCols(Wf, k, n); wq = w.q; ws = w.s; }
    else {
      wq = new Int8Array(batch * k * n); ws = new Float32Array(batch * n);
      for (let bz = 0; bz < batch; bz++) {
        const w = V.quantizeCols(Wf.subarray(bz * k * n, (bz + 1) * k * n), k, n);
        wq.set(w.q, bz * k * n); ws.set(w.s, bz * n);
      }
    }
    if (bug === "batchStride") {          // ignores the batch offset on W
      const d2 = { ...d };
      const out = new Float32Array(batch * m * n);
      for (let bz = 0; bz < batch; bz++) {
        const o = V.bgemmJS(x.q.subarray(bz * m * k, (bz + 1) * m * k), wq.subarray(0, k * n),
                            x.s.subarray(bz * m, (bz + 1) * m), ws.subarray(0, n),
                            { ...d2, batch: 1 }, L);
        out.set(o, bz * m * n);
      }
      return out;
    }
    if (bug === "rowSwap") {              // transposes two output rows
      const out = V.bgemmJS(x.q, wq, x.s, ws, d, L);
      if (m > 1) for (let j = 0; j < n; j++) { const t = out[0 * n + j]; out[0 * n + j] = out[1 * n + j]; out[1 * n + j] = t; }
      return out;
    }
    return V.bgemmJS(x.q, wq, x.s, ws, d, L);
  };
}

const eq = (a, b) => { for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false; return true; };

// ---- the properties ---------------------------------------------------------
const PROPS = {
  // NON-TRIVIALITY. Added after an external bug corpus scored the suite below
  // 0/4: the zero function satisfies every relation here, because zero is
  // permutation-equivariant, zero-row-preserving and batch-decomposable. Without
  // this, a kernel can pass the whole suite by doing nothing at all.
  async nonTriviality(K) {
    const d = { m: 6, k: 32, n: 5, batch: 1 };
    const out = await K(randf(d.m * d.k), randf(d.k * d.n), d);
    let nz = 0;
    for (let i = 0; i < out.length; i++) if (out[i] !== 0) nz++;
    if (nz < out.length / 4) return `output is ${out.length - nz}/${out.length} zeros`;
    return null;
  },
  // SENSITIVITY. Every element of A must be able to move its output row. Catches
  // an accumulator that overwrites instead of accumulating (acc= for acc+=):
  // every structural relation still holds, but only the last k contributes.
  // Measured on the raw accumulator, perturbed by a sign flip that leaves the
  // row absmax alone — otherwise the perturbation moves the output through the
  // quantization SCALE and the property proves nothing.
  async sensitivity(K) {
    const d = { m: 6, k: 32, n: 5, batch: 1, acc: true };
    const A = randf(d.m * d.k), B = randf(d.k * d.n);
    for (let p = 0; p < d.k; p++) A[1 * d.k + p] = 0.3;
    A[1 * d.k + 0] = 1.0;                               // pin the absmax at p=0
    const s0 = V.quantizeRows(A, d.m, d.k).s[1];
    const base = await K(A, B, d);
    for (let p = 1; p < d.k; p++) {
      const A2 = Float32Array.from(A);
      A2[1 * d.k + p] = -0.3;
      if (V.quantizeRows(A2, d.m, d.k).s[1] !== s0) continue;   // inconclusive
      const o2 = await K(A2, B, d);
      let moved = false;
      for (let j = 0; j < d.n; j++) if (o2[1 * d.n + j] !== base[1 * d.n + j]) { moved = true; break; }
      if (!moved) return `A[.,${p}] does not affect its output row`;
    }
    return null;
  },
  // a zero row of A must produce a zero row of output, whatever the scales are
  async zeroRow(K) {
    const d = { m: 6, k: 32, n: 5, batch: 1 };
    const A = randf(d.m * d.k), B = randf(d.k * d.n);
    for (let p = 0; p < d.k; p++) A[2 * d.k + p] = 0;
    const out = await K(A, B, d);
    for (let j = 0; j < d.n; j++) if (out[2 * d.n + j] !== 0) return `row 2 was zeroed but output[2,${j}] = ${out[2 * d.n + j]}`;
    return null;
  },
  // permuting rows of A must permute rows of the output the same way
  async rowPermutation(K) {
    const d = { m: 6, k: 32, n: 5, batch: 1 };
    const A = randf(d.m * d.k), B = randf(d.k * d.n);
    const perm = [3, 1, 5, 0, 4, 2];
    const Ap = new Float32Array(A.length);
    perm.forEach((src, dst) => Ap.set(A.subarray(src * d.k, (src + 1) * d.k), dst * d.k));
    const out = await K(A, B, d), outP = await K(Ap, B, d);
    for (let r = 0; r < d.m; r++)
      for (let j = 0; j < d.n; j++)
        if (outP[r * d.n + j] !== out[perm[r] * d.n + j])
          return `permuting rows of A did not permute the output at [${r},${j}]`;
    return null;
  },
  // permuting columns of B must permute columns of the output the same way
  async colPermutation(K) {
    const d = { m: 4, k: 32, n: 5, batch: 1 };
    const A = randf(d.m * d.k), B = randf(d.k * d.n);
    const perm = [2, 0, 4, 1, 3];
    const Bp = new Float32Array(B.length);
    for (let p = 0; p < d.k; p++) perm.forEach((src, dst) => { Bp[p * d.n + dst] = B[p * d.n + src]; });
    const out = await K(A, B, d), outP = await K(A, Bp, d);
    for (let r = 0; r < d.m; r++)
      for (let j = 0; j < d.n; j++)
        if (outP[r * d.n + j] !== out[r * d.n + perm[j]])
          return `permuting columns of B did not permute the output at [${r},${j}]`;
    return null;
  },
  // a batched call must equal running each batch element on its own
  async batchDecomposition(K) {
    const d = { m: 4, k: 32, n: 5, batch: 3 };
    const A = randf(d.batch * d.m * d.k), B = randf(d.batch * d.k * d.n);
    const together = await K(A, B, d);
    for (let bz = 0; bz < d.batch; bz++) {
      const alone = await K(A.subarray(bz * d.m * d.k, (bz + 1) * d.m * d.k),
                           B.subarray(bz * d.k * d.n, (bz + 1) * d.k * d.n), { ...d, batch: 1 });
      const slice = together.subarray(bz * d.m * d.n, (bz + 1) * d.m * d.n);
      if (!eq(alone, slice)) return `batch element ${bz} differs when computed alone vs batched`;
    }
    return null;
  },
};

(async () => {
  let pass = true;
  const ok = (c, msg) => { console.log(`${c ? "  ok  " : "  FAIL"}  ${msg}`); if (!c) pass = false; };

  console.log("\nproperties hold for the real kernel (no reference implementation used):");
  for (const [name, prop] of Object.entries(PROPS)) {
    const bad = await prop(makeKernel(null));
    ok(bad === null, `${name}${bad ? "  -> " + bad : ""}`);
  }

  // The point: these catch bugs with NO oracle. A mutation living in source
  // shared by every replica would sail past the probe and past a mirror tuned
  // to agree with it. It cannot sail past arithmetic that must be true.
  console.log("\nthe same properties catch bugs with no oracle to compare against:");
  const strideBad = await PROPS.batchDecomposition(makeKernel("batchStride"));
  ok(strideBad !== null, `batchDecomposition catches a dropped batch stride  (${strideBad || "MISSED"})`);
  const swapBad = await PROPS.rowPermutation(makeKernel("rowSwap"));
  ok(swapBad !== null, `rowPermutation catches swapped output rows  (${swapBad || "MISSED"})`);
  const swapZero = await PROPS.zeroRow(makeKernel("rowSwap"));
  console.log(`  note   zeroRow vs the same rowSwap bug: ${swapZero ? "caught" : "missed (properties are partial, not a proof)"}`);
  // a bug from someone else's taxonomy, not mine: acc= instead of acc+=. Every
  // structural relation survives it, which is why sensitivity had to exist.
  const accSens = await PROPS.sensitivity(makeKernel("accOverwrite"));
  const accPerm = await PROPS.rowPermutation(makeKernel("accOverwrite"));
  ok(accSens !== null, `sensitivity catches acc= instead of acc+=  (${accSens || "MISSED"})`);
  console.log(`  note   rowPermutation vs that same acc= bug: ${accPerm ? "caught" : "missed — it is structure-preserving, which is the whole trap"}`);

  console.log(pass ? "\nMETAMORPHIC TEST PASSED" : "\nMETAMORPHIC TEST FAILED");
  process.exit(pass ? 0 : 1);
})();
