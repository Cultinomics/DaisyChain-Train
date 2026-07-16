// Mutation-scoring the ORACLES, using a bug taxonomy someone else wrote.
//
// Bugs ported from dipankarsarkar/gpuemu-corpus (the 10 buggy variants). Their
// kernels are Triton/numpy and mine are WGSL/JS, so nothing runs across. What
// ports is the taxonomy: each bug is specified, so it can be re-injected into a
// different codebase. The point is that the bug list has a different author than
// the checks, which is the one thing I cannot manufacture myself.
//
// The question is NOT "do my kernels pass". It is "which of my checks catch a
// bug I did not think of". Every bug an oracle waves through is a hole with a
// name on it instead of a feeling about my own coverage.
const fs = require("fs");
const path = require("path");
const V = require("./public/verified_core.js");
const L = { mul: new Int16Array(fs.readFileSync(path.join(__dirname, "public", "mul_lut.bin")).buffer.slice(0)) };

const f32 = Math.fround;
const randf = (n) => Float32Array.from({ length: n }, () => Math.random() * 2 - 1);

// ---- MY kernel, with one corpus bug injected at a time ----------------------
// Faithful to the real bgemmJS/WGSL structure, so the injected bug is the only
// difference. `bug` names map to corpus variants; see TAXONOMY below.
function kernel(bug) {
  return (Xq, Wq, rs, cs, d) => {
    const { m, k, n } = d, batch = d.batch || 1, relu = !!d.relu, mul = L.mul, raw = !!d.acc;
    const out = raw ? new Int32Array(batch * m * n) : new Float32Array(batch * m * n);
    const acc = new Int32Array(n);
    for (let bz = 0; bz < batch; bz++) {
      const xo = bz * m * k, wo = bz * k * n, oo = bz * m * n, co = bz * n;
      for (let i = 0; i < m; i++) {
        acc.fill(0);
        const xrow = xo + i * k;
        for (let p = 0; p < k; p++) {
          const au = (Xq[xrow + p] & 0xFF) * 256, wrow = wo + p * n;
          for (let j = 0; j < n; j++) {
            const prod = mul[au + (Wq[wrow + j] & 0xFF)];
            // corpus: matmul_triton_buggy — acc= instead of acc+=, "correct only when K=1"
            if (bug === "accOverwrite") acc[j] = prod; else acc[j] += prod;
          }
        }
        const orow = oo + i * n;
        if (raw) {
          for (let j = 0; j < n; j++) out[orow + j] = acc[j];
          continue;
        }
        const rscale = rs[bz * m + i];
        for (let j = 0; j < n; j++) {
          // corpus: softmax_llm_buggy — "correct only when last dim is a multiple
          // of 128". My analogue: workgroup_size(8,8), so a missing bounds guard
          // is correct only when n is a multiple of 8.
          if (bug === "boundsMult8" && j >= (n - (n % 8))) { out[orow + j] = 0; continue; }
          let v = V.epi(acc[j], rscale, cs[co + j]);
          // corpus: gelu_triton_buggy — a dropped constant factor, uniformly ~2x off
          if (bug === "droppedFactor") v = f32(v * 2);
          // corpus: leaky_relu_triton_buggy — wrong alpha (0.1 vs 0.01)
          if (relu && bug === "leakyAlpha") v = v < 0 ? f32(v * 0.1) : v;
          else if (relu && v < 0) v = 0;
          out[orow + j] = v;
        }
      }
    }
    return out;
  };
}

// what each injected bug is, and where it lives
const TAXONOMY = {
  accOverwrite:  { from: "matmul_triton_buggy",  where: "loop", desc: "acc= instead of acc+= (only correct at K=1)" },
  boundsMult8:   { from: "softmax_llm_buggy",    where: "loop", desc: "correct only when n is a multiple of the workgroup width" },
  droppedFactor: { from: "gelu_triton_buggy",    where: "math", desc: "constant factor dropped, uniformly 2x" },
  leakyAlpha:    { from: "leaky_relu_buggy",     where: "math", desc: "wrong negative-slope alpha" },
};

// ---- oracle 1: the property suite (no reference implementation) -------------
const SHAPES = { m: 6, k: 32, n: 5, batch: 1 };
function quantize(Xf, Wf, d) {
  const batch = d.batch || 1;
  const x = V.quantizeRows(Xf, batch * d.m, d.k);
  let wq, ws;
  if (batch === 1) { const w = V.quantizeCols(Wf, d.k, d.n); wq = w.q; ws = w.s; }
  else {
    wq = new Int8Array(batch * d.k * d.n); ws = new Float32Array(batch * d.n);
    for (let bz = 0; bz < batch; bz++) {
      const w = V.quantizeCols(Wf.subarray(bz * d.k * d.n, (bz + 1) * d.k * d.n), d.k, d.n);
      wq.set(w.q, bz * d.k * d.n); ws.set(w.s, bz * d.n);
    }
  }
  return { x, wq, ws };
}
const call = (K, Xf, Wf, d) => { const q = quantize(Xf, Wf, d); return K(q.x.q, q.wq, q.x.s, q.ws, d); };

function propertySuite(K) {
  // NON-TRIVIALITY. Added after the corpus scored my suite 0/4: a kernel that
  // returns all zeros satisfies every relation below, because zero is
  // permutation-equivariant, zero-row-preserving and batch-decomposable. Every
  // algebraic property suite needs this or it can be passed by doing nothing.
  {
    const d = { ...SHAPES };
    const A = randf(d.m * d.k), B = randf(d.k * d.n);
    const o = call(K, A, B, d);
    let nz = 0;
    for (let i = 0; i < o.length; i++) if (o[i] !== 0) nz++;
    if (nz < o.length / 4) return "nonTriviality";
  }
  // SENSITIVITY. Every element of A must be able to move its output row. This is
  // what catches an accumulator that overwrites instead of accumulating: the
  // structure is perfect, every relation holds, but only the last k contributes.
  //
  // Measured against the RAW accumulator, and perturbed by a sign flip that
  // leaves |value| untouched. Both matter: block scaling coupled the row absmax
  // to every output in the row, so a naive perturbation moves the output through
  // the SCALE and the test passes while proving nothing about the accumulation.
  // That is how the first version of this property fooled me.
  {
    const d = { ...SHAPES, acc: true };
    const A = randf(d.m * d.k), B = randf(d.k * d.n);
    for (let p = 0; p < d.k; p++) A[1 * d.k + p] = 0.3;
    A[1 * d.k + 0] = 1.0;                          // pin the row absmax at p=0
    const q0 = quantize(A, B, d);
    const base = K(q0.x.q, q0.wq, q0.x.s, q0.ws, d);
    for (let p = 1; p < d.k; p++) {
      const A2 = Float32Array.from(A);
      A2[1 * d.k + p] = -0.3;                      // sign flip: absmax unchanged
      const q2 = quantize(A2, B, d);
      if (q2.x.s[1] !== q0.x.s[1]) continue;       // scale moved anyway: inconclusive
      const o2 = K(q2.x.q, q2.wq, q2.x.s, q2.ws, d);
      let moved = false;
      for (let j = 0; j < d.n; j++) if (o2[1 * d.n + j] !== base[1 * d.n + j]) { moved = true; break; }
      if (!moved) return `sensitivity(A[.,${p}] ignored)`;
    }
  }
  // RELU RANGE. relu(x) >= 0 is part of the kernel's DEFINITION when the
  // fused ReLU is on — a range constraint, not a relation between calls.
  // This is what catches a wrong negative slope: the structure survives a
  // leak, the sign does not.
  {
    const d = { ...SHAPES, relu: true };
    const A = randf(d.m * d.k), B = randf(d.k * d.n);
    const o = call(K, A, B, d);
    for (let i = 0; i < o.length; i++) if (o[i] < 0) return "reluRange";
  }
  // UNIT-SCALE ANCHOR. With rs = cs = 1 the dequant is the identity, so the
  // definition pins ABSOLUTE values: out must equal the exact integer dot
  // product (f32-exact far below 2^24). No RELATION can do this — if out
  // satisfies every relation, so does c*out — so the suite needs one point
  // where the spec fixes the scale. Expected values come from plain integer
  // arithmetic: no reference implementation, no LUT, no mirror.
  {
    const d = { m: 3, k: 5, n: 4, batch: 1 };
    const Xq = new Int8Array(d.m * d.k), Wq = new Int8Array(d.k * d.n);
    for (let i = 0; i < Xq.length; i++) Xq[i] = ((i * 37 + 11) % 25) - 12;
    for (let i = 0; i < Wq.length; i++) Wq[i] = ((i * 53 + 7) % 25) - 12;
    const rs = new Float32Array(d.m).fill(1), cs = new Float32Array(d.n).fill(1);
    const o = K(Xq, Wq, rs, cs, d);
    for (let i = 0; i < d.m; i++)
      for (let j = 0; j < d.n; j++) {
        let dot = 0;
        for (let p = 0; p < d.k; p++) dot += Xq[i * d.k + p] * Wq[p * d.n + j];
        if (o[i * d.n + j] !== dot) return "unitScaleAnchor";
      }
  }
  // zero row of A -> zero row out
  {
    const d = { ...SHAPES };
    const A = randf(d.m * d.k), B = randf(d.k * d.n);
    for (let p = 0; p < d.k; p++) A[2 * d.k + p] = 0;
    const o = call(K, A, B, d);
    for (let j = 0; j < d.n; j++) if (o[2 * d.n + j] !== 0) return "zeroRow";
  }
  // permuting rows of A permutes output rows
  {
    const d = { ...SHAPES };
    const A = randf(d.m * d.k), B = randf(d.k * d.n);
    const perm = [3, 1, 5, 0, 4, 2];
    const Ap = new Float32Array(A.length);
    perm.forEach((src, dst) => Ap.set(A.subarray(src * d.k, (src + 1) * d.k), dst * d.k));
    const o = call(K, A, B, d), op = call(K, Ap, B, d);
    for (let r = 0; r < d.m; r++) for (let j = 0; j < d.n; j++)
      if (op[r * d.n + j] !== o[perm[r] * d.n + j]) return "rowPermutation";
  }
  // permuting columns of B permutes output columns
  {
    const d = { ...SHAPES };
    const A = randf(d.m * d.k), B = randf(d.k * d.n);
    const perm = [2, 0, 4, 1, 3];
    const Bp = new Float32Array(B.length);
    for (let p = 0; p < d.k; p++) perm.forEach((src, dst) => { Bp[p * d.n + dst] = B[p * d.n + src]; });
    const o = call(K, A, B, d), op = call(K, A, Bp, d);
    for (let r = 0; r < d.m; r++) for (let j = 0; j < d.n; j++)
      if (op[r * d.n + j] !== o[r * d.n + perm[j]]) return "colPermutation";
  }
  // batched == each element on its own
  {
    const d = { m: 4, k: 32, n: 5, batch: 3 };
    const A = randf(d.batch * d.m * d.k), B = randf(d.batch * d.k * d.n);
    const together = call(K, A, B, d);
    for (let bz = 0; bz < d.batch; bz++) {
      const alone = call(K, A.subarray(bz * d.m * d.k, (bz + 1) * d.m * d.k),
                            B.subarray(bz * d.k * d.n, (bz + 1) * d.k * d.n), { ...d, batch: 1 });
      for (let i = 0; i < alone.length; i++)
        if (alone[i] !== together[bz * d.m * d.n + i]) return "batchDecomposition";
    }
  }
  return null;
}

// ---- oracle 2: the exact differential gate (kernel vs the JS mirror) --------
function differentialGate(K) {
  for (const d of [{ m: 5, k: 9, n: 6, batch: 3, relu: true },
                   { m: 32, k: 64, n: 32, batch: 1, relu: false },
                   { m: 7, k: 253, n: 5, batch: 2, relu: true },
                   { m: 17, k: 33, n: 9, batch: 1, relu: true }]) {
    const A = randf(d.batch * d.m * d.k), B = randf(d.batch * d.k * d.n);
    const q = quantize(A, B, d);
    for (const acc of [true, false]) {
      const dd = { ...d, acc };
      const hw = K(q.x.q, q.wq, q.x.s, q.ws, dd);
      const ref = V.bgemmJS(q.x.q, q.wq, q.x.s, q.ws, dd, L);
      for (let i = 0; i < ref.length; i++) if (hw[i] !== ref[i]) return (acc ? "accumulator" : "epilogue");
    }
  }
  return null;
}

// ---- run --------------------------------------------------------------------
console.log("\nBugs from dipankarsarkar/gpuemu-corpus, re-injected into my kernel.");
console.log("Scoring the ORACLES, not the kernel.\n");
console.log("bug              lives in   properties        differential");
console.log("---------------------------------------------------------------");
const score = { prop: 0, diff: 0, total: 0 };
for (const [bug, meta] of Object.entries(TAXONOMY)) {
  const K = kernel(bug);
  const p = propertySuite(K), dgate = differentialGate(K);
  score.total++;
  if (p) score.prop++;
  if (dgate) score.diff++;
  console.log(`${bug.padEnd(16)} ${meta.where.padEnd(10)} ${(p ? "CAUGHT (" + p + ")" : "MISSED").padEnd(17)} ${dgate ? "CAUGHT (" + dgate + ")" : "MISSED"}`);
}
// control: the real kernel must pass both
const cleanP = propertySuite(kernel(null)), cleanD = differentialGate(kernel(null));
console.log(`${"(no bug)".padEnd(16)} ${"-".padEnd(10)} ${(cleanP ? "FALSE POSITIVE" : "clean").padEnd(17)} ${cleanD ? "FALSE POSITIVE" : "clean"}`);

console.log(`\nmutation score, properties:   ${score.prop}/${score.total}`);
console.log(`mutation score, differential: ${score.diff}/${score.total}`);
console.log("\nby where the bug lives:");
for (const w of ["loop", "math"]) {
  const bugs = Object.entries(TAXONOMY).filter(([, m]) => m.where === w);
  const caught = bugs.filter(([b]) => propertySuite(kernel(b)) !== null).length;
  console.log(`  ${w.padEnd(5)} properties caught ${caught}/${bugs.length}`);
}
const ok = !cleanP && !cleanD;
console.log(ok ? "\nCONTROL OK (clean kernel passes both oracles)" : "\nCONTROL FAILED");
process.exit(ok ? 0 : 1);
