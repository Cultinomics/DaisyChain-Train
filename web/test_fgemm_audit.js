// Live audit of the split-K f32 GEMM (2026-10-14 review).
// auditFgemm recomputes single cells with the mirror's partition and rounding
// schedule; it must (a) accept the mirror's own output at the LIVE shapes
// (transA embedding gradient, split-K dlnf), and (b) reject a one-ulp error
// in a corner cell and in an interior cell, under both rounding schedules.
const V = require("./public/verified_core.js");

let pass = 0, fail = 0;
const t = (name, ok, detail) => {
  if (ok) { pass++; console.log("  PASS  " + name); }
  else { fail++; console.log(`  FAIL  ${name}  [${detail || ""}]`); }
};
const rnd = (n) => Float32Array.from({ length: n }, () => Math.random() - 0.5);
const bumpUlp = (x) => { const f = new Float32Array([x]), u = new Uint32Array(f.buffer); u[0] += 1; return f[0]; };

// live-like shapes, scaled down: transA leg (m=vocab, k=BT) and the split-K leg (k=vocab > 4096)
const shapes = [{ m: 600, k: 64, n: 12, transA: true }, { m: 64, k: 5000, n: 12 }, { m: 3, k: 9000, n: 2, transA: true }];
for (const fma of [false, true]) {
  for (const d of shapes) {
    const A = rnd(d.m * d.k), B = rnd(d.k * d.n);
    const out = V.fgemmMirror(A, B, d, fma);
    const tag = `${d.m}x${d.k}x${d.n}${d.transA ? " transA" : ""} fma=${fma}`;
    t(`audit accepts the mirror (${tag})`, V.auditFgemm(A, B, d, out, fma, 16) === null);
    const corner = out.slice(); corner[d.m * d.n - 1] = bumpUlp(corner[d.m * d.n - 1]);
    t(`audit catches a 1-ulp error in the last cell (${tag})`, V.auditFgemm(A, B, d, corner, fma, 4) !== null);
    // interior: the random cells must eventually find it; 0 corners, 400 random draws on a small grid
    const small = { m: 5, k: d.k, n: 3, transA: d.transA };
    const As = rnd(small.m * small.k), Bs = rnd(small.k * small.n);
    const o2 = V.fgemmMirror(As, Bs, small, fma); o2[1 * 3 + 1] = bumpUlp(o2[1 * 3 + 1]);
    t(`audit's random cells find an interior error (${tag})`, V.auditFgemm(As, Bs, small, o2, fma, 400) !== null);
  }
}
// The schedule matters: on split-K data the two schedules differ somewhere, so auditing
// with the wrong schedule must be able to fail -- otherwise the fma flag is untested.
{
  const d = { m: 4, k: 6000, n: 4 };
  const A = Float32Array.from({ length: d.m * d.k }, () => (Math.random() - 0.5) * 3);
  const B = Float32Array.from({ length: d.k * d.n }, () => (Math.random() - 0.5) * 3);
  const step = V.fgemmMirror(A, B, d, false), fused = V.fgemmMirror(A, B, d, true);
  let differ = false;
  for (let i = 0; i < step.length; i++) if (step[i] !== fused[i]) { differ = true; break; }
  t("the two rounding schedules differ on split-K data (fma flag is load-bearing)", differ);
  t("auditing fused output with the two-rounding schedule fails", V.auditFgemm(A, B, d, fused, false, 200) !== null);
}
console.log(`test_fgemm_audit.js: ${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
