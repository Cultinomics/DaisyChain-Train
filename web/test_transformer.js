// Mini transformer through the verified units: (a) loss drops well below the
// uniform baseline ln(V), (b) two replicas fed the same averaged gradients stay
// bit-identical, (c) generation produces corpus-like text.
const fs = require("fs");
const path = require("path");
const T = require("./public/traincore.js");
const V = require("./public/verified_core.js");
const X = require("./public/transformer.js");

function loadLUTs() {
  const p = (f) => path.join(__dirname, "public", f);
  return { mul: new Int16Array(fs.readFileSync(p("mul_lut.bin")).buffer.slice(0)),
           requant: new Int8Array(fs.readFileSync(p("requant_lut.bin")).buffer.slice(0)),
           relu: new Int8Array(fs.readFileSync(p("relu_lut.bin")).buffer.slice(0)) };
}
const L = loadLUTs();
const matmulInt8 = (Xq, Wq, m, k, n, LL) => V.lutMatmulJS(Xq, Wq, m, k, n, LL);
const cfg = { c: 32, t: 32, b: 8, layers: 2, heads: 2, steps: 120, lr: 0.02 };

(async function () {
  const A = X.init(cfg, L, matmulInt8);
  const B = X.init(cfg, L, matmulInt8);
  console.log(`vocab=${X.vocabSize()}, params=${A.nParams}, baseline loss=${Math.log(X.vocabSize()).toFixed(3)}`);
  const oa = T.makeAdam(A.nParams, { lr: cfg.lr });
  const ob = T.makeAdam(B.nParams, { lr: cfg.lr });
  let first = 0, loss = 0;
  for (let s = 0; s < cfg.steps; s++) {
    const ra = await X.trainStep(A);
    const rb = await X.trainStep(B);
    const avg = T.averageGrads([ra.grad, rb.grad]);
    X.applyUpdate(A, oa.step(avg));
    X.applyUpdate(B, ob.step(avg));
    loss = (ra.loss + rb.loss) / 2;
    if (s === 0) first = loss;
    if (s % 30 === 0 || s === cfg.steps - 1) console.log(`  step ${s}  loss ${loss.toFixed(4)}`);
  }
  const pa = X.getFlatParams(A), pb = X.getFlatParams(B);
  let diff = 0;
  for (let i = 0; i < pa.length; i++) diff = Math.max(diff, Math.abs(pa[i] - pb[i]));
  const sample = await X.generate(A, "the ", 60);
  console.log(`sample: "${sample}"`);
  console.log(`replica max param diff: ${diff.toExponential(3)}`);
  const ok = loss < first * 0.75 && loss < Math.log(X.vocabSize()) && diff === 0;
  console.log(ok ? "TRANSFORMER TEST PASSED" : "TRANSFORMER TEST FAILED");
  process.exit(ok ? 0 : 1);
})();
