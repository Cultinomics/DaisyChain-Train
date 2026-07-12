// Verifies DaisyAdam (TrainCore.makeAdam) on the path that matters: training
// THROUGH the verified INT8 units, where STE gradients are noisy and plain SGD
// plateaus. Checks (a) two replicas fed the same averaged gradients stay
// bit-identical, (b) Adam reaches a lower loss than SGD in the same steps.
const fs = require("fs");
const path = require("path");
const T = require("./public/traincore.js");
const V = require("./public/verified_core.js");

function loadLUTs() {
  const p = (f) => path.join(__dirname, "public", f);
  return {
    mul: new Int16Array(fs.readFileSync(p("mul_lut.bin")).buffer.slice(0)),
    requant: new Int8Array(fs.readFileSync(p("requant_lut.bin")).buffer.slice(0)),
    relu: new Int8Array(fs.readFileSync(p("relu_lut.bin")).buffer.slice(0)),
  };
}
function randn(n, rng) { const r = rng || Math.random; const o = new Float32Array(n); for (let i = 0; i < n; i += 2) { let u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); const m = Math.sqrt(-2 * Math.log(u)); o[i] = m * Math.cos(2 * Math.PI * v); if (i + 1 < n) o[i + 1] = m * Math.sin(2 * Math.PI * v); } return o; }
function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

const L = loadLUTs();
const D = { n: 128, din: 16, h: 16, dout: 4 }, steps = 300;

const Wtrue1 = randn(D.din * D.h, mulberry32(42)), Wtrue2 = randn(D.h * D.dout, mulberry32(43));
function target(X) { const hpre = T.matmul(X, Wtrue1, D.n, D.din, D.h); for (let i = 0; i < hpre.length; i++) hpre[i] = Math.max(0, hpre[i]); return T.matmul(hpre, Wtrue2, D.n, D.h, D.dout); }
const XA = randn(D.n * D.din, mulberry32(11)), yA = target(XA);
const XB = randn(D.n * D.din, mulberry32(12)), yB = target(XB);

async function run(useAdam) {
  const W1a = randn(D.din * D.h, mulberry32(7)), W2a = randn(D.h * D.dout, mulberry32(8));
  const W1b = Float32Array.from(W1a), W2b = Float32Array.from(W2a);
  const dim = W1a.length + W2a.length;
  const oa = T.makeAdam(dim, { lr: 0.2 }), ob = T.makeAdam(dim, { lr: 0.2 });
  let loss = 0;
  for (let s = 0; s < steps; s++) {
    const fa = await V.forward(XA, yA, W1a, W2a, D, L), ga = V.backward(XA, W1a, W2a, fa, D);
    const fb = await V.forward(XB, yB, W1b, W2b, D, L), gb = V.backward(XB, W1b, W2b, fb, D);
    const avg = T.averageGrads([ga, gb]);
    if (useAdam) {
      V.splitApply(W1a, W2a, oa.step(avg), 1);
      V.splitApply(W1b, W2b, ob.step(avg), 1);
    } else {
      V.splitApply(W1a, W2a, avg, 0.03);
      V.splitApply(W1b, W2b, avg, 0.03);
    }
    loss = (fa.loss + fb.loss) / 2;
  }
  let diff = 0;
  for (let i = 0; i < W1a.length; i++) diff = Math.max(diff, Math.abs(W1a[i] - W1b[i]));
  for (let i = 0; i < W2a.length; i++) diff = Math.max(diff, Math.abs(W2a[i] - W2b[i]));
  return { loss, diff };
}

(async function () {
  const sgd = await run(false);
  const adam = await run(true);
  console.log(`SGD(lr=0.03)      final loss ${sgd.loss.toFixed(5)}`);
  console.log(`DaisyAdam(lr=0.2) final loss ${adam.loss.toFixed(5)}  replica diff ${adam.diff.toExponential(3)}`);
  const ok = adam.diff === 0 && adam.loss < sgd.loss;
  console.log(ok ? "\nOPTIMIZER TEST PASSED — deterministic replicas, beats SGD through the verified units."
                 : "\nOPTIMIZER TEST FAILED");
  process.exit(ok ? 0 : 1);
})();
