// Proves 2-peer training THROUGH the verified units (LUTs) converges and stays
// in sync — the browser will do exactly this, with the matmul on WebGPU.
const fs = require("fs");
const path = require("path");
const T = require("./public/traincore.js");
const V = require("./public/verified_core.js");

function loadLUTs() {
  const p = (f) => path.join(__dirname, "public", f);
  const mul = new Int16Array(fs.readFileSync(p("mul_lut.bin")).buffer.slice(0));
  const requant = new Int8Array(fs.readFileSync(p("requant_lut.bin")).buffer.slice(0));
  const relu = new Int8Array(fs.readFileSync(p("relu_lut.bin")).buffer.slice(0));
  return { mul, requant, relu };
}

function randn(n, rng) { const r = rng || Math.random; const o = new Float32Array(n); for (let i = 0; i < n; i += 2) { let u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); const m = Math.sqrt(-2 * Math.log(u)); o[i] = m * Math.cos(2 * Math.PI * v); if (i + 1 < n) o[i + 1] = m * Math.sin(2 * Math.PI * v); } return o; }
function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

const L = loadLUTs();
const D = { n: 128, din: 16, h: 16, dout: 4 }, lr = 0.03, steps = 300;

// shared deterministic truth + init; per-peer data shard
const Wtrue1 = randn(D.din * D.h, mulberry32(42)), Wtrue2 = randn(D.h * D.dout, mulberry32(43));
function target(X) { const hpre = T.matmul(X, Wtrue1, D.n, D.din, D.h); for (let i = 0; i < hpre.length; i++) hpre[i] = Math.max(0, hpre[i]); return T.matmul(hpre, Wtrue2, D.n, D.h, D.dout); }
const XA = randn(D.n * D.din), yA = target(XA);
const XB = randn(D.n * D.din), yB = target(XB);

const W1a = randn(D.din * D.h, mulberry32(7)), W2a = randn(D.h * D.dout, mulberry32(8));
const W1b = Float32Array.from(W1a), W2b = Float32Array.from(W2a);

(async function () {
  let loss = 0, loss0 = 0;
  for (let s = 0; s < steps; s++) {
    const fa = await V.forward(XA, yA, W1a, W2a, D, L), ga = V.backward(XA, W1a, W2a, fa, D);
    const fb = await V.forward(XB, yB, W1b, W2b, D, L), gb = V.backward(XB, W1b, W2b, fb, D);
    const avg = T.averageGrads([ga, gb]);               // <-- exchanged P2P
    V.splitApply(W1a, W2a, avg, lr);
    V.splitApply(W1b, W2b, avg, lr);
    loss = (fa.loss + fb.loss) / 2; if (s === 0) loss0 = loss;
    if (s % 60 === 0 || s === steps - 1) console.log(`  step ${s}  cluster-avg loss ${loss.toFixed(5)}`);
  }
  let diff = 0; for (let i = 0; i < W1a.length; i++) diff = Math.max(diff, Math.abs(W1a[i] - W1b[i]));
  for (let i = 0; i < W2a.length; i++) diff = Math.max(diff, Math.abs(W2a[i] - W2b[i]));
  console.log(`\nreplica max param diff: ${diff.toExponential(3)}`);
  const ok = loss < loss0 * 0.3 && diff < 1e-9;
  console.log(ok ? "VERIFIED TEST PASSED — trained through the units, converged, replicas in sync." : "FAILED");
  process.exit(ok ? 0 : 1);
})();
