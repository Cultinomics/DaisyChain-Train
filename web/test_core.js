// Verifies the training loop + 2-peer gradient averaging in pure Node (no
// browser). Simulates two peers each holding half the data; they average
// gradients every step. Proves: (a) it converges, (b) both replicas stay
// identical — the same guarantees the browser P2P version needs.
const T = require("./public/traincore.js");

function randn(n) {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    a[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  return a;
}

const din = 16, dout = 4, nPer = 128, steps = 400, lr = 0.05;

// ground-truth weights
const Wtrue = randn(din * dout);
function makeShard() {
  const X = randn(nPer * din);
  const y = T.matmul(X, Wtrue, nPer, din, dout);   // clean targets
  return { X, y };
}
const A = makeShard(), B = makeShard();

// both peers start from the SAME W0 (initiator broadcasts it)
const W0 = randn(din * dout);
const Wa = Float32Array.from(W0), Wb = Float32Array.from(W0);

let loss = 0;
for (let s = 0; s < steps; s++) {
  const ra = T.forwardLossGrad(A.X, A.y, Wa, nPer, din, dout);
  const rb = T.forwardLossGrad(B.X, B.y, Wb, nPer, din, dout);
  const avg = T.averageGrads([ra.gradW, rb.gradW]);   // <-- exchanged P2P
  T.applyGrad(Wa, avg, lr);
  T.applyGrad(Wb, avg, lr);
  loss = (ra.loss + rb.loss) / 2;
  if (s % 80 === 0 || s === steps - 1)
    console.log(`  step ${s}  cluster-avg loss ${loss.toFixed(6)}`);
}

let maxDiff = 0;
for (let i = 0; i < Wa.length; i++) maxDiff = Math.max(maxDiff, Math.abs(Wa[i] - Wb[i]));
let recovery = 0;
for (let i = 0; i < Wtrue.length; i++) recovery = Math.max(recovery, Math.abs(Wa[i] - Wtrue[i]));

console.log(`\nreplica max param diff: ${maxDiff.toExponential(3)}`);
console.log(`max |W - W_true|:       ${recovery.toExponential(3)}`);
const ok = loss < 1e-3 && maxDiff < 1e-9 && recovery < 0.05;
console.log(ok ? "\nCORE TEST PASSED — converged, replicas in sync." : "\nCORE TEST FAILED");
process.exit(ok ? 0 : 1);
