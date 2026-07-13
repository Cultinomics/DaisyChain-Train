// Verified INT8 compute — the emulated GPU logic, in the browser.
// A layer's forward runs THROUGH the units: quantize -> LUT multiply -> requant
// -> optional ReLU -> dequant. Backward is a straight-through estimator (the
// integer path has no gradient), so ordinary float weights still learn.
// Same units as the Python/Docker DaisyChain; here they're lookup tables.
(function (root) {
  "use strict";

  let TC;   // TrainCore (matmul/transpose) — resolved per environment at the end

  function quantize(X) {
    let mx = 0; for (let i = 0; i < X.length; i++) { const a = Math.abs(X[i]); if (a > mx) mx = a; }
    const scale = Math.max(mx / 127, 1e-8);
    const q = new Int8Array(X.length);
    for (let i = 0; i < X.length; i++) { let v = Math.round(X[i] / scale); q[i] = v < -128 ? -128 : v > 127 ? 127 : v; }
    return { q, scale };
  }

  // int8 matmul via the verified multiply LUT: acc(m×n) = sum_k mulLUT[Xq,Wq]
  function lutMatmulJS(Xq, Wq, m, k, n, L) {
    const C = new Int32Array(m * n), mul = L.mul;
    for (let i = 0; i < m; i++) {
      for (let p = 0; p < k; p++) {
        const au = (Xq[i * k + p] & 0xFF) * 256, wo = p * n, co = i * n;
        for (let j = 0; j < n; j++) C[co + j] += mul[au + (Wq[wo + j] & 0xFF)];
      }
    }
    return C;
  }

  // ---- 3xINT8 fast-accurate GEMM --------------------------------------------
  // The CUTLASS example-27 "3xTF32" scheme, ported to the verified units:
  // split each float into a coarse int8 part plus an int8-quantized residual,
  // run three EXACT LUT GEMMs (hi·hi, hi·lo, lo·hi), drop the negligible
  // lo·lo, and recombine. Same big/small decomposition NVIDIA uses to recover
  // near-fp32 accuracy from TF32 tensor cores — here it recovers ~14-bit
  // accuracy from the 8-bit units, at 3× the unit ops. Every product still
  // goes through the verified mul8 LUT.
  function quantize2(X) {
    const hi = quantize(X);
    const r = new Float32Array(X.length);
    for (let i = 0; i < X.length; i++) r[i] = X[i] - hi.q[i] * hi.scale;
    const lo = quantize(r);
    return { hi, lo };
  }
  function combine3(hh, hl, lh, x, w, len) {
    const out = new Float32Array(len);
    const shh = x.hi.scale * w.hi.scale, shl = x.hi.scale * w.lo.scale, slh = x.lo.scale * w.hi.scale;
    for (let i = 0; i < len; i++) out[i] = hh[i] * shh + hl[i] * shl + lh[i] * slh;
    return out;
  }
  function lutMatmul3JS(Xf, Wf, m, k, n, L) {          // sync, CPU LUT path
    const x = quantize2(Xf), w = quantize2(Wf);
    return combine3(lutMatmulJS(x.hi.q, w.hi.q, m, k, n, L),
                    lutMatmulJS(x.hi.q, w.lo.q, m, k, n, L),
                    lutMatmulJS(x.lo.q, w.hi.q, m, k, n, L), x, w, m * n);
  }
  async function lutMatmul3(Xf, Wf, m, k, n, L, matmulInt8) {  // any backend
    const x = quantize2(Xf), w = quantize2(Wf);
    const mm = matmulInt8 || lutMatmulJS;
    const [hh, hl, lh] = await Promise.all([
      mm(x.hi.q, w.hi.q, m, k, n, L),
      mm(x.hi.q, w.lo.q, m, k, n, L),
      mm(x.lo.q, w.hi.q, m, k, n, L),
    ]);
    return combine3(hh, hl, lh, x, w, m * n);
  }

  // one verified layer forward; returns float out (+ cache for STE backward).
  // Every product goes through the verified INT8 multiply (mul8 LUT) with exact
  // int32 accumulation — i.e. an emulated INT8 tensor-core GEMM — then dequant.
  async function linearFwd(X, W, m, k, n, L, useRelu, matmulInt8) {
    const xq = quantize(X), wq = quantize(W);
    const acc = await (matmulInt8 || lutMatmulJS)(xq.q, wq.q, m, k, n, L);  // verified multiply
    const dq = xq.scale * wq.scale;
    const out = new Float32Array(m * n);
    const mask = useRelu ? new Uint8Array(m * n) : null;
    for (let i = 0; i < m * n; i++) {
      let v = acc[i] * dq;
      if (useRelu) { if (v > 0) mask[i] = 1; else v = 0; }
      out[i] = v;
    }
    return { out, mask };
  }

  // 2-layer MLP: X→H (relu) →dout. Forward through verified units, MSE loss.
  async function forward(X, y, W1, W2, D, L, matmulInt8) {
    const { n, din, h, dout } = D;
    const l1 = await linearFwd(X, W1, n, din, h, L, true, matmulInt8);
    const l2 = await linearFwd(l1.out, W2, n, h, dout, L, false, matmulInt8);
    const resid = new Float32Array(n * dout); let loss = 0;
    for (let i = 0; i < resid.length; i++) { const r = l2.out[i] - y[i]; resid[i] = r; loss += r * r; }
    loss /= resid.length;
    return { loss, resid, z1: l1.out, mask1: l1.mask };
  }

  // STE backward (verified matmul treated as float X@W). Returns flat [gW1, gW2].
  function backward(X, W1, W2, fwd, D) {
    const { n, din, h, dout } = D;
    const { resid, z1, mask1 } = fwd;
    const s = 2 / n;
    const dout_ = new Float32Array(resid.length);
    for (let i = 0; i < resid.length; i++) dout_[i] = resid[i] * s;
    const mm = TC.matmul, tr = TC.transpose;
    const gW2 = mm(tr(z1, n, h), dout_, h, n, dout);          // z1ᵀ @ dout
    const dz1 = mm(dout_, tr(W2, h, dout), n, dout, h);        // dout @ W2ᵀ
    for (let i = 0; i < dz1.length; i++) if (!mask1[i]) dz1[i] = 0;  // relu grad
    const gW1 = mm(tr(X, n, din), dz1, din, n, h);            // Xᵀ @ dz1
    const g = new Float32Array(gW1.length + gW2.length);
    g.set(gW1, 0); g.set(gW2, gW1.length);
    return g;
  }

  function splitApply(W1, W2, gAvg, lr) {
    for (let i = 0; i < W1.length; i++) W1[i] -= lr * gAvg[i];
    for (let j = 0; j < W2.length; j++) W2[j] -= lr * gAvg[W1.length + j];
  }

  const api = { quantize, quantize2, lutMatmulJS, lutMatmul3JS, lutMatmul3, linearFwd, forward, backward, splitApply };
  if (typeof module !== "undefined" && module.exports) { TC = require("./traincore.js"); module.exports = api; }
  else { TC = root.TrainCore; root.Verified = api; }
})(typeof self !== "undefined" ? self : this);
