// WebGPU INT8 matmul via the verified multiply LUT — the emulated GPU logic
// running on the browser's GPU. Automatic CPU fallback (same LUT) for machines
// without WebGPU (e.g. old PCs via Supermium). initCompute() returns
//   { backend, label, matmulInt8(Xq, Wq, m, k, n, L) -> Int32Array }
// matching Verified.lutMatmulJS, so the trainer is device-blind.
//
// High-throughput path: when the browser exposes WGSL's
// packed_4x8_integer_dot_product feature, we use dot4I8Packed — it compiles to
// the GPU's DP4A/INT8 dot-product hardware (the same units tensor-core INT8
// paths are built on): 4 exact int8 MACs per instruction, int32 accumulation,
// and 4× less memory traffic from packing. Because int8×int8→int32 is exact,
// it is bit-identical to the verified mul8 LUT — and we PROVE that at init by
// cross-checking random matmuls against the LUT before trusting it. If the
// hardware ever disagrees with the units, we fall back to the LUT shader.
(function (root) {
  "use strict";

  const WGSL_LUT = `
    @group(0) @binding(0) var<storage, read> Xq  : array<i32>;   // int8 byte per elem
    @group(0) @binding(1) var<storage, read> Wq  : array<i32>;
    @group(0) @binding(2) var<storage, read> lut : array<i32>;   // 65536 signed products
    @group(0) @binding(3) var<storage, read_write> C : array<i32>;
    @group(0) @binding(4) var<uniform> dims : vec3<u32>;         // m, k, n
    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
      let m = dims.x; let k = dims.y; let n = dims.z;
      let row = gid.x; let col = gid.y;
      if (row >= m || col >= n) { return; }
      var s : i32 = 0;
      for (var p = 0u; p < k; p = p + 1u) {
        let au = u32(Xq[row * k + p] & 255);
        let bu = u32(Wq[p * n + col] & 255);
        s = s + lut[au * 256u + bu];
      }
      C[row * n + col] = s;
    }`;

  // X packed 4×int8 along k; W transposed then packed the same way, so each
  // thread streams two contiguous rows of u32 words through dot4I8Packed.
  const WGSL_DP4 = `
    @group(0) @binding(0) var<storage, read> Xp : array<u32>;
    @group(0) @binding(1) var<storage, read> Wp : array<u32>;    // Wᵀ, packed
    @group(0) @binding(2) var<storage, read_write> C : array<i32>;
    @group(0) @binding(3) var<uniform> dims : vec3<u32>;         // m, kw=ceil(k/4), n
    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
      let m = dims.x; let kw = dims.y; let n = dims.z;
      let row = gid.x; let col = gid.y;
      if (row >= m || col >= n) { return; }
      var s : i32 = 0;
      for (var p = 0u; p < kw; p = p + 1u) {
        s = s + dot4I8Packed(Xp[row * kw + p], Wp[col * kw + p]);
      }
      C[row * n + col] = s;
    }`;

  // Batched block-scaled GEMM with a FUSED EPILOGUE (CUTLASS ex. 05/24 + 12):
  // grid z = batch index, so ALL attention heads run in ONE dispatch, and the
  // epilogue (block dequant rs·cs + optional ReLU) happens before the data
  // leaves the GPU — f32 out, no int32 readback, no second pass in JS.
  const WGSL_BG_LUT = `
    @group(0) @binding(0) var<storage, read> Xq  : array<i32>;   // int8 byte per elem
    @group(0) @binding(1) var<storage, read> Wq  : array<i32>;
    @group(0) @binding(2) var<storage, read> lut : array<i32>;
    @group(0) @binding(3) var<storage, read> rs  : array<f32>;   // per (batch,row)
    @group(0) @binding(4) var<storage, read> cs  : array<f32>;   // per (batch,col)
    @group(0) @binding(5) var<storage, read_write> O : array<f32>;
    @group(0) @binding(6) var<uniform> dims : vec4<u32>;         // m, k, n, flags(1=relu)
    @compute @workgroup_size(8, 8, 1)
    fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
      let m = dims.x; let k = dims.y; let n = dims.z;
      let row = gid.x; let col = gid.y; let bz = gid.z;
      if (row >= m || col >= n) { return; }
      var s : i32 = 0;
      let xo = (bz * m + row) * k;
      let wo = bz * k * n + col;
      for (var p = 0u; p < k; p = p + 1u) {
        let au = u32(Xq[xo + p] & 255);
        let bu = u32(Wq[wo + p * n] & 255);
        s = s + lut[au * 256u + bu];
      }
      var v = f32(s) * rs[bz * m + row] * cs[bz * n + col];
      if ((dims.w & 1u) == 1u && v < 0.0) { v = 0.0; }
      O[(bz * m + row) * n + col] = v;
    }`;

  // same fused/batched kernel on the DP4A hardware path (Wᵀ packed per batch)
  const WGSL_BG_DP4 = `
    @group(0) @binding(0) var<storage, read> Xp : array<u32>;
    @group(0) @binding(1) var<storage, read> Wp : array<u32>;    // per-batch Wᵀ, packed
    @group(0) @binding(2) var<storage, read> rs : array<f32>;
    @group(0) @binding(3) var<storage, read> cs : array<f32>;
    @group(0) @binding(4) var<storage, read_write> O : array<f32>;
    @group(0) @binding(5) var<uniform> dims : vec4<u32>;         // m, kw, n, flags
    @compute @workgroup_size(8, 8, 1)
    fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
      let m = dims.x; let kw = dims.y; let n = dims.z;
      let row = gid.x; let col = gid.y; let bz = gid.z;
      if (row >= m || col >= n) { return; }
      var s : i32 = 0;
      let xo = (bz * m + row) * kw;
      let wo = (bz * n + col) * kw;
      for (var p = 0u; p < kw; p = p + 1u) {
        s = s + dot4I8Packed(Xp[xo + p], Wp[wo + p]);
      }
      var v = f32(s) * rs[bz * m + row] * cs[bz * n + col];
      if ((dims.w & 1u) == 1u && v < 0.0) { v = 0.0; }
      O[(bz * m + row) * n + col] = v;
    }`;

  // Gather-fused attention (CUTLASS ex. 36/52): kernels index q/k/v directly in
  // their BT×C layout (head-strided) — no gather copies, no kᵀ transpose — and
  // the ctx kernel scatters straight back into BT×C. int8×int8→i32 is exact, so
  // these are bit-identical to the LUT mirrors (proved at init before use).
  const WGSL_ATT_SCORES = `
    @group(0) @binding(0) var<storage, read> Q : array<i32>;     // int8 per elem, BT×C
    @group(0) @binding(1) var<storage, read> K : array<i32>;
    @group(0) @binding(2) var<storage, read> qs : array<f32>;    // per (token,head)
    @group(0) @binding(3) var<storage, read> ks : array<f32>;
    @group(0) @binding(4) var<storage, read_write> O : array<f32>;
    @group(0) @binding(5) var<uniform> dims : vec4<u32>;         // T, heads, hd, _
    @compute @workgroup_size(8, 8, 1)
    fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
      let T = dims.x; let heads = dims.y; let hd = dims.z;
      let ti = gid.x; let tj = gid.y; let bz = gid.z;
      if (ti >= T || tj >= T) { return; }
      let bi = bz / heads; let h = bz % heads;
      let C = heads * hd;
      let qo = (bi * T + ti) * C + h * hd;
      let ko = (bi * T + tj) * C + h * hd;
      var s : i32 = 0;
      for (var p = 0u; p < hd; p = p + 1u) { s = s + Q[qo + p] * K[ko + p]; }
      O[(bz * T + ti) * T + tj] = f32(s) * qs[(bi * T + ti) * heads + h] * ks[(bi * T + tj) * heads + h];
    }`;
  const WGSL_ATT_CTX = `
    @group(0) @binding(0) var<storage, read> A : array<i32>;     // int8, BH×T×T
    @group(0) @binding(1) var<storage, read> V : array<i32>;     // int8, BT×C
    @group(0) @binding(2) var<storage, read> as_ : array<f32>;   // per (bz,row)
    @group(0) @binding(3) var<storage, read> vs : array<f32>;    // per (batch,head,chan)
    @group(0) @binding(4) var<storage, read_write> O : array<f32>;  // BT×C (scatter fused)
    @group(0) @binding(5) var<uniform> dims : vec4<u32>;         // T, heads, hd, _
    @compute @workgroup_size(8, 8, 1)
    fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
      let T = dims.x; let heads = dims.y; let hd = dims.z;
      let ti = gid.x; let j = gid.y; let bz = gid.z;
      if (ti >= T || j >= hd) { return; }
      let bi = bz / heads; let h = bz % heads;
      let C = heads * hd;
      let ao = (bz * T + ti) * T;
      var s : i32 = 0;
      for (var tj = 0u; tj < T; tj = tj + 1u) { s = s + A[ao + tj] * V[(bi * T + tj) * C + h * hd + j]; }
      O[(bi * T + ti) * C + h * hd + j] = f32(s) * as_[bz * T + ti] * vs[(bi * heads + h) * hd + j];
    }`;

  // Split-K f32 GEMM (CUTLASS ex. 06) for the STE BACKWARD only — the backward
  // was always float (the integer path has no gradient); this just moves that
  // exact float math off the JS thread. Split-K matters for dlnf: M=256, N=32,
  // K=16512 — 8k outputs with a huge inner loop would idle the GPU, so slices
  // of K run on separate workgroups and a second tiny pass reduces partials.
  const WGSL_FGEMM = `
    @group(0) @binding(0) var<storage, read> A : array<f32>;
    @group(0) @binding(1) var<storage, read> Bm : array<f32>;
    @group(0) @binding(2) var<storage, read_write> P : array<f32>;   // S partials
    @group(0) @binding(3) var<uniform> dims : vec4<u32>;             // m, k, n, flags(bit0=transA, rest=S)
    @compute @workgroup_size(8, 8, 1)
    fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
      let m = dims.x; let k = dims.y; let n = dims.z;
      let transA = (dims.w & 1u) == 1u;
      let S = dims.w >> 1u;
      let row = gid.x; let col = gid.y; let z = gid.z;
      if (row >= m || col >= n) { return; }
      let ks = (k + S - 1u) / S;
      let p0 = z * ks;
      let p1 = min(k, p0 + ks);
      var s : f32 = 0.0;
      for (var p = p0; p < p1; p = p + 1u) {
        let a = select(A[row * k + p], A[p * m + row], transA);
        s = s + a * Bm[p * n + col];
      }
      P[(z * m + row) * n + col] = s;
    }`;
  const WGSL_FREDUCE = `
    @group(0) @binding(0) var<storage, read> P : array<f32>;
    @group(0) @binding(1) var<storage, read_write> O : array<f32>;
    @group(0) @binding(2) var<uniform> dims : vec4<u32>;             // mn, S, _, _
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
      let mn = dims.x; let S = dims.y;
      let i = gid.x;
      if (i >= mn) { return; }
      var s : f32 = 0.0;
      for (var z = 0u; z < S; z = z + 1u) { s = s + P[z * mn + i]; }
      O[i] = s;
    }`;

  async function loadLUTs(base) {
    base = base || "";
    const [mulB, reqB, reluB, meta] = await Promise.all([
      fetch(base + "mul_lut.bin").then(r => r.arrayBuffer()),
      fetch(base + "requant_lut.bin").then(r => r.arrayBuffer()),
      fetch(base + "relu_lut.bin").then(r => r.arrayBuffer()),
      fetch(base + "luts_meta.json").then(r => r.json()),
    ]);
    return { mul: new Int16Array(mulB), requant: new Int8Array(reqB),
             relu: new Int8Array(reluB), shift: meta.shift };
  }

  // pack a row-major int8 matrix (rows×cols) into u32 words of 4 bytes along
  // cols, zero-padded to kw words per row (zeros contribute 0 to the dot)
  function packRows(Q, rows, cols, kw) {
    const out = new Uint32Array(rows * kw);
    const bytes = new Uint8Array(out.buffer);
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++) bytes[(r * kw * 4) + c] = Q[r * cols + c] & 0xFF;
    return out;
  }
  function transposeI8(Q, rows, cols) {
    const out = new Int8Array(rows * cols);
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) out[c * rows + r] = Q[r * cols + c];
    return out;
  }

  async function initCompute(L) {
    const cpu = { backend: "cpu", label: "CPU (JS)",
                  matmulInt8: (Xq, Wq, m, k, n, LL) => root.Verified.lutMatmulJS(Xq, Wq, m, k, n, LL) };
    if (!(root.navigator && navigator.gpu)) return cpu;
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return cpu;
      const device = await adapter.requestDevice();
      const info = adapter.info || {};
      const gpuName = info.description || info.vendor || "WebGPU";

      // LUT pipeline (always built — the fallback and the verification oracle)
      const lutModule = device.createShaderModule({ code: WGSL_LUT });
      const lutPipe = device.createComputePipeline({ layout: "auto", compute: { module: lutModule, entryPoint: "main" } });
      const lut32 = new Int32Array(L.mul);   // widen int16 -> int32
      const lutBuf = device.createBuffer({ size: lut32.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(lutBuf, 0, lut32);
      const bgLutModule = device.createShaderModule({ code: WGSL_BG_LUT });
      const bgLutPipe = device.createComputePipeline({ layout: "auto", compute: { module: bgLutModule, entryPoint: "main" } });

      // gather-fused attention kernels — must reproduce the LUT mirrors exactly
      const mkPipe = (code) => device.createComputePipeline({ layout: "auto",
        compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
      const scoresPipe = mkPipe(WGSL_ATT_SCORES), ctxPipe = mkPipe(WGSL_ATT_CTX);
      let att = { scores: (qq, kq, qs, ks, d) => gpuAttScores(device, scoresPipe, qq, kq, qs, ks, d),
                  ctx: (aq, vq, as, vs, d) => gpuAttCtx(device, ctxPipe, aq, vq, as, vs, d) };
      try {
        const d0 = { B: 2, T: 8, heads: 2, hd: 8 };
        const nQ = d0.B * d0.T * d0.heads * d0.hd;
        const qq = new Int8Array(nQ), kq = new Int8Array(nQ), vq = new Int8Array(nQ);
        for (let i = 0; i < nQ; i++) { qq[i] = (Math.random() * 256 - 128) | 0; kq[i] = (Math.random() * 256 - 128) | 0; vq[i] = (Math.random() * 256 - 128) | 0; }
        const qs = Float32Array.from({ length: d0.B * d0.T * d0.heads }, () => Math.random() + 0.5);
        const ks = Float32Array.from(qs, () => Math.random() + 0.5);
        const aq = new Int8Array(d0.B * d0.heads * d0.T * d0.T);
        for (let i = 0; i < aq.length; i++) aq[i] = (Math.random() * 127) | 0;
        const as = Float32Array.from({ length: d0.B * d0.heads * d0.T }, () => Math.random() + 0.5);
        const vs = Float32Array.from({ length: d0.B * d0.heads * d0.hd }, () => Math.random() + 0.5);
        const [hwS, hwC] = await Promise.all([att.scores(qq, kq, qs, ks, d0), att.ctx(aq, vq, as, vs, d0)]);
        const refS = root.Verified.attScoresJS(qq, kq, qs, ks, d0, L);
        const refC = root.Verified.attCtxJS(aq, vq, as, vs, d0, L);
        const close = (a, b) => Math.abs(a - b) <= Math.abs(b) * 1e-6 + 1e-6;
        for (let i = 0; i < refS.length; i++) if (!close(hwS[i], refS[i])) throw new Error("scores mismatch");
        for (let i = 0; i < refC.length; i++) if (!close(hwC[i], refC[i])) throw new Error("ctx mismatch");
      } catch (e) {
        console.warn("fused attention kernels failed verification — using CPU LUT mirrors:", e.message);
        att = null;
      }

      // split-K f32 GEMM for the STE backward (self-tested vs JS float matmul)
      const fPipes = { gemm: mkPipe(WGSL_FGEMM), reduce: mkPipe(WGSL_FREDUCE) };
      let fgemm = (A, Bm, d) => gpuFgemm(device, fPipes, A, Bm, d);
      try {
        const m0 = 7, k0 = 4500, n0 = 5;                      // k big enough to exercise split-K
        const A = Float32Array.from({ length: m0 * k0 }, () => Math.random() - 0.5);
        const Bm = Float32Array.from({ length: k0 * n0 }, () => Math.random() - 0.5);
        const hw = await fgemm(A, Bm, { m: m0, k: k0, n: n0 });
        const ref = root.TrainCore.matmul(A, Bm, m0, k0, n0);
        for (let i = 0; i < ref.length; i++)
          if (Math.abs(hw[i] - ref[i]) > Math.abs(ref[i]) * 1e-3 + 1e-3) throw new Error("fgemm mismatch");
      } catch (e) {
        console.warn("split-K f32 GEMM failed verification — backward stays in JS:", e.message);
        fgemm = null;
      }

      const viaLUT = { backend: "webgpu", label: `${gpuName} (LUT shader)`,
                       matmulInt8: (Xq, Wq, m, k, n) => gpuMatmulLUT(device, lutPipe, lutBuf, Xq, Wq, m, k, n),
                       bgemm: (Xq, Wq, rs, cs, d) => gpuBgemmLUT(device, bgLutPipe, lutBuf, Xq, Wq, rs, cs, d),
                       att, fgemm };

      // DP4A pipeline — only if the WGSL feature exists AND it reproduces the
      // verified units exactly on random self-tests
      if (!(navigator.gpu.wgslLanguageFeatures && navigator.gpu.wgslLanguageFeatures.has("packed_4x8_integer_dot_product")))
        return viaLUT;
      const dp4Module = device.createShaderModule({ code: WGSL_DP4 });
      const dp4Pipe = device.createComputePipeline({ layout: "auto", compute: { module: dp4Module, entryPoint: "main" } });
      const dp4mm = (Xq, Wq, m, k, n) => gpuMatmulDP4(device, dp4Pipe, Xq, Wq, m, k, n);
      for (let trial = 0; trial < 3; trial++) {         // hardware vs verified units
        const m0 = 5 + trial, k0 = 7 + 3 * trial, n0 = 6 + trial;
        const Xq = new Int8Array(m0 * k0), Wq = new Int8Array(k0 * n0);
        for (let i = 0; i < Xq.length; i++) Xq[i] = (Math.random() * 256 - 128) | 0;
        for (let i = 0; i < Wq.length; i++) Wq[i] = (Math.random() * 256 - 128) | 0;
        const hw = await dp4mm(Xq, Wq, m0, k0, n0);
        const ref = root.Verified.lutMatmulJS(Xq, Wq, m0, k0, n0, L);
        for (let i = 0; i < ref.length; i++)
          if (hw[i] !== ref[i]) { console.warn("DP4A disagreed with the verified units — using LUT shader"); return viaLUT; }
      }
      // fused/batched DP4A kernel, gated by the same verification: it must
      // reproduce the CPU LUT mirror bit-for-bit on the integer accumulate
      const bgDp4Module = device.createShaderModule({ code: WGSL_BG_DP4 });
      const bgDp4Pipe = device.createComputePipeline({ layout: "auto", compute: { module: bgDp4Module, entryPoint: "main" } });
      const bg = (Xq, Wq, rs, cs, d) => gpuBgemmDP4(device, bgDp4Pipe, Xq, Wq, rs, cs, d);
      {
        const d0 = { m: 5, k: 9, n: 6, batch: 3, relu: true };
        const Xq = new Int8Array(d0.batch * d0.m * d0.k), Wq = new Int8Array(d0.batch * d0.k * d0.n);
        for (let i = 0; i < Xq.length; i++) Xq[i] = (Math.random() * 256 - 128) | 0;
        for (let i = 0; i < Wq.length; i++) Wq[i] = (Math.random() * 256 - 128) | 0;
        const rs = new Float32Array(d0.batch * d0.m).fill(0).map(() => Math.random() + 0.5);
        const cs = new Float32Array(d0.batch * d0.n).fill(0).map(() => Math.random() + 0.5);
        const hw = await bg(Xq, Wq, rs, cs, d0);
        const ref = root.Verified.bgemmJS(Xq, Wq, rs, cs, d0, L);
        for (let i = 0; i < ref.length; i++)
          if (Math.abs(hw[i] - ref[i]) > Math.abs(ref[i]) * 1e-6 + 1e-6) {
            console.warn("batched DP4A disagreed with the verified units — using LUT bgemm");
            return { backend: "webgpu", label: `${gpuName} (DP4A int8 dot — HW verified vs units)`,
                     matmulInt8: dp4mm, bgemm: viaLUT.bgemm, att, fgemm };
          }
      }
      return { backend: "webgpu", label: `${gpuName} (DP4A int8 dot — HW verified vs units)`,
               matmulInt8: dp4mm, bgemm: bg, att, fgemm };
    } catch (e) { console.warn("WebGPU init failed, CPU fallback:", e); return cpu; }
  }

  // shared dispatch/readback plumbing
  async function runPass(device, pipeline, entries, m, n) {
    const bytesC = m * n * 4;
    const bufC = mk(device, bytesC, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
      entries: entries(bufC) });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(m / 8), Math.ceil(n / 8)); pass.end();
    const read = mk(device, bytesC, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    enc.copyBufferToBuffer(bufC, 0, read, 0, bytesC);
    device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const out = new Int32Array(read.getMappedRange().slice(0));
    read.unmap();
    return { out, bufC, read };
  }

  async function gpuMatmulLUT(device, pipeline, lutBuf, Xq, Wq, m, k, n) {
    const X32 = Int32Array.from(Xq), W32 = Int32Array.from(Wq);   // byte -> i32
    const bufX = up(device, X32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufW = up(device, W32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufD = up(device, new Uint32Array([m, k, n, 0]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const r = await runPass(device, pipeline, (bufC) => [
      { binding: 0, resource: { buffer: bufX } }, { binding: 1, resource: { buffer: bufW } },
      { binding: 2, resource: { buffer: lutBuf } }, { binding: 3, resource: { buffer: bufC } },
      { binding: 4, resource: { buffer: bufD } } ], m, n);
    [bufX, bufW, bufD, r.bufC, r.read].forEach(b => b.destroy());
    return r.out;
  }

  async function gpuMatmulDP4(device, pipeline, Xq, Wq, m, k, n) {
    const kw = Math.ceil(k / 4);
    const Xp = packRows(Xq, m, k, kw);
    const Wp = packRows(transposeI8(Wq, k, n), n, k, kw);   // Wᵀ so col j is a row
    const bufX = up(device, Xp, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufW = up(device, Wp, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufD = up(device, new Uint32Array([m, kw, n, 0]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const r = await runPass(device, pipeline, (bufC) => [
      { binding: 0, resource: { buffer: bufX } }, { binding: 1, resource: { buffer: bufW } },
      { binding: 2, resource: { buffer: bufC } }, { binding: 3, resource: { buffer: bufD } } ], m, n);
    [bufX, bufW, bufD, r.bufC, r.read].forEach(b => b.destroy());
    return r.out;
  }

  // attention kernels: int8 (widened i32) in, f32 out, strided head indexing
  async function gpuAttScores(device, pipeline, qq, kq, qs, ks, d) {
    const { B, T, heads } = d;
    const bufQ = up(device, Int32Array.from(qq), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufK = up(device, Int32Array.from(kq), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufQs = up(device, qs, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufKs = up(device, ks, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufD = up(device, new Uint32Array([d.T, d.heads, d.hd, 0]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const r = await runBgPass(device, pipeline, (bufO) => [
      { binding: 0, resource: { buffer: bufQ } }, { binding: 1, resource: { buffer: bufK } },
      { binding: 2, resource: { buffer: bufQs } }, { binding: 3, resource: { buffer: bufKs } },
      { binding: 4, resource: { buffer: bufO } }, { binding: 5, resource: { buffer: bufD } } ], T, T, B * heads);
    [bufQ, bufK, bufQs, bufKs, bufD, r.bufO, r.read].forEach(b => b.destroy());
    return r.out;
  }
  async function gpuAttCtx(device, pipeline, aq, vq, as, vs, d) {
    const { B, T, heads, hd } = d;
    const bufA = up(device, Int32Array.from(aq), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufV = up(device, Int32Array.from(vq), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufAs = up(device, as, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufVs = up(device, vs, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufD = up(device, new Uint32Array([T, heads, hd, 0]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const r = await runBgPass(device, pipeline, (bufO) => [
      { binding: 0, resource: { buffer: bufA } }, { binding: 1, resource: { buffer: bufV } },
      { binding: 2, resource: { buffer: bufAs } }, { binding: 3, resource: { buffer: bufVs } },
      { binding: 4, resource: { buffer: bufO } }, { binding: 5, resource: { buffer: bufD } } ], T, hd, B * heads);
    [bufA, bufV, bufAs, bufVs, bufD, r.bufO, r.read].forEach(b => b.destroy());
    return r.out;
  }

  // split-K f32 GEMM (backward): partial pass + reduce pass
  async function gpuFgemm(device, pipes, A, Bm, d) {
    const { m, k, n } = d, transA = d.transA ? 1 : 0;
    const S = k > 4096 ? Math.min(16, Math.ceil(k / 2048)) : 1;
    const bufA = up(device, A, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufB = up(device, Bm, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufP = mk(device, S * m * n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const bufD1 = up(device, new Uint32Array([m, k, n, transA | (S << 1)]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const bufO = mk(device, m * n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const bufD2 = up(device, new Uint32Array([m * n, S, 0, 0]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const enc = device.createCommandEncoder();
    let pass = enc.beginComputePass();
    pass.setPipeline(pipes.gemm);
    pass.setBindGroup(0, device.createBindGroup({ layout: pipes.gemm.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: bufA } }, { binding: 1, resource: { buffer: bufB } },
      { binding: 2, resource: { buffer: bufP } }, { binding: 3, resource: { buffer: bufD1 } } ] }));
    pass.dispatchWorkgroups(Math.ceil(m / 8), Math.ceil(n / 8), S); pass.end();
    pass = enc.beginComputePass();
    pass.setPipeline(pipes.reduce);
    pass.setBindGroup(0, device.createBindGroup({ layout: pipes.reduce.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: bufP } }, { binding: 1, resource: { buffer: bufO } },
      { binding: 2, resource: { buffer: bufD2 } } ] }));
    pass.dispatchWorkgroups(Math.ceil(m * n / 64)); pass.end();
    const read = mk(device, m * n * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    enc.copyBufferToBuffer(bufO, 0, read, 0, m * n * 4);
    device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(read.getMappedRange().slice(0));
    read.unmap();
    [bufA, bufB, bufP, bufD1, bufO, bufD2, read].forEach(b => b.destroy());
    return out;
  }

  // fused batched dispatch: f32 out, epilogue done on-device
  async function runBgPass(device, pipeline, entries, m, n, batch) {
    const bytesO = batch * m * n * 4;
    const bufO = mk(device, bytesO, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: entries(bufO) });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(m / 8), Math.ceil(n / 8), batch); pass.end();
    const read = mk(device, bytesO, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    enc.copyBufferToBuffer(bufO, 0, read, 0, bytesO);
    device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(read.getMappedRange().slice(0));
    read.unmap();
    return { out, bufO, read };
  }

  async function gpuBgemmLUT(device, pipeline, lutBuf, Xq, Wq, rs, cs, d) {
    const { m, k, n } = d, batch = d.batch || 1, flags = d.relu ? 1 : 0;
    const bufX = up(device, Int32Array.from(Xq), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufW = up(device, Int32Array.from(Wq), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufR = up(device, rs, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufS = up(device, cs, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufD = up(device, new Uint32Array([m, k, n, flags]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const r = await runBgPass(device, pipeline, (bufO) => [
      { binding: 0, resource: { buffer: bufX } }, { binding: 1, resource: { buffer: bufW } },
      { binding: 2, resource: { buffer: lutBuf } }, { binding: 3, resource: { buffer: bufR } },
      { binding: 4, resource: { buffer: bufS } }, { binding: 5, resource: { buffer: bufO } },
      { binding: 6, resource: { buffer: bufD } } ], m, n, batch);
    [bufX, bufW, bufR, bufS, bufD, r.bufO, r.read].forEach(b => b.destroy());
    return r.out;
  }

  async function gpuBgemmDP4(device, pipeline, Xq, Wq, rs, cs, d) {
    const { m, k, n } = d, batch = d.batch || 1, flags = d.relu ? 1 : 0;
    const kw = Math.ceil(k / 4);
    const Xp = packRows(Xq, batch * m, k, kw);            // rows are (batch·m)
    const Wp = new Uint32Array(batch * n * kw);           // per-batch Wᵀ, packed
    for (let bz = 0; bz < batch; bz++) {
      const wt = transposeI8(Wq.subarray(bz * k * n, (bz + 1) * k * n), k, n);
      Wp.set(packRows(wt, n, k, kw), bz * n * kw);
    }
    const bufX = up(device, Xp, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufW = up(device, Wp, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufR = up(device, rs, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufS = up(device, cs, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufD = up(device, new Uint32Array([m, kw, n, flags]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const r = await runBgPass(device, pipeline, (bufO) => [
      { binding: 0, resource: { buffer: bufX } }, { binding: 1, resource: { buffer: bufW } },
      { binding: 2, resource: { buffer: bufR } }, { binding: 3, resource: { buffer: bufS } },
      { binding: 4, resource: { buffer: bufO } }, { binding: 5, resource: { buffer: bufD } } ], m, n, batch);
    [bufX, bufW, bufR, bufS, bufD, r.bufO, r.read].forEach(b => b.destroy());
    return r.out;
  }

  function mk(device, size, usage) { return device.createBuffer({ size, usage }); }
  function up(device, arr, usage) {
    const b = mk(device, Math.max(16, arr.byteLength), usage);
    device.queue.writeBuffer(b, 0, arr);
    return b;
  }

  root.Compute = { initCompute, loadLUTs };
})(self);
