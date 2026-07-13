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
      const viaLUT = { backend: "webgpu", label: `${gpuName} (LUT shader)`,
                       matmulInt8: (Xq, Wq, m, k, n) => gpuMatmulLUT(device, lutPipe, lutBuf, Xq, Wq, m, k, n) };

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
      return { backend: "webgpu", label: `${gpuName} (DP4A int8 dot — HW verified vs units)`, matmulInt8: dp4mm };
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

  function mk(device, size, usage) { return device.createBuffer({ size, usage }); }
  function up(device, arr, usage) {
    const b = mk(device, Math.max(16, arr.byteLength), usage);
    device.queue.writeBuffer(b, 0, arr);
    return b;
  }

  root.Compute = { initCompute, loadLUTs };
})(self);
