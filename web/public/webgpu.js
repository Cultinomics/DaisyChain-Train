// WebGPU INT8 matmul via the verified multiply LUT — the emulated GPU logic
// running on the browser's GPU. Automatic CPU fallback (same LUT) for machines
// without WebGPU (e.g. old PCs via Supermium). initCompute() returns
//   { backend, label, matmulInt8(Xq, Wq, m, k, n, L) -> Int32Array }
// matching Verified.lutMatmulJS, so the trainer is device-blind.
(function (root) {
  "use strict";

  const WGSL = `
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

  async function initCompute(L) {
    const cpu = { backend: "cpu", label: "CPU (JS)",
                  matmulInt8: (Xq, Wq, m, k, n, LL) => root.Verified.lutMatmulJS(Xq, Wq, m, k, n, LL) };
    if (!(root.navigator && navigator.gpu)) return cpu;
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return cpu;
      const device = await adapter.requestDevice();
      const module = device.createShaderModule({ code: WGSL });
      const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
      // upload the multiply LUT once (as i32)
      const lut32 = new Int32Array(L.mul);   // widen int16 -> int32
      const lutBuf = device.createBuffer({ size: lut32.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(lutBuf, 0, lut32);
      const info = adapter.info || {};
      return { backend: "webgpu", label: info.description || info.vendor || "WebGPU",
               matmulInt8: (Xq, Wq, m, k, n) => gpuMatmul(device, pipeline, lutBuf, Xq, Wq, m, k, n) };
    } catch (e) { console.warn("WebGPU init failed, CPU fallback:", e); return cpu; }
  }

  async function gpuMatmul(device, pipeline, lutBuf, Xq, Wq, m, k, n) {
    const X32 = Int32Array.from(Xq), W32 = Int32Array.from(Wq);   // byte -> i32
    const bufX = mk(device, X32.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufW = mk(device, W32.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bytesC = m * n * 4;
    const bufC = mk(device, bytesC, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const bufD = mk(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(bufX, 0, X32);
    device.queue.writeBuffer(bufW, 0, W32);
    device.queue.writeBuffer(bufD, 0, new Uint32Array([m, k, n]));
    const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: bufX } }, { binding: 1, resource: { buffer: bufW } },
      { binding: 2, resource: { buffer: lutBuf } }, { binding: 3, resource: { buffer: bufC } },
      { binding: 4, resource: { buffer: bufD } } ] });
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
    [bufX, bufW, bufC, bufD, read].forEach(b => b.destroy());
    return out;
  }
  function mk(device, size, usage) { return device.createBuffer({ size, usage }); }

  root.Compute = { initCompute, loadLUTs };
})(self);
