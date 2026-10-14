// "0 = unproven" for the kernel probe hash (2026-10-14).
// app.js now treats phash 0 as "this device's probe did not run" and halts on it
// (the comparison used to SKIP such a peer). That is only sound if a probe that
// DID run can never hash to 0: check the canonical probe on the CPU backend and
// on a fake bgemm that returns the same result, and that a wrong kernel moves it.
const fs = require("fs");
const path = require("path");
const V = require("./public/verified_core.js");

const root = { Verified: V };
new Function("self", fs.readFileSync(path.join(__dirname, "public", "webgpu.js"), "utf8"))(root);
const mul = new Int16Array(fs.readFileSync(path.join(__dirname, "public", "mul_lut.bin")).buffer.slice(0));
const L = { mul };

let pass = 0, fail = 0;
const t = (name, ok, detail) => {
  if (ok) { pass++; console.log("  PASS  " + name); }
  else { fail++; console.log(`  FAIL  ${name}  [${detail || ""}]`); }
};

(async () => {
  const cpu = await root.Compute.kernelProbe(null, L);
  t("the canonical probe on the CPU mirror is non-zero", cpu !== 0, cpu);
  t("...and deterministic", (await root.Compute.kernelProbe(null, L)) === cpu);
  const same = { bgemm: async (Xq, Wq, rs, cs, d) => V.bgemmJS(Xq, Wq, rs, cs, d, L) };
  t("a backend computing the same values hashes the same", (await root.Compute.kernelProbe(same, L)) === cpu);
  const off = { bgemm: async (Xq, Wq, rs, cs, d) => {
    const o = V.bgemmJS(Xq, Wq, rs, cs, d, L); o[o.length - 1] += 1; return o; } };
  const h = await root.Compute.kernelProbe(off, L);
  t("a kernel wrong in one cell hashes differently (and non-zero)", h !== cpu && h !== 0, h);
  const app = fs.readFileSync(path.join(__dirname, "public", "app.js"), "utf8");
  t("app.js resets probeHash to 0 when the probe throws", /catch \(e\) \{ probeHash = 0;/.test(app));
  t("app.js halts on a roster member with phash 0", app.includes("roster.find(id => hs.has(id) && !hs.get(id).phash)"));
  console.log(`test_probe_unproven.js: ${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
})();
