// A float -> int8 quantize has no answer for NaN/Inf. Before the guard, an Inf made the |max|
// scale Infinity and every value quantized to 0 ([0.5, Inf, 3] -> [0, 0, 0]), and a NaN was
// skipped by the |max| scan and then stored as 0 by the Int8Array -- training continued on
// zeros with no error. Every quantizer must now refuse, and finite inputs must be unchanged.
"use strict";
const V = require("./public/verified_core.js");

let fails = 0;
function ck(name, cond) { console.log((cond ? "  ok    " : "  FAIL  ") + name); if (!cond) fails++; }
function throwsNonFinite(fn) { try { fn(); return false; } catch (e) { return e.name === "NonFiniteError"; } }

const bad = [Float32Array.from([0.5, Infinity, 3, -1]), Float32Array.from([0.5, NaN, 3, -1]),
             Float32Array.from([-Infinity, 1, 2, 3])];
for (const X of bad) {
  const tag = Array.from(X).join(",");
  ck("quantize refuses [" + tag + "]", throwsNonFinite(() => V.quantize(X)));
  ck("quantizeRows refuses [" + tag + "]", throwsNonFinite(() => V.quantizeRows(X, 2, 2)));
  ck("quantizeCols refuses [" + tag + "]", throwsNonFinite(() => V.quantizeCols(X, 2, 2)));
  ck("rowAbsMax refuses [" + tag + "]", throwsNonFinite(() => V.rowAbsMax(X, 2, 2)));
}
const q = V.quantize(Float32Array.from([0.5, -1, 3]));
ck("finite input quantizes as before: [21, -42, 127]", Array.from(q.q).join() === "21,-42,127");

if (fails) { console.log("NONFINITE TEST FAILED: " + fails); process.exit(1); }
console.log("NONFINITE TEST PASSED — every quantizer refuses NaN/Inf instead of zeroing.");
