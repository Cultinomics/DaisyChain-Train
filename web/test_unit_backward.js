// Does computing the STE gradient THROUGH the verified units (block-scaled int8)
// instead of in float damage convergence?
//
// Identical seeds, identical data, identical optimizer. The only difference is
// whether backward()'s matmuls run in f32 or through mul8. Runs on the CPU LUT
// mirrors, so this measures the arithmetic question and nothing else.
const fs = require("fs");
const path = require("path");
const T = require("./public/traincore.js");
const V = require("./public/verified_core.js");
const X = require("./public/transformer.js");

const p = (f) => path.join(__dirname, "public", f);
const L = { mul: new Int16Array(fs.readFileSync(p("mul_lut.bin")).buffer.slice(0)),
            requant: new Int8Array(fs.readFileSync(p("requant_lut.bin")).buffer.slice(0)),
            relu: new Int8Array(fs.readFileSync(p("relu_lut.bin")).buffer.slice(0)) };

// the real Spikewhale tokenizer if asked — 16512 tokens is where gradient
// dynamic range actually hurts, so the 96-char fallback would flatter the result
if (process.env.REAL_TOK) {
  X.loadTokenizerData(JSON.parse(fs.readFileSync(p("tokenizer.json"), "utf8")));
  console.log(`tokenizer: ${X.tokenizerName()}`);
}
const STEPS = +(process.env.STEPS || 150);
const base = { c: 32, t: 32, b: 8, layers: 2, heads: 2, steps: STEPS, lr: 0.02 };

// deterministic batches so both runs see byte-identical data
function makeBatches(n, cfg) {
  let seed = 1234;
  const rnd = () => { seed = (Math.imul(seed, 1103515245) + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
  const out = [];
  for (let s = 0; s < n; s++) {
    const ids = [];
    for (let i = 0; i < cfg.b; i++) ids.push(Math.floor(rnd() * 1e6));
    out.push(ids);
  }
  return out;
}

async function run(label, unitBackward, dataSeed, lr) {
  const cfg = { ...base, unitBackward, lr: lr || base.lr };
  const m = X.init(cfg, L, null);
  const opt = T.makeAdam(m.nParams, { lr: cfg.lr });
  // pin Math.random so both runs draw identical batch windows
  let seed = dataSeed || 777;
  const orig = Math.random;
  Math.random = () => { seed = (Math.imul(seed, 1103515245) + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
  const curve = [];
  const t0 = Date.now();
  for (let s = 0; s < STEPS; s++) {
    const r = await X.trainStep(m);
    X.applyUpdate(m, opt.step(r.grad));
    curve.push(r.loss);
  }
  Math.random = orig;
  const ms = (Date.now() - t0) / STEPS;
  const tail = curve.slice(-10).reduce((a, b) => a + b) / 10;
  return { label, curve, tail, ms, first: curve[0] };
}

(async () => {
  console.log(`\nvocab=${X.vocabSize()}  steps=${STEPS}  (CPU LUT mirrors, identical seeds)\n`);
  const f = await run("float backward", false);
  const u = await run("unit backward ", true);
  console.log("step        float      units");
  for (const s of [0, 25, 50, 75, 100, Math.min(125, STEPS - 1), STEPS - 1]) {
    if (s >= STEPS) continue;
    console.log(`${String(s).padStart(4)}   ${f.curve[s].toFixed(4).padStart(9)}  ${u.curve[s].toFixed(4).padStart(9)}`);
  }
  console.log(`\nfinal (avg last 10)   float ${f.tail.toFixed(4)}   units ${u.tail.toFixed(4)}`);
  console.log(`ms/step               float ${f.ms.toFixed(0)}       units ${u.ms.toFixed(0)}`);
  const ratio = u.tail / f.tail;
  console.log(`\nunits/float loss ratio: ${ratio.toFixed(3)}  (1.00 = no damage)`);
  const converged = u.tail < u.first * 0.75 && u.tail < Math.log(X.vocabSize());
  console.log(converged ? "unit backward CONVERGES" : "unit backward FAILED TO CONVERGE");
  console.log(ratio < 1.15 ? "damage within 15% — viable" : "damage exceeds 15% — needs work (try 2-pass gradients)");
})();
