// Mutation-scoring the oracles against MY OWN bugs — July 2026.
//
// test_corpus.js scores the checks against someone else's bug taxonomy. That
// answers "how do my oracles do on kernel bugs an outsider wrote down". This
// file asks the harder question: this month produced four real bugs with
// names and fixes. What does the corpus of MY OWN bugs score?
//
//   strippedBinding  a layout:auto pipeline silently dropped the unused scale
//                    bindings, so the verify variant bound the wrong buffers.
//                    Modeled as: the kernel ignores the column scales.
//   roundOnce        the fused q+k+v sum ran in f64 and rounded once where the
//                    old code rounded to f32 after each add — a last-ulp spec
//                    fork that would split replicas.
//   deadGate         a gate whose pass was vacuous (tolerance compare, or no
//                    discriminating input) — a checker bug, not a kernel bug.
//   rosterStall      a gradient reached the leader but not one follower; the
//                    sync guard held SAFETY (no fork) but LIVENESS died — a
//                    live fleet parked for hours. A protocol bug: every
//                    computed value everywhere was correct.
//
// The instruments scored:
//   properties    the reference-free suite (relations + definitional anchors)
//   differential  the exact bit-level differential gate
//   metaTest      mutation-testing the GATES themselves (test_gates.js's job)
//   liveness      a protocol simulation: deliver gradients asymmetrically and
//                 require the run to finish with identical weights
const fs = require("fs");
const path = require("path");
const V = require("./public/verified_core.js");
const C = require("./test_corpus.js");           // propertySuite, differentialGate, quantize
const L = { mul: new Int16Array(fs.readFileSync(path.join(__dirname, "public", "mul_lut.bin")).buffer.slice(0)) };
const f32 = Math.fround;

let pass = true;
const ok = (c, msg) => { console.log(`${c ? "  ok  " : "  FAIL"}  ${msg}`); if (!c) pass = false; };

// ---- bug 1: strippedBinding (kernel ignores the column scales) --------------
// The real bug bound the wrong buffer; the observable effect is a kernel whose
// per-column scale is not the one the mirror used — a per-column scalar error.
function kStripped(Xq, Wq, rs, cs, d) {
  return V.bgemmJS(Xq, Wq, rs, new Float32Array(cs.length).fill(1), d, L);
}
// ---- bug 2: roundOnce (epilogue analogue: one rounding instead of the chain) -
// The real bug was in a fused backward sum; its species is "right value, wrong
// rounding schedule". Injected here in the epilogue where the oracles look.
function kRoundOnce(Xq, Wq, rs, cs, d) {
  const o = V.bgemmJS(Xq, Wq, rs, cs, { ...d, acc: true }, L);
  if (d.acc) return o;
  const { m, n } = d, batch = d.batch || 1, relu = !!d.relu;
  const out = new Float32Array(o.length);
  for (let bz = 0; bz < batch; bz++)
    for (let i = 0; i < m; i++)
      for (let j = 0; j < n; j++) {
        let v = f32(o[(bz * m + i) * n + j] * rs[bz * m + i] * cs[bz * n + j]);   // f64 chain, ONE rounding
        if (relu && v < 0) v = 0;
        out[(bz * m + i) * n + j] = v;
      }
  return out;
}

console.log("\nMy own July-2026 bugs, scored against every instrument I have.\n");
console.log("bug              lives in    properties   differential   metaTest   liveness");
console.log("------------------------------------------------------------------------------");

// data-plane bugs: properties vs differential
{
  const p1 = C.propertySuite(kStripped), d1 = C.differentialGate(kStripped);
  console.log(`strippedBinding  data/scale  ${(p1 ? "CAUGHT" : "MISSED").padEnd(12)} ${(d1 ? "CAUGHT" : "MISSED").padEnd(14)} -          -`);
  // the theorem, again: ignoring cs is a per-column scalar — invisible to every
  // relation AND to the unit-scale anchor (which pins cs=1, exactly what the
  // bug substitutes). If a future property catches it, this assertion should
  // be UPDATED, not deleted — it documents where the blindness boundary is.
  ok(p1 === null, "properties are blind to strippedBinding (per-column scalar: the c*out theorem)");
  ok(d1 !== null, `differential catches strippedBinding (${d1})`);

  const p2 = C.propertySuite(kRoundOnce), d2 = C.differentialGate(kRoundOnce);
  console.log(`roundOnce        data/round  ${(p2 ? "CAUGHT" : "MISSED").padEnd(12)} ${(d2 ? "CAUGHT" : "MISSED").padEnd(14)} -          -`);
  ok(p2 === null, "properties are blind to roundOnce (last-ulp; the anchor sits at rs=cs=1 where both schedules agree)");
  ok(d2 !== null, `differential catches roundOnce (${d2})`);
}

// ---- bug 3: deadGate — a CHECKER bug; only a meta-test can see it -----------
// gateLive is the shipped exact gate; gateDead is the old tolerance gate. The
// meta-test feeds a known-bad kernel to a gate and demands rejection: a gate
// that accepts a known bug is dead, whatever it accepts elsewhere.
{
  const randf = (n) => Float32Array.from({ length: n }, () => Math.random() * 2 - 1);
  const mkGate = (exact) => (K) => {
    for (const d of [{ m: 7, k: 33, n: 6, batch: 2, relu: true }, { m: 5, k: 9, n: 4, batch: 1, relu: false }]) {
      const A = randf(d.batch * d.m * d.k), B = randf(d.batch * d.k * d.n);
      const q = C.quantize(A, B, d);
      const hw = K(q.x.q, q.wq, q.x.s, q.ws, d);
      const ref = V.bgemmJS(q.x.q, q.wq, q.x.s, q.ws, d, L);
      for (let i = 0; i < ref.length; i++) {
        if (exact ? V.bitDiff(hw[i], ref[i]) : Math.abs(hw[i] - ref[i]) > Math.abs(ref[i]) * 1e-6 + 1e-6)
          return "mismatch";
      }
    }
    return null;
  };
  const gateLive = mkGate(true), gateDead = mkGate(false);
  const metaTest = (gate) => gate(kRoundOnce) !== null;     // must reject a known bug
  const deadCaught = !metaTest(gateDead), liveOk = metaTest(gateLive);
  console.log(`deadGate         checker     ${"-".padEnd(12)} ${"-".padEnd(14)} ${deadCaught ? "CAUGHT" : "MISSED"}     -`);
  ok(deadCaught, "metaTest catches the dead gate (tolerance gate waves the roundOnce bug through)");
  ok(liveOk, "metaTest passes the live gate (exact gate rejects the same bug)");
}

// ---- bug 4: rosterStall — a PROTOCOL bug; only a liveness test can see it ----
// Three peers, deterministic per-step gradients, and a bus that delivers peer
// C's step-1 gradient to the leader but NOT to follower B (the production
// incident's exact shape). Every computed value on every peer is correct, so
// no data oracle can fire. The old protocol (wait, then halt) loses liveness;
// the repair protocol (ask the leader, who retains 8 steps) finishes the run
// with identical weights everywhere.
{
  const STEPS = 5, PEERS = ["L", "B", "C"];
  const gradOf = (peer, step) => (peer.charCodeAt(0) * 31 + step * 7) % 100;   // deterministic
  function run(repair) {
    const weights = { L: 0, B: 0, C: 0 };
    const retained = new Map();                                 // leader's grad store
    for (let s = 0; s < STEPS; s++) {
      const inbox = {};
      for (const p of PEERS) inbox[p] = { [p]: gradOf(p, s) };
      for (const from of PEERS)
        for (const to of PEERS) {
          if (from === to) continue;
          if (s === 1 && from === "C" && to === "B") continue;  // the asymmetric drop
          inbox[to][from] = gradOf(from, s);
        }
      for (const p of PEERS) retained.set(`${s}:${p}`, gradOf(p, s));          // leader retains
      for (const p of PEERS) {
        const missing = PEERS.filter((q) => !(q in inbox[p]));
        for (const q of missing) {
          if (!repair) return { halted: `${p} missing ${q}@${s}` };            // old: guard stops
          inbox[p][q] = retained.get(`${s}:${q}`);                             // new: leader repairs, bit-exact
        }
        // strict roster order, like the real averaging
        weights[p] += PEERS.reduce((a, q) => a + inbox[p][q], 0) / PEERS.length;
      }
    }
    return { weights };
  }
  const oldP = run(false), newP = run(true);
  const identical = newP.weights && newP.weights.L === newP.weights.B && newP.weights.B === newP.weights.C;
  console.log(`rosterStall      protocol    ${"-".padEnd(12)} ${"-".padEnd(14)} -          ${oldP.halted && identical ? "CAUGHT" : "MISSED"}`);
  ok(!!oldP.halted, `liveness test catches the old protocol (halts: ${oldP.halted})`);
  ok(identical, "repair protocol finishes the run with identical weights on all three peers");
}

console.log(`
scoreboard, my own bugs:
  properties    0/2  (of the data-plane bugs they could address — the c*out theorem, both times)
  differential  2/2  (of the data-plane bugs)
  metaTest      1/1  (the checker bug — only mutation-testing the gate sees it)
  liveness      1/1  (the protocol bug — no data oracle can fire when every value is correct)

Half of this month's bugs did not live in the kernels at all. The instruments
that caught them are not better oracles — they are DIFFERENT instruments:
tests of the checkers, and tests of the protocol. Union: 4/4, but only
because the population defines the instrument, not the other way round.`);

console.log(pass ? "\nSELF-CORPUS TEST PASSED" : "\nSELF-CORPUS TEST FAILED");
process.exit(pass ? 0 : 1);
