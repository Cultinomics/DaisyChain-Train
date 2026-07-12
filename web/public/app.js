// DaisyChain-Web client: connect P2P (WebRTC), compute (WebGPU or CPU), and
// train a shared model together — averaging gradients over the data channels.
"use strict";

// model + task now live in transformer.js — a mini transformer LM trained
// through the verified INT8 units. Settings come from the sliders.
const STUN = [{ urls: "stun:stun.l.google.com:19302" }];

const ui = {
  status: document.getElementById("status"),
  backend: document.getElementById("backend"),
  me: document.getElementById("me"),
  peers: document.getElementById("peers"),
  loss: document.getElementById("loss"),
  step: document.getElementById("step"),
  bar: document.getElementById("bar"),
  diff: document.getElementById("diff"),
  log: document.getElementById("log"),
  start: document.getElementById("start"),
  save: document.getElementById("save"),
  load: document.getElementById("load"),
  loadBtn: document.getElementById("loadBtn"),
  requests: document.getElementById("requests"),
  roomInfo: document.getElementById("roomInfo"),
  roomCode: document.getElementById("roomCode"),
  copyLink: document.getElementById("copyLink"),
  cfgC: document.getElementById("cfgC"),
  cfgT: document.getElementById("cfgT"),
  cfgB: document.getElementById("cfgB"),
  cfgSteps: document.getElementById("cfgSteps"),
  cfgLr: document.getElementById("cfgLr"),
};
function log(m) { ui.log.textContent = `${new Date().toLocaleTimeString()}  ${m}\n` + ui.log.textContent; }
function setStatus(s) { ui.status.textContent = s; }

// ---- deterministic RNG so every peer agrees on W_true and W0 (no broadcast) --
function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function randn(n, rng) { const r = rng || Math.random; const o = new Float32Array(n); for (let i = 0; i < n; i += 2) { let u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); const m = Math.sqrt(-2 * Math.log(u)); o[i] = m * Math.cos(2 * Math.PI * v); if (i + 1 < n) o[i + 1] = m * Math.sin(2 * Math.PI * v); } return o; }

// ---- state -----------------------------------------------------------------
// friendly device name (cottagecore, Snapdrop-style)
const ADJ = ["Mossy", "Golden", "Amber", "Fern", "Hazel", "Cozy", "Wandering", "Little",
             "Sunny", "Misty", "Wild", "Quiet", "Brave", "Dusty", "Merry"];
const NOUN = ["Fox", "Hare", "Owl", "Badger", "Toad", "Sparrow", "Otter", "Deer",
              "Hedgehog", "Mushroom", "Acorn", "Willow", "Robin", "Fawn", "Moth"];
const deviceName = ADJ[Math.floor(Math.random() * ADJ.length)] + " " +
                   NOUN[Math.floor(Math.random() * NOUN.length)];

let myId = null, compute = null, ws = null, L = null, wasDenied = false;
const pcs = new Map(), chans = new Map();      // peerId -> RTCPeerConnection / DataChannel
const names = new Map();                       // peerId -> device name
const incoming = new Map();                    // step -> Map(peerId -> Float32Array)
let model = null, training = false;            // the mini transformer (transformer.js)
let trainedSteps = 0;                          // steps baked into the current weights
function nmeOf(id) { return names.get(id) || id; }

function room() { return new URLSearchParams(location.search).get("room"); }   // null -> group by network
function updatePeers() { ui.peers.textContent = chans.size ? [...chans.keys()].map(nmeOf).join(", ") : "(none yet)"; }

// ---- signaling + WebRTC ----------------------------------------------------
function connectSignaling() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const params = new URLSearchParams();
  params.set("name", deviceName);
  if (room()) params.set("room", room());                             // no code -> group by network
  ws = new WebSocket(`${proto}://${location.host}/?${params}`);
  ws.onopen = () => setStatus(room() ? `connected — private room "${room()}"` : "connected — grouping with devices on your network");
  ws.onclose = () => { if (!wasDenied) setStatus("signaling disconnected"); };
  ws.onmessage = async (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "welcome") {
      myId = msg.id;
      if (msg.room) log(`group: ${msg.room.startsWith("net:") ? "your network" : "private room " + msg.room.replace("room:", "")}`);
      if (msg.host) { log("you host this room — joiners wait for your approval"); setStatus(`hosting private room "${room()}"`); }
      else if (room()) setStatus(`accepted into private room "${room()}"`);
      // I'm newest: initiate to everyone already here
      for (const p of msg.peers) { names.set(p.id, p.name); initiatePeer(p.id); }
    } else if (msg.type === "waiting") {
      setStatus("knocking — waiting for the room's host to let you in…");
    } else if (msg.type === "denied") {
      wasDenied = true;
      setStatus("the host declined your request to join");
      log("join request declined by the host");
    } else if (msg.type === "host") {
      log("the host left — you are now the host of this room");
      setStatus(`hosting private room "${room()}"`);
    } else if (msg.type === "join-request") {
      addJoinRequest(msg.id, msg.name);
    } else if (msg.type === "peer-joined") {
      names.set(msg.id, msg.name);
      log(`${msg.name} joined (they will connect to me)`);
    } else if (msg.type === "peer-left") {
      log(`${nmeOf(msg.id)} left`); cleanupPeer(msg.id); names.delete(msg.id); updatePeers();
    } else if (msg.type === "signal") {
      await onSignal(msg.from, msg.data);
    }
  };
}
function signal(to, data) { ws.send(JSON.stringify({ type: "signal", to, data })); }

// host-side Accept/Deny row for a knocking device (textContent only — the name
// comes off the wire and must never be parsed as HTML)
function addJoinRequest(id, name) {
  const row = document.createElement("div");
  row.style.cssText = "display:flex;gap:8px;align-items:center;justify-content:space-between;margin-top:10px;flex-wrap:wrap";
  const who = document.createElement("span");
  who.textContent = `🚪 ${name} wants to join`;
  const btn = (label, allow) => {
    const b = document.createElement("button");
    b.textContent = label;
    b.style.cssText = "padding:6px 14px;font-size:.85rem" + (allow ? "" : ";background:#8b2e25");
    b.onclick = () => {
      ws.send(JSON.stringify({ type: "admit", id, allow }));
      row.remove();
      log(allow ? `you let ${name} in` : `you declined ${name}`);
    };
    return b;
  };
  row.append(who, btn("Accept", true), btn("Deny", false));
  ui.requests.appendChild(row);
}

function newPC(peerId) {
  const pc = new RTCPeerConnection({ iceServers: STUN });
  pc.onicecandidate = (e) => { if (e.candidate) signal(peerId, { candidate: e.candidate }); };
  pc.onconnectionstatechange = () => { if (pc.connectionState === "failed") cleanupPeer(peerId); };
  pcs.set(peerId, pc);
  return pc;
}
function initiatePeer(peerId) {
  const pc = newPC(peerId);
  const dc = pc.createDataChannel("daisy");
  setupChannel(peerId, dc);
  pc.createOffer().then(o => pc.setLocalDescription(o)).then(() => signal(peerId, { sdp: pc.localDescription }));
}
async function onSignal(from, data) {
  let pc = pcs.get(from);
  if (data.sdp) {
    if (!pc) { pc = newPC(from); pc.ondatachannel = (e) => setupChannel(from, e.channel); }
    await pc.setRemoteDescription(data.sdp);
    if (data.sdp.type === "offer") {
      const ans = await pc.createAnswer(); await pc.setLocalDescription(ans);
      signal(from, { sdp: pc.localDescription });
    }
  } else if (data.candidate && pc) {
    try { await pc.addIceCandidate(data.candidate); } catch (e) {}
  }
}
function setupChannel(peerId, dc) {
  dc.binaryType = "arraybuffer";
  dc.onopen = () => { chans.set(peerId, dc); updatePeers(); log(`connected to ${nmeOf(peerId)}`); ui.start.disabled = false; };
  dc.onclose = () => { chans.delete(peerId); updatePeers(); wake(); };
  dc.onmessage = (e) => onGrad(peerId, e.data);
}
function cleanupPeer(id) { const pc = pcs.get(id); if (pc) pc.close(); pcs.delete(id); chans.delete(id); wake(); }

// ---- checkpoints ------------------------------------------------------------
// File layout (also the broadcast payload after the sentinel):
//   8 bytes magic "DAISYPT2" | int32 c,t,vocab,steps | f32 flat params
// DaisyChain's own format (not torch-pickle) — .pt extension for familiarity.
const CKPT_MAGIC = "DAISYPT2";
const CKPT_SENTINEL = -2;                      // wire: [int32 -2][checkpoint bytes]

function packCheckpoint() {
  const flat = Transformer.getFlatParams(model);
  const buf = new ArrayBuffer(8 + 16 + flat.length * 4);
  new Uint8Array(buf, 0, 8).set([...CKPT_MAGIC].map(c => c.charCodeAt(0)));
  new Int32Array(buf, 8, 4).set([model.cfg.c, model.cfg.t, model.cfg.vocab, trainedSteps]);
  new Float32Array(buf, 24).set(flat);
  return buf;
}
function parseCheckpoint(buf) {
  const magic = String.fromCharCode(...new Uint8Array(buf, 0, 8));
  if (magic !== CKPT_MAGIC) throw new Error("not a DaisyChain v2 checkpoint");
  const [c, t, vocab, steps] = new Int32Array(buf, 8, 4);
  if (vocab !== Transformer.VOCAB) throw new Error(`vocab mismatch (file ${vocab}, this build ${Transformer.VOCAB})`);
  if (c < 16 || c > 64 || t < 16 || t > 64) throw new Error(`bad dims in checkpoint (width ${c}, seq ${t})`);
  return { c, t, steps, flat: new Float32Array(buf.slice(24)) };
}
function applyCheckpoint(ck, from) {
  if (!model || model.cfg.c !== ck.c || model.cfg.t !== ck.t) {
    buildModel({ ...readCfgFromUI(), c: ck.c, t: ck.t });
    showCfgInUI({ ...readCfgFromUI(), c: ck.c, t: ck.t });
  }
  if (ck.flat.length !== model.nParams) throw new Error("truncated checkpoint");
  Transformer.setFlatParams(model, ck.flat);
  trainedSteps = ck.steps;
  ui.save.disabled = false;
  ui.step.textContent = `${ck.steps} baked in`;
  log(`checkpoint loaded (${ck.steps} steps) ${from ? "from " + from : "from file"} — all set to resume`);
}
function broadcastCheckpoint() {
  const ck = packCheckpoint();
  const msg = new ArrayBuffer(4 + ck.byteLength);
  new Int32Array(msg, 0, 1)[0] = CKPT_SENTINEL;
  new Uint8Array(msg, 4).set(new Uint8Array(ck));
  let n = 0;
  for (const dc of chans.values()) if (dc.readyState === "open") { dc.send(msg); n++; }
  log(`checkpoint pushed to ${n} device(s)`);
}
function saveCheckpoint() {
  const blob = new Blob([packCheckpoint()], { type: "application/octet-stream" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `daisychain-lm-w${model.cfg.c}-step${trainedSteps}.pt`;
  a.click();
  URL.revokeObjectURL(a.href);
}

// ---- training config: whoever presses Start sets it for the whole group ----
const CFG_SENTINEL = -3;                       // wire: [int32 -3][int32 c,t,b,steps][f32 lr]
function readCfgFromUI() {
  return { c: +ui.cfgC.value, t: +ui.cfgT.value, b: +ui.cfgB.value,
           steps: +ui.cfgSteps.value, lr: +ui.cfgLr.value / 1000 };
}
function showCfgInUI(cfg) {
  const set = (el, vid, val) => { el.value = val; document.getElementById(vid).textContent = val; };
  set(ui.cfgC, "vcfgC", cfg.c); set(ui.cfgT, "vcfgT", cfg.t); set(ui.cfgB, "vcfgB", cfg.b);
  set(ui.cfgSteps, "vcfgSteps", cfg.steps);
  set(ui.cfgLr, "vcfgLr", Math.round(cfg.lr * 1000));
}
function broadcastConfig(cfg) {
  const buf = new ArrayBuffer(24);
  new Int32Array(buf, 0, 5).set([CFG_SENTINEL, cfg.c, cfg.t, cfg.b, cfg.steps]);
  new Float32Array(buf, 20, 1)[0] = cfg.lr;
  for (const dc of chans.values()) if (dc.readyState === "open") dc.send(buf);
}
function onConfig(peerId, buf) {
  if (training) { log(`ignored settings from ${nmeOf(peerId)} (already training)`); return; }
  const [, c, t, b, steps] = new Int32Array(buf, 0, 5);
  const lr = new Float32Array(buf, 20, 1)[0];
  if (!(c >= 16 && c <= 64 && c % 2 === 0 && t >= 16 && t <= 64 && b >= 1 && b <= 32 &&
        steps >= 1 && steps <= 10000 && lr > 0 && lr <= 0.2)) {
    log(`rejected bad settings from ${nmeOf(peerId)}`); return;
  }
  const cfg = { c, t, b, steps, lr };
  showCfgInUI(cfg);
  log(`${nmeOf(peerId)} started the group: width=${c}, seq=${t}, batch=${b}, ${steps} steps, lr=${lr}`);
  buildModel(cfg);
  train(cfg);                                  // follow automatically
}

// ---- gradient wire format:  [int32 step][float32 grad...] -----------------
function packGrad(step, grad) {
  const buf = new ArrayBuffer(4 + grad.byteLength);
  new Int32Array(buf, 0, 1)[0] = step;
  new Float32Array(buf, 4).set(grad);
  return buf;
}
const waiters = new Set();                       // pending waitForGrads checkers
function wake() { for (const w of waiters) w(); }
function onGrad(peerId, buf) {
  const step = new Int32Array(buf, 0, 1)[0];
  if (step === CFG_SENTINEL) { onConfig(peerId, buf); return; }
  if (step === CKPT_SENTINEL) {                  // a peer pushed a checkpoint
    if (training) { log(`ignored checkpoint from ${nmeOf(peerId)} (training in progress)`); return; }
    try { applyCheckpoint(parseCheckpoint(buf.slice(4)), nmeOf(peerId)); }
    catch (e) { log(`bad checkpoint from ${nmeOf(peerId)}: ${e.message}`); }
    return;
  }
  const grad = new Float32Array(buf.slice(4));
  if (!incoming.has(step)) incoming.set(step, new Map());
  incoming.get(step).set(peerId, grad);
  wake();                                        // resolve waits immediately (no polling)
}
function broadcastGrad(step, grad) { const b = packGrad(step, grad); for (const dc of chans.values()) if (dc.readyState === "open") dc.send(b); }
// Event-driven: re-checked on every gradient arrival and peer departure, plus a
// coarse fallback timer (background tabs throttle timers to ~1s, so the old
// 15ms poll was the bottleneck there). Peers that left are dropped from the
// wait — a device dying no longer costs the full timeout every step.
function waitForGrads(step, cohort, timeoutMs = 8000) {
  return new Promise((resolve) => {
    const t0 = Date.now();
    let timer = null;
    const check = () => {
      const live = cohort.filter(id => chans.has(id));   // prune departed peers
      const got = incoming.get(step) || new Map();
      const have = live.filter(id => got.has(id));
      if (have.length === live.length || Date.now() - t0 > timeoutMs) {
        waiters.delete(check); clearInterval(timer);
        resolve(have.map(id => got.get(id)));
      }
    };
    waiters.add(check);
    timer = setInterval(check, 500);                     // safety net only
    check();
  });
}

// ---- compute: one async training step THROUGH the verified units -----------
async function localStep() {
  // transformer forward runs through the verified INT8 multiply; STE backward
  return Transformer.trainStep(model);
}

// ---- the training loop -----------------------------------------------------
async function train(cfg) {
  if (training) return; training = true; ui.start.disabled = true;
  cfgSliders().forEach(el => el.disabled = true);
  const cohort = [...chans.keys()];                    // lock the cohort (departed peers are pruned per-step)
  const steps = cfg.steps;
  const opt = TrainCore.makeAdam(model.nParams, { lr: cfg.lr });
  log(`training started — cohort ${cohort.length} peer(s), world ${cohort.length + 1}, ` +
      `width=${cfg.c} seq=${cfg.t} batch=${cfg.b}×${cohort.length + 1}, optimizer ${opt.name}`);
  for (let s = 0; s < steps; s++) {
    const { loss, grad } = await localStep();
    broadcastGrad(s, grad);
    const remote = cohort.length ? await waitForGrads(s, cohort) : [];
    const all = [grad, ...remote];
    const avg = TrainCore.averageGrads(all);
    const upd = opt.step(avg);                      // DaisyAdam on the cluster-avg grad
    Transformer.applyUpdate(model, upd);            // W -= upd (lr folded into upd)
    incoming.delete(s);
    trainedSteps++;
    if (s % 10 === 0 || s === steps - 1) {
      ui.loss.textContent = loss.toFixed(5);
      ui.step.textContent = `${s + 1} / ${steps}`;
      ui.bar.style.width = `${Math.round(100 * (s + 1) / steps)}%`;
      await new Promise(r => setTimeout(r, 0));         // yield to UI
    }
  }
  log(`training done — final loss ${ui.loss.textContent}`);
  try {
    const sample = await Transformer.generate(model, "the ", 70);
    ui.diff.textContent = `the model speaks: “${sample.trim()}”`;
  } catch (e) {
    ui.diff.textContent = "done — trained through the verified units; all peers share one model.";
  }
  training = false;
  ui.save.disabled = false;
  ui.start.disabled = false;
  cfgSliders().forEach(el => el.disabled = false);
}
function cfgSliders() { return [ui.cfgC, ui.cfgT, ui.cfgB, ui.cfgSteps, ui.cfgLr]; }

// (re)build the mini transformer — deterministic shared init (same seeds on
// every peer); each device samples its own batch windows from the shared corpus
function buildModel(cfg) {
  model = Transformer.init(cfg, L, compute.matmulInt8);
  trainedSteps = 0;
}

// ---- boot ------------------------------------------------------------------
(async function () {
  // Room-first deployments (the HF Space sets DAISY_FORCE_ROOMS): there is no
  // LAN auto-grouping between strangers — visitors without a room code choose
  // to create their own room (they become its host) or join one by code.
  try {
    const mode = await (await fetch("mode")).json();
    if (mode.forceRooms && !room()) {
      const lobby = document.getElementById("lobby");
      for (const c of document.querySelectorAll(".card")) if (c !== lobby) c.style.display = "none";
      lobby.style.display = "";
      document.getElementById("createRoom").onclick = () => {
        const code = (ADJ[Math.floor(Math.random() * ADJ.length)] + "-" +
                      NOUN[Math.floor(Math.random() * NOUN.length)] + "-" +
                      (100 + Math.floor(Math.random() * 900))).toLowerCase();
        location.href = "?room=" + code;
      };
      const join = () => {
        const code = document.getElementById("joinCode").value.trim().toLowerCase();
        if (code) location.href = "?room=" + encodeURIComponent(code);
      };
      document.getElementById("joinRoom").onclick = join;
      document.getElementById("joinCode").onkeydown = (e) => { if (e.key === "Enter") join(); };
      return;                                            // wait for the choice
    }
  } catch (e) {}                                         // no /mode: local default

  // Neural Units are mandatory: no LUTs -> no training, period. There is no
  // float fallback path anywhere in this app; both backends (WebGPU shader and
  // CPU JS) compute every product through the verified mul8 LUT.
  try {
    L = await Compute.loadLUTs();                        // the verified units, as tables
    if (!(L.mul instanceof Int16Array) || L.mul.length !== 65536)
      throw new Error("mul8 LUT malformed");
    // self-test: the unit must reproduce a known product before we trust it
    if (L.mul[((7 & 0xFF) * 256) + (-3 & 0xFF)] !== -21)
      throw new Error("mul8 LUT self-test failed (7 × -3 ≠ -21)");
    compute = await Compute.initCompute(L);
  } catch (e) {
    setStatus("NEURAL UNITS UNAVAILABLE — training disabled");
    ui.backend.textContent = "unavailable";
    log(`FATAL: verified neural units failed to load (${e.message}). ` +
        `This build only trains through the units — there is no fallback.`);
    ui.start.disabled = true;
    return;                                              // no signaling, no training
  }
  ui.backend.textContent = `${compute.backend.toUpperCase()} — ${compute.label} · through verified INT8 units`;
  // slider readouts
  for (const [el, v] of [[ui.cfgC, "vcfgC"], [ui.cfgT, "vcfgT"], [ui.cfgB, "vcfgB"],
                         [ui.cfgSteps, "vcfgSteps"], [ui.cfgLr, "vcfgLr"]])
    el.oninput = () => document.getElementById(v).textContent = el.value;
  buildModel(readCfgFromUI());
  ui.me.textContent = deviceName;
  if (room()) {
    ui.roomInfo.style.display = "";
    ui.roomCode.textContent = room();
    ui.copyLink.onclick = async () => {
      try { await navigator.clipboard.writeText(location.href); ui.copyLink.textContent = "Copied!"; }
      catch { ui.copyLink.textContent = location.href; }
      setTimeout(() => ui.copyLink.textContent = "Copy invite link", 2000);
    };
  }
  updatePeers();
  connectSignaling();
  ui.start.onclick = () => {
    const cfg = readCfgFromUI();
    buildModel(cfg);
    broadcastConfig(cfg);                                // everyone follows these settings
    train(cfg);
  };
  ui.save.onclick = saveCheckpoint;
  ui.loadBtn.onclick = () => { if (training) { log("can't load a checkpoint mid-training"); return; } ui.load.click(); };
  ui.load.onchange = async () => {
    const f = ui.load.files[0]; ui.load.value = "";
    if (!f) return;
    try {
      const ck = parseCheckpoint(await f.arrayBuffer());
      applyCheckpoint(ck, null);
      broadcastCheckpoint();                             // every device resumes from this
    } catch (e) { log(`checkpoint rejected: ${e.message}`); }
  };
  log(`ready — this device is "${deviceName}", computing through verified units on ${compute.backend.toUpperCase()}`);
})();
