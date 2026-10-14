// DaisyChain-Web client: connect P2P (WebRTC), compute (WebGPU or CPU), and
// train a shared model together — averaging gradients over the data channels.
"use strict";

// model + task now live in transformer.js — a mini transformer LM trained
// through the verified INT8 units. Settings come from the sliders.
// STUN discovers the direct path; the TURN relays (Open Relay, free) carry the
// traffic when both sides sit behind symmetric NATs (mobile carriers, corp
// networks) — without TURN those users can never connect across networks.
const STUN = [
  { urls: "stun:stun.l.google.com:19302" },
  { urls: "turn:openrelay.metered.ca:80", username: "openrelayproject", credential: "openrelayproject" },
  { urls: "turn:openrelay.metered.ca:443", username: "openrelayproject", credential: "openrelayproject" },
  { urls: "turns:openrelay.metered.ca:443?transport=tcp", username: "openrelayproject", credential: "openrelayproject" },
];
let rtcConfig = { iceServers: STUN };          // /mode can override (own TURN etc.)

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
  dataset: document.getElementById("dataset"),
  tokenizer: document.getElementById("tokenizer"),
  kit: document.getElementById("kit"),
  genPrompt: document.getElementById("genPrompt"),
  genBtn: document.getElementById("genBtn"),
  genOut: document.getElementById("genOut"),
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
// ---- sync guard state -------------------------------------------------------
// The peer that presses Start is the round leader. Every step the leader
// publishes the exact set of contributors to average (the roster); followers
// apply that set verbatim or stop — no device may silently average a
// different set, which is what used to fork the weights.
let leaderId = null;                           // null => I lead (I pressed Start)
const rosters = new Map();                     // step -> [peerId,...] from the leader
const peerHashes = new Map();                  // step -> Map(peerId -> uint32 pre-step weight hash)
function nmeOf(id) { return names.get(id) || id; }

function room() { return new URLSearchParams(location.search).get("room"); }   // null -> group by network
// show EVERY device the room knows about, not just open data channels —
// a peer whose WebRTC is still connecting (or failed) must stay visible
function updatePeers() {
  if (!names.size) { ui.peers.textContent = "(none yet)"; return; }
  ui.peers.textContent = [...names.entries()]
    .map(([id, n]) => `${n} ${chans.has(id) ? "✓" : "(connecting…)"}`).join(", ");
}

// ---- signaling + WebRTC ----------------------------------------------------
function connectSignaling() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const params = new URLSearchParams();
  params.set("name", deviceName);
  if (room()) params.set("room", room());                             // no code -> group by network
  ws = new WebSocket(`${proto}://${location.host}/?${params}`);
  ws.onopen = () => setStatus(room() ? `connected — private room "${room()}"` : "connected — grouping with devices on your network");
  ws.onclose = () => {
    if (wasDenied) return;
    setStatus("signaling disconnected — reconnecting…");
    setTimeout(connectSignaling, 3000);        // data channels survive; rejoin the room
  };
  ws.onmessage = async (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "welcome") {
      myId = msg.id;
      if (msg.room) log(`group: ${msg.room.startsWith("net:") ? "your network" : "private room " + msg.room.replace("room:", "")}`);
      if (msg.host) { log("you host this room — joiners wait for your approval"); setStatus(`hosting private room "${room()}"`); }
      else if (room()) setStatus(`accepted into private room "${room()}"`);
      // I'm newest: initiate to everyone already here (skip channels that
      // survived a signaling blip — no duplicate connections)
      for (const p of msg.peers) { names.set(p.id, p.name); if (!chans.has(p.id)) initiatePeer(p.id); }
      updatePeers();
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
      updatePeers();
      log(`${msg.name} joined (they will connect to me)`);
    } else if (msg.type === "peer-left") {
      log(`${nmeOf(msg.id)} left`); cleanupPeer(msg.id); names.delete(msg.id); updatePeers();
    } else if (msg.type === "signal") {
      await onSignal(msg.from, msg.data);
    } else if (msg.type === "ping") {          // keepalive (proxies close idle sockets)
      ws.send(JSON.stringify({ type: "pong" }));
    } else if (msg.type === "relay") {         // payload relayed via the server (WS fallback)
      onGrad(msg.from, bufFromB64(msg.data));
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
  const pc = new RTCPeerConnection(rtcConfig);
  pc.onicecandidate = (e) => { if (e.candidate) signal(peerId, { candidate: e.candidate }); };
  pc.onconnectionstatechange = () => {
    if (pc.connectionState === "failed") { cleanupPeer(peerId); scheduleReconnect(peerId); }
    // "disconnected" often self-heals; give it a grace period, then treat as failed
    if (pc.connectionState === "disconnected")
      setTimeout(() => {
        if (pcs.get(peerId) === pc && pc.connectionState === "disconnected") {
          log(`${nmeOf(peerId)} connection did not recover — redialing`);
          cleanupPeer(peerId); scheduleReconnect(peerId);
        }
      }, 4000);
  };
  pcs.set(peerId, pc);
  return pc;
}

// ---- auto-reconnect ----------------------------------------------------------
// If a data channel drops but the peer is still in the room (signaling knows
// them), redial with backoff. Only the higher-numbered peer initiates, so both
// sides don't collide (offer glare). If the peer truly left, names loses them
// and the retries stop on their own.
// ---- WebSocket relay fallback (PairDrop's WSPeer pattern) ---------------------
// A peer still in the room after 5 failed WebRTC attempts is reachable through
// the signaling server even though no direct path exists (symmetric NATs, no
// TURN). Install a pseudo data channel that relays binary over the WS instead
// of dropping them. Devices that actually left are removed via peer-left.
function b64FromBuf(buf) {
  const u = new Uint8Array(buf); let s = "";
  for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode(...u.subarray(i, i + 0x8000));
  return btoa(s);
}
function bufFromB64(s) { return Uint8Array.from(atob(s), c => c.charCodeAt(0)).buffer; }
function makeRelayChannel(peerId) {
  return {
    isRelay: true,
    get readyState() { return ws && ws.readyState === 1 && names.has(peerId) ? "open" : "closed"; },
    get bufferedAmount() { return ws ? ws.bufferedAmount : 0; },
    send(buf) { ws.send(JSON.stringify({ type: "relay", to: peerId, data: b64FromBuf(buf) })); },
    close() { chans.delete(peerId); },
  };
}

const reconnectTimers = new Map();             // peerId -> attempt count
function scheduleReconnect(peerId, attempt = 0) {
  if (!names.has(peerId) || chans.has(peerId) || reconnectTimers.has(peerId)) return;
  if (attempt >= 5) {                          // no direct path — fall back to server relay
    if (names.has(peerId)) {
      log(`no direct WebRTC path to ${nmeOf(peerId)} after 5 attempts — relaying through the server instead`);
      chans.set(peerId, makeRelayChannel(peerId));
      updatePeers(); wake();
      ui.start.disabled = false;
    }
    return;
  }
  const delay = 1500 * Math.pow(2, attempt);
  reconnectTimers.set(peerId, setTimeout(() => {
    reconnectTimers.delete(peerId);
    if (!names.has(peerId) || chans.has(peerId)) return;
    if (+myId.slice(1) > +peerId.slice(1)) {   // deterministic initiator
      log(`reconnecting to ${nmeOf(peerId)} (attempt ${attempt + 1})…`);
      cleanupPeer(peerId);
      initiatePeer(peerId);
    }
    setTimeout(() => scheduleReconnect(peerId, attempt + 1), 6000);
  }, delay));
}
function initiatePeer(peerId) {
  const pc = newPC(peerId);
  const dc = pc.createDataChannel("daisy");
  setupChannel(peerId, dc);
  pc.createOffer().then(o => pc.setLocalDescription(o)).then(() => signal(peerId, { sdp: pc.localDescription }));
}
// ICE candidates can arrive while setRemoteDescription is still awaiting (the
// async handlers interleave) — adding one before the description is set throws
// and the candidate is lost, so the connection silently fails. Queue them and
// flush once the remote description lands.
const pendingCand = new Map();                 // peerId -> [candidate,...]
async function onSignal(from, data) {
  let pc = pcs.get(from);
  if (data.sdp) {
    if (!pc) { pc = newPC(from); pc.ondatachannel = (e) => setupChannel(from, e.channel); }
    await pc.setRemoteDescription(data.sdp);
    for (const c of pendingCand.get(from) || [])
      try { await pc.addIceCandidate(c); } catch (e) {}
    pendingCand.delete(from);
    if (data.sdp.type === "offer") {
      const ans = await pc.createAnswer(); await pc.setLocalDescription(ans);
      signal(from, { sdp: pc.localDescription });
    }
  } else if (data.candidate) {
    if (pc && pc.remoteDescription) {
      try { await pc.addIceCandidate(data.candidate); } catch (e) {}
    } else {                                   // pc missing or not ready: hold it
      if (!pendingCand.has(from)) pendingCand.set(from, []);
      pendingCand.get(from).push(data.candidate);
    }
  }
}
function setupChannel(peerId, dc) {
  dc.binaryType = "arraybuffer";
  dc.onopen = () => {
    chans.set(peerId, dc); updatePeers(); log(`connected to ${nmeOf(peerId)}`); ui.start.disabled = false;
    // I'm leading a live run (leaderId null while training) — sync them in at
    // the next step boundary instead of making them wait for the next run
    if (training && leaderId === null) { pendingJoins.add(peerId); log(`${nmeOf(peerId)} will be synced into the running training`); }
  };
  dc.onclose = () => { chans.delete(peerId); dropFragments(peerId); updatePeers(); wake(); scheduleReconnect(peerId); };
  dc.onmessage = (e) => onGrad(peerId, e.data);
}
function cleanupPeer(id) {
  const pc = pcs.get(id); if (pc) pc.close();
  pcs.delete(id); chans.delete(id); pendingCand.delete(id); pendingJoins.delete(id);
  dropFragments(id);
  wake();
}
// A message cut off by a closed channel is never completed (the sender does not
// resume it), so its parts must go when the channel does. Before 2026-10-14 only
// cleanupPeer dropped them, and only the reconnect INITIATOR calls that: the
// other side kept up to a resume bundle's worth of dead parts per drop.
function dropFragments(id) {
  for (const k of [...fragIn.keys()]) if (k.startsWith(id + ":")) fragIn.delete(k);
}

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
  if (vocab !== Transformer.vocabSize())
    throw new Error(`tokenizer mismatch — checkpoint has a ${vocab}-token vocab, this device loaded ` +
                    `${Transformer.vocabSize()} (${Transformer.tokenizerName()}); use matching builds`);
  if (c < 16 || c > 128 || t < 16 || t > 128) throw new Error(`bad dims in checkpoint (width ${c}, seq ${t})`);
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
  modelReady();
  ui.step.textContent = `${ck.steps} baked in`;
  log(`checkpoint loaded (${ck.steps} steps) ${from ? "from " + from : "from file"} — all set to resume`);
}
function broadcastCheckpoint() {
  const ck = packCheckpoint();
  const msg = new ArrayBuffer(4 + ck.byteLength);
  new Int32Array(msg, 0, 1)[0] = CKPT_SENTINEL;
  new Uint8Array(msg, 4).set(new Uint8Array(ck));
  let n = 0;
  for (const dc of chans.values()) if (dc.readyState === "open") { dcSend(dc, msg); n++; }
  log(`checkpoint pushed to ${n} device(s) (${(msg.byteLength / 1048576).toFixed(1)} MB)`);
}
// ---- inference kit: one self-contained HTML file with the trained weights --
// Bundles the model code + current checkpoint (base64) + a prompt box into a
// single file that runs generations offline — no server, nothing to install.
// The mul8 LUT is rebuilt in the file itself (it is exactly a×b for int8).
async function downloadInferenceKit() {
  const srcs = await Promise.all(["traincore.js", "verified_core.js", "transformer.js"]
    .map(f => fetch(f).then(r => r.text())));
  let tokData = "null";                          // embed the tokenizer (offline file)
  try { tokData = await (await fetch("tokenizer.json")).text(); } catch (e) {}
  const ck = new Uint8Array(packCheckpoint());
  let b64 = ""; const CH = 0x8000;
  for (let i = 0; i < ck.length; i += CH) b64 += String.fromCharCode(...ck.subarray(i, i + CH));
  b64 = btoa(b64);
  const esc = (s) => s.replace(/<\/script/gi, "<\\/script");
  const html = `<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DaisyChain inference — ${trainedSteps} steps</title>
<style>body{font-family:sans-serif;max-width:680px;margin:0 auto;padding:24px 16px;background:#efe4c9;color:#2a1d0a}
input{width:100%;padding:10px;border-radius:8px;border:1px solid #8b6f47;font-family:monospace;box-sizing:border-box}
button{margin-top:10px;padding:10px 24px;border:0;border-radius:8px;background:#4a7c2e;color:#f5ecd9;font-weight:700;cursor:pointer}
pre{background:#fbf6e8;border-radius:8px;padding:12px;white-space:pre-wrap;min-height:60px}</style></head><body>
<h2>🌼 DaisyChain model — inference</h2>
<p>Trained ${trainedSteps} steps · width ${model.cfg.c} · seq ${model.cfg.t} · dataset: ${Transformer.datasetName()}.
Runs entirely in this file through the verified INT8 units (CPU LUT).</p>
<input id="p" value="the "><button id="g">Generate</button><pre id="o"></pre>
<script>${esc(srcs[0])}<\/script><script>${esc(srcs[1])}<\/script><script>${esc(srcs[2])}<\/script>
<script>
const mul = new Int16Array(65536);
for (let a = 0; a < 256; a++) for (let b = 0; b < 256; b++)
  mul[a*256+b] = ((a>127?a-256:a) * (b>127?b-256:b));
const L = { mul };
const TOKDATA = ${esc(tokData)};
if (TOKDATA) Transformer.loadTokenizerData(TOKDATA);
const bytes = Uint8Array.from(atob("${b64}"), c => c.charCodeAt(0));
const buf = bytes.buffer;
const dims = new Int32Array(buf, 8, 4);            // c, t, vocab, steps
const m = Transformer.init({ c: dims[0], t: dims[1], b: 1, steps: 0, lr: 0 }, L,
  (Xq, Wq, mm, kk, nn, LL) => Verified.lutMatmulJS(Xq, Wq, mm, kk, nn, LL));
Transformer.setFlatParams(m, new Float32Array(buf.slice(24)));
document.getElementById("g").onclick = async () => {
  const b = document.getElementById("g"), o = document.getElementById("o");
  b.disabled = true; o.textContent = "generating…";
  try { o.textContent = await Transformer.generate(m, document.getElementById("p").value || "the ", 150); }
  catch (e) { o.textContent = "error: " + e.message; }
  b.disabled = false;
};
<\/script></body></html>`;
  const blob = new Blob([html], { type: "text/html" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `daisychain-inference-step${trainedSteps}.html`;
  a.click();
  URL.revokeObjectURL(a.href);
  log("inference kit downloaded — a single HTML file: open it anywhere, type a prompt, generate");
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
  // lr crosses the wire as float32 — fround here so the leader trains with
  // the exact same value the followers decode (else Adam forks the weights)
  // ds is FIXED: FineWeb-Edu is the only dataset (still broadcast, so mixed
  // old/new clients keep interoperating — followers just ignore the choice)
  return { c: +ui.cfgC.value, t: +ui.cfgT.value, b: +ui.cfgB.value,
           steps: +ui.cfgSteps.value, lr: Math.fround(+ui.cfgLr.value / 1000),
           ds: "HuggingFaceFW/fineweb-edu" };
}
function showCfgInUI(cfg) {
  const set = (el, vid, val) => { el.value = val; document.getElementById(vid).textContent = val; };
  set(ui.cfgC, "vcfgC", cfg.c); set(ui.cfgT, "vcfgT", cfg.t); set(ui.cfgB, "vcfgB", cfg.b);
  set(ui.cfgSteps, "vcfgSteps", cfg.steps);
  set(ui.cfgLr, "vcfgLr", Math.round(cfg.lr * 1000));
}
// switch the training text to a chosen dataset (whole group uses the same one);
// on failure the built-in corpus stays and we log why
async function loadDataset(ds) {
  if (!ds) return;
  ui.dataset.textContent = `streaming ${ds}…`;
  try {
    const { name, chars } = await Transformer.streamDataset(ds);
    ui.dataset.textContent = name;
    log(`dataset: ${name} — ${(chars / 1000).toFixed(0)}k chars (this device's own slice)`);
  } catch (e) {
    ui.dataset.textContent = "built-in corpus";
    log(`dataset "${ds}" unavailable (${e.message}) — using the built-in corpus`);
  }
}
function broadcastConfig(cfg) {
  const dsBytes = new TextEncoder().encode((cfg.ds || "").slice(0, 96));
  const buf = new ArrayBuffer(32 + dsBytes.length);
  new Int32Array(buf, 0, 5).set([CFG_SENTINEL, cfg.c, cfg.t, cfg.b, cfg.steps]);
  new Float32Array(buf, 20, 1)[0] = cfg.lr;
  new Int32Array(buf, 24, 1)[0] = Transformer.vocabSize();   // tokenizer must match
  new Int32Array(buf, 28, 1)[0] = dsBytes.length;
  new Uint8Array(buf, 32).set(dsBytes);
  broadcast(buf);
}
async function onConfig(peerId, buf) {
  if (training) { log(`ignored settings from ${nmeOf(peerId)} (already training)`); return; }
  const [, c, t, b, steps] = new Int32Array(buf, 0, 5);
  const lr = new Float32Array(buf, 20, 1)[0];
  const vocab = buf.byteLength >= 28 ? new Int32Array(buf, 24, 1)[0] : -1;
  let ds = "";
  if (buf.byteLength >= 32) {
    const dsLen = new Int32Array(buf, 28, 1)[0];
    if (dsLen > 0 && dsLen <= 96 && buf.byteLength === 32 + dsLen)
      ds = new TextDecoder().decode(new Uint8Array(buf, 32, dsLen));
  }
  if (vocab !== Transformer.vocabSize()) {
    log(`NOT JOINING: ${nmeOf(peerId)} uses a ${vocab}-token tokenizer, this device has ` +
        `${Transformer.vocabSize()} (${Transformer.tokenizerName()}) — refresh so all devices match`);
    return;
  }
  if (!(c >= 16 && c <= 128 && c % 2 === 0 && t >= 16 && t <= 128 && b >= 1 && b <= 32 &&
        steps >= 1 && steps <= 10000 && lr > 0 && lr <= 0.2)) {
    log(`rejected bad settings from ${nmeOf(peerId)}`); return;
  }
  const cfg = { c, t, b, steps, lr: Math.fround(lr), ds };
  leaderId = peerId;                           // the starter leads sync for this run
  showCfgInUI(cfg);
  log(`${nmeOf(peerId)} started the group: width=${c}, seq=${t}, batch=${b}, ${steps} steps, lr=${lr}` +
      (ds ? `, dataset=${ds}` : ""));
  if (ds) await loadDataset(ds);               // same dataset as the starter
  buildModel(cfg);
  train(cfg);                                  // follow automatically
}

// ---- big-message fragmentation ----------------------------------------------
// WebRTC data channels cap a single message (~256KB Chrome, 64KB elsewhere).
// Gradients with the Spikewhale vocab are multi-MB, and checkpoints always
// were at width ≥ 64 — so anything large goes out in 48KB chunks:
//   [int32 -5][int32 msgId][int32 seq][int32 total][bytes...]
// Channels are ordered+reliable, so chunks arrive in order per peer.
const FRAG_SENTINEL = -5, FRAG_CHUNK = 48 * 1024;
const DC_MAXBUF = 4 * 1024 * 1024;               // pause sending above this backlog
let fragSeq = 1;
const fragIn = new Map();                        // peerId -> {id, parts, got, total}
function dcDrain(dc) {                           // wait for the send buffer to empty out
  return new Promise((res) => {
    if (dc.bufferedAmount <= DC_MAXBUF || dc.readyState !== "open") return res();
    const t = setInterval(() => {
      if (dc.bufferedAmount <= DC_MAXBUF || dc.readyState !== "open") { clearInterval(t); res(); }
    }, 50);
  });
}
async function dcSend(dc, buf) {
  // multi-MB gradients can overflow the channel's send buffer, which makes
  // dc.send throw and kills the training loop — so apply backpressure
  if (buf.byteLength <= FRAG_CHUNK) {
    await dcDrain(dc);
    if (dc.readyState === "open") dc.send(buf);
    return;
  }
  const id = fragSeq++, src = new Uint8Array(buf);
  const total = Math.ceil(src.length / FRAG_CHUNK);
  for (let s = 0; s < total; s++) {
    await dcDrain(dc);
    if (dc.readyState !== "open") return;        // peer left mid-send
    const part = src.subarray(s * FRAG_CHUNK, Math.min((s + 1) * FRAG_CHUNK, src.length));
    const msg = new ArrayBuffer(16 + part.length);
    new Int32Array(msg, 0, 4).set([FRAG_SENTINEL, id, s, total]);
    new Uint8Array(msg, 16).set(part);
    dc.send(msg);
  }
}
// parallel: one slow channel (e.g. a joiner mid-download of a resume bundle)
// must not delay the gradient broadcast to everyone else
function broadcast(buf) {
  return Promise.all([...chans.values()].filter(dc => dc.readyState === "open").map(dc => dcSend(dc, buf)));
}

// ---- mid-run join: the leader ships weights + Adam state + step -------------
// A device that connects (or reconnects) while training gets a resume bundle:
//   [i32 -6][i32 startStep][i32 adamT][i32 c,t,b,steps][f32 lr]
//   [f32 weights ×nP][f32 adam.m ×nP][f32 adam.v ×nP]
// From that snapshot it applies the exact same roster updates as everyone else
// (bit-identical), and its own gradients join the roster as soon as they
// arrive on time — no restart of the run.
const RESUME_SENTINEL = -6;
const pendingJoins = new Set();                // peers to sync in at the next step boundary
async function sendResume(peerId, startStep, opt, cfg) {
  const dc = chans.get(peerId);
  if (!dc || dc.readyState !== "open") return;
  const st = opt.getState();                   // snapshot NOW (top of step, post-update)
  const w = Transformer.getFlatParams(model);
  const nP = w.length;
  const buf = new ArrayBuffer(32 + nP * 12);
  new Int32Array(buf, 0, 7).set([RESUME_SENTINEL, startStep, st.t, cfg.c, cfg.t, cfg.b, cfg.steps]);
  new Float32Array(buf, 28, 1)[0] = cfg.lr;
  new Float32Array(buf, 32, nP).set(w);
  new Float32Array(buf, 32 + nP * 4, nP).set(st.m);
  new Float32Array(buf, 32 + nP * 8, nP).set(st.v);
  log(`syncing ${nmeOf(peerId)} into the run at step ${startStep + 1} (${(buf.byteLength / 1048576).toFixed(1)} MB state transfer)`);
  await dcSend(dc, buf);
}
function onResume(peerId, buf) {
  if (training) { log(`ignored resume bundle from ${nmeOf(peerId)} (already training)`); return; }
  const iv = new Int32Array(buf, 0, 7);
  const lr = Math.fround(new Float32Array(buf, 28, 1)[0]);
  const cfg = { c: iv[3], t: iv[4], b: iv[5], steps: iv[6], lr };
  if (!(cfg.c >= 16 && cfg.c <= 128 && cfg.t >= 16 && cfg.t <= 128 && cfg.b >= 1 && cfg.b <= 32 &&
        cfg.steps >= 1 && cfg.steps <= 10000 && lr > 0 && lr <= 0.2)) {
    log(`rejected bad resume bundle from ${nmeOf(peerId)}`); return;
  }
  buildModel(cfg); showCfgInUI(cfg);
  const nP = model.nParams;
  if (buf.byteLength !== 32 + nP * 12) { log(`resume bundle size mismatch from ${nmeOf(peerId)}`); return; }
  Transformer.setFlatParams(model, new Float32Array(buf.slice(32, 32 + nP * 4)));
  const adam = { m: new Float32Array(buf.slice(32 + nP * 4, 32 + nP * 8)),
                 v: new Float32Array(buf.slice(32 + nP * 8)), t: iv[2] };
  leaderId = peerId;
  trainedSteps = iv[1];
  log(`${nmeOf(peerId)} synced me into the running training at step ${iv[1] + 1} — joining the cohort`);
  train(cfg, { startStep: iv[1], adam });
}
function onFragment(peerId, buf) {               // returns full message when complete
  const [, id, seq, total] = new Int32Array(buf, 0, 4);
  // key by message id too: a resume bundle and step gradients can interleave
  // on the same channel, and each must reassemble independently
  const key = peerId + ":" + id;
  let st = fragIn.get(key);
  if (!st) { st = { id, parts: [], got: 0, total }; fragIn.set(key, st); }
  if (seq < 0 || seq >= st.total || total !== st.total) {     // malformed: drop the whole message, loudly
    log(`dropped malformed message ${id} from ${nmeOf(peerId)} (fragment ${seq}/${total}, expected /${st.total})`);
    fragIn.delete(key);
    return null;
  }
  if (!st.parts[seq]) st.got++;                  // a repeated part must not count twice
  st.parts[seq] = new Uint8Array(buf, 16).slice(0);
  if (st.got < st.total) return null;
  fragIn.delete(key);
  let len = 0; for (const p of st.parts) len += p.length;
  const out = new Uint8Array(len);
  let off = 0; for (const p of st.parts) { out.set(p, off); off += p.length; }
  return out.buffer;
}

// ---- gradient wire: [i32 step][u32 whash][u32 phash][f32 loss][f32 grad...]
// whash = FNV-1a of the sender's weights BEFORE this step's update — proves the
//   whole group is still applying the same updates to the same model.
// phash = FNV-1a of the sender's canonical kernel probe — proves the sender's
//   arithmetic still agrees with the fleet's. whash CANNOT do this: a device
//   with a broken kernel broadcasts a bad gradient, everyone averages the same
//   bad bytes, and every replica stays bit-identical. Replica agreement is not
//   kernel correctness, so the two hashes check genuinely different things.
// loss = the sender's local batch loss, for the true cluster-average display.
function packGrad(step, whash, phash, loss, grad) {
  const buf = new ArrayBuffer(16 + grad.byteLength);
  new Int32Array(buf, 0, 1)[0] = step;
  new Uint32Array(buf, 4, 2).set([whash, phash]);
  new Float32Array(buf, 12, 1)[0] = loss;
  new Float32Array(buf, 16).set(grad);
  return buf;
}
// ---- continuous kernel verification ----------------------------------------
// The boot gate only ever sees toy shapes at one moment. These two keep proving
// it while the run is live: a random-cell audit against the units at the REAL
// shapes, and the canonical probe re-run on a cadence so thermal/driver drift
// shows up mid-run rather than never.
const PROBE_EVERY = 25;                        // steps between probe refreshes
let probeHash = 0, auditFailure = null;
// Audit EVERY GEMM, not 2% of them. The old rate was chosen to bound overhead,
// but the overhead was never measured: an audit costs k multiply-adds per cell
// in JS, and at live shapes that is 2.8 us for 16 cells — auditing all ~10
// GEMMs of a step costs ~0.03 ms against a ~320 ms step, under 0.01%. The old
// 2%/6-cell setting sampled ~1.2 cells per step and bought nothing for it.
// 12 cells: the first 6 are the structural danger points (see auditTile's
// stratified sampling), the rest random.
const gemmAudit = {
  cells: 12,
  due: () => true,                            // measured at <0.01% of a step
  fail: (msg) => { if (!auditFailure) { auditFailure = msg; log(`KERNEL AUDIT FAILED: ${msg}`); } },
};
// Logged once so a probe mismatch is diagnosable: the probe folds the kernel
// hash and the JS-transcendental hash together, and this says which library
// this browser has. Peers whose math hash differs have a different JS engine
// math implementation, which forks weight INIT (Box-Muller) even when every
// kernel is perfect.
let mathHash = 0;
async function refreshProbe() {
  if (!mathHash) {
    mathHash = Compute.mathProbe();
    log(`JS math library hash: ${mathHash.toString(16)} (peers must match; ECMA-262 does not require exp/log/cos/sin to be correctly rounded)`);
  }
  // A failed probe must not leave the PREVIOUS hash in place: that hash would
  // go out with every gradient as a proof of kernels this run never re-checked.
  // 0 means "unproven", and the roster check halts on it (2026-10-14; the same
  // shape as an emulator call that reported success without doing the work).
  try { probeHash = await Compute.kernelProbe(compute, L); }
  catch (e) { probeHash = 0; log(`kernel probe failed to run: ${e.message} — this device's kernels are UNPROVEN`); }
  return probeHash;
}
function hashWeights() {                        // FNV-1a over the flat param bytes
  const f = Transformer.getFlatParams(model);
  const b = new Uint8Array(f.buffer, f.byteOffset, f.byteLength);
  let h = 0x811c9dc5;
  for (let i = 0; i < b.length; i++) { h ^= b[i]; h = Math.imul(h, 0x01000193); }
  return h >>> 0;
}
// ---- roster-gradient repair --------------------------------------------------
// Gradients travel peer-to-peer over a mesh, and the mesh can be asymmetric: a
// peer's gradient reaches the leader (so it enters the roster) but never
// reaches some follower — one flaky WebRTC link. The follower then can't
// average (partial average = forked weights) and used to hard-stop the whole
// run. But the LEADER holds every roster gradient by construction (the roster
// IS the set that reached it), so the follower asks the leader to forward the
// missing bytes instead of dying. Bit-exact: the same Float32Array bytes the
// origin peer broadcast. The leader retains a few recent steps to serve these
// (it steps at ~8s while a follower is stalled — its own cohort timeout — so a
// small window covers the repair round trip).
// wires: request  [i32 -7][i32 step][i32 peerNum]
//        response [i32 -8][i32 step][i32 peerNum][u32 whash][u32 phash][f32 loss][f32 grad...]
const REPAIRQ_SENTINEL = -7, REPAIRG_SENTINEL = -8;
const REPAIR_RETAIN = 8;                        // steps of grads kept for repair
function requestRepair(step, peerId2) {
  const dc = chans.get(leaderId);
  if (!dc || dc.readyState !== "open") return;
  const rq = new ArrayBuffer(12);
  new Int32Array(rq).set([REPAIRQ_SENTINEL, step, +peerId2.slice(1)]);
  return dcSend(dc, rq);
}
// roster wire: [int32 -4][int32 step][int32 n][int32 peerNums...]
const ROSTER_SENTINEL = -4;
function broadcastRoster(step, ids) {
  const nums = ids.map(id => +id.slice(1));     // "p12" -> 12
  const buf = new ArrayBuffer(12 + 4 * nums.length);
  const iv = new Int32Array(buf);
  iv[0] = ROSTER_SENTINEL; iv[1] = step; iv[2] = nums.length;
  iv.set(nums, 3);
  broadcast(buf);
}
const waiters = new Set();                       // pending waitForGrads checkers
function wake() { for (const w of waiters) w(); }
function onGrad(peerId, buf) {
  const step = new Int32Array(buf, 0, 1)[0];
  if (step === FRAG_SENTINEL) {                  // chunk of a large message
    const whole = onFragment(peerId, buf);
    if (whole) onGrad(peerId, whole);
    return;
  }
  if (step === CFG_SENTINEL) { onConfig(peerId, buf); return; }
  if (step === RESUME_SENTINEL) { onResume(peerId, buf); return; }
  if (step === CKPT_SENTINEL) {                  // a peer pushed a checkpoint
    if (training) { log(`ignored checkpoint from ${nmeOf(peerId)} (training in progress)`); return; }
    try { applyCheckpoint(parseCheckpoint(buf.slice(4)), nmeOf(peerId)); }
    catch (e) { log(`bad checkpoint from ${nmeOf(peerId)}: ${e.message}`); }
    return;
  }
  if (step === ROSTER_SENTINEL) {                // the leader's contributor set for a step
    if (training && peerId !== leaderId) return;  // only the run's leader may steer sync
    const iv = new Int32Array(buf);
    rosters.set(iv[1], [...iv.slice(3, 3 + iv[2])].map(n => "p" + n));
    wake(); return;
  }
  if (step === REPAIRQ_SENTINEL) {               // a peer asks for a grad it never got
    const [, rstep, num] = new Int32Array(buf, 0, 3);
    const pid = "p" + num;
    const g = (incoming.get(rstep) || new Map()).get(pid);
    const meta = (peerHashes.get(rstep) || new Map()).get(pid);
    if (!g || !meta) return;                     // already evicted — requester halts as before
    const out = new ArrayBuffer(24 + g.byteLength);
    new Int32Array(out, 0, 3).set([REPAIRG_SENTINEL, rstep, num]);
    new Uint32Array(out, 12, 2).set([meta.hash, meta.phash]);
    new Float32Array(out, 20, 1)[0] = meta.loss;
    new Float32Array(out, 24).set(g);
    const dc = chans.get(peerId);
    if (dc && dc.readyState === "open") dcSend(dc, out);
    return;
  }
  if (step === REPAIRG_SENTINEL) {               // leader-forwarded copy of a missing grad
    if (peerId !== leaderId) return;             // only the leader may vouch for another peer's grad
    const iv = new Int32Array(buf, 0, 3);
    const rstep = iv[1], pid = "p" + iv[2];
    const [rwhash, rphash] = new Uint32Array(buf, 12, 2);
    const rloss = new Float32Array(buf, 20, 1)[0];
    if (!incoming.has(rstep)) incoming.set(rstep, new Map());
    if (!incoming.get(rstep).has(pid)) incoming.get(rstep).set(pid, new Float32Array(buf.slice(24)));
    if (!peerHashes.has(rstep)) peerHashes.set(rstep, new Map());
    if (!peerHashes.get(rstep).has(pid)) peerHashes.get(rstep).set(pid, { hash: rwhash, phash: rphash, loss: rloss });
    wake(); return;
  }
  const [whash, phash] = new Uint32Array(buf, 4, 2);
  const loss = new Float32Array(buf, 12, 1)[0];
  const grad = new Float32Array(buf.slice(16));
  if (!incoming.has(step)) incoming.set(step, new Map());
  incoming.get(step).set(peerId, grad);
  if (!peerHashes.has(step)) peerHashes.set(step, new Map());
  peerHashes.get(step).set(peerId, { hash: whash, phash, loss });
  wake();                                        // resolve waits immediately (no polling)
}
function broadcastGrad(step, whash, phash, loss, grad) { return broadcast(packGrad(step, whash, phash, loss, grad)); }
// Event-driven: re-checked on every gradient arrival and peer departure, plus a
// coarse fallback timer (background tabs throttle timers to ~1s, so the old
// 15ms poll was the bottleneck there). Peers that left are dropped from the
// wait — a device dying no longer costs the full timeout every step.
function waitFor(pred, timeoutMs) {              // resolves true if pred held, false on timeout
  return new Promise((resolve) => {
    const t0 = Date.now();
    let timer = null;
    const check = () => {
      const ok = pred();
      if (ok || Date.now() - t0 > timeoutMs) {
        waiters.delete(check); clearInterval(timer);
        resolve(!!ok);
      }
    };
    waiters.add(check);
    timer = setInterval(check, 500);                     // safety net only
    check();
  });
}
async function waitForGradIds(step, cohort, timeoutMs = 8000) {
  await waitFor(() => {
    const live = cohort.filter(id => chans.has(id));     // prune departed peers
    const got = incoming.get(step) || new Map();
    return live.every(id => got.has(id));
  }, timeoutMs);
  const got = incoming.get(step) || new Map();
  return cohort.filter(id => chans.has(id) && got.has(id));
}

// ---- compute: one async training step THROUGH the verified units -----------
async function localStep() {
  // transformer forward runs through the verified INT8 multiply; STE backward
  return Transformer.trainStep(model);
}

// ---- the training loop -----------------------------------------------------
async function train(cfg, resume) {
  if (training) return; training = true; ui.start.disabled = true;
  cfgSliders().forEach(el => el.disabled = true);
  // resume runs keep `incoming` — grads that arrived while the bundle was
  // downloading are exactly the steps we need to catch up with
  if (!resume) { incoming.clear(); rosters.clear(); peerHashes.clear(); }
  let iLead = !resume && leaderId === null;            // set by Start (null) or onConfig/onResume
  let lastRoster = null;                               // the promotion electorate: last applied roster
  let lastLocal = null;                                // this step's own grad (reused on failover redo)
  const deadLeaders = new Set();                       // leaders that left this run
  const cohort = [...chans.keys()];                    // grows when a synced-in peer starts contributing
  const steps = cfg.steps;
  const opt = TrainCore.makeAdam(model.nParams, { lr: cfg.lr });
  if (resume) opt.setState(resume.adam);               // bit-identical moments from the leader
  pendingJoins.clear();
  auditFailure = null;
  await refreshProbe();                                // fresh kernel proof for this run
  log(`training started — cohort ${cohort.length} peer(s), world ${cohort.length + 1}, ` +
      `width=${cfg.c} seq=${cfg.t} batch=${cfg.b}×${cohort.length + 1}, optimizer ${opt.name}` +
      (cohort.length ? `, sync ${iLead ? "led by me" : "led by " + nmeOf(leaderId)}` : "") +
      ` · data: ${Transformer.datasetName()}` +
      ` · grad payload ${(model.nParams * 4 / 1048576).toFixed(1)} MB/step/peer`);
  if ((cfg.c > 64 || cfg.t > 64) && cohort.length + 1 < 4)
    log(`⚠ large model (width ${cfg.c}, seq ${cfg.t}) with only ${cohort.length + 1} device(s) — ` +
        `steps will be slow and the effective batch small. 4+ devices recommended at this size.`);
  let halted = null;                                   // sync-guard stop reason
  const contrib = new Map([[myId, 0]]);                // peerId -> grads contributed
  for (const id of cohort) contrib.set(id, 0);
  try {
  for (let s = resume ? resume.startStep : 0; s < steps; s++) {
    // leader: ship the current state to any device that connected mid-run —
    // snapshot here (top of step, post-update) so their replay starts exact
    if (iLead && pendingJoins.size) {
      for (const pid of [...pendingJoins]) { pendingJoins.delete(pid); sendResume(pid, s, opt, cfg); }
    }
    const whash = hashWeights();                       // pre-step fingerprint, sent with the grad
    // catching up after a mid-run join: if the leader's roster for this step
    // already exists and doesn't include me, skip the local compute — just
    // apply the roster updates and race forward to the live step
    const known = rosters.get(s);
    const skipMine = !iLead && known && !known.includes(myId);
    let loss = 0, grad = null;
    if (!skipMine) {
      // compute each step's gradient EXACTLY once: a leader-failover redo
      // must reuse the grad already broadcast — recomputing would give peers
      // a different gradient than the one they hold, forking the weights
      if (lastLocal && lastLocal.s === s) {
        ({ loss, grad } = lastLocal);
      } else {
        // re-prove this device's kernel against the units on a cadence, so
        // drift that appears mid-run (thermal, driver, shape) is caught while
        // it runs rather than only at boot on toy inputs
        if (s % PROBE_EVERY === 0) await refreshProbe();
        ({ loss, grad } = await localStep());
        if (auditFailure) { halted = auditFailure; break; }
        lastLocal = { s, loss, grad };
        await broadcastGrad(s, whash, probeHash, loss, grad);
        // keep my own grad addressable too, so repair requests for MY grad
        // (a peer that never received it) can be served like any other
        if (!incoming.has(s)) incoming.set(s, new Map());
        incoming.get(s).set(myId, grad);
        if (!peerHashes.has(s)) peerHashes.set(s, new Map());
        peerHashes.get(s).set(myId, { hash: whash, phash: probeHash, loss });
      }
    }
    let all;
    if (!cohort.length && iLead) {
      all = [grad];                                    // solo run (joiners can still sync in)
    } else if (iLead) {
      const ids = await waitForGradIds(s, cohort);
      // synced-in peers join the cohort the first step their grad shows up
      const extra = [...((incoming.get(s) || new Map()).keys())]
        .filter(id => !cohort.includes(id) && !ids.includes(id) && chans.has(id));
      for (const e of extra) { cohort.push(e); contrib.set(e, contrib.get(e) || 0); log(`${nmeOf(e)} joined the training cohort at step ${s + 1}`); }
      // sync guard: publish the exact contributor set so every device
      // averages the same gradients (or stops) — never a silent fork
      const roster = [myId, ...ids, ...extra];
      broadcastRoster(s, roster);
      rosters.set(s, roster);
      // CRITICAL: average strictly in roster order. Float addition is
      // commutative but NOT associative — if any device sums in a different
      // order (e.g. self-first), 3+ devices each get a microscopically
      // different average and the weights fork. Roster order is the canon.
      all = roster.map(id => id === myId ? grad : incoming.get(s).get(id));
    } else {
      // follower: apply the leader's roster verbatim, or stop
      await waitFor(() => rosters.has(s) || !chans.has(leaderId), 15000);
      if (!rosters.has(s)) {
        if (!chans.has(leaderId)) {
          // leader failover: everyone promotes the same next peer — the
          // lowest-numbered member of the LAST APPLIED roster (identical on
          // every device) that hasn't already died as leader. All followers
          // hold bit-identical weights + Adam state, so the new leader
          // continues the run seamlessly; no state transfer needed.
          deadLeaders.add(leaderId);
          const electorate = (lastRoster || [myId, leaderId]);
          const next = electorate.filter(id => !deadLeaders.has(id))
                                 .sort((a, b) => +a.slice(1) - +b.slice(1))[0];
          if (!next) { halted = "the sync leader left and no one remains to promote"; break; }
          if (next === myId) {
            iLead = true; leaderId = null;
            cohort.length = 0; cohort.push(...chans.keys());
            log(`the leader left — I take over sync leadership at step ${s + 1} (cohort ${cohort.length})`);
          } else {
            leaderId = next;
            log(`the leader left — ${nmeOf(next)} takes over sync from step ${s + 1}`);
          }
          s--; continue;                       // redo this step under the new leader
        }
        halted = `no roster from ${nmeOf(leaderId)} for step ${s + 1}`;
        break;
      }
      const roster = rosters.get(s);
      const need = roster.filter(id => id !== myId);
      const haveAll = () => need.every(id => (incoming.get(s) || new Map()).has(id));
      let ok = await waitFor(haveAll, 2500);
      if (!ok) {
        // mesh asymmetry: the grad reached the leader (it's in the roster) but
        // not me — ask the leader to forward it rather than stopping the run.
        // If the leader itself has left, still wait the full window: the
        // missing grads may be in flight from their origin peers directly.
        const missing = need.filter(id => !(incoming.get(s) || new Map()).has(id));
        if (chans.has(leaderId)) {
          log(`step ${s + 1}: still missing ${missing.map(nmeOf).join(", ")} — requesting repair from ${nmeOf(leaderId)}`);
          for (const id of missing) await requestRepair(s, id);
        }
        ok = await waitFor(haveAll, 12000);
        if (ok) log(`step ${s + 1}: gradient(s) recovered — continuing`);
      }
      if (!ok) {
        halted = `missing a roster gradient at step ${s + 1} — applying a partial average would fork the weights`;
        break;
      }
      // same canonical order as the leader (self's grad in its roster slot);
      // if the leader dropped my late grad from the roster, I skip it too
      all = roster.map(id => id === myId ? grad : incoming.get(s).get(id)).filter(Boolean);
    }
    // divergence check: every contributor's pre-step weight hash must match mine
    const hs = peerHashes.get(s);
    const roster = rosters.get(s) || [myId];
    lastRoster = roster;                               // promotion electorate if the leader dies
    if (hs) {
      const bad = roster.find(id => id !== myId && hs.has(id) && hs.get(id).hash !== whash);
      if (bad) { halted = `weights diverged from ${nmeOf(bad)} (detected at step ${s + 1})`; break; }
      // kernel disagreement: same seeded probe, same exact integer arithmetic
      // expected on every backend. This is the check the weight hash can't make.
      // phash 0 = that device's probe did not run. It used to make this
      // comparison skip the peer (`phash && ...`), so a device whose kernels
      // could not be proven trained on unchecked; now it stops the run.
      const unproven = roster.find(id => hs.has(id) && !hs.get(id).phash);
      if (unproven) {
        halted = `${unproven === myId ? "this device's" : nmeOf(unproven) + "'s"} kernel probe failed to run ` +
                 `at step ${s + 1} — its arithmetic is unproven, so its gradients cannot be trusted`;
        break;
      }
      const badK = roster.find(id => id !== myId && hs.has(id) && hs.get(id).phash !== probeHash);
      if (badK) {
        halted = `kernel disagreement with ${nmeOf(badK)} at step ${s + 1} — their verified-unit probe ` +
                 `hashes ${hs.get(badK).phash} vs mine ${probeHash}. One of us is computing different ` +
                 `arithmetic; the shared weights would look fine either way, so stopping.`;
        break;
      }
    }
    // cluster-average loss over this step's actual contributors
    let lossSum = 0, lossN = 0;
    for (const id of roster) {
      if (id === myId) { lossSum += loss; lossN++; contrib.set(myId, (contrib.get(myId) || 0) + 1); }
      else if (hs && hs.has(id)) { lossSum += hs.get(id).loss; lossN++; contrib.set(id, (contrib.get(id) || 0) + 1); }
    }
    const clusterLoss = lossSum / Math.max(1, lossN);
    let avg;
    try { avg = TrainCore.averageGrads(all); }
    catch (e) {
      // every device averages the same roster, so every device stops HERE, at the same step,
      // with weights and Adam state untouched (still a valid checkpoint to resume from)
      if (e.name !== "GradientError") throw e;
      halted = `step ${s + 1}: ${e.message} -- refusing to apply it (weights and Adam state unchanged)`;
      break;
    }
    const upd = opt.step(avg);                      // DaisyAdam on the cluster-avg grad
    Transformer.applyUpdate(model, upd);            // W -= upd (lr folded into upd)
    // retain a few applied steps so stalled peers can repair (evict beyond that)
    incoming.delete(s - REPAIR_RETAIN); rosters.delete(s - REPAIR_RETAIN); peerHashes.delete(s - REPAIR_RETAIN);
    trainedSteps++;
    if (s % 10 === 0 || s === steps - 1) {
      ui.loss.textContent = clusterLoss.toFixed(5);
      ui.step.textContent = `${s + 1} / ${steps}`;
      ui.bar.style.width = `${Math.round(100 * (s + 1) / steps)}%`;
      await new Promise(r => setTimeout(r, 0));         // yield to UI
    }
    if (cohort.length && (s + 1) % 50 === 0)          // who is contributing what
      log(`step ${s + 1}: contributions — ` +
          [...contrib.entries()].map(([id, n]) => `${id === myId ? "me" : nmeOf(id)} ${n}/${s + 1}`).join(", "));
  }
  } catch (e) {                                       // a device that errors REPORTS it
    halted = `error during training: ${e.message}`;   // (instead of freezing at "—")
    console.error(e);
  }
  if (cohort.length)
    log(`contribution totals — ` +
        [...contrib.entries()].map(([id, n]) => `${id === myId ? "me" : nmeOf(id)} ${n} grad(s)`).join(", "));
  if (halted) {
    log(`SYNC GUARD: stopped — ${halted}. The model here is intact (${trainedSteps} steps); ` +
        `to re-sync the group, load/push a checkpoint and start again.`);
    ui.diff.textContent = "stopped by the sync guard — see log";
  } else {
    log(`training done — final loss ${ui.loss.textContent}`);
    try {
      const sample = await Transformer.generate(model, "the ", 70);
      ui.diff.textContent = `the model speaks: “${sample.trim()}”`;
    } catch (e) {
      ui.diff.textContent = "done — trained through the verified units; all peers share one model.";
    }
  }
  incoming.clear(); rosters.clear(); peerHashes.clear();
  leaderId = null;
  training = false;
  modelReady();
  ui.start.disabled = false;
  cfgSliders().forEach(el => el.disabled = false);
}
function modelReady() {                 // trained weights exist: enable save/kit/generate
  ui.save.disabled = false;
  ui.kit.disabled = false;
  ui.genBtn.disabled = false;
}
function cfgSliders() { return [ui.cfgC, ui.cfgT, ui.cfgB, ui.cfgSteps, ui.cfgLr]; }

// (re)build the mini transformer — deterministic shared init (same seeds on
// every peer); each device samples its own batch windows from the shared corpus
function buildModel(cfg) {
  model = Transformer.init(cfg, L, compute, gemmAudit);  // compute.bgemm = fused batched GPU path
  trainedSteps = 0;
}

// ---- boot ------------------------------------------------------------------
(async function () {
  // Room-first deployments (the HF Space sets DAISY_FORCE_ROOMS): there is no
  // LAN auto-grouping between strangers — visitors without a room code choose
  // to create their own room (they become its host) or join one by code.
  try {
    const mode = await (await fetch("mode")).json();
    if (mode.rtc) rtcConfig = mode.rtc;                  // operator-provided ICE config
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
  ui.backend.textContent = `${compute.backend.toUpperCase()} — ${compute.label} · block-scaled INT8, batched + fused epilogue, through verified units`;
  // canonical kernel probe: a fixed seeded GEMM through this device's live
  // kernel. Every device that agrees with the units produces the same number,
  // GPU or CPU — so it is comparable across the fleet, and re-run mid-training.
  await refreshProbe();
  log(`kernel probe: ${probeHash} (exact int32 through the units — every honest device gets this same number)`);
  // tokenizer must load BEFORE the model is built (it sets the vocab size)
  try {
    const name = await Transformer.loadTokenizer();
    ui.tokenizer.textContent = name;
    log(`tokenizer: ${name}`);
  } catch (e) {
    ui.tokenizer.textContent = Transformer.tokenizerName();
    log(`tokenizer.json unavailable (${e.message}) — using the ${Transformer.tokenizerName()} vocab`);
  }
  if (compute.backend === "cpu" && Transformer.vocabSize() > 1000)
    log(`⚠ CPU backend with a ${Transformer.vocabSize()}-token vocab — steps will take seconds; ` +
        `a WebGPU-capable browser will be much faster`);
  // slider readouts
  for (const [el, v] of [[ui.cfgC, "vcfgC"], [ui.cfgT, "vcfgT"], [ui.cfgB, "vcfgB"],
                         [ui.cfgSteps, "vcfgSteps"], [ui.cfgLr, "vcfgLr"]])
    el.oninput = () => document.getElementById(v).textContent = el.value;
  buildModel(readCfgFromUI());
  ui.start.disabled = false;                             // solo training works too
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
  // phones suspend the page (screen off, app switch) — reconnect the moment we
  // come back, and on network changes (wifi <-> cellular), PairDrop-style
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && (!ws || ws.readyState > 1)) { log("page woke — reconnecting"); connectSignaling(); }
  });
  if (navigator.connection && navigator.connection.addEventListener)
    navigator.connection.addEventListener("change", () => {
      if (!ws || ws.readyState > 1) { log("network changed — reconnecting"); connectSignaling(); }
    });
  // dataset: FineWeb-Edu (hardcoded), streamed from this Space's /data endpoint
  loadDataset("HuggingFaceFW/fineweb-edu");
  ui.start.onclick = async () => {
    if (training) return;
    const cfg = readCfgFromUI();
    ui.start.disabled = true;
    if (cfg.ds) await loadDataset(cfg.ds);               // my pick becomes the group's dataset
    leaderId = null;                                     // I pressed Start: I lead sync
    buildModel(cfg);
    broadcastConfig(cfg);                                // everyone follows these settings
    train(cfg);
  };
  ui.save.onclick = saveCheckpoint;
  ui.kit.onclick = downloadInferenceKit;
  ui.genBtn.onclick = async () => {
    if (training) { log("can't generate mid-training"); return; }
    ui.genBtn.disabled = true;
    ui.genOut.textContent = "generating…";
    try { ui.genOut.textContent = await Transformer.generate(model, ui.genPrompt.value || "the ", 150); }
    catch (e) { ui.genOut.textContent = `error: ${e.message}`; }
    ui.genBtn.disabled = false;
  };
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
