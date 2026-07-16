// DaisyChain-Web signaling + static host.
// - Serves public/ (the page users open).
// - WebSocket signaling: introduces peers in a room and relays WebRTC
//   offers/answers/ICE. It never sees the compute — that's P2P over WebRTC.
// Only dependency: `ws`.  Run:  npm install && node server.js
const http = require("http");
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const { WebSocketServer } = require("ws");

// Snapdrop-style: peers are grouped by their PUBLIC IP, so only devices on the
// same network auto-discover each other. An explicit ?room=CODE overrides this
// to connect across networks (share the code with people you invite).
function clientIP(req) {
  const xff = req.headers["x-forwarded-for"];
  if (xff) return xff.split(",")[0].trim();
  return (req.socket.remoteAddress || "unknown").replace(/^::ffff:/, "");
}
function roomFor(req) {
  const u = new URL(req.url, "http://x");
  const code = u.searchParams.get("room");
  if (code) return "room:" + code;
  const h = crypto.createHash("sha256").update(clientIP(req)).digest("hex").slice(0, 10);
  return "net:" + h;               // same network -> same room (IP not exposed in the id)
}

const PORT = process.env.PORT || 8787;
const PUB = path.join(__dirname, "public");
const TYPES = { ".html": "text/html", ".js": "text/javascript",
                ".css": "text/css", ".json": "application/json" };

// ---- training text: FineWeb-Edu, served by THIS server ----------------------
// The browser used to call datasets-server.huggingface.co/rows directly, which
// 503s routinely under load. The parquet files themselves sit on the HF CDN
// behind plain resolve/ URLs — public, range-request friendly, and far more
// reliable than the rows API. So this endpoint reads a random slice of a
// random shard of the 10BT sample straight off the CDN with hyparquet (pure
// JS, no native deps) and hands the browser plain text. FineWeb-Edu is the
// ONLY dataset; there is no pass-through of client-chosen dataset ids.
const FWE_BASE = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/";
const FWE_SHARDS = Array.from({ length: 14 }, (_, i) => `${String(i).padStart(3, "0")}_00000.parquet`);
const FWE_ROWS_PER_SLICE = 130;          // ~550k chars, what one training run consumes
const fweCache = new Map();              // shard -> { file, metadata } (metadata is ~kBs)
let hyparquet = null;                    // ESM — loaded on first use

async function fweSlice() {
  hyparquet = hyparquet || await import("hyparquet");
  const shard = FWE_SHARDS[Math.floor(Math.random() * FWE_SHARDS.length)];
  let entry = fweCache.get(shard);
  if (!entry) {
    const file = await hyparquet.asyncBufferFromUrl({ url: FWE_BASE + shard });
    const metadata = await hyparquet.parquetMetadataAsync(file);
    entry = { file, metadata };
    fweCache.set(shard, entry);
  }
  const total = Number(entry.metadata.num_rows);
  const start = Math.floor(Math.random() * Math.max(1, total - FWE_ROWS_PER_SLICE));
  const rows = await hyparquet.parquetReadObjects({
    file: entry.file, metadata: entry.metadata, columns: ["text"],
    rowStart: start, rowEnd: start + FWE_ROWS_PER_SLICE,
  });
  return { shard, start, text: rows.map(r => r.text).join("\n") };
}

// keepalive: proxies (incl. HuggingFace's) close idle WebSockets — ping every
// 30s keeps the pipe warm and detects dead clients within a minute
setInterval(() => {
  for (const room of rooms.values())
    for (const [pid, v] of room.peers) {
      if (v.ws.isAlive === false) { v.ws.terminate(); continue; }
      v.ws.isAlive = false;
      send(v.ws, { type: "ping" });
    }
}, 30000);

const server = http.createServer((req, res) => {
  let p = decodeURIComponent(req.url.split("?")[0]);
  if (p === "/mode") {                 // deployment flags for the client
    res.writeHead(200, { "Content-Type": "application/json" });
    // DAISY_FORCE_ROOMS=1 (set on the HF Space): no LAN auto-grouping — every
    // visitor creates their own private room and invites devices by link.
    // DAISY_RTC_CONFIG: optional JSON RTCPeerConnection config (own TURN etc.,
    // PairDrop-style server-provided rtcConfig).
    let rtc = null;
    try { if (process.env.DAISY_RTC_CONFIG) rtc = JSON.parse(process.env.DAISY_RTC_CONFIG); } catch (e) {}
    return res.end(JSON.stringify({ forceRooms: !!process.env.DAISY_FORCE_ROOMS, rtc }));
  }
  if (p === "/data") {                 // a fresh random slice of FineWeb-Edu
    return fweSlice()
      .then(({ shard, start, text }) => {
        res.writeHead(200, { "Content-Type": "text/plain; charset=utf-8",
                             "X-FWE-Shard": shard, "X-FWE-Row": String(start) });
        res.end(text);
      })
      .catch((e) => {
        console.error("fineweb slice failed:", e.message);
        res.writeHead(503, { "Content-Type": "text/plain" });
        res.end("fineweb-edu unavailable: " + e.message);
      });
  }
  if (p === "/") p = "/index.html";
  const file = path.join(PUB, path.normalize(p));
  if (!file.startsWith(PUB)) { res.writeHead(403); return res.end(); }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); return res.end("not found"); }
    res.writeHead(200, { "Content-Type": TYPES[path.extname(file)] || "application/octet-stream" });
    res.end(data);
  });
});

const wss = new WebSocketServer({ server });
// roomId -> { peers: Map(id -> {ws,name}), host: id|null, pending: Map(id -> {ws,name}) }
// Private rooms (room:CODE) are gated: the creator is the host, and everyone
// arriving later waits until the host accepts them — knowing the code is not
// enough. Network rooms (net:) keep auto-join (same LAN, Snapdrop-style).
const rooms = new Map();
let nextId = 1;

function send(ws, obj) { if (ws.readyState === 1) ws.send(JSON.stringify(obj)); }

wss.on("connection", (ws, req) => {
  const roomId = roomFor(req);
  const name = (new URL(req.url, "http://x").searchParams.get("name") || "").slice(0, 40) || ("p" + nextId);
  const id = "p" + (nextId++);
  ws.peerId = id; ws.roomId = roomId;
  if (!rooms.has(roomId)) rooms.set(roomId, { peers: new Map(), host: null, pending: new Map() });
  const room = rooms.get(roomId);
  const isPrivate = roomId.startsWith("room:");

  function admit(pid, peer) {
    const roster = [...room.peers.entries()].map(([qid, v]) => ({ id: qid, name: v.name }));
    send(peer.ws, { type: "welcome", id: pid, room: roomId, peers: roster, host: room.host === pid });
    for (const [, v] of room.peers) send(v.ws, { type: "peer-joined", id: pid, name: peer.name });
    room.peers.set(pid, peer);
  }

  if (isPrivate && room.peers.size === 0) room.host = id;   // creator hosts
  if (isPrivate && room.host !== id) {
    room.pending.set(id, { ws, name });
    send(ws, { type: "waiting" });
    const h = room.peers.get(room.host);
    if (h) send(h.ws, { type: "join-request", id, name });
  } else {
    admit(id, { ws, name });
  }

  ws.on("message", (buf) => {
    let msg; try { msg = JSON.parse(buf); } catch { return; }
    if (msg.type === "signal" && msg.to && room.peers.has(id)) {   // relay WebRTC signaling
      const target = room.peers.get(msg.to);
      if (target) send(target.ws, { type: "signal", from: id, data: msg.data });
    } else if (msg.type === "pong") {                              // keepalive reply
      ws.isAlive = true;
    } else if (msg.type === "relay" && msg.to && room.peers.has(id)) {
      // PairDrop-style WS fallback: when two devices can't form a direct
      // WebRTC path (symmetric NATs, no TURN), the training payloads flow
      // through here instead of dropping the peer
      const target = room.peers.get(msg.to);
      if (target) send(target.ws, { type: "relay", from: id, data: msg.data });
    } else if (msg.type === "admit" && id === room.host) {         // host verdict on a joiner
      const p = room.pending.get(msg.id);
      if (!p) return;
      room.pending.delete(msg.id);
      if (msg.allow) admit(msg.id, p);
      else { send(p.ws, { type: "denied" }); p.ws.close(); }
    }
  });

  ws.on("close", () => {
    room.pending.delete(id);
    if (room.peers.delete(id))
      for (const [, v] of room.peers) send(v.ws, { type: "peer-left", id });
    if (room.host === id) {                                        // host left: promote oldest
      room.host = room.peers.keys().next().value ?? null;
      const h = room.peers.get(room.host);
      if (h) {
        send(h.ws, { type: "host" });
        for (const [pid, p] of room.pending) send(h.ws, { type: "join-request", id: pid, name: p.name });
      }
    }
    if (room.peers.size === 0 && room.pending.size === 0) rooms.delete(roomId);
  });
});

server.listen(PORT, () => console.log(`DaisyChain-Web on http://localhost:${PORT}`));
