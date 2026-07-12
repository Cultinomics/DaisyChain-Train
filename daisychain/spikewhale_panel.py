"""SpikeWhale training control panel (CLI: python -m daisychain.spikewhale_panel).

A web page with sliders for the SpikeWhale config. Pick a size your hardware can
handle, hit Start, and it launches the real DaisyChain training (SpikeWhale +
FineWeb-Edu) and streams the live loss. The exact env is shown so you can run the
same command on other machines to train distributed.
"""
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("SW_PANEL_PORT", "8899"))
_proc = None
_log = []          # rolling training log
_lock = threading.Lock()


def _pump(proc):
    for line in iter(proc.stdout.readline, ""):
        with _lock:
            _log.append(line.rstrip("\n"))
            if len(_log) > 400:
                del _log[:200]
    proc.stdout.close()


def start_training(cfg):
    global _proc
    if _proc and _proc.poll() is None:
        return False, "already running"
    env = dict(os.environ)
    env.update({
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29610", "WORLD_SIZE": "1",
        "RANK": "0", "USE_LIBUV": "0", "PYTHONUNBUFFERED": "1",
        "DAISY_TASK": "daisychain.spikewhale_task:SpikeWhaleTask",
        "DAISY_SW_HIDDEN": str(cfg["hidden"]), "DAISY_SW_LAYERS": str(cfg["layers"]),
        "DAISY_SW_HEADS": str(cfg["heads"]), "DAISY_SW_EXPERTS": str(cfg["experts"]),
        "DAISY_SW_SEQLEN": str(cfg["seqlen"]),
        "DAISY_STEPS": str(cfg["steps"]), "DAISY_LR": str(cfg["lr"]),
        "DAISY_OPTIMIZER": "adam", "DAISY_BASE_BATCH": str(cfg["batch"]),
    })
    if cfg.get("dataset"):
        env["DAISY_SW_DATASET"] = cfg["dataset"]
        # blank subset means the dataset's default config; unset any inherited one
        if cfg.get("subset"):
            env["DAISY_SW_SUBSET"] = cfg["subset"]
        else:
            env.pop("DAISY_SW_SUBSET", None)
    with _lock:
        _log.clear()
        _log.append(f"launching training: hidden={cfg['hidden']} layers={cfg['layers']} "
                    f"experts={cfg['experts']} seqlen={cfg['seqlen']} steps={cfg['steps']} "
                    f"dataset={env.get('DAISY_SW_DATASET', 'HuggingFaceFW/fineweb-edu')}"
                    + (f":{env['DAISY_SW_SUBSET']}" if env.get("DAISY_SW_SUBSET") else ""))
    _proc = subprocess.Popen([sys.executable, "-u", "-m", "daisychain.train"],
                             env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
    threading.Thread(target=_pump, args=(_proc,), daemon=True).start()
    return True, "started"


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SpikeWhale · DaisyChain trainer</title>
<style>
 body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:720px;margin:0 auto;
      padding:22px;background:#efe4c9;color:#2a1d0a}
 @media(prefers-color-scheme:dark){body{background:#14100a;color:#ede1c3}}
 h1{margin:0 0 2px}.sub{color:#6b4423;margin:0 0 16px}
 @media(prefers-color-scheme:dark){.sub{color:#c9b072}}
 .card{background:#fbf6e8;border:1px solid rgba(139,111,71,.3);border-radius:10px;padding:16px;margin:12px 0}
 @media(prefers-color-scheme:dark){.card{background:#1f1a12;border-color:rgba(201,176,114,.35)}}
 label{display:flex;justify-content:space-between;font-size:.92rem;margin:10px 0 4px;font-weight:600}
 input[type=range]{width:100%;accent-color:#4a7c2e}
 input[type=text]{width:100%;padding:8px 10px;border-radius:8px;border:1px solid rgba(139,111,71,.4);
        background:rgba(255,255,255,.5);color:inherit;font-family:'Courier New',monospace;font-size:.9rem}
 @media(prefers-color-scheme:dark){input[type=text]{background:rgba(0,0,0,.25);border-color:rgba(201,176,114,.35)}}
 .val{font-family:'Courier New',monospace;color:#4a7c2e;font-weight:700}
 @media(prefers-color-scheme:dark){.val{color:#9bc466}}
 button{background:linear-gradient(135deg,#4a7c2e,#2d5016);color:#f5ecd9;border:0;border-radius:8px;
        padding:12px 26px;font-weight:800;font-size:1rem;cursor:pointer}
 .num{font-family:'Courier New',monospace;font-size:2rem;font-weight:700;text-align:center;
      color:#f5ecd9;background:linear-gradient(135deg,#2d5016,#1f3a0f);border-radius:10px;padding:14px}
 pre{background:rgba(0,0,0,.06);border-radius:8px;padding:10px;max-height:220px;overflow:auto;
     font-size:.78rem;white-space:pre-wrap;font-family:'Courier New',monospace}
 @media(prefers-color-scheme:dark){pre{background:rgba(0,0,0,.3)}}
 .lbl{font-size:11px;font-weight:800;letter-spacing:1.5px;text-transform:uppercase;color:#6b4423;margin-bottom:8px}
 @media(prefers-color-scheme:dark){.lbl{color:#c9b072}}
</style></head><body>
<h1>🐋 SpikeWhale · DaisyChain</h1>
<p class="sub">Pick a size your hardware can train, then start. Trains the real SpikeWhale on streamed FineWeb-Edu, distributed by DaisyChain. Smaller = faster on old hardware.</p>
<div id="settings">
<div class="card"><div class="lbl">Model size</div>
 <label>Hidden size <span class="val" id="vhidden">256</span></label><input type="range" id="hidden" min="64" max="768" step="64" value="256">
 <label>Layers <span class="val" id="vlayers">4</span></label><input type="range" id="layers" min="1" max="12" step="1" value="4">
 <label>Attention heads <span class="val" id="vheads">4</span></label><input type="range" id="heads" min="1" max="8" step="1" value="4">
 <label>MoE experts <span class="val" id="vexperts">4</span></label><input type="range" id="experts" min="1" max="8" step="1" value="4">
 <label>Sequence length <span class="val" id="vseqlen">128</span></label><input type="range" id="seqlen" min="32" max="512" step="32" value="128">
</div>
<div class="card"><div class="lbl">Training</div>
 <label>Learning rate ×1e-4 <span class="val" id="vlr">30</span></label><input type="range" id="lr" min="1" max="100" step="1" value="30">
 <label>Batch (per step) <span class="val" id="vbatch">4</span></label><input type="range" id="batch" min="1" max="16" step="1" value="4">
 <label>Steps <span class="val" id="vsteps">200</span></label><input type="range" id="steps" min="20" max="2000" step="20" value="200">
</div>
<div class="card"><div class="lbl">Data</div>
 <label for="dataset">HuggingFace dataset</label>
 <input type="text" id="dataset" value="HuggingFaceFW/fineweb-edu" spellcheck="false">
 <label for="subset">Config / subset <span style="font-weight:400">(blank = default)</span></label>
 <input type="text" id="subset" value="sample-10BT" spellcheck="false">
 <p class="sub" style="margin:.6rem 0 0;font-size:.82rem">Any streamable text dataset with a <code>text</code> column works. For gated/private datasets, log in first with <code>huggingface-cli login</code> on this machine — the trainer inherits your token.</p>
</div>
</div>
<div class="card" style="text-align:center">
 <button id="startbtn" onclick="start()">Start training</button>
 <button id="backbtn" onclick="goBack()" style="display:none;background:linear-gradient(135deg,#6b4423,#4a2f18)">&#8592; Back to settings</button>
 <p class="sub" style="margin:.6rem 0 0" id="status">idle</p></div>
<div class="card"><div class="lbl">Live loss</div><div class="num" id="loss">—</div></div>
<div class="card"><div class="lbl">Log</div><pre id="log"></pre></div>
<script>
const ids=["hidden","layers","heads","experts","seqlen","lr","batch","steps"];
ids.forEach(k=>{const el=document.getElementById(k);const v=document.getElementById("v"+k);
  el.oninput=()=>v.textContent=el.value;});
function cfg(){const c={};ids.forEach(k=>c[k]=+document.getElementById(k).value);c.lr=c.lr/1e4;
  c.dataset=document.getElementById("dataset").value.trim();
  c.subset=document.getElementById("subset").value.trim();return c;}
function showSettings(on){document.getElementById("settings").style.display=on?"":"none";
  document.getElementById("startbtn").style.display=on?"":"none";
  document.getElementById("backbtn").style.display=on?"none":"";}
async function start(){document.getElementById("status").textContent="starting…";
  showSettings(false);
  await fetch("/start",{method:"POST",body:JSON.stringify(cfg())});}
async function goBack(){await fetch("/stop",{method:"POST"});
  showSettings(true);document.getElementById("status").textContent="stopped — adjust and start again";}
async function poll(){try{const r=await fetch("/log");const d=await r.json();
  document.getElementById("log").textContent=d.log.slice().reverse().join("\\n");
  let last="—";for(const l of d.log){const m=l.match(/cluster-avg loss ([0-9.]+)/);if(m)last=m[1];}
  document.getElementById("loss").textContent=last;
  if(document.getElementById("backbtn").style.display!=="none")
    document.getElementById("status").textContent=d.running?"training…":"idle / done";}catch(e){}}
setInterval(poll,1000);poll();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def _send(self, body, ctype="text/html; charset=utf-8", code=200):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/log"):
            with _lock:
                running = _proc is not None and _proc.poll() is None
                self._send(json.dumps({"log": list(_log), "running": running}), "application/json")
        else:
            self._send(PAGE)

    def do_POST(self):
        if self.path.startswith("/start"):
            n = int(self.headers.get("Content-Length", 0))
            cfg = json.loads(self.rfile.read(n) or "{}")
            ok, msg = start_training(cfg)
            self._send(json.dumps({"ok": ok, "msg": msg}), "application/json")
        elif self.path.startswith("/stop"):
            global _proc
            if _proc and _proc.poll() is None:
                _proc.terminate()
                with _lock:
                    _log.append("training stopped from the panel")
            self._send(json.dumps({"ok": True}), "application/json")

    def log_message(self, *a):
        pass


def main():
    print(f"[spikewhale-panel] http://localhost:{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()


if __name__ == "__main__":
    main()
