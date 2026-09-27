"""DaisyChain dashboard (CLI: `daisychain-dashboard`). Tailwind-styled page with
a readiness banner, P2P connectivity scan, pooled resource + capacity plan, and
live training status. Config via DAISY_NODES_FILE. Serves on :8080."""
import json
from html import escape as _html_escape
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    from daisychain.dashboard.scanner import scan_cluster
except Exception:
    from scanner import scan_cluster  # running from the folder directly

NODES_FILE = os.environ.get("DAISY_NODES_FILE", "config/nodes.example.json")
PORT = int(os.environ.get("DAISY_DASH_PORT", "8080"))
EXPECTED_WORLD = os.environ.get("DAISY_EXPECTED_WORLD")
BASE_BATCH = int(os.environ.get("DAISY_BASE_BATCH", "32"))


def load_nodes():
    with open(NODES_FILE) as f:
        return json.load(f)


def _chip(ok, yes="OK", no="DOWN"):
    cls = ("bg-emerald-500/20 text-emerald-600 dark:text-emerald-400" if ok
           else "bg-rose-500/20 text-rose-600 dark:text-rose-400")
    return f'<span class="px-2 py-0.5 rounded text-xs font-semibold {cls}">{yes if ok else no}</span>'


#: Remote-supplied strings are escaped before they reach the page.
#:
#: `device` and the GPU `name` come from each node's /resources JSON -- i.e. from
#: the OTHER machine, not from local config -- and were interpolated raw into the
#: dashboard HTML. A node serving `{"gpu": {"name": "<script>...</script>"}}`
#: would run script in the operator's browser on the next 3-second refresh.
#: The node list itself is local config and trusted, but escaping it too costs
#: nothing and keeps the rule simple: nothing reaches the page unescaped.
def _e(v) -> str:
    return _html_escape("" if v is None else str(v))


def render(d):
    ready = d["ready"]
    banner = ("bg-emerald-500", "✓ CLUSTER READY — all nodes connected") if ready \
        else ("bg-rose-500", f"✗ NOT READY — {d['reachable']}/{d['total']} nodes reachable")
    rows = ""
    for n in d["nodes"]:
        lat = f'{n["latency_ms"]} ms' if n["latency_ms"] is not None else "—"
        res = n.get("resources") or {}
        dev = res.get("device", "—")
        gpu = (res.get("gpu") or {}).get("name", "")
        devlabel = _e(dev) + (f" ({_e(gpu)})" if gpu else "")
        rows += f"""<tr class="border-b border-slate-100 dark:border-slate-800">
          <td class="py-2 px-3 font-medium">{_e(n['name'])}</td>
          <td class="py-2 px-3 text-slate-500">{_e(n['host'])}</td>
          <td class="py-2 px-3">{_chip(n['reachable'],'reachable','unreachable')}</td>
          <td class="py-2 px-3">{devlabel}</td>
          <td class="py-2 px-3 tabular-nums">{lat}</td></tr>"""
    plan = d["plan"]
    pr = ""
    for p in plan["per_node"]:
        dv = _e(p["device"]) + (f" ({_e(p['gpu'])})" if p.get("gpu") else "")
        ram = f'{p["ram_gb"]} GB' if p.get("ram_gb") is not None else "—"
        pr += f"""<tr class="border-b border-slate-100 dark:border-slate-800">
          <td class="py-2 px-3 font-medium">{_e(p['name'])}</td>
          <td class="py-2 px-3">{dv}</td>
          <td class="py-2 px-3 tabular-nums">{ram}</td>
          <td class="py-2 px-3 tabular-nums">{p['capacity']}</td>
          <td class="py-2 px-3 tabular-nums">{p['weight']}</td>
          <td class="py-2 px-3 tabular-nums">{p['batch']}</td></tr>"""
    tot_ram = f'{plan["total_ram_gb"]} GB' if plan["total_ram_gb"] is not None else "—"
    t = d["training"]
    if t:
        step, total = t.get("step", 0), t.get("total_steps", 1)
        pct = int(100 * (step + 1) / max(1, total)); loss = t.get("cluster_avg_loss")
        badge = _chip(True, "DONE") if t.get("done") else '<span class="text-xs text-sky-500 animate-pulse">training…</span>'
        train = f"""<div class="flex items-center justify-between mb-2">
          <span class="text-sm text-slate-500">step {step} / {total}</span>{badge}</div>
        <div class="w-full h-3 rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden">
          <div class="h-3 bg-sky-500" style="width:{pct}%"></div></div>
        <p class="mt-3 text-2xl font-semibold tabular-nums">{loss:.5f}
          <span class="text-sm font-normal text-slate-500">cluster-avg loss</span></p>"""
    else:
        train = '<p class="text-slate-400 text-sm">No active run detected (waiting for rank 0 status)…</p>'
    return f"""<!doctype html><html><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta http-equiv="refresh" content="3"><title>DaisyChain — Cluster</title>
    <script src="https://cdn.tailwindcss.com"></script></head>
    <body class="bg-slate-50 dark:bg-slate-900 text-slate-800 dark:text-slate-100 min-h-screen">
    <div class="max-w-5xl mx-auto p-6 space-y-6">
      <header class="flex items-center justify-between">
        <div><h1 class="text-2xl font-bold">\U0001f33c DaisyChain</h1>
        <p class="text-sm text-slate-500">Old Hardware Training Pipeline · scanned {d['scanned_at']}</p></div>
        <span class="text-xs text-slate-400">auto-refresh 3s</span></header>
      <div class="{banner[0]} text-white rounded-xl px-5 py-4 font-semibold text-lg shadow">{banner[1]}</div>
      <div class="grid md:grid-cols-3 gap-4">
        <div class="rounded-xl border border-slate-200 dark:border-slate-700 p-4"><p class="text-xs uppercase tracking-wide text-slate-400">Nodes</p><p class="text-3xl font-bold tabular-nums">{d['reachable']}/{d['total']}</p></div>
        <div class="rounded-xl border border-slate-200 dark:border-slate-700 p-4"><p class="text-xs uppercase tracking-wide text-slate-400">Total cores</p><p class="text-3xl font-bold tabular-nums">{plan['total_cores']}</p></div>
        <div class="rounded-xl border border-slate-200 dark:border-slate-700 p-4"><p class="text-xs uppercase tracking-wide text-slate-400">Total RAM</p><p class="text-3xl font-bold tabular-nums">{tot_ram}</p></div></div>
      <section class="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
        <h2 class="px-4 py-3 font-semibold border-b border-slate-100 dark:border-slate-800">Connectivity scan</h2>
        <table class="w-full text-sm"><thead class="text-left text-slate-400"><tr>
          <th class="py-2 px-3">Node</th><th class="py-2 px-3">Host</th><th class="py-2 px-3">Agent</th>
          <th class="py-2 px-3">Device</th><th class="py-2 px-3">Latency</th></tr></thead><tbody>{rows}</tbody></table></section>
      <div class="grid md:grid-cols-2 gap-6">
        <section class="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
          <h2 class="px-4 py-3 font-semibold border-b border-slate-100 dark:border-slate-800">Capacity plan · global batch {plan['global_batch']}</h2>
          <table class="w-full text-sm"><thead class="text-left text-slate-400"><tr>
            <th class="py-2 px-3">Node</th><th class="py-2 px-3">Device</th><th class="py-2 px-3">RAM</th>
            <th class="py-2 px-3">Capacity</th><th class="py-2 px-3">Weight</th><th class="py-2 px-3">Batch</th></tr></thead><tbody>{pr}</tbody></table></section>
        <section class="rounded-xl border border-slate-200 dark:border-slate-700 p-4">
          <h2 class="font-semibold mb-3">Live training</h2>{train}</section></div>
      <footer class="text-center text-xs text-slate-400 pt-2">DaisyChainAI · pools compute, not memory · small models on spare hardware</footer>
    </div></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        data = scan_cluster(load_nodes(),
                            int(EXPECTED_WORLD) if EXPECTED_WORLD else None, BASE_BATCH)
        if self.path.startswith("/api"):
            body = json.dumps(data).encode(); ctype = "application/json"
        else:
            body = render(data).encode(); ctype = "text/html; charset=utf-8"
        self.send_response(200); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def main():
    print(f"[dashboard] :{PORT} nodes={NODES_FILE}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
