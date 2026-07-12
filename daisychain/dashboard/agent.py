"""Per-node agent (CLI: `daisychain-agent`). Serves health/resources/status so
the dashboard can scan this machine. Pure stdlib."""
import json
import os
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    from daisychain.cluster import survey_node
except Exception:
    def survey_node(cpu_fraction=0.9):
        cores = int(os.environ.get("DAISY_CORES", os.cpu_count() or 1))
        return {"host": socket.gethostname(), "cores": cores,
                "threads": max(1, int(cores * cpu_fraction)),
                "ram_gb": None, "device": "cpu", "gpu": None, "capacity": cores}

RANK = int(os.environ.get("RANK", "0"))
STATUS_FILE = os.environ.get("DAISY_STATUS_FILE", "status.json")
PORT = int(os.environ.get("DAISY_AGENT_PORT", "8900"))


def _status():
    try:
        with open(STATUS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


class Handler(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send({"ok": True, "rank": RANK, "host": socket.gethostname()})
        elif self.path.startswith("/resources"):
            r = survey_node(); r["rank"] = RANK; self._send(r)
        elif self.path.startswith("/status"):
            self._send(_status())
        else:
            self._send({"error": "not found"}, 404)

    def log_message(self, *a):
        pass


def main():
    print(f"[agent] rank {RANK} on :{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
