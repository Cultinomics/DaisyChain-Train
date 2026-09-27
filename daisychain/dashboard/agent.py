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


#: Cached capacity score. `survey_node()` defaults to measure=True, which runs
#: `capacity_score()` -- 0.3 s of timed 512x512 matmuls, measured at **0.47 s**
#: wall including warmup. The dashboard page carries
#: `<meta http-equiv="refresh" content="3">`, so every 3 s it re-scans every node
#: and each `/resources` request re-ran that benchmark: **~16% duty cycle of a
#: core, per node, continuously, while training** -- on hardware this project
#: targets precisely because it is small.
#:
#: It is also self-perturbing. The score feeds capacity-weighted batch sharding,
#: and was being measured while competing with both the training step and the
#: benchmark's own previous run, so the number it reported was a function of how
#: often the dashboard was open.
#:
#: Capacity is a property of the machine, not a live metric: measure once, reuse.
#: DAISY_CAPACITY still overrides (survey_node honours it), and a restart
#: re-measures.
_CAPACITY = None


def _resources():
    """Node resources for /resources, without re-benchmarking on every request."""
    global _CAPACITY
    try:
        if _CAPACITY is None:
            _CAPACITY = survey_node().get("capacity")     # measure once, at first ask
        r = survey_node(measure=False)                    # cheap fields only
        r["capacity"] = _CAPACITY
        return r
    except TypeError:
        # the stdlib fallback survey_node above takes no `measure` kwarg
        return survey_node()


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
            r = _resources(); r["rank"] = RANK; self._send(r)
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
