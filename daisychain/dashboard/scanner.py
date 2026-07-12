"""P2P cluster scanner: probe each node's agent, measure latency, gather
resources + live status, compute the capacity plan and readiness verdict."""
import json
import socket
import time
import urllib.request


def _get(url, timeout=2.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def _latency_ms(host, port, timeout=2.0):
    t = time.time()
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return (time.time() - t) * 1000.0
    except Exception:
        return None


def scan_node(node):
    host, ap = node["host"], node.get("agent_port", 8900)
    base = f"http://{host}:{ap}"
    out = {"name": node["name"], "host": host, "reachable": False,
           "latency_ms": None, "resources": None, "status": None,
           "gloo_port_open": None}
    lat = _latency_ms(host, ap)
    out["latency_ms"] = round(lat, 1) if lat is not None else None
    if lat is None:
        return out
    try:
        h = _get(f"{base}/health")
        out["reachable"] = bool(h.get("ok"))
        out["rank"] = h.get("rank")
        out["resources"] = _get(f"{base}/resources")
        out["status"] = _get(f"{base}/status")
    except Exception as e:
        out["error"] = str(e)
    gp = node.get("gloo_port")
    if gp:
        out["gloo_port_open"] = _latency_ms(host, gp) is not None
    return out


def capacity_plan(scans, base_batch=32):
    live = [n for n in scans if n.get("resources")]
    caps = [float(n["resources"].get("capacity") or n["resources"].get("cores", 1))
            for n in live]
    total = sum(caps) or 1
    world = len(live)
    gb = base_batch * max(1, world)
    batches = [max(1, round(gb * c / total)) for c in caps]
    tb = sum(batches) or 1
    weights = [b / tb for b in batches]
    rams = [n["resources"].get("ram_gb") for n in live
            if n["resources"].get("ram_gb") is not None]
    return {"world": world, "total_cores": sum(n["resources"].get("cores", 0) for n in live),
            "total_ram_gb": round(sum(rams), 1) if rams else None,
            "per_node": [{"name": n["name"], "device": n["resources"].get("device", "cpu"),
                          "cores": n["resources"].get("cores"),
                          "gpu": (n["resources"].get("gpu") or {}).get("name"),
                          "capacity": round(c, 1), "weight": round(w, 3), "batch": b,
                          "ram_gb": n["resources"].get("ram_gb")}
                         for n, c, w, b in zip(live, caps, weights, batches)],
            "global_batch": sum(batches)}


def scan_cluster(nodes, expected_world=None, base_batch=32):
    scans = [scan_node(n) for n in nodes]
    plan = capacity_plan(scans, base_batch)
    reachable = [s for s in scans if s["reachable"]]
    want = expected_world if expected_world is not None else len(nodes)
    ready = (len(reachable) == len(nodes) == want)
    train = None
    for s in scans:
        st = s.get("status") or {}
        if st.get("rank") == 0 and st:
            train = st
            break
    return {"nodes": scans, "plan": plan, "ready": ready,
            "reachable": len(reachable), "total": len(nodes),
            "expected_world": want, "training": train,
            "scanned_at": time.strftime("%H:%M:%S")}
