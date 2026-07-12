# Connecting machines with Tailscale (recommended)

DaisyChain needs every node reachable by a **stable IP on one interface**.
Tailscale gives you exactly that: a private mesh where each machine gets a
`100.x.y.z` address on the `tailscale0` interface, reachable P2P across NATs and
different networks — no port-forwarding.

## 1. Install on every machine
- Linux: `curl -fsSL https://tailscale.com/install.sh | sh`
- Windows / macOS: <https://tailscale.com/download>

Bring each up (same tailnet / account) and note its IP:

```bash
sudo tailscale up
tailscale ip -4          # e.g. 100.101.102.10
tailscale status         # see every machine + IP
```

## 2. Configure the cluster
Pick one machine as **rank 0** and use its Tailscale IP as `MASTER_ADDR`. On
every machine:

```bash
export MASTER_ADDR=100.101.102.10
export MASTER_PORT=29560
export WORLD_SIZE=3
export GLOO_SOCKET_IFNAME=tailscale0   # pin gloo to the mesh NIC
export RANK=0                           # 1, 2, ... on the others
daisychain-train
```

`GLOO_SOCKET_IFNAME=tailscale0` is the important line — it stops gloo from
wandering onto a LAN/VPN/Docker interface.

## 3. Verify
Run `daisychain-dashboard` (point `config/nodes.example.json` at the Tailscale
IPs). Green banner = every node reachable and ports open → launch training.

> On Windows the Tailscale interface name differs and gloo tensor collectives are
> unstable anyway — prefer Linux nodes or Docker for the actual training.
