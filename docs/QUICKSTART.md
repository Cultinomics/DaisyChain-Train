# DaisyChain Quickstart

Two ways to run. **Docker** is the most reliable (especially on Windows).

## A. Docker — one command (demo cluster on one machine)

```bash
docker compose -f docker/docker-compose.yml up --build
# open http://localhost:8080
```

This starts 3 node containers + the dashboard on a Docker network — the whole
pipeline, so you can see connectivity, the capacity plan, and live training.
Stop with `docker compose -f docker/docker-compose.yml down`.

Windows: just run `scripts\setup.bat` and pick **[1] Docker**.

## B. Python — real machines

On **every** machine:

```bash
pip install torch numpy psutil
pip install -e .            # from the repo, or `pip install daisychain`
```

Set the cluster env (copy `config/cluster.example.env`), changing only `RANK`
per machine, then run:

```bash
export MASTER_ADDR=100.101.102.10   # coordinator IP (Tailscale 100.x recommended)
export MASTER_PORT=29560
export WORLD_SIZE=3
export RANK=0                        # 1, 2, ... on the others
export GLOO_SOCKET_IFNAME=tailscale0 # your mesh/LAN NIC
daisychain-train
```

Windows: run `scripts\setup.bat` and pick **[2] Python** (it prompts for these).

## Watch it

Run the dashboard anywhere that can reach the nodes:

```bash
# edit config/nodes.example.json with your node hosts, then:
daisychain-dashboard        # http://localhost:8080
```

## Train your own model

The default is a tiny example task. To train **your** model, see
[CUSTOM_TASK.md](CUSTOM_TASK.md) — copy `examples/my_task_template.py`, fill in
`build_model` / `sample` / `loss`, and set `DAISY_TASK=your_module:YourTask`.

**Before you rely on it, read [LIMITS.md](LIMITS.md).** DaisyChain pools compute,
not memory, and is for *small* models on spare hardware — not GPU-class training.
