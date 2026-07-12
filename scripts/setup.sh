#!/usr/bin/env bash
# DaisyChain setup helper (Linux/macOS).
set -e
cd "$(dirname "$0")/.."

echo "🌼 DaisyChain — Old Hardware Training Pipeline"
echo "  [1] Docker (demo cluster on this box)"
echo "  [2] Python node (join a real cluster)"
echo "  [3] Install deps only"
read -rp "  Choice: " c

case "$c" in
  1)
    command -v docker >/dev/null || { echo "Install Docker first."; exit 1; }
    docker compose -f docker/docker-compose.yml up --build -d
    echo "Dashboard: http://localhost:8080   (stop: docker compose -f docker/docker-compose.yml down)"
    ;;
  3|2)
    python3 -m pip install --upgrade pip
    python3 -m pip install torch numpy psutil
    python3 -m pip install -e .
    echo "Installed."
    if [ "$c" = "2" ]; then
      read -rp "  MASTER_ADDR (coordinator IP): " MA
      read -rp "  WORLD_SIZE: " WS
      read -rp "  RANK (0=coordinator): " RK
      read -rp "  GLOO_SOCKET_IFNAME (e.g. tailscale0): " IF
      export MASTER_ADDR="$MA" MASTER_PORT=29560 WORLD_SIZE="$WS" RANK="$RK"
      export GLOO_SOCKET_IFNAME="$IF" USE_LIBUV=0
      python3 -m daisychain.train
    fi
    ;;
  *) echo "bye" ;;
esac
