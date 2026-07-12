#!/bin/sh
# A DaisyChain node: run the dashboard agent in the background, then train.
# After training, keep the agent alive so the dashboard can still show status.
set -e
python -m daisychain.dashboard.agent &
python -m daisychain.train || echo "[node] training exited"
echo "[node] training done — agent still serving status"
wait
