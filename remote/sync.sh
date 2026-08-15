#!/usr/bin/env bash
# Sync the repo to the GPU host (code + assets, not runs/checkpoints).
set -euo pipefail
cd "$(dirname "$0")/.."
rsync -az --delete \
  --exclude runs --exclude .git --exclude __pycache__ --exclude '*.pyc' \
  ./ claude_svpn:~/es_parkour_recreation/
echo synced
