#!/usr/bin/env bash
# Render one GIF per terrain type from the latest teacher checkpoint.
# Usage (on the GPU host): ~/es_parkour_recreation/remote/make_gifs.sh [level]
set -euo pipefail
cd ~/es_parkour_recreation
LEVEL="${1:-3}"
mkdir -p gifs
for T in gap step hurdle parkour; do
  MUJOCO_GL=osmesa ~/esparkour_venv/bin/python scripts/render_rollout.py \
    --teacher runs/teacher/ckpt_latest.pt --type "$T" --level "$LEVEL" \
    --events --out "gifs/${T}_L${LEVEL}.gif" 2>&1 | tail -1
done
echo GIFS_DONE
