#!/usr/bin/env bash
# Render one GIF per terrain type from the latest checkpoint(s).
# Usage (on the GPU host): make_gifs.sh [level] [student_ckpt]
# With a student checkpoint, renders the SNN student instead of the teacher.
set -euo pipefail
cd ~/es_parkour_recreation
LEVEL="${1:-3}"
STUDENT="${2:-}"
EXTRA=""
TAG="teacher"
if [ -n "$STUDENT" ]; then
  EXTRA="--student $STUDENT"
  TAG="student"
fi
mkdir -p gifs
for T in gap step hurdle parkour; do
  MUJOCO_GL=osmesa ~/esparkour_venv/bin/python scripts/render_rollout.py \
    --teacher runs/teacher/ckpt_latest.pt $EXTRA --type "$T" --level "$LEVEL" \
    --events --out "gifs/${TAG}_${T}_L${LEVEL}.gif" 2>&1 | tail -1
done
echo GIFS_DONE
