#!/usr/bin/env bash
# Bootstrap a fresh interactive GPU pod for autoresearch training.
# Reproduces the manual bring-up: system headers -> repo -> uv env -> data -> secrets.
# Idempotent: safe to re-run. Usage:  WANDB_API_KEY=xxxx bash setup_env.sh
set -euo pipefail

REPO_URL="${REPO_URL:-git@github.com:Aochong-Li/autoresearch.git}"
WORKDIR="${WORKDIR:-/data/s3_autosync/autoresearch}"
PERSIST_SSH="${PERSIST_SSH:-/data/s3_autosync/.ssh}"   # survives pod restarts

echo "==> [1/6] system deps (python dev headers for torch.compile/Triton)"
# Triton compiles a small CUDA util against Python.h at first run; the slim image lacks headers.
apt-get update -qq && apt-get install -y -qq python3.10-dev python3-dev

echo "==> [2/6] persistent SSH key -> git remote"
mkdir -p "$PERSIST_SSH" /root/.ssh
if [ ! -f "$PERSIST_SSH/id_ed25519" ]; then
  ssh-keygen -t ed25519 -f "$PERSIST_SSH/id_ed25519" -N "" -C "h100-pod"
  echo "   NEW deploy key created. Register its .pub on the repo with write access:"
  echo "   gh repo deploy-key add $PERSIST_SSH/id_ed25519.pub --repo Aochong-Li/autoresearch --allow-write --title h100-pod"
fi
ln -sf "$PERSIST_SSH/id_ed25519" /root/.ssh/id_ed25519
ln -sf "$PERSIST_SSH/id_ed25519.pub" /root/.ssh/id_ed25519.pub
chmod 600 "$PERSIST_SSH/id_ed25519"
ssh-keyscan -t ed25519 github.com >> /root/.ssh/known_hosts 2>/dev/null || true

echo "==> [3/6] repo at $WORKDIR"
if [ -d "$WORKDIR/.git" ]; then
  git -C "$WORKDIR" remote set-url origin "$REPO_URL"
  git -C "$WORKDIR" fetch --all --quiet
else
  git clone "$REPO_URL" "$WORKDIR"
fi
git -C "$WORKDIR" config user.name  "${GIT_NAME:-Aochong-Li}"
git -C "$WORKDIR" config user.email "${GIT_EMAIL:-oliverli@datologyai.com}"

echo "==> [4/6] uv environment (torch cu128 + wandb, from uv.lock)"
cd "$WORKDIR"
uv sync

echo "==> [5/6] data + tokenizer (prepare.py) if not cached"
if [ ! -d /root/.cache/autoresearch/data ] || [ ! -f /root/.cache/autoresearch/tokenizer/tokenizer.pkl ]; then
  uv run prepare.py
else
  echo "   cache present, skipping prepare.py"
fi

echo "==> [6/6] secrets + self-check"
if [ -n "${WANDB_API_KEY:-}" ]; then
  printf '%s' "$WANDB_API_KEY" > /root/.wandb_key && chmod 600 /root/.wandb_key
  echo "   staged /root/.wandb_key"
else
  echo "   WARN: WANDB_API_KEY not set; export it or write /root/.wandb_key before training"
fi
uv run python -c "import torch, wandb; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'ngpu', torch.cuda.device_count(), '| wandb', wandb.__version__)"
nvidia-smi --query-gpu=index,name,memory.total,utilization.gpu --format=csv,noheader
echo "==> done. Train with:  export WANDB_API_KEY=\$(cat /root/.wandb_key); CUDA_VISIBLE_DEVICES=0 RUN_NAME=baseline uv run train.py"
