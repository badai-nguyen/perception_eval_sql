#!/usr/bin/env bash
# 04 — Start the full stack, or if it is already running: up -d (apply compose/scale) then restart all services.
# Extra args: e.g. ./04_START.sh --scale worker=3
set -euo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEPLOY_DIR"
if [[ ! -f .env ]]; then
  echo "Error: .env not found. Run 01_SETUP_ENV.sh first, then edit .env" >&2
  exit 1
fi

dc() { docker compose --env-file .env "$@"; }

if [[ -n "$(dc ps -q --status running 2>/dev/null || true)" ]]; then
  echo "Stack already running — updating with up -d, then restarting all services."
  dc up -d "$@"
  dc restart
else
  dc up -d "$@"
fi
