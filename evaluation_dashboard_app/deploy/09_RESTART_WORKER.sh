#!/usr/bin/env bash
# 09 — Restart worker containers (pick up worker/ or lib/ code changes without full rebuild).
set -euo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEPLOY_DIR"
if [[ -f .env ]]; then
  exec docker compose --env-file .env restart worker "$@"
fi
exec docker compose restart worker "$@"
