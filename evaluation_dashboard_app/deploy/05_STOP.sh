#!/usr/bin/env bash
# 05 — Stop containers (keeps volumes e.g. postgres_data). Optional: ./05_STOP.sh -v
set -euo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEPLOY_DIR"
if [[ -f .env ]]; then
  exec docker compose --env-file .env down "$@"
fi
exec docker compose down "$@"
