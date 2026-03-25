#!/usr/bin/env bash
# 07 — Follow logs (all services by default). Narrow: ./07_LOGS.sh worker
set -euo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEPLOY_DIR"
if [[ -f .env ]]; then
  exec docker compose --env-file .env logs -f "$@"
fi
exec docker compose logs -f "$@"
