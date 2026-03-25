#!/usr/bin/env bash
# 06 — Show running containers for this stack.
set -euo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEPLOY_DIR"
if [[ -f .env ]]; then
  exec docker compose --env-file .env ps "$@"
fi
exec docker compose ps "$@"
