#!/usr/bin/env bash
# 02 — Build images from docker-compose.yml (from this directory).
# Extra args: e.g. ./02_BUILD.sh --no-cache
set -euo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEPLOY_DIR"
exec docker compose build "$@"
