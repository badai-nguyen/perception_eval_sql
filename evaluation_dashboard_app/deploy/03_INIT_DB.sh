#!/usr/bin/env bash
# 03 — One-time: start Postgres if needed, then run init_db (creates task tables).
set -euo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEPLOY_DIR"
if [[ ! -f .env ]]; then
  echo "Error: .env not found. Run 01_SETUP_ENV.sh first, then edit .env" >&2
  exit 1
fi
docker compose --env-file .env up -d postgres
docker compose --env-file .env run --rm init_db
echo "Database init finished."
