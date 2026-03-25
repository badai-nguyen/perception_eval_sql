#!/usr/bin/env bash
# 01 — Copy .env.example to .env if .env does not exist (then edit .env).
set -euo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEPLOY_DIR"
if [[ -f .env ]]; then
  echo ".env already exists; not overwriting."
  exit 0
fi
if [[ ! -f .env.example ]]; then
  echo "Error: .env.example not found in $DEPLOY_DIR" >&2
  exit 1
fi
cp .env.example .env
echo "Created .env from .env.example — edit it, then: 02_BUILD.sh, 03_INIT_DB.sh (once), 04_START.sh."
