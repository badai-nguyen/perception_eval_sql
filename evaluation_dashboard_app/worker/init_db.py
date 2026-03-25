"""
Create Postgres tasks table. Run once before starting Streamlit/worker in production.
Usage: python -m worker.init_db
Requires: DATABASE_URL environment variable.
"""

import sys
import os

# Ensure app root is on path so lib.db is importable
_APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

from lib.db import init_db, get_database_url


def main():
    if not get_database_url():
        print("DATABASE_URL is not set. Skipping init.", file=sys.stderr)
        sys.exit(0)
    if init_db():
        print("Tasks table created or already exists.")
        sys.exit(0)
    print("Failed to create tasks table. Check DATABASE_URL and Postgres.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
