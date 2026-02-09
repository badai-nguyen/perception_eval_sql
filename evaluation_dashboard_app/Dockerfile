# syntax=docker/dockerfile:1
# Evaluation Dashboard (Streamlit)
# Build with host SSH key:  docker build --secret id=ssh,src=$HOME/.ssh/id_rsa -t ...
# See Readme.md "Docker" section for build/run and required mount paths.
# Python 3.12 required. webauto-auth-py is installed with --no-deps to skip driving-log-replayer (Python <3.11 only).
FROM python:3.10-slim

WORKDIR /app

# Entrypoint, Shapely, git + openssh-client (for pip install git+ssh), pipx (for v_and_v_util)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgeos-c1v5 bash git openssh-client \
    && rm -rf /var/lib/apt/lists/*

# pipx needed before pipx install
RUN pip install --no-cache-dir pipx && pipx ensurepath
ENV PATH="/root/.local/bin:${PATH}"

# Private deps first (fail fast if SSH/key or clone fails). Build with: --secret id=ssh,src=$HOME/.ssh/id_rsa
RUN --mount=type=secret,id=ssh,dst=/tmp/ssh_key \
    mkdir -p /root/.ssh && cp /tmp/ssh_key /root/.ssh/id_rsa && chmod 700 /root/.ssh && chmod 600 /root/.ssh/id_rsa && \
    GIT_SSH_COMMAND="ssh -i /root/.ssh/id_rsa -o StrictHostKeyChecking=accept-new" \
    pip install --no-cache-dir "git+ssh://git@github.com/tier4/webauto-auth-py.git"
RUN --mount=type=secret,id=ssh,dst=/tmp/ssh_key \
    mkdir -p /root/.ssh && cp /tmp/ssh_key /root/.ssh/id_rsa && chmod 700 /root/.ssh && chmod 600 /root/.ssh/id_rsa && \
    GIT_SSH_COMMAND="ssh -i /root/.ssh/id_rsa -o StrictHostKeyChecking=accept-new" \
    pipx install "git+ssh://git@github.com/tier4/v_and_v_util.git"

# Public Python dependencies (after SSH installs so failures surface early)
COPY requirements-docker.txt requirements-docker.txt
RUN pip install --no-cache-dir -r requirements-docker.txt

# App code and config
COPY Overview.py .
COPY pages/ pages/
COPY lib/ lib/
COPY configs/ configs/
COPY docs/ docs/
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Default data dir: mount over this so the app sees your runs (Summary.csv, Score.csv, parquet).
# Optional: mount configs to persist autoware_evaluator_dl_config.json.
# Optional: mount pilot-auto and set PILOT_INSTALL_SETUP for Summary/Score CSV generation.
RUN mkdir -p /app/data

EXPOSE 8501

# Entrypoint optionally sources PILOT_INSTALL_SETUP then runs Streamlit.
# Start with proper mounts, e.g.:
#   docker run -p 8501:8501 \
#     -v /host/path/to/data:/app/data \
#     -v /host/path/to/pilot-auto:/mnt/pilot \
#     -e PILOT_INSTALL_SETUP=/mnt/pilot/install/setup.bash \
#     your-image
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD []
