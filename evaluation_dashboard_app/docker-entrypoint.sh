#!/usr/bin/env bash
set -e
# Optional: source pilot-auto (ROS 2) so Summary/Score generation can use perception_eval.
# Set PILOT_INSTALL_SETUP to the path of setup.bash inside the container (e.g. after mounting pilot-auto).
if [[ -n "${PILOT_INSTALL_SETUP}" && -f "${PILOT_INSTALL_SETUP}" ]]; then
  source "${PILOT_INSTALL_SETUP}"
fi
exec streamlit run Overview.py --server.address=0.0.0.0 --server.port=8501 "$@"
