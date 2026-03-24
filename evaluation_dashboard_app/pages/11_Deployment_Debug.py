"""
Deployment debug: environment (redacted), Postgres/Redis/RQ health, task counts, optional Docker list/logs.
"""
import os
from datetime import timedelta

import pandas as pd
import streamlit as st

from lib.deploy_debug import (
    EXEC_TIMEOUT_SEC,
    MAX_LOG_TAIL_LINES,
    compose_project_filter,
    container_exec_command,
    container_logs_tail,
    docker_client_or_none,
    is_docker_debug_enabled,
    is_exec_enabled,
    list_containers_for_debug,
    postgres_check,
    redacted_deployment_env_rows,
    redis_ping_check,
    rq_overview,
    task_counts_by_status,
)
from lib.page_chrome import inject_app_page_styles, render_page_hero, section_header

st.set_page_config(
    layout="wide",
    page_title="Deployment debug",
    page_icon="🐳",
    initial_sidebar_state="expanded",
)
inject_app_page_styles()
render_page_hero(
    kicker="Operations",
    title="Deployment & Docker debug",
    description=(
        "Check Postgres, Redis, and the RQ queue; inspect redacted environment variables; "
        "optionally list containers and tail logs when Docker socket access is enabled; "
        "optional one-shot shell commands when `EVAL_DEPLOYMENT_DEBUG_EXEC=1`."
    ),
    mode="Single Run",
)

tab_env, tab_dep, tab_tasks, tab_docker = st.tabs(
    ["Environment", "Dependencies", "Tasks", "Docker"]
)

with tab_env:
    section_header("Deployment environment", "Sensitive connection strings are redacted.")
    env_df = pd.DataFrame(redacted_deployment_env_rows(), columns=["Variable", "Value"])
    st.dataframe(env_df, use_container_width=True, hide_index=True)

with tab_dep:
    section_header("Postgres")
    ok, msg = postgres_check()
    if ok:
        st.success(msg)
    else:
        st.error(msg)

    section_header("Redis")
    ok_r, msg_r = redis_ping_check()
    if ok_r:
        st.success(msg_r)
    else:
        st.error(msg_r)

    section_header("RQ queue")
    ok_q, msg_q, details = rq_overview()
    if ok_q and details:
        st.success(msg_q)
        st.json(details)
    else:
        st.warning(msg_q if not ok_q else "No queue details")

with tab_tasks:
    section_header("Task rows by status", "From Postgres `tasks` when the task queue is enabled.")
    ok_t, msg_t, counts = task_counts_by_status()
    if counts is None:
        st.info(msg_t)
    elif ok_t and counts:
        st.success(msg_t)
        cdf = pd.DataFrame(
            [{"status": k, "count": v} for k, v in sorted(counts.items())]
        )
        st.dataframe(cdf, use_container_width=True, hide_index=True)
    elif ok_t:
        st.success("No task rows yet (empty table).")
    else:
        st.error(msg_t)


def _render_docker_disabled(reason: str) -> None:
    st.warning(reason)
    st.markdown(
        """
**Enable Docker debug (trusted operators only)**

1. From the `deploy/` directory, ensure `docker-compose.yml` mounts `/var/run/docker.sock` into the `streamlit` service and sets `EVAL_DEPLOYMENT_DEBUG_DOCKER=1`, then run `docker compose up -d` (or `docker compose up -d --force-recreate streamlit` after editing compose).

2. Set `EVAL_DEPLOYMENT_DEBUG_COMPOSE_PROJECT` in `.env` to your Compose project name
   (same value as in `docker compose ls`) so the UI lists only this stack’s containers.

3. Rebuild or restart the Streamlit service after changing dependencies so `docker` (docker-py) is installed.

Anyone who can open this page with socket access can read container logs for listed containers — use network ACLs or auth in front of the app.

With `EVAL_DEPLOYMENT_DEBUG_EXEC=1`, the Docker tab can also run `sh -c` inside a selected container — treat that like full shell access.
        """
    )


def _render_docker_exec_ui(client, full_id: str) -> None:
    if not is_exec_enabled():
        st.caption(
            "To run commands in the selected container, set `EVAL_DEPLOYMENT_DEBUG_EXEC=1` in `.env` "
            "and recreate Streamlit (high risk — same as `docker exec`)."
        )
        return

    prev = st.session_state.get("deploy_debug_exec_cid")
    if prev is not None and prev != full_id:
        st.session_state.pop("deploy_debug_exec_result", None)
    st.session_state["deploy_debug_exec_cid"] = full_id

    st.markdown("**Run command in container**")
    st.caption(
        "Runs `sh -c \"…\"` in the **currently selected** container. Output is capped; long commands time out "
        f"after ~{int(EXEC_TIMEOUT_SEC)}s."
    )
    st.text_input("Shell command", key="deploy_debug_exec_cmd", placeholder="ls -la /app")
    if st.button("Run", key="deploy_debug_exec_run"):
        cmd = (st.session_state.get("deploy_debug_exec_cmd") or "").strip()
        if not cmd:
            st.warning("Enter a command.")
        else:
            with st.spinner("Executing…"):
                code, out = container_exec_command(client, full_id, cmd)
            st.session_state["deploy_debug_exec_result"] = (code, out)
    res = st.session_state.get("deploy_debug_exec_result")
    if res:
        code, out = res
        st.caption(f"Exit code: {code}")
        st.code(out or "(no output)", language=None)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


with tab_docker:
    section_header("Containers & logs", "Requires `EVAL_DEPLOYMENT_DEBUG_DOCKER` and `/var/run/docker.sock` in the Streamlit container.")

    client = docker_client_or_none()
    if client is None:
        if not _env_flag("EVAL_DEPLOYMENT_DEBUG_DOCKER"):
            _render_docker_disabled(
                "Docker debug is off: set `EVAL_DEPLOYMENT_DEBUG_DOCKER=1` and apply the optional compose override "
                "that mounts the host Docker socket (see below)."
            )
        elif not is_docker_debug_enabled():
            _render_docker_disabled(
                "`EVAL_DEPLOYMENT_DEBUG_DOCKER` is set, but the Docker Unix socket is not available inside this container "
                "(or `DOCKER_HOST` points to a non-Unix endpoint that is unreachable)."
            )
        else:
            try:
                import docker as _docker_check  # noqa: F401
            except ImportError:
                _render_docker_disabled(
                    "Docker debug is enabled and the socket is present, but the `docker` Python package is not installed."
                )
            else:
                _render_docker_disabled(
                    "`docker.from_env()` failed — check socket permissions (Streamlit user must read/write the socket) "
                    "or daemon availability."
                )
    else:
        proj = compose_project_filter()
        if proj:
            st.caption(f"Filtering by Compose project label: `{proj}`")
        else:
            st.warning(
                "Listing all containers on this Docker host. Set `EVAL_DEPLOYMENT_DEBUG_COMPOSE_PROJECT` in `.env` "
                "to match `docker compose ls` and restrict the list."
            )

        _use_fragment = getattr(st, "fragment", None) is not None

        if _use_fragment:

            @st.fragment(run_every=timedelta(seconds=6))
            def _docker_fragment():
                rows, list_warn = list_containers_for_debug(client)
                if list_warn and isinstance(list_warn, str) and list_warn.startswith("Docker list failed"):
                    st.error(list_warn)
                    return
                if list_warn:
                    st.markdown(list_warn)
                if not rows:
                    st.info("No containers match the current filter.")
                    return
                display_df = pd.DataFrame(rows).drop(columns=["full_id"], errors="ignore")
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                options = [f"{r['name']} ({r['id']})" for r in rows]
                id_by_label = {f"{r['name']} ({r['id']})": r["full_id"] for r in rows}

                prev_cid = st.session_state.get("deploy_debug_cid")
                default_ix = 0
                if prev_cid:
                    for i, opt in enumerate(options):
                        if id_by_label[opt] == prev_cid:
                            default_ix = i
                            break

                pick = st.selectbox("Container", options=options, index=default_ix, key="deploy_debug_pick")
                full_id = id_by_label[pick]
                st.session_state.deploy_debug_cid = full_id

                tail = st.slider(
                    "Log tail (lines)",
                    min_value=50,
                    max_value=MAX_LOG_TAIL_LINES,
                    value=300,
                    step=50,
                    key="deploy_debug_tail",
                )
                logs = container_logs_tail(client, full_id, tail)
                # Use st.code (not st.text_area with a fixed key): keyed text_area keeps stale
                # session_state and ignores new value= when the container selection changes.
                st.markdown("**Logs**")
                st.code(logs or "(empty)", language=None)
                _render_docker_exec_ui(client, full_id)

            _docker_fragment()
        else:
            rows, list_warn = list_containers_for_debug(client)
            if list_warn and isinstance(list_warn, str) and list_warn.startswith("Docker list failed"):
                st.error(list_warn)
            elif list_warn:
                st.markdown(list_warn)
            if not rows:
                st.info("No containers match the current filter.")
            else:
                df = pd.DataFrame(rows)
                st.dataframe(
                    df.drop(columns=["full_id"], errors="ignore"),
                    use_container_width=True,
                    hide_index=True,
                )
                options = [f"{r['name']} ({r['id']})" for r in rows]
                id_by_label = {f"{r['name']} ({r['id']})": r["full_id"] for r in rows}
                pick = st.selectbox("Container", options=options, key="deploy_debug_pick_legacy")
                tail = st.slider(
                    "Log tail (lines)",
                    min_value=50,
                    max_value=MAX_LOG_TAIL_LINES,
                    value=300,
                    step=50,
                    key="deploy_debug_tail_legacy",
                )
                full_id_legacy = id_by_label[pick]
                logs = container_logs_tail(client, full_id_legacy, tail)
                st.markdown("**Logs**")
                st.code(logs or "(empty)", language=None)
                _render_docker_exec_ui(client, full_id_legacy)
                if st.button("Refresh container list"):
                    st.rerun()
