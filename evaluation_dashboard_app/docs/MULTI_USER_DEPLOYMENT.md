# Multi-User Server Deployment Guide

This document describes what is needed when deploying the evaluation dashboard on a server where **multiple users** (e.g. a **local team**) will download results, run evaluations, view results, share them with others, and manage data.

**Design choice: no per-user authentication.** The app is intended as a **shared team tool**. All data is visible and shareable by everyone who can access the server. There is no login; team members simply open the web UI and use shareable links to point each other to specific runs or comparisons.

---

## 1. Overview of Requirements

| Area | Requirement |
|------|-------------|
| **Access** | Team members access the same server (browser). No per-user login. |
| **Download / Eval** | Anyone can download job results and run eval; paths are restricted to the data root so users don’t step on each other. |
| **View** | Everyone can view **all** runs (Overview, TP Summary, Criteria, etc.). |
| **Share** | Share a specific run or comparison via link; anyone with server access sees the same view. |
| **Data management** | List runs, see sizes, delete unnecessary data (with confirmation). |

---

## 2. What Has Been Implemented

### 2.1 Single shared data root

- **`EVAL_DASHBOARD_DATA_ROOT`**: All evaluation data lives under one root (default: `data/`). The server can set this to e.g. `/var/eval_dashboard/data` so that paths are consistent and manageable.
- **Path safety**: User-provided paths (Output Path, Eval Root) are resolved and **restricted to this root**. Paths like `../../../etc` or absolute paths outside the root are rejected.
- **Data Management page** (`7_Data_Management.py`): List all runs under the data root, show size and last modified, delete runs (with confirmation), and get a **shareable link** to a run or comparison.
- **Shareable links**: Overview supports URL query params `?mode=single|compare&run_a=...&run_b=...`. The Data Management page and Overview provide a “Copy shareable link” action so users can share exact views with others.

### 2.2 Config and credentials (server-side)

- The app runs in **one process** (e.g. Docker on a server). Users only **access the web UI** in their browser; they do not log in with API credentials.
- **Download API credentials** (e.g. `~/.webauto` or `AUTH_PROFILE`) are **server-side**: one set of credentials for the whole server/container. When any user triggers a download from the web page, the server uses these credentials to call the Evaluator/Download API. **Per-user API credentials are not required.**
- **Config file** (`configs/autoware_evaluator_dl_config.json`) is shared (one per server/container). Last-used Project ID, Job ID, Output Path, etc. are shared by all users of that server.

---

## 3. Data Layout (Current)

```
EVAL_DASHBOARD_DATA_ROOT/   (default: data/)
├── run_1/                  ← One “Run” in Overview
│   ├── Summary.csv
│   ├── Score.csv
│   └── (optional .parquet, result.txt, score.json, ...)
├── run_2/
└── ...
```

- **Runs** are top-level subdirectories of the data root. **All runs are shared**—everyone who can access the server can see and use them (no per-user filtering).
- **Parquet** for Detection Stats / Bounding Box Viewer is currently under `data/*.parquet` (flat). You can later move to per-run parquet if needed (e.g. `data/<run_id>/*.parquet`).

---

## 4. Security and Path Safety

- **Path containment**: All paths used for listing, download output, and eval root are resolved with `lib/path_utils.py` and must lie **under** `EVAL_DASHBOARD_DATA_ROOT`. This prevents path traversal and writing outside the intended area.
- **Delete**: Delete is only allowed for directories under the data root and only for run-level folders (no recursive delete of arbitrary subpaths from the UI).
- **Credentials**: Download API credentials are the **server’s** (e.g. mounted `~/.webauto` in Docker). Keep them out of the repo and expose only to the server process; users never provide API credentials in the UI.

---

## 5. Sharing Results

- **URL-based sharing**: Use query parameters on the Overview page:
  - `?mode=single&run_a=<run_name>`
  - `?mode=compare&run_a=<run_name>&run_b=<run_name>`
- **Copy shareable link**: Available from Overview and from the Data Management page so users can copy a link and send it to colleagues. Anyone with server access will see the same run(s) when opening the link.

---

## 6. Data Management

- **Data Management page** provides:
  - List of runs (name, size, last modified, presence of Summary.csv/Score.csv).
  - **Delete** a run (with confirmation) to free space.
  - **Copy shareable link** for a run (single or compare with another run).
- Only run-level directories under the data root are listed and deletable; no arbitrary path input for delete.

---

## 7. No Per-User Authentication (By Design)

Per-user authentication is **not** used. The dashboard is a **local-team tool**: all data is shared so the team can collaborate easily. Everyone who can reach the server sees the same runs, can trigger downloads (using the server’s API credentials), and can share links to specific runs or comparisons. Access control, if needed, is handled at the network level (e.g. firewall, VPN, or reverse proxy in front of the app), not with per-user login inside the app.

---

## 8. Deployment Checklist

- [ ] Set **`EVAL_DASHBOARD_DATA_ROOT`** to the desired data directory (e.g. `/var/eval_dashboard/data`) and ensure the process has read/write permissions.
- [ ] Mount or create that directory when using Docker; ensure it is persistent and backed up if needed.
- [ ] Provide **server-side credentials** for the Download API (e.g. mount `~/.webauto` into the container). All users’ download requests use this single set; no per-user API credentials.
- [ ] Restrict access as needed (firewall, VPN, or reverse proxy) so only your team can reach the app. No in-app login required.
- [ ] Optionally run behind a **reverse proxy** (nginx/Caddy) for TLS and rate limiting.
- [ ] Inform users to use **“Copy shareable link”** and the **Data Management** page to share results and clean up old runs.

---

## 9. Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `EVAL_DASHBOARD_DATA_ROOT` | Root directory for all evaluation runs. All user paths are restricted to this root. | `data` (relative to app CWD) |

---

## 10. Summary

- **Implemented**: Single shared data root, path safety, Data Management page (list/delete runs, shareable links), and shareable Overview URLs. **No per-user authentication**—all data is shared for the local team.
- **Download API credentials** are server-side (one set for the server); team members only use the web UI.
