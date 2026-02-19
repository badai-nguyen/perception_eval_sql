"""
Path utilities for multi-user deployment: single data root and path safety.

All user-facing paths (output_path, eval_root, run directories) should be
resolved against EVAL_DASHBOARD_DATA_ROOT so that:
- Path traversal (e.g. ../../../etc) is rejected.
- Absolute paths outside the root are rejected.
- Listing and delete operations stay within the data root.
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple

# Root for all evaluation data. Set EVAL_DASHBOARD_DATA_ROOT to override (e.g. /var/eval_dashboard/data).
_DATA_ROOT: Optional[Path] = None


def get_data_root() -> Path:
    """Return the canonical data root directory. Created if it does not exist."""
    global _DATA_ROOT
    if _DATA_ROOT is None:
        raw = os.environ.get("EVAL_DASHBOARD_DATA_ROOT", "data")
        p = Path(raw)
        if not p.is_absolute():
            # Resolve relative to CWD (app root when running streamlit)
            p = Path.cwd() / p
        _DATA_ROOT = p.resolve()
    return _DATA_ROOT


def get_data_root_display() -> str:
    """Return a short display path for the data root (e.g. 'data' or 'data/')."""
    root = get_data_root()
    try:
        rel = root.relative_to(Path.cwd())
        return str(rel).replace("\\", "/") or "."
    except ValueError:
        return root.name or "data"


def path_display(path: Path) -> str:
    """Return a short display path for a path under the data root (e.g. 'data/run_name')."""
    root = get_data_root()
    try:
        path.resolve().relative_to(root)
    except (ValueError, OSError):
        return path.name or str(path)
    prefix = get_data_root_display()
    try:
        rel = path.resolve().relative_to(root)
        suffix = str(rel).replace("\\", "/")
        return f"{prefix}/{suffix}" if suffix else prefix
    except (ValueError, OSError):
        return path.name or str(path)


def resolve_under_data_root(
    user_path: str,
    allow_create: bool = False,
    allow_missing: bool = False,
) -> Tuple[Optional[Path], str]:
    """
    Resolve a user-provided path so it lies under the data root.
    Returns (resolved_path, error_message). If error_message is non-empty, resolved_path is None.

    - allow_create: if True, create the path (and parents) if missing.
    - allow_missing: if True, do not require the path to exist (e.g. for eval_root before first run).
    """
    if not user_path or not str(user_path).strip():
        return None, "Path is empty."
    root = get_data_root()
    try:
        p = Path(user_path.strip())
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        else:
            p = p.resolve()
        # Ensure it is under root (use resolve() for both for consistent comparison)
        try:
            p.relative_to(root)
        except ValueError:
            return None, f"Path must be under the data root: {root}"
        if allow_create:
            p.mkdir(parents=True, exist_ok=True)
        elif not allow_missing and not p.exists():
            return None, f"Path does not exist: {p}"
        return p, ""
    except Exception as e:
        return None, str(e)


def list_run_directories() -> List[Path]:
    """Return sorted list of run directories (immediate subdirs of data root) that exist."""
    root = get_data_root()
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])


def count_tlr_scenarios(path: Path) -> int:
    """Count TLR scenarios in path: direct subdirs with result.json or suite subdirs with testcase/result.json."""
    if not path.exists() or not path.is_dir():
        return 0
    count = 0
    for child in path.iterdir():
        if not child.is_dir():
            continue
        if (child / "result.json").exists():
            count += 1
        else:
            for tc in child.iterdir():
                if tc.is_dir() and (tc / "result.json").exists():
                    count += 1
    return count


def list_tlr_result_directories() -> List[Tuple[Path, int]]:
    """Return sorted list of (path, scenario_count) under data root that contain TLR result.json.
    Scans data root and up to two levels deep (root, root/X, root/X/Y). Only includes paths
    that have at least one result.json (flat or suite layout)."""
    root = get_data_root()
    if not root.exists():
        return []
    candidates: List[Tuple[Path, int]] = []
    # depth 0
    n = count_tlr_scenarios(root)
    if n > 0:
        candidates.append((root, n))
    # depth 1
    for child in root.iterdir():
        if not child.is_dir():
            continue
        n = count_tlr_scenarios(child)
        if n > 0:
            candidates.append((child, n))
        # depth 2
        for grand in child.iterdir():
            if not grand.is_dir():
                continue
            n = count_tlr_scenarios(grand)
            if n > 0:
                candidates.append((grand, n))
    return sorted(candidates, key=lambda x: str(x[0]))


def get_run_info(run_path: Path) -> dict:
    """Return dict with name, path, size_bytes, mtime, has_summary, has_score, has_parquet."""
    size_bytes = 0
    try:
        for entry in run_path.rglob("*"):
            if entry.is_file():
                try:
                    size_bytes += entry.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    try:
        mtime = run_path.stat().st_mtime
    except OSError:
        mtime = 0
    has_summary = (run_path / "Summary.csv").exists()
    has_score = (run_path / "Score.csv").exists()
    has_parquet = any(run_path.glob("*.parquet"))
    return {
        "name": run_path.name,
        "path": run_path,
        "size_bytes": size_bytes,
        "mtime": mtime,
        "has_summary": has_summary,
        "has_score": has_score,
        "has_parquet": has_parquet,
    }


def delete_run(run_name: str) -> Tuple[bool, str]:
    """
    Delete a run directory by name (must be a direct child of data root).
    Returns (success, message).
    """
    root = get_data_root()
    if not run_name or run_name.strip() != run_name:
        return False, "Invalid run name."
    # Avoid path traversal: only allow a single path component
    if os.sep in run_name or "/" in run_name or ".." in run_name:
        return False, "Invalid run name."
    run_path = root / run_name
    if not run_path.exists():
        return False, f"Run does not exist: {run_name}"
    if not run_path.is_dir():
        return False, "Not a directory."
    try:
        run_path.relative_to(root)
    except ValueError:
        return False, "Run is not under data root."
    try:
        import shutil
        shutil.rmtree(run_path)
        return True, f"Deleted run: {run_name}"
    except Exception as e:
        return False, str(e)


def format_size(size_bytes: int) -> str:
    """Human-readable size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
