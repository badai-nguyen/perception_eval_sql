"""
I/O helpers for perception_catalog_analyzer: download pkl.z scene results and generate parquet.

Uses perception_catalog_analyzer.stream_io.export_scene_results_to_pkl for downloads
and SceneDataFrame / scenarios_to_df for building parquet from existing pkl.z files.
"""

from __future__ import annotations

import gc
import glob
import os
from pathlib import Path
from typing import Callable

try:
    import joblib
    import pandas as pd
    from perception_catalog_analyzer.dataframe import SceneDataFrame, scene2df
    from perception_catalog_analyzer.stream_io import export_scene_results_to_pkl, scenarios_to_df
    _ANALYZER_AVAILABLE = True
except ImportError as e:
    _ANALYZER_AVAILABLE = False
    _IMPORT_ERROR = e


def _require_analyzer() -> None:
    if not _ANALYZER_AVAILABLE:
        raise ImportError(
            "perception_catalog_analyzer is required for pkl download and parquet generation. "
            f"Install it first. Original error: {_IMPORT_ERROR!s}"
        ) from _IMPORT_ERROR


def download_scene_results_to_pkl(
    project_id: str,
    job_id: str,
    out_dir: str | Path,
    *,
    overwrite: bool = False,
) -> list[str]:
    """
    Download scene report as pkl.z files using perception_catalog_analyzer.

    Creates out_dir and an "archive" subdir, then calls export_scene_results_to_pkl
    to download and save *.pkl.z files there.

    Args:
        project_id: Project ID of the report.
        job_id: Job ID of the report.
        out_dir: Base output directory; archive subdir will be out_dir/archive.
        overwrite: If True, overwrite existing pkl.z files.

    Returns:
        List of paths to downloaded *.pkl.z files (under out_dir/archive).
    """
    _require_analyzer()
    out_dir = Path(out_dir)
    archive_outdir = out_dir / "archive"
    archive_outdir.mkdir(parents=True, exist_ok=True)
    archive_str = os.fspath(archive_outdir)
    pkl_files = export_scene_results_to_pkl(
        project_id=project_id,
        job_id=job_id,
        out_dir=archive_str,
        overwrite=overwrite,
    )
    return pkl_files


def build_scene_dataframe_from_pkl_dir(
    archive_dir: str | Path,
    *,
    max_files: int | None = None,
    skip_empty: bool = True,
    skip_bad_dtype: bool = True,
    on_skip: Callable[[str, str], None] | None = None,
) -> SceneDataFrame:
    """
    Build a SceneDataFrame from all *.pkl.z files in a directory.

    Args:
        archive_dir: Directory containing *.pkl.z files.
        max_files: If set, process at most this many files (for testing).
        skip_empty: If True, skip pkl.z that yield an empty dataframe.
        skip_bad_dtype: If True, skip pkl.z where df.current["x_error"].dtype != "float64".
        on_skip: Optional callback (file_path, reason) when a file is skipped.

    Returns:
        SceneDataFrame concatenated from all valid pkl.z files.
    """
    _require_analyzer()
    archive_dir = Path(archive_dir)
    pattern = os.path.join(archive_dir, "*.pkl.z")
    pkl_files = sorted(glob.glob(pattern))
    if max_files is not None:
        pkl_files = pkl_files[:max_files]

    df = SceneDataFrame(current=pd.DataFrame())
    for pkl_file in pkl_files:
        data = joblib.load(pkl_file)
        df_ = scenarios_to_df(data, scenario_parser_function=scene2df)
        del data
        if df_.empty():
            if skip_empty:
                if on_skip:
                    on_skip(pkl_file, "empty")
                continue
        if skip_bad_dtype and hasattr(df_, "current") and "x_error" in getattr(df_.current, "columns", []):
            if df_.current["x_error"].dtype != "float64":
                if on_skip:
                    on_skip(pkl_file, f"bad dtype x_error={df_.current['x_error'].dtype}")
                continue
        df = df.concatenate(df_)
        del df_
        gc.collect()
    return df


def pkl_archive_to_parquet(
    archive_dir: str | Path,
    parquet_path: str | Path | None = None,
    *,
    max_files: int | None = None,
    skip_empty: bool = True,
    skip_bad_dtype: bool = True,
    on_skip: Callable[[str, str], None] | None = None,
) -> str:
    """
    Generate a single parquet file from all *.pkl.z files in archive_dir.

    Args:
        archive_dir: Directory containing *.pkl.z files (e.g. .../output/project/job/archive).
        parquet_path: Output parquet path. Default: archive_dir / "scene_result.parquet".
        max_files: If set, use at most this many pkl.z files.
        skip_empty: Skip pkl.z that yield empty dataframe.
        skip_bad_dtype: Skip pkl.z where x_error is not float64.
        on_skip: Optional callback (file_path, reason) when a file is skipped.

    Returns:
        Path to the written parquet file.
    """
    _require_analyzer()
    archive_dir = Path(archive_dir)
    if parquet_path is None:
        parquet_path = archive_dir / "scene_result.parquet"
    parquet_path = Path(parquet_path)

    df = build_scene_dataframe_from_pkl_dir(
        archive_dir,
        max_files=max_files,
        skip_empty=skip_empty,
        skip_bad_dtype=skip_bad_dtype,
        on_skip=on_skip,
    )
    df.to_parquet(parquet_path)
    return os.fspath(parquet_path)
