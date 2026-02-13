"""
I/O helpers for perception_catalog_analyzer: generate parquet from pkl (or pkl.z) scene results.

Uses SceneDataFrame / scenarios_to_df for building parquet from existing .pkl or .pkl.z files.
"""

from __future__ import annotations

import gc
import glob
import os
import pickle
from pathlib import Path
from typing import Any, Callable, Iterable, List

try:
    import joblib
    import pandas as pd
    from perception_catalog_analyzer.dataframe import SceneDataFrame, scene2df
    from perception_catalog_analyzer.stream_io import export_scene_results_to_pkl, scenarios_to_df
    _ANALYZER_AVAILABLE = True
except ImportError as e:
    _ANALYZER_AVAILABLE = False
    _IMPORT_ERROR = e


def _scenarios_to_df_local(
    scenarios: Any,
    scenario_parser_function: Callable[[list], SceneDataFrame],
    topic_name_list: List[str] | None = None,
    *,
    debug: bool = True,
) -> SceneDataFrame:
    """
    Local implementation of scenarios_to_df with debug and support for flat list of frames.
    scenario_parser_function should accept list of PerceptionFrameResult and return SceneDataFrame.
    """
    if topic_name_list is None:
        topic_name_list = []

    # Normalize to list of "scenarios"
    # Single scenario object (has frame_results) -> wrap in list
    if not isinstance(scenarios, Iterable) or hasattr(scenarios, "frame_results"):
        scenarios = [scenarios]
    # Flat list of PerceptionFrameResult (e.g. from scene_result.pkl) -> treat whole list as one scenario
    elif (
        isinstance(scenarios, list)
        and len(scenarios) > 0
        and not hasattr(scenarios[0], "frame_results")
        and hasattr(scenarios[0], "pass_fail_result")
    ):
        scenarios = [scenarios]  # one scenario = entire list of frames

    if debug:
        print("[scenarios_to_df] type(scenarios) =", type(scenarios))
        print("[scenarios_to_df] len(scenarios) =", len(scenarios))
        if scenarios:
            first = scenarios[0]
            print("[scenarios_to_df] type(first) =", type(first))
            print("[scenarios_to_df] has frame_results =", getattr(first, "frame_results", None) is not None)
            if hasattr(first, "frame_results"):
                fr = getattr(first, "frame_results", {})
                print("[scenarios_to_df] frame_results type =", type(fr))
                if isinstance(fr, dict):
                    print("[scenarios_to_df] frame_results keys =", list(fr.keys()))
                    for k, v in list(fr.items())[:2]:
                        print(f"[scenarios_to_df]   frame_results[{k!r}] len =", len(v) if v is not None else 0, " type(elem) =", type(v[0]).__name__ if v and len(v) else "n/a")
            # Check if first element looks like a frame (list of frames format)
            if isinstance(first, list):
                print("[scenarios_to_df] first element is list, len =", len(first))
                if first:
                    print("[scenarios_to_df] first[0] type =", type(first[0]).__name__)
            elif not hasattr(first, "frame_results") and hasattr(first, "pass_fail_result"):
                print("[scenarios_to_df] first looks like a single frame (has pass_fail_result, no frame_results)")

    output = SceneDataFrame(current=pd.DataFrame())

    for sc_idx, sc in enumerate(scenarios):
        if debug:
            print(f"[scenarios_to_df] scenario index = {sc_idx}, type(sc) =", type(sc).__name__)

        suite_name = getattr(sc, "suite_name", "")
        scenario_name = getattr(sc, "scenario_name", "")
        frame_results_dict = getattr(sc, "frame_results", None)

        # Handle flat list of PerceptionFrameResult (e.g. from scene_result.pkl)
        if frame_results_dict is None or (isinstance(frame_results_dict, dict) and len(frame_results_dict) == 0):
            if isinstance(sc, list):
                frame_list = sc
            else:
                frame_list = [sc] if (hasattr(sc, "pass_fail_result") or not hasattr(sc, "frame_results")) else []
            if frame_list:
                if debug:
                    print(f"[scenarios_to_df] treating as flat list of frames, len = {len(frame_list)}, first type = {type(frame_list[0]).__name__}")
                frame_results_dict = {"default_topic": frame_list}
            else:
                if debug:
                    print("[scenarios_to_df] skip: no frame_results and not a list of frames")
                continue
        elif not isinstance(frame_results_dict, dict):
            if debug:
                print("[scenarios_to_df] skip: frame_results is not a dict, type =", type(frame_results_dict))
            continue

        topics_this_scenario = topic_name_list if topic_name_list else list(frame_results_dict.keys())
        if debug:
            print("[scenarios_to_df] topics_this_scenario =", topics_this_scenario)

        scenario_df = SceneDataFrame(current=pd.DataFrame())

        for topic_name in topics_this_scenario:
            if topic_name not in frame_results_dict:
                if debug and sc_idx == 0:
                    print(f"[scenarios_to_df] skip topic {topic_name!r} (not in frame_results)")
                continue
            frame_results = frame_results_dict[topic_name]
            if debug and sc_idx == 0:
                print(f"[scenarios_to_df] topic {topic_name!r} -> {len(frame_results)} frames")
            df_ = scenario_parser_function(frame_results)
            if debug and sc_idx == 0:
                print(f"[scenarios_to_df] parser returned current shape = {getattr(getattr(df_, 'current', None), 'shape', None)}, future = {getattr(df_, 'future', None) is not None}")
            df_["topic_name"] = topic_name
            scenario_df = scenario_df.concatenate(df_, ignore_index=True)

        if scenario_df.empty():
            if debug:
                print(f"[scenarios_to_df] scenario_df is empty after topics, skip")
            continue
        scenario_df["t4dataset_id"] = getattr(sc, "t4dataset_id", None)
        scenario_df["suite_name"] = suite_name
        scenario_df["scenario_name"] = scenario_name
        scenario_df["t4dataset_name"] = getattr(sc, "t4dataset_name", None)
        output = output.concatenate(scenario_df)

    if debug:
        print("[scenarios_to_df] output.current shape =", getattr(output.current, "shape", None), " output.future =", output.future is not None)
    return output


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
    pkl_dir: str | Path,
    *,
    max_files: int | None = None,
    skip_empty: bool = True,
    skip_bad_dtype: bool = True,
    on_skip: Callable[[str, str], None] | None = None,
) -> SceneDataFrame:
    """
    Build a SceneDataFrame from all *.pkl and *.pkl.z files in a directory.

    Args:
        pkl_dir: Directory containing *.pkl (and/or *.pkl.z) files.
        max_files: If set, process at most this many files (for testing).
        skip_empty: If True, skip files that yield an empty dataframe.
        skip_bad_dtype: If True, skip files where df.current["x_error"].dtype != "float64".
        on_skip: Optional callback (file_path, reason) when a file is skipped.

    Returns:
        SceneDataFrame concatenated from all valid pkl/pkl.z files.
    """
    _require_analyzer()
    pkl_dir = Path(pkl_dir)
    pkl_z = sorted(pkl_dir.rglob("*.pkl.z"))
    pkl_plain = sorted(pkl_dir.rglob("*.pkl"))
    pkl_files = sorted(set(pkl_z + pkl_plain))
    if max_files is not None:
        pkl_files = pkl_files[:max_files]

    df = SceneDataFrame(current=pd.DataFrame())
    for pkl_file in pkl_files:
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        df_ = _scenarios_to_df_local(data, scenario_parser_function=scene2df, debug=False)
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
    pkl_dir: str | Path,
    parquet_path: str | Path | None = None,
    *,
    max_files: int | None = None,
    skip_empty: bool = True,
    skip_bad_dtype: bool = True,
    on_skip: Callable[[str, str], None] | None = None,
) -> str:
    """
    Generate a single parquet file from all *.pkl and *.pkl.z files in a directory.

    Args:
        pkl_dir: Directory containing *.pkl (and/or *.pkl.z) files.
        parquet_path: Output parquet path. Default: pkl_dir / "scene_result.parquet".
        max_files: If set, use at most this many files.
        skip_empty: Skip files that yield empty dataframe.
        skip_bad_dtype: Skip files where x_error is not float64.
        on_skip: Optional callback (file_path, reason) when a file is skipped.

    Returns:
        Path to the written parquet file.
    """
    _require_analyzer()
    pkl_dir = Path(pkl_dir)
    df = build_scene_dataframe_from_pkl_dir(
        pkl_dir,
        max_files=max_files,
        skip_empty=skip_empty,
        skip_bad_dtype=skip_bad_dtype,
        on_skip=on_skip,
    )
    df.to_parquet(pkl_dir)
    parquet_file = pkl_dir / "current.parquet"
    # Check if the file is generated successfully and return its path
    if parquet_file.exists():
        return os.fspath(parquet_file)
    else:
        raise ValueError(f"Failed to generate parquet file: {parquet_file}")
