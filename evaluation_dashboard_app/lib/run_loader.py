from pathlib import Path
import pandas as pd

SUMMARY_DTYPES = {
    "id": "string",
    "TP": "float64",
    "xave": "float64",
    "xstd": "float64",
    "xrms": "float64",
    "yave": "float64",
    "ystd": "float64",
    "yrms": "float64",
    "vx": "float64",
    "vy": "float64",
    # The following are included in dtypes, but may not be in Summary.csv
    "perception_label": "string",
    "product_label": "string"
}

def _has_parquet_files(run_dir: Path) -> bool:
    """Return True if run_dir contains at least one .parquet file."""
    return any(run_dir.glob("*.parquet"))


def load_run(run_dir: Path):
    summary_path = run_dir / "Summary.csv"
    score_path = run_dir / "Score.csv"

    if not summary_path.exists():
        if _has_parquet_files(run_dir):
            # Parquet-only run: allow load for Detection Stats and Bounding Box Viewer
            score = pd.read_csv(
                score_path,
                header=None,
                engine="python",
                names=[
                    "Scenario", "Option", "GT_OBJ", "Distance0", "NM0", "TP/TN0", "ADD0", "AIL0", "UIL0", "PFN/PFP0", "UUID Num0", "Practical Pass Rate0", "MAX_DIST_THRESH0", "OBJ_CNTS0",
                    "Distance1", "NM1", "TP/TN1", "ADD1", "AIL1", "UIL1", "PFN/PFP1", "UUID Num1", "Practical Pass Rate1", "MAX_DIST_THRESH1", "OBJ_CNTS1",
                    "Distance2", "NM2", "TP/TN2", "ADD2", "AIL2", "UIL2", "PFN/PFP2", "UUID Num2", "Practical Pass Rate2", "MAX_DIST_THRESH2", "OBJ_CNTS2",
                    "Distance3", "NM3", "TP/TN3", "ADD3", "AIL3", "UIL3", "PFN/PFP3", "UUID Num3", "Practical Pass Rate3", "MAX_DIST_THRESH3", "OBJ_CNTS3",
                ]
            ) if score_path.exists() else None
            return {
                "path": run_dir,
                "summary": None,
                "score": score,
            }
        raise FileNotFoundError(f"{summary_path} not found and no .parquet files in run directory")

    # Try reading with all columns; missing columns will be filled with NaN.
    summary = pd.read_csv(
        summary_path,
        header=None,
        names=SUMMARY_DTYPES.keys(),
        dtype={k: v for k, v in SUMMARY_DTYPES.items() if k not in ["perception_label", "product_label"]},
    )

    # If perception_label/product_label not in columns, add them as empty string type for consistency
    for col in ["perception_label", "product_label"]:
        if col not in summary.columns:
            summary[col] = pd.Series([""] * len(summary), dtype="string")

    score = pd.read_csv(
        score_path,
        header=None,
        engine="python",
        names=[
            "Scenario", "Option", "GT_OBJ", "Distance0", "NM0", "TP/TN0", "ADD0", "AIL0", "UIL0", "PFN/PFP0", "UUID Num0", "Practical Pass Rate0", "MAX_DIST_THRESH0", "OBJ_CNTS0",
            "Distance1", "NM1", "TP/TN1", "ADD1", "AIL1", "UIL1", "PFN/PFP1", "UUID Num1", "Practical Pass Rate1", "MAX_DIST_THRESH1", "OBJ_CNTS1",
            "Distance2", "NM2", "TP/TN2", "ADD2", "AIL2", "UIL2", "PFN/PFP2", "UUID Num2", "Practical Pass Rate2", "MAX_DIST_THRESH2", "OBJ_CNTS2",
            "Distance3", "NM3", "TP/TN3", "ADD3", "AIL3", "UIL3", "PFN/PFP3", "UUID Num3", "Practical Pass Rate3", "MAX_DIST_THRESH3", "OBJ_CNTS3",
        ]
    ) if score_path.exists() else None

    return {
        "path": run_dir,
        "summary": summary,
        "score": score,
    }
