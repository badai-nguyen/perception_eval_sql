from pathlib import Path
import pandas as pd

def load_run(run_dir: str):
    run_dir = Path(run_dir)

    summary = pd.read_csv(
        run_dir / "summary.csv",
        header=None,
        names=[
            "id", "TP",
            "xave", "xstd", "xrms",
            "yave", "ystd", "yrms",
            "vx", "vy",
        ],
        dtype={
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
        },
    )

    score = pd.read_csv(run_dir / "score.csv")

    return {
        "path": str(run_dir),
        "summary": summary,
        "score": score,
    }
