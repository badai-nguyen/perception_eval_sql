"""
Central check for numpy/pandas binary compatibility.
Use before importing pandas in Streamlit pages to show a clear error when
system and pip packages are mixed (e.g. system pandas vs user numpy).
"""


def check_numpy_pandas():
    """
    Try to import numpy and pandas. On ABI errors return (False, message).
    Returns (True, None) on success.
    """
    try:
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        return True, None
    except ImportError as e:
        msg = str(e)
        if "numpy.core.multiarray" in msg or "multiarray" in msg.lower() or "numpy" in msg.lower():
            return False, (
                "**NumPy compatibility error.** pandas/pyarrow were built against a different "
                "NumPy version. Use a virtual environment and run: "
                "`pip install --upgrade numpy pandas pyarrow`"
            )
        raise
    except ValueError as e:
        msg = str(e)
        if "numpy.dtype" in msg or "binary incompatibility" in msg.lower():
            return False, (
                "**NumPy/pandas binary incompatibility.** You are likely mixing system pandas "
                "(`/usr/lib/python3/dist-packages/`) with a different NumPy (e.g. from pip). "
                "Use a **virtual environment** and install everything with pip:\n\n"
                "```bash\npython3 -m venv .venv\nsource .venv/bin/activate  # or .venv\\\\Scripts\\\\activate on Windows\n"
                "pip install -r requirements.txt\n```\n\n"
                "Then run the app with that environment activated."
            )
        raise
