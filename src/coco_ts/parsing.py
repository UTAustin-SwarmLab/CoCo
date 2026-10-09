"""Forecast parser preserved from the evaluation pipeline."""
from typing import Any

def parse_forecast(
    forecast_str: str | None,
    pred_len: int,
    sample_idx: int,
    values,
) -> tuple[list[float], dict[str, Any]]:
    info: dict[str, Any] = {"parse_ok": False, "method": None, "error": None}

    if forecast_str is None or not isinstance(forecast_str, str):
        info["method"] = "none_or_non_string"
        info["error"] = "Forecast string is None or not a string"
        print(f"Error: Forecast string is None or not a string for sample {sample_idx}")
        print(f"Forecast string: {forecast_str}")
        return [values[-1]] * pred_len, info

    raw = forecast_str.strip()
    if not raw:
        info["method"] = "empty"
        info["error"] = "Empty forecast string"
        print(f"Error: Empty forecast string for sample {sample_idx}")
        return [values[-1]] * pred_len, info

    # First try: parse from the last non-empty line (supports reasoning + final numeric line).
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    for ln in reversed(lines):
        ln2 = ln
        if ":" in ln2 and "forecast" in ln2.lower():
            ln2 = ln2.split(":", 1)[1].strip()
        ln2 = ln2.strip().strip("[]")
        try:
            vals = [float(x.strip()) for x in ln2.split(",") if x.strip()]
        except Exception:
            vals = []
        if len(vals) == pred_len:
            info["parse_ok"] = True
            info["method"] = "last_line_csv"
            return vals, info

    # Fallback: extract floats from entire response, take last pred_len.
    import re

    nums = re.findall(r"[-+]?(?:\d+\.\d+|\d+\.|\.\d+|\d+)(?:[eE][-+]?\d+)?", raw)
    if len(nums) >= pred_len:
        try:
            info["parse_ok"] = True
            info["method"] = "regex_last_k"
            return [float(x) for x in nums[-pred_len:]], info
        except Exception as e:
            info["error"] = f"Regex float conversion error: {e}"

    info["method"] = "failed"
    info["error"] = f"Could not parse {pred_len} forecast values"
    print(
        f"Error: Could not parse {pred_len} forecast values for sample {sample_idx}. "
        f"Raw response:\n{raw}"
    )
    return [values[-1]] * pred_len, info
