"""Build LLM forecast prompt from history values and text (matches gpt_forecast baseline)."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Sequence


def _format_timestamp(x: object) -> str:
    """Format a timestamp-like value into a readable calendar form when possible."""
    if x is None:
        return ""
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return ""
        if re.fullmatch(r"\d{8}", s):
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.hour == 0 and dt.minute == 0 and dt.second == 0 and dt.microsecond == 0:
                return dt.date().isoformat()
            return dt.replace(microsecond=0).isoformat(sep=" ")
        except Exception:
            pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(s, fmt)
                if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
                    return dt.date().isoformat()
                return dt.isoformat(sep=" ")
            except Exception:
                continue
        return s

    if isinstance(x, datetime):
        dt = x
        if dt.hour == 0 and dt.minute == 0 and dt.second == 0 and dt.microsecond == 0:
            return dt.date().isoformat()
        return dt.replace(microsecond=0).isoformat(sep=" ")

    return str(x)


def _looks_like_datetime_timestamps(ts: Sequence[object] | None) -> bool:
    """Only treat timestamps as datetimes if they look like string date/times, not indices."""
    if not ts:
        return False
    if not all(isinstance(x, str) and x.strip() for x in ts):
        return False
    s = [x.strip() for x in ts]

    if all(re.fullmatch(r"\d+", x) for x in s):
        if all(len(x) == 8 for x in s):
            try:
                years = [int(x[:4]) for x in s]
            except Exception:
                return False
            return all(1600 <= y <= 2500 for y in years)
        return False

    def _parse_one(x: str) -> bool:
        x2 = x.strip()
        try:
            datetime.fromisoformat(x2.replace("Z", "+00:00"))
            return True
        except Exception:
            pass
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d", "%Y/%m/%d %H:%M:%S"):
            try:
                datetime.strptime(x2, fmt)
                return True
            except Exception:
                continue
        return False

    return all(_parse_one(x) for x in s)


def _format_forecast_targets_section(pred_len: int, forecast_timestamps: Sequence[object] | None) -> str:
    if not forecast_timestamps or len(forecast_timestamps) != pred_len:
        return ""
    if not _looks_like_datetime_timestamps(forecast_timestamps):
        return ""
    targets_str = "\n".join([f"{i + 1}. {_format_timestamp(forecast_timestamps[i])}" for i in range(pred_len)])
    return (
        "FORECAST TARGET TIMESTAMPS (next steps, in order):\n"
        f"{targets_str}\n\n"
        f"IMPORTANT: Your {pred_len} forecast numbers MUST correspond to these timestamps in the same order.\n\n"
        f"IMPORTANT: When referring to dates, ONLY use the timestamps listed above. Do not invent dates.\n\n"
    )


def _forecast_instruction(pred_len: int, reasoning: bool) -> str:
    if reasoning:
        return (
            f"First, briefly explain your reasoning: identify key trends, "
            f"predictive signals, or regime changes that inform your forecast.\n"
            f"Then, on the final line, output exactly {pred_len} comma-separated "
            f"numbers representing your forecast.\n\n"
            f"Forecast:"
        )
    return (
        f"IMPORTANT: Output ONLY {pred_len} comma-separated numbers "
        f"representing your forecast. Do not include any explanations or other text.\n\n"
        f"Forecast:"
    )


def _lean_forecast_instruction(pred_len: int, reasoning: bool) -> str:
    if reasoning:
        return (
            f"Briefly explain your reasoning, then on the final line output "
            f"exactly {pred_len} comma-separated numbers.\n\n"
            f"Forecast:"
        )
    return (
        f"Output exactly {pred_len} comma-separated numbers. "
        f"No other text.\n\n"
        f"Forecast:"
    )


def _build_target_info_section(
    target_description: str | None = None,
    source_dataset: str | None = None,
    include_text: bool = True,
) -> str:
    """Build a short section describing the target variable and source."""
    parts: list[str] = []
    if target_description:
        parts.append(f"Target: {target_description}")
    if source_dataset and not include_text:
        parts.append(f"Source: {source_dataset}")
    if not parts:
        return ""
    return "\n".join(parts) + "\n"


def _build_lean_prompt(
    text_list: list,
    values: list[float],
    pred_len: int,
    include_text: bool = True,
    timestamps: list | None = None,
    forecast_timestamps: list | None = None,
    reasoning: bool = False,
    target_description: str | None = None,
    source_dataset: str | None = None,
) -> str:
    """CiK-inspired lean prompt: compact values, XML tags, minimal instructions."""
    seq_len = len(values)
    use_ts = timestamps is not None and len(timestamps) == seq_len

    if use_ts:
        history_str = "\n".join(
            f"({_format_timestamp(timestamps[i])}, {values[i]:.6g})"
            for i in range(seq_len)
        )
    else:
        history_str = ", ".join(f"{v:.6g}" for v in values)

    # Build context from text annotations (only non-empty ones, last 10)
    context_parts = []
    if include_text:
        text_window = min(10, seq_len)
        for i in range(text_window):
            idx = seq_len - text_window + i
            t = text_list[idx] if idx < len(text_list) else None
            if t and isinstance(t, str) and t.strip():
                label = _format_timestamp(timestamps[idx]) if use_ts else f"Step {idx + 1}"
                context_parts.append(f"{label}: {t}")

    # Forecast target timestamps
    target_str = ""
    if (forecast_timestamps and len(forecast_timestamps) == pred_len
            and _looks_like_datetime_timestamps(forecast_timestamps)):
        target_str = ", ".join(_format_timestamp(forecast_timestamps[i]) for i in range(pred_len))

    instruction = _lean_forecast_instruction(pred_len, reasoning)

    parts = [f"I have a time series forecasting task for you.\n"]

    target_info = _build_target_info_section(target_description, source_dataset, include_text)
    if target_info:
        parts.append(target_info)

    if context_parts:
        parts.append(f"<context>\n" + "\n".join(context_parts) + "\n</context>\n")

    parts.append(f"<history>\n{history_str}\n</history>\n")

    if target_str:
        parts.append(f"Predict the next {pred_len} values (at timestamps: {target_str}).\n")
    else:
        parts.append(f"Predict the next {pred_len} values.\n")

    parts.append(
        f"Example output format: 1.23, 4.56, 7.89\n\n"
        f"{instruction}"
    )

    return "\n".join(parts)


def build_forecast_prompt(
    text_list: list,
    values: list[float],
    pred_len: int,
    include_text: bool = True,
    timestamps: list | None = None,
    forecast_timestamps: list | None = None,
    reasoning: bool = False,
    include_reasoning: bool | None = None,
    prompt_style: str = "default",
    target_description: str | None = None,
    source_dataset: str | None = None,
    text_window: int = 10,
) -> str:
    """Build the LLM forecast prompt.

    Uses actual (unscaled) values in the prompt, matching eval/baselines/gpt_forecast.py.
    When *reasoning* is True the model is asked to explain before the forecast.

    prompt_style:
        "default" - original verbose prompt
        "lean"    - CiK-inspired compact prompt with XML tags
    target_description:
        Optional one-liner describing what the target variable represents.
    source_dataset:
        Optional dataset source name; shown when *include_text* is False
        to give the model semantic grounding.
    """
    if include_reasoning is not None:
        reasoning = bool(include_reasoning)

    if prompt_style == "lean":
        return _build_lean_prompt(
            text_list, values, pred_len,
            include_text=include_text,
            timestamps=timestamps,
            forecast_timestamps=forecast_timestamps,
            reasoning=reasoning,
            target_description=target_description,
            source_dataset=source_dataset,
        )

    seq_len = len(values)

    all_values_str = "\n".join([f"{v:.4f}" for v in values])
    if timestamps is not None and len(timestamps) == len(values):
        all_values_with_time_str = "\n".join(
            [f"{_format_timestamp(timestamps[i])}: {values[i]:.4f}" for i in range(len(values))]
        )
    else:
        all_values_with_time_str = all_values_str

    forecast_targets_section = _format_forecast_targets_section(pred_len, forecast_timestamps)
    instruction = _forecast_instruction(pred_len, reasoning)
    target_info = _build_target_info_section(target_description, source_dataset, include_text)

    if not include_text:
        return f"""You are a time series forecasting expert. Analyze the historical data to predict future values.

{target_info}FULL TIME SERIES ({seq_len} timesteps):
[
{all_values_with_time_str}
]

TASK:
Based on the full time series pattern, provide a forecast for the next {pred_len} steps.

{forecast_targets_section}{instruction}"""

    text_window = min(text_window, seq_len)
    detailed_context_items = []
    use_ts = timestamps is not None and len(timestamps) == len(values)
    for i in range(text_window):
        idx = seq_len - text_window + i
        t = text_list[idx] if idx < len(text_list) else None
        v = values[idx]
        ts_label = _format_timestamp(timestamps[idx]) if use_ts else f"Timestep {idx + 1}"
        if t and isinstance(t, str) and t.strip():
            detailed_context_items.append(f"{ts_label} (value: {v:.4f}): {t}")
        else:
            detailed_context_items.append(f"{ts_label} (value: {v:.4f}): No annotation")
    detailed_context = "\n".join(detailed_context_items)

    return f"""You are a time series forecasting expert. Analyze the historical data with text annotations to predict future values.

{target_info}FULL TIME SERIES ({seq_len} timesteps):
[
{all_values_with_time_str}
]

DETAILED CONTEXT (most recent {text_window} timesteps with text annotations):
{detailed_context}

TASK:
Analyze the full time series pattern and the text annotations for:
1. FACTUAL PATTERNS: What trends, events, or behaviors are described in the recent history?
2. PREDICTIVE SIGNALS: Are there any forward-looking statements, expectations, or indicators of future change?

Based on this analysis, provide a forecast for the next {pred_len} steps.

{forecast_targets_section}{instruction}"""
