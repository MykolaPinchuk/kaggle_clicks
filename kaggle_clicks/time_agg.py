from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeAggConfig:
    entities: tuple[str, ...] = ("device_ip", "device_id", "app_id", "site_id")
    windows: tuple[int, ...] = (1, 6, 24)
    alpha: float = 1.0
    beta: float = 10.0


class _WindowState:
    __slots__ = ("window", "events", "imps", "clicks")

    def __init__(self, window: int) -> None:
        self.window = int(window)
        self.events: deque[tuple[int, float, float]] = deque()
        self.imps: float = 0.0
        self.clicks: float = 0.0

    def drop_until(self, hour_idx: int) -> None:
        cutoff = hour_idx - self.window
        while self.events and self.events[0][0] < cutoff:
            _, imps, clicks = self.events.popleft()
            self.imps -= imps
            self.clicks -= clicks

    def snapshot(self) -> tuple[float, float]:
        return self.imps, self.clicks

    def add(self, hour_idx: int, imps: float, clicks: float) -> None:
        self.events.append((hour_idx, imps, clicks))
        self.imps += imps
        self.clicks += clicks


@dataclass
class _EntityState:
    windows: dict[int, _WindowState]
    last_seen_hour: int | None = None

    def advance(self, hour_idx: int) -> None:
        for state in self.windows.values():
            state.drop_until(hour_idx)

    def snapshot(self, hour_idx: int, alpha: float, beta: float) -> tuple[dict[int, tuple[float, float]], float]:
        self.advance(hour_idx)
        stats: dict[int, tuple[float, float]] = {}
        for window, state in self.windows.items():
            imps, clicks = state.snapshot()
            stats[window] = (
                float(np.log1p(imps)),
                float((clicks + alpha) / (imps + alpha + beta)),
            )
        recency = np.nan if self.last_seen_hour is None else float(hour_idx - self.last_seen_hour)
        return stats, recency

    def add_counts(self, hour_idx: int, imps: float, clicks: float) -> None:
        for state in self.windows.values():
            state.add(hour_idx, imps, clicks)
        self.last_seen_hour = hour_idx


def _hour_index(series: pd.Series) -> np.ndarray:
    base = series.min()
    if pd.isna(base):
        raise ValueError("Cannot compute hour index on empty series.")
    diffs = series.values.astype("datetime64[ns]") - base.to_datetime64()
    return (diffs / np.timedelta64(1, "h")).astype(np.int64)


def add_time_agg_features(df: pd.DataFrame, cfg: TimeAggConfig | None = None) -> tuple[pd.DataFrame, list[str]]:
    if cfg is None:
        cfg = TimeAggConfig()
    if not cfg.entities:
        return df, []
    for col in cfg.entities:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
    if "hour_dt" not in df.columns:
        raise ValueError("DataFrame must contain 'hour_dt'.")

    if not df["hour_dt"].is_monotonic_increasing:
        raise ValueError("DataFrame must be sorted by hour_dt in ascending order.")
    hour_idx = _hour_index(df["hour_dt"])

    n = len(df)
    click_arr = df["click"].to_numpy(dtype=np.float32, copy=False)
    entity_arrays = {col: df[col].to_numpy(copy=False) for col in cfg.entities}

    feature_arrays: dict[str, np.ndarray] = {}
    default_ctr = float(cfg.alpha / (cfg.alpha + cfg.beta))
    for col in cfg.entities:
        feature_arrays[f"{col}__recency_hours"] = np.full(n, np.nan, dtype=np.float32)
        for window in cfg.windows:
            feature_arrays[f"{col}__imps_log_w{window}h"] = np.zeros(n, dtype=np.float32)
            feature_arrays[f"{col}__ctr_w{window}h"] = np.full(n, default_ctr, dtype=np.float32)

    states: dict[str, dict[Any, _EntityState]] = {col: {} for col in cfg.entities}
    pending: dict[str, dict[Any, list[float]]] = {col: {} for col in cfg.entities}

    def flush(hour: int) -> None:
        for col in cfg.entities:
            col_pending = pending[col]
            if not col_pending:
                continue
            col_states = states[col]
            for key, (imps, clicks) in col_pending.items():
                state = col_states.setdefault(key, _EntityState({w: _WindowState(w) for w in cfg.windows}))
                state.add_counts(hour, imps, clicks)
            pending[col] = {}

    current_hour = None
    for idx in range(n):
        hour = int(hour_idx[idx])
        if current_hour is None:
            current_hour = hour
        elif hour != current_hour:
            flush(current_hour)
            current_hour = hour

        click = float(click_arr[idx])

        for col in cfg.entities:
            key = entity_arrays[col][idx]
            col_states = states[col]
            state = col_states.get(key)
            if state is None:
                state = _EntityState({w: _WindowState(w) for w in cfg.windows})
                col_states[key] = state

            stats_by_window, recency = state.snapshot(hour, cfg.alpha, cfg.beta)
            feature_arrays[f"{col}__recency_hours"][idx] = np.float32(recency)

            for window, (log_imps, ctr) in stats_by_window.items():
                feature_arrays[f"{col}__imps_log_w{window}h"][idx] = np.float32(log_imps)
                feature_arrays[f"{col}__ctr_w{window}h"][idx] = np.float32(ctr)

            pend = pending[col].get(key)
            if pend is None:
                pending[col][key] = [1.0, click]
            else:
                pend[0] += 1.0
                pend[1] += click

    if current_hour is not None:
        flush(current_hour)

    output = df.copy()
    new_cols: list[str] = []
    for name, values in feature_arrays.items():
        output[name] = values
        new_cols.append(name)

    return output, new_cols
