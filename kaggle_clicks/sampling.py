from __future__ import annotations

import pandas as pd


def deterministic_pct_mask(values: pd.Series, pct: int) -> pd.Series:
    if not (0 < pct <= 100):
        raise ValueError(f"pct must be in [1,100], got {pct}")
    h = pd.util.hash_pandas_object(values, index=False).astype("uint64")
    return (h % 100) < pct
