"""Create a deliberately corrupted copy of the dataset (failure injection for the validation stage)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "failure_injection" / "churn_corrupt.csv"


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "churn.csv")
    df.loc[df.index % 9 == 0, "tenure_months"] = None  # ~11% of the column missing, as after a broken upstream export
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)} with {int(df['tenure_months'].isna().sum())} null tenure values")


if __name__ == "__main__":
    main()
