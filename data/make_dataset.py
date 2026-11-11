"""Deterministic synthetic customer-churn dataset generator.

Two dataset versions are used to demonstrate DVC provenance:
  v1 -> 1500 rows, seed 11
  v2 -> 3000 rows, seed 22, slightly drifted feature distribution
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

VERSIONS = {"v1": {"rows": 1500, "seed": 11, "drift": 0.0}, "v2": {"rows": 3000, "seed": 22, "drift": 0.35}}
FEATURES = ["tenure_months", "monthly_charges", "support_tickets", "contract_months", "usage_hours"]


def generate(version: str) -> pd.DataFrame:
    cfg = VERSIONS[version]
    rng = np.random.default_rng(cfg["seed"])
    n = cfg["rows"]
    drift = cfg["drift"]
    tenure = rng.integers(1, 72, n)
    monthly = rng.normal(65 + 10 * drift, 20, n).clip(15, 140)
    tickets = rng.poisson(1.5 + drift, n)
    contract = rng.choice([1, 12, 24], n, p=[0.55, 0.3, 0.15])
    usage = rng.normal(40 - 5 * drift, 12, n).clip(0, 100)
    logit = (
        -1.2 - 0.045 * tenure + 0.02 * (monthly - 65) + 0.45 * tickets - 0.06 * (contract - 1) / 11 * 6 - 0.03 * (usage - 40)
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n) < prob).astype(int)
    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": monthly.round(2),
            "support_tickets": tickets,
            "contract_months": contract,
            "usage_hours": usage.round(2),
            "churn": churn,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=sorted(VERSIONS), required=True)
    parser.add_argument("--out", default="data/churn.csv")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    generate(args.version).to_csv(out, index=False)
    print(f"wrote {out} ({args.version})")


if __name__ == "__main__":
    main()
