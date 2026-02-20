#!/usr/bin/env python3
"""Compare replication descriptive stats to paper Table 3 Panel A/C."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


PANEL_A_PAPER: Dict[str, Tuple[float, float]] = {
    "abs_dacc": (0.104, 0.134),
    "dacc": (0.005, 0.164),
    "size": (5.470, 2.286),
    "sigma_cfo": (0.195, 0.652),
    "cfo": (-0.001, 0.298),
    "lev": (0.168, 0.230),
    "loss": (0.395, 0.489),
    "mb": (3.022, 5.653),
    "lit": (0.258, 0.438),
    "altman": (-0.069, 7.324),
    "tenure_ln": (1.803, 0.737),
    # Paper reports ACCR_1 and |ACCR_1|; code has ta_lag_abs for |ACCR_1|.
    "ta_lag_abs": (0.126, 0.207),
    "big4": (0.694, 0.461),
    "sec_tier": (0.102, 0.303),
}


PANEL_C_PAPER: Dict[str, Tuple[float, float]] = {
    "gc": (0.299, 0.458),
    "size": (3.700, 1.939),
    "sigma_earn": (3.311, 18.372),
    "lev": (0.200, 0.476),
    "loss": (0.859, 0.348),
    "roa": (-0.384, 0.395),
    "mb": (3.258, 16.872),
    "lit": (0.288, 0.453),
    "altman": (-1.524, 2.956),
    "tenure_ln": (1.702, 0.738),
    "accr": (-0.773, 3.949),
    "big4": (0.449, 0.497),
    "sec_tier": (0.106, 0.308),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose Table 3 Panel A/C descriptive-stat gaps.")
    p.add_argument("--t5", default="data/processed/table5_input.csv")
    p.add_argument("--t8", default="data/processed/table8_input.csv")
    p.add_argument("--out", default="outputs/table3_panelA_C_compare.csv")
    return p.parse_args()


def summarize(df: pd.DataFrame, panel: str, paper: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    rows = []
    for var, (paper_mean, paper_sd) in paper.items():
        if var in df.columns:
            s = pd.to_numeric(df[var], errors="coerce")
            rep_mean = float(s.mean())
            rep_sd = float(s.std(ddof=1))
            n = int(s.notna().sum())
        else:
            rep_mean = float("nan")
            rep_sd = float("nan")
            n = 0
        rows.append(
            {
                "panel": panel,
                "var": var,
                "N_nonmissing": n,
                "rep_mean": rep_mean,
                "paper_mean": paper_mean,
                "diff_mean": rep_mean - paper_mean,
                "rep_sd": rep_sd,
                "paper_sd": paper_sd,
                "diff_sd": rep_sd - paper_sd,
                "abs_mean_gap": abs(rep_mean - paper_mean),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    t5 = pd.read_csv(args.t5)
    t8 = pd.read_csv(args.t8)

    a = summarize(t5, "A", PANEL_A_PAPER)
    c = summarize(t8, "C", PANEL_C_PAPER)
    out = pd.concat([a, c], ignore_index=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Panel A N: {len(t5)} (paper 13,771)")
    print(f"Panel C N: {len(t8)} (paper 4,969)")
    print("\nTop 6 mean gaps in Panel A:")
    print(
        a.sort_values("abs_mean_gap", ascending=False)
        .head(6)[["var", "rep_mean", "paper_mean", "diff_mean", "rep_sd", "paper_sd", "diff_sd"]]
        .to_string(index=False)
    )
    print("\nTop 6 mean gaps in Panel C:")
    print(
        c.sort_values("abs_mean_gap", ascending=False)
        .head(6)[["var", "rep_mean", "paper_mean", "diff_mean", "rep_sd", "paper_sd", "diff_sd"]]
        .to_string(index=False)
    )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
