#!/usr/bin/env python3
"""Replicate Table 2 Panel A descriptive statistics (auditor specialist indicators)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


PAPER_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "National Specialist": {"d1_mean": 0.116, "d1_sd": 0.320, "d2_mean": 0.214, "d2_sd": 0.410},
    "City Specialist": {"d1_mean": 0.350, "d1_sd": 0.477, "d2_mean": 0.327, "d2_sd": 0.469},
    "Both National and City Specialist": {"d1_mean": 0.076, "d1_sd": 0.265, "d2_mean": 0.124, "d2_sd": 0.330},
    "National Specialist Only": {"d1_mean": 0.040, "d1_sd": 0.195, "d2_mean": 0.090, "d2_sd": 0.287},
    "City Specialist Only": {"d1_mean": 0.274, "d1_sd": 0.446, "d2_mean": 0.203, "d2_sd": 0.402},
}

ROW_SPECS: List[Tuple[str, str, str]] = [
    ("National Specialist", "nat_spec_d1", "nat_spec_d2"),
    ("City Specialist", "city_spec_d1", "city_spec_d2"),
    ("Both National and City Specialist", "both_d1", "both_d2"),
    ("National Specialist Only", "nat_only_d1", "nat_only_d2"),
    ("City Specialist Only", "city_only_d1", "city_only_d2"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replicate Table 2 Panel A descriptive statistics.")
    p.add_argument(
        "--input",
        default="data/processed/table5_input.csv",
        help="Input sample for |DACC| analysis (paper N=13,771).",
    )
    p.add_argument("--output-csv", default="outputs/table2_panelA_replication.csv")
    p.add_argument("--output-md", default="outputs/table2_panelA_replication.md")
    return p.parse_args()


def f3(x: float) -> float:
    return round(float(x), 3)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)

    needed = [c for _, c1, c2 in ROW_SPECS for c in [c1, c2]]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"Input is missing required columns: {missing}")

    n = len(df)
    rows = []
    for label, c1, c2 in ROW_SPECS:
        d1_mean = df[c1].mean()
        d1_sd = df[c1].std(ddof=1)
        d2_mean = df[c2].mean()
        d2_sd = df[c2].std(ddof=1)

        paper = PAPER_BENCHMARKS[label]
        rows.append(
            {
                "variable": label,
                "n_sample": n,
                "d1_mean": f3(d1_mean),
                "d1_sd": f3(d1_sd),
                "d2_mean": f3(d2_mean),
                "d2_sd": f3(d2_sd),
                "paper_d1_mean": paper["d1_mean"],
                "paper_d1_sd": paper["d1_sd"],
                "paper_d2_mean": paper["d2_mean"],
                "paper_d2_sd": paper["d2_sd"],
                "diff_d1_mean": f3(d1_mean - paper["d1_mean"]),
                "diff_d1_sd": f3(d1_sd - paper["d1_sd"]),
                "diff_d2_mean": f3(d2_mean - paper["d2_mean"]),
                "diff_d2_sd": f3(d2_sd - paper["d2_sd"]),
            }
        )

    out = pd.DataFrame(rows)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    md_lines = [
        "# TABLE 2 Panel A (Replication Check)",
        "",
        f"Input sample: `{args.input}`",
        "",
        f"N = {n} (paper benchmark N = 13,771)",
        "",
        out.to_markdown(index=False),
    ]
    md_path = Path(args.output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines))

    print(out_path)
    print(md_path)


if __name__ == "__main__":
    main()
