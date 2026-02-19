#!/usr/bin/env python3
"""A/B diagnostics for ALTMAN/MB/ROA construction against paper Table 3 targets."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


PANEL_A_TARGETS: Dict[str, Tuple[float, float]] = {
    "mb": (3.022, 5.653),
    "altman": (-0.069, 7.324),
}

PANEL_C_TARGETS: Dict[str, Tuple[float, float]] = {
    "mb": (3.258, 16.872),
    "altman": (-1.524, 2.956),
    "roa": (-0.384, 0.395),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose ALTMAN/MB/ROA formula variants.")
    p.add_argument("--t5", default="data/processed/table5_input.csv")
    p.add_argument("--t8", default="data/processed/table8_input.csv")
    p.add_argument("--comp", default="data/raw/compustat/funda_1996_2009_manual.dta")
    p.add_argument("--out", default="outputs/control_formula_ab_compare.csv")
    return p.parse_args()


def norm_cik(v) -> str:
    if pd.isna(v):
        return ""
    s = re.sub(r"\D", "", str(v))
    s = s.lstrip("0")
    return s if s else "0"


def winsorize(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().empty:
        return s
    return s.clip(s.quantile(0.01), s.quantile(0.99))


def load_comp(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".dta":
        d = pd.read_stata(path, convert_categoricals=False)
    else:
        d = pd.read_csv(path, dtype={"cik": "string"}, low_memory=False)
    d.columns = [c.lower() for c in d.columns]
    keep = [
        "cik",
        "fyear",
        "indfmt",
        "datafmt",
        "consol",
        "at",
        "lt",
        "ni",
        "ib",
        "csho",
        "prcl_f",
        "ceq",
        "seq",
        "act",
        "lct",
        "re",
        "ebit",
        "sale",
    ]
    keep = [c for c in keep if c in d.columns]
    d = d[keep].copy()
    for c in [
        "fyear",
        "at",
        "lt",
        "ni",
        "ib",
        "csho",
        "prcl_f",
        "ceq",
        "seq",
        "act",
        "lct",
        "re",
        "ebit",
        "sale",
    ]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    for col, val in [("indfmt", "INDL"), ("datafmt", "STD"), ("consol", "C")]:
        if col in d.columns:
            d = d[d[col].astype(str).str.upper() == val]
    d = d[(d["fyear"] >= 1996) & (d["fyear"] <= 2009)].copy()
    d["cik_norm"] = d["cik"].apply(norm_cik)
    d = d.sort_values(["cik_norm", "fyear"])
    d["lag_at"] = d.groupby("cik_norm")["at"].shift(1)
    d["mve"] = d["csho"].abs() * d["prcl_f"].abs()

    # MB variants
    d["mb_at_lt"] = d["mve"] / (d["at"] - d["lt"])
    if "ceq" in d.columns:
        d["mb_ceq"] = d["mve"] / d["ceq"]
    if "seq" in d.columns:
        d["mb_seq"] = d["mve"] / d["seq"]

    # ROA variants
    d["roa_ni_at"] = d["ni"] / d["at"]
    d["roa_ib_at"] = d["ib"] / d["at"]
    d["roa_ni_lag_at"] = d["ni"] / d["lag_at"]
    d["roa_ib_lag_at"] = d["ib"] / d["lag_at"]

    # Altman building blocks
    x1 = (d["act"] - d["lct"]) / d["at"] if {"act", "lct", "at"}.issubset(d.columns) else np.nan
    x2 = d["re"] / d["at"] if {"re", "at"}.issubset(d.columns) else np.nan
    x3 = d["ebit"] / d["at"] if {"ebit", "at"}.issubset(d.columns) else np.nan
    x4_mve = d["mve"] / d["lt"] if {"lt"}.issubset(d.columns) else np.nan
    x4_bve = (d["at"] - d["lt"]) / d["lt"] if {"at", "lt"}.issubset(d.columns) else np.nan
    x5 = d["sale"] / d["at"] if {"sale", "at"}.issubset(d.columns) else np.nan

    d["altman_1968_mve"] = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4_mve + 1.0 * x5
    d["altman_1968_bve"] = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4_bve + 1.0 * x5
    # Commonly used revised forms for private/non-manufacturing settings.
    d["altman_zprime_private"] = 0.717 * x1 + 0.847 * x2 + 3.107 * x3 + 0.420 * x4_bve + 0.998 * x5
    d["altman_zdoubleprime_nonman"] = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4_bve

    keep2 = [
        "cik_norm",
        "fyear",
        "mb_at_lt",
        "mb_ceq",
        "mb_seq",
        "roa_ni_at",
        "roa_ib_at",
        "roa_ni_lag_at",
        "roa_ib_lag_at",
        "altman_1968_mve",
        "altman_1968_bve",
        "altman_zprime_private",
        "altman_zdoubleprime_nonman",
    ]
    keep2 = [c for c in keep2 if c in d.columns]
    return d[keep2].copy()


def evaluate_variants(
    panel_name: str,
    sample: pd.DataFrame,
    comp_var: pd.DataFrame,
    targets: Dict[str, Tuple[float, float]],
) -> pd.DataFrame:
    s = sample.copy()
    s["cik_norm"] = s["cik_norm"].astype(str).str.replace(r"\D", "", regex=True).str.lstrip("0").replace("", "0")
    s["fiscal_year"] = pd.to_numeric(s["fiscal_year"], errors="coerce")

    merged = s[["cik_norm", "fiscal_year"]].drop_duplicates().merge(
        comp_var,
        left_on=["cik_norm", "fiscal_year"],
        right_on=["cik_norm", "fyear"],
        how="left",
    )

    var_groups: Dict[str, List[str]] = {
        "mb": [c for c in ["mb_at_lt", "mb_ceq", "mb_seq"] if c in merged.columns],
        "roa": [c for c in ["roa_ni_at", "roa_ib_at", "roa_ni_lag_at", "roa_ib_lag_at"] if c in merged.columns],
        "altman": [
            c
            for c in [
                "altman_1968_mve",
                "altman_1968_bve",
                "altman_zprime_private",
                "altman_zdoubleprime_nonman",
            ]
            if c in merged.columns
        ],
    }

    rows = []
    for concept, (paper_mean, paper_sd) in targets.items():
        for formula in var_groups.get(concept, []):
            x = winsorize(merged[formula])
            rep_mean = float(x.mean())
            rep_sd = float(x.std(ddof=1))
            rows.append(
                {
                    "panel": panel_name,
                    "concept": concept,
                    "formula": formula,
                    "n_nonmissing": int(x.notna().sum()),
                    "rep_mean": rep_mean,
                    "paper_mean": paper_mean,
                    "diff_mean": rep_mean - paper_mean,
                    "rep_sd": rep_sd,
                    "paper_sd": paper_sd,
                    "diff_sd": rep_sd - paper_sd,
                    "score_abs_mean_plus_sd": abs(rep_mean - paper_mean) + abs(rep_sd - paper_sd),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    t5 = pd.read_csv(args.t5)
    t8 = pd.read_csv(args.t8)
    comp = load_comp(Path(args.comp))

    a = evaluate_variants("A", t5, comp, PANEL_A_TARGETS)
    c = evaluate_variants("C", t8, comp, PANEL_C_TARGETS)
    out = pd.concat([a, c], ignore_index=True)
    out = out.sort_values(["panel", "concept", "score_abs_mean_plus_sd"], ascending=[True, True, True])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print("\nBest formulas by panel/concept:")
    best = out.groupby(["panel", "concept"], as_index=False).first()
    print(best[["panel", "concept", "formula", "rep_mean", "paper_mean", "rep_sd", "paper_sd", "score_abs_mean_plus_sd"]].to_string(index=False))


if __name__ == "__main__":
    main()

