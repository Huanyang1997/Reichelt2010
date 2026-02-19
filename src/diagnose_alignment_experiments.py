#!/usr/bin/env python3
"""Run alignment diagnostics against paper Table 3/5/8 under multiple scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


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

PAPER_T5_M3 = {
    1: {"both": -0.009, "nat_only": -0.005, "city_only": -0.004},
    2: {"both": -0.011, "nat_only": -0.004, "city_only": -0.003},
}
PAPER_T8_M3 = {
    1: {"both": 0.590, "nat_only": 0.286, "city_only": 0.240},
    2: {"both": 0.483, "nat_only": 0.258, "city_only": 0.223},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose Table 3/5/8 gaps under alignment scenarios.")
    p.add_argument("--t5", default="data/processed/table5_input.csv")
    p.add_argument("--t8", default="data/processed/table8_input.csv")
    p.add_argument("--comp", default="data/raw/compustat/funda_1996_2009_manual.dta")
    p.add_argument("--out", default="outputs/diagnose_alignment_scenarios.csv")
    return p.parse_args()


def load_comp(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".dta":
        d = pd.read_stata(path, convert_categoricals=False)
    else:
        d = pd.read_csv(path, dtype={"cik": "string"}, low_memory=False)
    d.columns = [c.lower() for c in d.columns]
    d["cik"] = d["cik"].astype("string")
    for c in ["fyear", "at", "lt"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    for col, val in [("indfmt", "INDL"), ("datafmt", "STD"), ("consol", "C")]:
        if col in d.columns:
            d = d[d[col].astype(str).str.upper() == val]
    d = d[(d["fyear"] >= 1996) & (d["fyear"] <= 2009)].copy()
    d["cik_norm"] = d["cik"].astype(str).str.replace(r"\D", "", regex=True).str.lstrip("0")
    d["cik_norm"] = d["cik_norm"].replace("", "0")
    d = d.sort_values(["cik_norm", "fyear"])
    d["lag_at"] = d.groupby("cik_norm")["at"].shift(1)
    return d[["cik_norm", "fyear", "at", "lt", "lag_at"]].dropna(subset=["cik_norm", "fyear"])


def winsorize(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().empty:
        return s
    lo = s.quantile(0.01)
    hi = s.quantile(0.99)
    return s.clip(lo, hi)


def summarize_panel(df: pd.DataFrame, paper: Dict[str, Tuple[float, float]], panel: str) -> List[Dict[str, object]]:
    out = []
    for var, (paper_mean, paper_sd) in paper.items():
        if var not in df.columns:
            out.append(
                {
                    "panel": panel,
                    "var": var,
                    "rep_mean": np.nan,
                    "paper_mean": paper_mean,
                    "diff_mean": np.nan,
                    "rep_sd": np.nan,
                    "paper_sd": paper_sd,
                    "diff_sd": np.nan,
                    "abs_mean_gap": np.nan,
                }
            )
            continue
        s = pd.to_numeric(df[var], errors="coerce")
        rep_mean = float(s.mean())
        rep_sd = float(s.std(ddof=1))
        out.append(
            {
                "panel": panel,
                "var": var,
                "rep_mean": rep_mean,
                "paper_mean": paper_mean,
                "diff_mean": rep_mean - paper_mean,
                "rep_sd": rep_sd,
                "paper_sd": paper_sd,
                "diff_sd": rep_sd - paper_sd,
                "abs_mean_gap": abs(rep_mean - paper_mean),
            }
        )
    return out


def run_t5_m3(df: pd.DataFrame, dnum: int):
    y = "abs_dacc"
    x = [
        "size",
        "sigma_cfo",
        "cfo",
        "lev",
        "loss",
        "mb",
        "lit",
        "altman",
        "tenure_ln",
        "ta_lag_abs",
        "big4",
        "sec_tier",
        f"both_d{dnum}",
        f"nat_only_d{dnum}",
        f"city_only_d{dnum}",
    ]
    use = df[[y] + x + ["company_fkey"]].dropna().copy()
    if use.empty:
        return None, use
    X = sm.add_constant(use[x], has_constant="add")
    m = sm.OLS(use[y], X)
    if use["company_fkey"].nunique() >= 2:
        r = m.fit(cov_type="cluster", cov_kwds={"groups": use["company_fkey"]})
    else:
        r = m.fit()
    return r, use


def run_t8_m3(df: pd.DataFrame, dnum: int):
    controls = [
        "size",
        "sigma_earn",
        "lev",
        "loss",
        "roa",
        "mb",
        "lit",
        "altman",
        "tenure_ln",
        "accr",
        "big4",
        "sec_tier",
    ]
    use = df.copy()
    use["sic2_cat"] = use["sic2"].astype("Int64").astype(str)
    use["fiscal_year_cat"] = pd.to_numeric(use["fiscal_year"], errors="coerce").astype("Int64").astype(str)
    need = ["gc"] + controls + [f"both_d{dnum}", f"nat_only_d{dnum}", f"city_only_d{dnum}", "sic2_cat", "fiscal_year_cat", "company_fkey"]
    use = use[need].dropna().copy()
    if use.empty:
        return None, use
    f_controls = " + ".join(controls)
    formula = (
        f"gc ~ {f_controls} + both_d{dnum} + nat_only_d{dnum} + city_only_d{dnum} + "
        "C(sic2_cat) + C(fiscal_year_cat)"
    )
    m = smf.glm(formula=formula, data=use, family=sm.families.Binomial())
    try:
        if use["company_fkey"].nunique() >= 2:
            r = m.fit(cov_type="cluster", cov_kwds={"groups": use["company_fkey"]})
        else:
            r = m.fit()
    except Exception:
        return None, use
    return r, use


def apply_rewinsor(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    cols = [
        "abs_dacc",
        "dacc",
        "size",
        "sigma_cfo",
        "cfo",
        "lev",
        "mb",
        "altman",
        "tenure_ln",
        "ta_lag_abs",
        "accr",
        "sigma_earn",
        "roa",
    ]
    for c in cols:
        if c in d.columns:
            d[c] = winsorize(d[c])
    return d


def apply_denom_filter(df: pd.DataFrame, comp_den: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["cik_norm"] = (
        d["cik_norm"]
        .astype("string")
        .str.replace(r"\D", "", regex=True)
        .str.lstrip("0")
        .replace("", "0")
    )
    d["fiscal_year"] = pd.to_numeric(d["fiscal_year"], errors="coerce")
    comp = comp_den.copy()
    comp["cik_norm"] = comp["cik_norm"].astype("string")
    comp["fyear"] = pd.to_numeric(comp["fyear"], errors="coerce")

    d = d.merge(comp, left_on=["cik_norm", "fiscal_year"], right_on=["cik_norm", "fyear"], how="left")
    keep = (
        pd.to_numeric(d["lag_at"], errors="coerce") > 0
    ) & (
        pd.to_numeric(d["at"], errors="coerce") > 0
    ) & (
        pd.to_numeric(d["lt"], errors="coerce") > 0
    ) & (
        pd.to_numeric(d["at"], errors="coerce") - pd.to_numeric(d["lt"], errors="coerce") > 0
    )
    d = d[keep].copy()
    return d.drop(columns=["fyear", "at", "lt", "lag_at"], errors="ignore")


def extract_key_metrics(name: str, t5: pd.DataFrame, t8: pd.DataFrame) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    # Table 3 comparison
    rows.extend([{**r, "scenario": name, "metric": "table3"} for r in summarize_panel(t5, PANEL_A_PAPER, "A")])
    rows.extend([{**r, "scenario": name, "metric": "table3"} for r in summarize_panel(t8, PANEL_C_PAPER, "C")])

    # Table 5 model 3
    for dnum in [1, 2]:
        r, use = run_t5_m3(t5, dnum)
        for key, term in [("both", f"both_d{dnum}"), ("nat_only", f"nat_only_d{dnum}"), ("city_only", f"city_only_d{dnum}")]:
            coef = float(r.params.get(term, np.nan)) if r is not None else np.nan
            pval = float(r.pvalues.get(term, np.nan)) if r is not None else np.nan
            rows.append(
                {
                    "scenario": name,
                    "metric": "table5_m3",
                    "panel": "B",
                    "var": f"d{dnum}_{key}",
                    "rep_mean": coef,
                    "paper_mean": PAPER_T5_M3[dnum][key],
                    "diff_mean": coef - PAPER_T5_M3[dnum][key] if np.isfinite(coef) else np.nan,
                    "rep_sd": pval,
                    "paper_sd": np.nan,
                    "diff_sd": np.nan,
                    "abs_mean_gap": abs(coef - PAPER_T5_M3[dnum][key]) if np.isfinite(coef) else np.nan,
                    "n_used": len(use),
                }
            )

    # Table 8 model 3
    for dnum in [1, 2]:
        r, use = run_t8_m3(t8, dnum)
        for key, term in [("both", f"both_d{dnum}"), ("nat_only", f"nat_only_d{dnum}"), ("city_only", f"city_only_d{dnum}")]:
            coef = float(r.params.get(term, np.nan)) if r is not None else np.nan
            pval = float(r.pvalues.get(term, np.nan)) if r is not None else np.nan
            rows.append(
                {
                    "scenario": name,
                    "metric": "table8_m3",
                    "panel": "D",
                    "var": f"d{dnum}_{key}",
                    "rep_mean": coef,
                    "paper_mean": PAPER_T8_M3[dnum][key],
                    "diff_mean": coef - PAPER_T8_M3[dnum][key] if np.isfinite(coef) else np.nan,
                    "rep_sd": pval,
                    "paper_sd": np.nan,
                    "diff_sd": np.nan,
                    "abs_mean_gap": abs(coef - PAPER_T8_M3[dnum][key]) if np.isfinite(coef) else np.nan,
                    "n_used": len(use),
                }
            )

    return rows


def main() -> None:
    args = parse_args()
    t5_base = pd.read_csv(args.t5)
    t8_base = pd.read_csv(args.t8)

    comp_den = load_comp(Path(args.comp))

    scenarios = []

    # S0: baseline
    scenarios.append(("baseline", t5_base.copy(), t8_base.copy()))

    # S1: re-winsorize within equation samples
    scenarios.append(("rewinsor_final_sample", apply_rewinsor(t5_base), apply_rewinsor(t8_base)))

    # S2: denominator-validity filter
    s2_t5 = apply_denom_filter(t5_base, comp_den)
    s2_t8 = apply_denom_filter(t8_base, comp_den)
    scenarios.append(("denom_filter", s2_t5, s2_t8))

    # S3: denominator filter + re-winsorize
    scenarios.append(("denom_filter_rewinsor", apply_rewinsor(s2_t5), apply_rewinsor(s2_t8)))

    out_rows: List[Dict[str, object]] = []
    for name, t5, t8 in scenarios:
        out_rows.extend(extract_key_metrics(name, t5, t8))

    out = pd.DataFrame(out_rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    # Console summary
    print(f"Saved: {out_path}")
    print("\nScenario sample sizes:")
    for name, t5, t8 in scenarios:
        print(f"- {name}: t5={len(t5):,}, t8={len(t8):,}")

    print("\nTable 5 model3 key terms (coef, p) by scenario:")
    for name in [s[0] for s in scenarios]:
        s = out[(out["scenario"] == name) & (out["metric"] == "table5_m3")]
        print(f"\n{name}")
        for dnum in [1, 2]:
            for key in ["both", "nat_only", "city_only"]:
                r = s[s["var"] == f"d{dnum}_{key}"].iloc[0]
                print(
                    f"d{dnum}_{key}: coef={r['rep_mean']:.4f}, p={r['rep_sd']:.4f}, "
                    f"paper={r['paper_mean']:.4f}, diff={r['diff_mean']:+.4f}"
                )

    print("\nLargest Table 3 mean gaps (Panel A/C) by scenario:")
    for name in [s[0] for s in scenarios]:
        s = out[(out["scenario"] == name) & (out["metric"] == "table3")].copy()
        s = s.dropna(subset=["abs_mean_gap"]).sort_values("abs_mean_gap", ascending=False).head(5)
        print(f"\n{name}")
        print(s[["panel", "var", "rep_mean", "paper_mean", "diff_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
