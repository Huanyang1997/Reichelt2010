#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence
try:
    from scipy.stats import chi2
except Exception:  # pragma: no cover
    chi2 = None


STATE_FIPS_TO_ABBR = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT",
    "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL",
    "18": "IN", "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD",
    "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT", "31": "NE",
    "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA", "54": "WV",
    "55": "WI", "56": "WY", "60": "AS", "66": "GU", "69": "MP", "72": "PR", "78": "VI",
}
US_STATE_ABBR = set(STATE_FIPS_TO_ABBR.values())


@dataclass
class Paths:
    aa_fee: Path
    aa_op: Path
    aa_during: Path
    comp_funda: Path
    output_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replicate Reichelt & Wang (2010) Table 5 and 8")
    parser.add_argument("--aa-fee", default="data/raw/audit_analytics/audit_fee_manual_2026-02-03.csv")
    parser.add_argument("--aa-op", default="data/raw/audit_analytics/revised_audit_opinions_manual_2026-02-02.csv")
    parser.add_argument("--aa-during", default="data/raw/audit_analytics/auditor_during_manual_2026-01-30.csv")
    parser.add_argument("--comp-funda", default="data/raw/compustat/funda_1996_2009_manual.dta")
    parser.add_argument(
        "--msa-place-map",
        default="data/raw/reference/msa/_check/place05-cbsa06.xls",
        help="Census place-to-CBSA mapping file (place05-cbsa06.xls).",
    )
    parser.add_argument(
        "--msa-fuzzy-threshold",
        type=float,
        default=0.90,
        help="Minimum SequenceMatcher score for state-constrained fuzzy city matching.",
    )
    parser.add_argument(
        "--msa-fuzzy-gap",
        type=float,
        default=0.03,
        help="Minimum score gap between best and second-best fuzzy city candidates.",
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--year-start", type=int, default=2003)
    parser.add_argument("--year-end", type=int, default=2007)
    parser.add_argument(
        "--tenure-source",
        choices=["during", "consecutive", "hybrid"],
        default="during",
        help="Tenure source: auditor_during only, fee/op consecutive only, or hybrid (during then fallback).",
    )
    parser.add_argument("--exclude-utility", action="store_true", help="Exclude SIC 4900-4999")
    parser.add_argument(
        "--domestic-filter",
        choices=["strict", "none"],
        default="none",
        help="Domestic sample filter in Panel A start. 'strict' keeps USA only; 'none' does not filter by country.",
    )
    parser.add_argument(
        "--comp-costat",
        choices=["all", "A"],
        default="all",
        help="Compustat COSTAT filter: all statuses or only active (A).",
    )
    parser.add_argument("--allow-reduced-model", action="store_true", help="Run without Altman if unavailable")
    parser.add_argument(
        "--stop-after",
        choices=["none", "panel", "model_data"],
        default="none",
        help="Stop after writing intermediate datasets.",
    )
    return parser.parse_args()


def norm_cik(v) -> str:
    if pd.isna(v):
        return ""
    s = re.sub(r"\D", "", str(v))
    s = s.lstrip("0")
    return s if s else "0"


def norm_text(v: str) -> str:
    if pd.isna(v):
        return ""
    s = str(v).upper()
    s = s.replace("&", " AND ")
    s = s.replace("SAINT", "ST")
    s = s.replace("FORT", "FT")
    s = re.sub(r"[^A-Z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def winsorize(s: pd.Series, q1: float = 0.01, q99: float = 0.99) -> pd.Series:
    if s.dropna().empty:
        return s
    lo = s.quantile(q1)
    hi = s.quantile(q99)
    return s.clip(lo, hi)


def winsorize_columns(df: pd.DataFrame, cols: List[str], q1: float = 0.01, q99: float = 0.99) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = winsorize(pd.to_numeric(out[c], errors="coerce"), q1=q1, q99=q99)
    return out


def truncate_1pct_rows(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    keep = pd.Series(True, index=out.index)
    for c in cols:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        if s.dropna().empty:
            continue
        lo = s.quantile(0.01)
        hi = s.quantile(0.99)
        keep = keep & s.between(lo, hi, inclusive="both")
    return out.loc[keep].copy()


def parse_place_to_cbsa(path: Path) -> pd.DataFrame:
    d = pd.read_excel(path)
    d.columns = [str(c).strip().upper() for c in d.columns]
    col_place = "PLACE"
    col_state = "STATE"
    col_cbsa = "CBSA CODE"
    col_title = "CBSA TITLE"
    col_lsad = "CBSA LSAD"
    need = [col_place, col_state, col_cbsa, col_title]
    if not set(need).issubset(d.columns):
        return pd.DataFrame(columns=["city_state_norm", "place_norm", "state", "cbsa_code", "cbsa_title"])

    m = d[need + ([col_lsad] if col_lsad in d.columns else [])].copy()
    if col_lsad in m.columns:
        m = m[m[col_lsad].astype(str).str.contains("Metropolitan", case=False, na=False)]
    m = m.dropna(subset=[col_place, col_state, col_cbsa])
    m[col_state] = m[col_state].astype(str).str.upper().str.strip()
    m[col_place] = (
        m[col_place]
        .astype(str)
        .str.replace(r"\s*\(.*?\)\s*", " ", regex=True)
        .str.strip()
        .str.replace(
            r"\s+(CITY|TOWN|VILLAGE|BOROUGH|MUNICIPALITY|CDP|CENSUS DESIGNATED PLACE)$",
            "",
            case=False,
            regex=True,
        )
        .str.strip()
    )
    m["cbsa_code"] = pd.to_numeric(m[col_cbsa], errors="coerce").astype("Int64")
    m = m.dropna(subset=["cbsa_code"]).copy()
    m["cbsa_code"] = m["cbsa_code"].astype(int).astype(str).str.zfill(5)
    m["cbsa_title"] = m[col_title].astype(str).str.strip()
    m["place_norm"] = m[col_place].apply(norm_text)
    m["state"] = m[col_state]
    m["city_state_norm"] = m.apply(
        lambda r: f"{r['place_norm']}|{r['state']}",
        axis=1,
    )
    # Resolve one city-state mapping to multiple CBSAs by choosing the most frequent.
    g = (
        m.groupby(["city_state_norm", "place_norm", "state", "cbsa_code", "cbsa_title"], as_index=False)
        .size()
        .sort_values(["city_state_norm", "size"], ascending=[True, False])
    )
    g = g.drop_duplicates(["city_state_norm"], keep="first")
    return g[["city_state_norm", "place_norm", "state", "cbsa_code", "cbsa_title"]]


def compute_consecutive_tenure(df: pd.DataFrame) -> pd.DataFrame:
    # df expected columns: company_fkey, fiscal_year, auditor_fkey
    d = df[["company_fkey", "fiscal_year", "auditor_fkey"]].dropna().copy()
    d = d.sort_values(["company_fkey", "fiscal_year"])

    def _streak(g: pd.DataFrame) -> pd.DataFrame:
        prev_aud = None
        prev_year = None
        streak = 0
        out = []
        for _, r in g.iterrows():
            aud = r["auditor_fkey"]
            yr = int(r["fiscal_year"])
            if prev_aud == aud and prev_year is not None and yr == prev_year + 1:
                streak += 1
            else:
                streak = 1
            out.append(streak)
            prev_aud = aud
            prev_year = yr
        g = g.copy()
        g["tenure_years"] = out
        g["tenure_ln"] = np.log(g["tenure_years"].astype(float))
        return g

    d = d.groupby("company_fkey", group_keys=False).apply(_streak)
    return d[["company_fkey", "fiscal_year", "tenure_years", "tenure_ln"]]


def compute_tenure_from_during(
    during: pd.DataFrame, year_start: int, year_end: int
) -> pd.DataFrame:
    need = ["company_fkey", "auditor_fkey", "event_date", "deduced_begin_date", "deduced_end_date"]
    d = during[[c for c in need if c in during.columns]].copy()
    if not {"company_fkey", "auditor_fkey"}.issubset(d.columns):
        return pd.DataFrame(columns=["company_fkey", "auditor_fkey", "fiscal_year", "tenure_years", "tenure_ln"])

    for c in ["event_date", "deduced_begin_date", "deduced_end_date"]:
        if c in d.columns:
            d[c] = pd.to_datetime(d[c], errors="coerce")
        else:
            d[c] = pd.NaT

    d["begin_date"] = d["deduced_begin_date"].fillna(d["event_date"])
    d["end_date"] = d["deduced_end_date"].fillna(d["event_date"])
    d = d.dropna(subset=["company_fkey", "auditor_fkey", "begin_date", "end_date"]).copy()
    if d.empty:
        return pd.DataFrame(columns=["company_fkey", "auditor_fkey", "fiscal_year", "tenure_years", "tenure_ln"])

    swap = d["end_date"] < d["begin_date"]
    if swap.any():
        tmp = d.loc[swap, "begin_date"].copy()
        d.loc[swap, "begin_date"] = d.loc[swap, "end_date"]
        d.loc[swap, "end_date"] = tmp

    d["begin_year"] = d["begin_date"].dt.year
    d["end_year"] = d["end_date"].dt.year
    d = d[(d["end_year"] >= year_start) & (d["begin_year"] <= year_end)].copy()
    if d.empty:
        return pd.DataFrame(columns=["company_fkey", "auditor_fkey", "fiscal_year", "tenure_years", "tenure_ln"])

    d["begin_clip"] = d["begin_year"].clip(lower=year_start, upper=year_end)
    d["end_clip"] = d["end_year"].clip(lower=year_start, upper=year_end)
    d = d[d["begin_clip"] <= d["end_clip"]].copy()
    if d.empty:
        return pd.DataFrame(columns=["company_fkey", "auditor_fkey", "fiscal_year", "tenure_years", "tenure_ln"])

    d["fiscal_year"] = d.apply(lambda r: list(range(int(r["begin_clip"]), int(r["end_clip"]) + 1)), axis=1)
    d = d.explode("fiscal_year")
    d["fiscal_year"] = pd.to_numeric(d["fiscal_year"], errors="coerce").astype("Int64")
    d = d.dropna(subset=["fiscal_year"]).copy()

    # If multiple records overlap the same company-auditor-year, use earliest inferred begin_year.
    d = d.sort_values(["company_fkey", "auditor_fkey", "fiscal_year", "begin_year"])
    d = d.drop_duplicates(["company_fkey", "auditor_fkey", "fiscal_year"], keep="first")
    d["tenure_years"] = (d["fiscal_year"].astype(int) - d["begin_year"].astype(int) + 1).clip(lower=1)
    d["tenure_ln"] = np.log(d["tenure_years"].astype(float))
    return d[["company_fkey", "auditor_fkey", "fiscal_year", "tenure_years", "tenure_ln"]]


def load_compustat(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".dta":
        d = pd.read_stata(path, convert_categoricals=False)
    else:
        d = pd.read_csv(path, dtype={"cik": "string"})
    d.columns = [c.lower() for c in d.columns]
    if "cik" in d.columns:
        d["cik"] = d["cik"].astype("string")
    return d


def specialist_flags_def1(shares: pd.DataFrame, market_cols: List[str]) -> pd.DataFrame:
    sort_cols = market_cols + ["share"]
    asc = [True] * len(market_cols) + [False]
    s = shares.sort_values(sort_cols, ascending=asc)
    top = s.groupby(market_cols, as_index=False).nth(0).reset_index(drop=True)
    second = s.groupby(market_cols, as_index=False).nth(1).reset_index(drop=True)
    second = second[market_cols + ["share"]].rename(columns={"share": "share2"})
    top = top.merge(second, on=market_cols, how="left")
    top["share2"] = top["share2"].fillna(0.0)
    top["is_specialist"] = (top["share"] - top["share2"]) >= 0.10
    top = top[top["is_specialist"]]
    return top[market_cols + ["auditor_fkey"]]


def build_specialist_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    national = (
        d.groupby(["fiscal_year", "sic2", "auditor_fkey"], as_index=False)["audit_fees"].sum()
    )
    national["market_total"] = national.groupby(["fiscal_year", "sic2"])["audit_fees"].transform("sum")
    national["share"] = national["audit_fees"] / national["market_total"]

    city = (
        d.groupby(["fiscal_year", "sic2", "msa_code", "auditor_fkey"], as_index=False)["audit_fees"].sum()
    )
    city["market_total"] = city.groupby(["fiscal_year", "sic2", "msa_code"])["audit_fees"].transform("sum")
    city["share"] = city["audit_fees"] / city["market_total"]

    # Definition 1
    nat_d1 = specialist_flags_def1(national, ["fiscal_year", "sic2"])
    nat_d1["nat_spec_d1"] = 1
    city_d1 = specialist_flags_def1(city, ["fiscal_year", "sic2", "msa_code"])
    city_d1["city_spec_d1"] = 1

    # Definition 2
    nat_d2 = national[national["share"] > 0.30][["fiscal_year", "sic2", "auditor_fkey"]].copy()
    nat_d2["nat_spec_d2"] = 1
    city_d2 = city[city["share"] > 0.50][["fiscal_year", "sic2", "msa_code", "auditor_fkey"]].copy()
    city_d2["city_spec_d2"] = 1

    out = d[["company_fkey", "fiscal_year", "sic2", "msa_code", "auditor_fkey"]].copy()
    out = out.drop_duplicates(["company_fkey", "fiscal_year", "auditor_fkey"])

    out = out.merge(nat_d1, on=["fiscal_year", "sic2", "auditor_fkey"], how="left")
    out = out.merge(city_d1, on=["fiscal_year", "sic2", "msa_code", "auditor_fkey"], how="left")
    out = out.merge(nat_d2, on=["fiscal_year", "sic2", "auditor_fkey"], how="left")
    out = out.merge(city_d2, on=["fiscal_year", "sic2", "msa_code", "auditor_fkey"], how="left")

    for c in ["nat_spec_d1", "city_spec_d1", "nat_spec_d2", "city_spec_d2"]:
        out[c] = out[c].fillna(0).astype(int)

    for dnum in [1, 2]:
        n = f"nat_spec_d{dnum}"
        c = f"city_spec_d{dnum}"
        out[f"both_d{dnum}"] = ((out[n] == 1) & (out[c] == 1)).astype(int)
        out[f"nat_only_d{dnum}"] = ((out[n] == 1) & (out[c] == 0)).astype(int)
        out[f"city_only_d{dnum}"] = ((out[n] == 0) & (out[c] == 1)).astype(int)

    return out


def run_ols_cluster(df: pd.DataFrame, y: str, x_cols: List[str], group_col: str):
    use = df[[y] + x_cols + [group_col]].dropna().copy()
    if use.empty:
        return None, use
    X = sm.add_constant(use[x_cols], has_constant="add")
    if X.shape[1] == 0:
        return None, use
    model = sm.OLS(use[y], X)
    if use[group_col].nunique() >= 2:
        res = model.fit(cov_type="cluster", cov_kwds={"groups": use[group_col]})
    else:
        res = model.fit(cov_type="HC1")
    return res, use


def run_logit_cluster_formula(df: pd.DataFrame, formula: str, group_col: str):
    use = df.dropna(subset=[group_col]).copy()
    if use.empty:
        return None, use
    model = smf.glm(formula=formula, data=use, family=sm.families.Binomial())
    try:
        if use[group_col].nunique() >= 2:
            res = model.fit(cov_type="cluster", cov_kwds={"groups": use[group_col]})
        else:
            res = model.fit()
        # Guard against singular robust covariance returning all-NaN inference.
        if hasattr(res, "pvalues") and np.isnan(np.asarray(res.pvalues)).all():
            res = model.fit()
    except Exception:
        return None, use
    return res, use


def fmt_est(x) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x):.3f}"


def fmt_p(x) -> str:
    if x is None or pd.isna(x):
        return ""
    xv = float(x)
    if xv < 0.001:
        return "<0.001"
    return f"{xv:.3f}"


def latex_escape(s: str) -> str:
    return (
        str(s)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def term_for_definition(term_spec, definition: int) -> str:
    if isinstance(term_spec, dict):
        if definition in term_spec:
            return term_spec[definition]
        return term_spec.get(str(definition), "")
    return str(term_spec)


def build_paper_table_df(
    row_specs: List[Tuple[str, object]],
    model_results: Dict[Tuple[int, str], object],
    summary_getter,
) -> pd.DataFrame:
    model_order = [(1, "model1"), (1, "model2"), (1, "model3"), (2, "model1"), (2, "model2"), (2, "model3")]
    cols = ["Variable"]
    for d, m in model_order:
        mnum = m[-1]
        cols += [f"D{d}M{mnum}_Est", f"D{d}M{mnum}_p"]

    rows = []
    for label, term_spec in row_specs:
        out = {"Variable": label}
        for d, m in model_order:
            term = term_for_definition(term_spec, d)
            res = model_results.get((d, m))
            est_col = f"D{d}M{m[-1]}_Est"
            p_col = f"D{d}M{m[-1]}_p"
            if res is None or term == "" or term not in res.params.index:
                out[est_col] = ""
                out[p_col] = ""
            else:
                out[est_col] = fmt_est(res.params.get(term, np.nan))
                out[p_col] = fmt_p(res.pvalues.get(term, np.nan))
        rows.append(out)

    for label in ["F-value", "Adj. R2", "Likelihood ratio", "Pseudo-R2"]:
        out = {"Variable": label}
        any_val = False
        for d, m in model_order:
            est_col = f"D{d}M{m[-1]}_Est"
            p_col = f"D{d}M{m[-1]}_p"
            res = model_results.get((d, m))
            est, p = summary_getter(label, res)
            out[est_col] = fmt_est(est)
            out[p_col] = fmt_p(p)
            any_val = any_val or (out[est_col] != "" or out[p_col] != "")
        if any_val:
            rows.append(out)

    return pd.DataFrame(rows, columns=cols)


def write_paper_table_markdown(path: Path, title: str, subtitle: str, df: pd.DataFrame) -> None:
    lines = [f"# {title}", "", subtitle, "", df.to_markdown(index=False)]
    path.write_text("\n".join(lines))


def write_paper_table_latex(path: Path, title: str, subtitle: str, df: pd.DataFrame, note: str) -> None:
    model_headers = [
        (1, "Model 1"), (1, "Model 2"), (1, "Model 3"),
        (2, "Model 1"), (2, "Model 2"), (2, "Model 3"),
    ]
    lines = []
    lines.append(r"\begin{table}[!htbp]\centering")
    lines.append(r"\scriptsize")
    lines.append(rf"\caption{{{latex_escape(title)}}}")
    lines.append(rf"\textit{{{latex_escape(subtitle)}}}")
    lines.append(r"\begin{tabular}{l" + "rr" * 6 + "}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{6}{c}{Auditor Industry Specialist Definition 1} & \multicolumn{6}{c}{Auditor Industry Specialist Definition 2}\\")
    lines.append(r"\cmidrule(lr){2-7}\cmidrule(lr){8-13}")
    head2 = [" "]
    for _, mname in model_headers:
        head2.extend([mname, ""])
    lines.append(" & ".join(head2) + r"\\")
    lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(lr){12-13}")
    head3 = ["Variable"] + ["Estimate", "p-value"] * 6
    lines.append(" & ".join(head3) + r"\\")
    lines.append(r"\midrule")

    for _, r in df.iterrows():
        vals = [r["Variable"]]
        for d, m in [(1, "model1"), (1, "model2"), (1, "model3"), (2, "model1"), (2, "model2"), (2, "model3")]:
            est = str(r[f"D{d}M{m[-1]}_Est"])
            pv = str(r[f"D{d}M{m[-1]}_p"])
            if est == "nan":
                est = ""
            if pv == "nan":
                pv = ""
            est = est.replace("<", "$<$")
            pv = pv.replace("<", "$<$")
            vals.extend([est, pv])
        row_label = latex_escape(vals[0])
        body = [row_label] + vals[1:]
        lines.append(" & ".join(body) + r"\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\vspace{{0.3em}}\par\footnotesize{{{latex_escape(note)}}}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    paths = Paths(
        aa_fee=Path(args.aa_fee),
        aa_op=Path(args.aa_op),
        aa_during=Path(args.aa_during),
        comp_funda=Path(args.comp_funda),
        output_dir=Path(args.output_dir),
    )
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    counts: Dict[str, int] = {}

    # 1) Load data
    fee = pd.read_csv(
        paths.aa_fee,
        dtype={"COMPANY_FKEY": "string", "COMPANY_KEY": "string", "AUDITOR_FKEY": "string"},
    )
    op = pd.read_csv(
        paths.aa_op,
        dtype={"COMPANY_FKEY": "string", "COMPANY_KEY": "string", "AUDITOR_FKEY": "string"},
    )
    during = pd.read_csv(
        paths.aa_during,
        dtype={"COMPANY_FKEY": "string", "COMPANY_KEY": "string"},
    )
    # Keep CIK as string to avoid accidental ".0" coercion effects.
    comp = load_compustat(paths.comp_funda)
    counts["auditors_city_missing_after_fill"] = 0
    counts["auditors_state_missing_after_fill"] = 0

    fee.columns = [c.lower() for c in fee.columns]
    op.columns = [c.lower() for c in op.columns]
    during.columns = [c.lower() for c in during.columns]
    comp.columns = [c.lower() for c in comp.columns]

    # Preserve identifier keys as strings for stable matching.
    for df in [fee, op, during]:
        for k in ["company_fkey", "company_key", "auditor_fkey"]:
            if k in df.columns:
                df[k] = df[k].astype("string").str.strip()

    # Harmonize AA columns
    op = op.rename(
        columns={
            "fiscal_year_of_op": "fiscal_year",
            "fiscal_year_end_op": "fiscal_year_ended",
        }
    )

    for col in ["fiscal_year", "audit_fees", "sic_code_fkey"]:
        if col in fee.columns:
            fee[col] = pd.to_numeric(fee[col], errors="coerce")
    for col in ["fiscal_year", "going_concern", "sic_code_fkey"]:
        if col in op.columns:
            op[col] = pd.to_numeric(op[col], errors="coerce")

    fee = fee.dropna(subset=["company_fkey", "auditor_fkey", "fiscal_year", "audit_fees"]) 
    op = op.dropna(subset=["company_fkey", "auditor_fkey", "fiscal_year"]) 

    # Opinion dedupe by key keep latest filing
    if "file_date" in op.columns:
        op["file_date"] = pd.to_datetime(op["file_date"], errors="coerce")
        op = op.sort_values("file_date")
    op = op.drop_duplicates(["company_fkey", "auditor_fkey", "fiscal_year"], keep="last")

    aa = fee.merge(
        op[
            [
                "company_fkey",
                "auditor_fkey",
                "fiscal_year",
                "going_concern",
                "auditor_city",
                "auditor_state",
                "loc_state_country",
                "bus_state_country",
                "mail_state_country",
            ]
        ],
        on=["company_fkey", "auditor_fkey", "fiscal_year"],
        how="left",
    )
    # Office location is strictly from Audit Opinions: AUDITOR_CITY/AUDITOR_STATE only.
    aa["auditor_city"] = aa["auditor_city"].astype("string").str.strip()
    aa["auditor_state"] = aa["auditor_state"].astype("string").str.upper().str.strip()
    aa["auditor_zip"] = pd.NA
    counts["auditors_city_missing_after_fill"] = int(aa["auditor_city"].isna().sum())
    counts["auditors_state_missing_after_fill"] = int(aa["auditor_state"].isna().sum())

    # Keep one row per company-year (largest audit fee)
    aa = aa.sort_values(["company_fkey", "fiscal_year", "audit_fees"], ascending=[True, True, False])
    aa = aa.drop_duplicates(["company_fkey", "fiscal_year"], keep="first")
    counts["aa_company_year_raw"] = len(aa)

    # Tenure construction
    tenure_during = compute_tenure_from_during(during, args.year_start, args.year_end)
    counts["tenure_rows_from_during"] = len(tenure_during)
    try:
        tenure_during.to_csv(processed_dir / "tenure_from_auditor_during_2003_2007.csv", index=False)
    except Exception:
        pass
    if args.tenure_source == "during":
        aa = aa.merge(
            tenure_during,
            on=["company_fkey", "auditor_fkey", "fiscal_year"],
            how="left",
        )
    elif args.tenure_source == "consecutive":
        tenure_consec = compute_consecutive_tenure(aa)
        aa = aa.merge(tenure_consec, on=["company_fkey", "fiscal_year"], how="left")
    else:  # hybrid
        aa = aa.merge(
            tenure_during,
            on=["company_fkey", "auditor_fkey", "fiscal_year"],
            how="left",
        )
        tenure_consec = compute_consecutive_tenure(aa)
        aa = aa.merge(
            tenure_consec[["company_fkey", "fiscal_year", "tenure_years", "tenure_ln"]].rename(
                columns={"tenure_years": "tenure_years_consec", "tenure_ln": "tenure_ln_consec"}
            ),
            on=["company_fkey", "fiscal_year"],
            how="left",
        )
        aa["tenure_years"] = aa["tenure_years"].fillna(aa["tenure_years_consec"])
        aa["tenure_ln"] = aa["tenure_ln"].fillna(aa["tenure_ln_consec"])
        aa = aa.drop(columns=["tenure_years_consec", "tenure_ln_consec"], errors="ignore")
    counts["aa_tenure_nonmissing"] = int(aa["tenure_ln"].notna().sum())

    # Domestic flag (Compustat-only): use FIC == USA.
    # Do not rely on AA country fields for domestic screening.
    aa["aa_comp_key"] = aa["company_fkey"].fillna(aa["company_key"])
    aa["cik_norm"] = aa["aa_comp_key"].apply(norm_cik)
    comp_dom = comp.copy()
    if "fyear" in comp_dom.columns:
        comp_dom["fyear"] = pd.to_numeric(comp_dom["fyear"], errors="coerce")
    comp_dom["cik_norm"] = comp_dom["cik"].apply(norm_cik)
    if "fic" in comp_dom.columns:
        comp_dom["fic"] = comp_dom["fic"].astype(str).str.upper().str.strip()
    else:
        comp_dom["fic"] = pd.NA
    for col, val in [("indfmt", "INDL"), ("datafmt", "STD"), ("consol", "C")]:
        if col in comp_dom.columns:
            comp_dom = comp_dom[comp_dom[col].astype(str).str.upper() == val]
    comp_dom = comp_dom[["cik_norm", "fyear", "fic"]].dropna(subset=["cik_norm", "fyear"])
    comp_dom = comp_dom.sort_values(["cik_norm", "fyear", "fic"], ascending=[True, True, False])
    comp_dom = comp_dom.drop_duplicates(["cik_norm", "fyear"], keep="first")
    aa = aa.merge(
        comp_dom,
        left_on=["cik_norm", "fiscal_year"],
        right_on=["cik_norm", "fyear"],
        how="left",
    )
    aa = aa.drop(columns=["fyear"], errors="ignore")
    aa["domestic"] = aa["fic"].eq("USA")

    # SIC filtering
    aa["sic4"] = pd.to_numeric(aa["sic_code_fkey"], errors="coerce")
    aa["sic2"] = (aa["sic4"] // 100).astype("Int64")
    aa["nonfinancial"] = aa["sic4"].notna() & (~aa["sic4"].between(6000, 6999, inclusive="both"))
    if args.exclude_utility:
        aa["nonutility"] = ~aa["sic4"].between(4900, 4999, inclusive="both")
    else:
        aa["nonutility"] = True

    # MSA mapping: place05 exact + state-constrained fuzzy city matching.
    msa_place_path = Path(args.msa_place_map)
    if not msa_place_path.exists():
        raise RuntimeError(f"MSA mapping requires place map file, but not found: {msa_place_path}")
    place_map = parse_place_to_cbsa(msa_place_path)
    if place_map.empty:
        raise RuntimeError(f"Failed to parse place map or no metropolitan rows: {msa_place_path}")

    msa_map_source = "place05_exact_plus_state_fuzzy"

    aa["auditor_state"] = aa["auditor_state"].astype("string").str.upper().str.strip()
    aa["city_norm"] = aa["auditor_city"].apply(norm_text)
    aa["city_state_norm"] = aa.apply(
        lambda r: f"{r['city_norm']}|{r['auditor_state']}"
        if (pd.notna(r["city_norm"]) and r["city_norm"] and pd.notna(r["auditor_state"]) and r["auditor_state"])
        else "",
        axis=1,
    )
    exact_map = place_map[["city_state_norm", "cbsa_code", "cbsa_title"]].drop_duplicates(["city_state_norm"], keep="first")
    aa = aa.merge(
        exact_map.rename(columns={"cbsa_code": "cbsa_code_exact", "cbsa_title": "cbsa_title_exact"}),
        on="city_state_norm",
        how="left",
    )

    place_states = set(place_map["state"].dropna().astype(str).unique())
    state_to_places = (
        place_map.dropna(subset=["state", "place_norm"])
        .groupby("state")["place_norm"]
        .apply(lambda s: sorted(set(s.astype(str))))
        .to_dict()
    )
    place_lookup = {
        (str(r.place_norm), str(r.state)): (str(r.cbsa_code), str(r.cbsa_title))
        for r in place_map[["place_norm", "state", "cbsa_code", "cbsa_title"]].itertuples(index=False)
    }

    exact_hit = aa["cbsa_code_exact"].notna()
    fuzzy_base = (
        (~exact_hit)
        & aa["city_norm"].astype(str).ne("")
        & aa["auditor_state"].isin(place_states)
        & aa["auditor_state"].isin(US_STATE_ABBR)
    )

    fuzzy_rows = []
    for r in aa.loc[fuzzy_base, ["city_state_norm", "city_norm", "auditor_state"]].drop_duplicates().itertuples(index=False):
        candidates = state_to_places.get(str(r.auditor_state), [])
        if not candidates:
            continue
        best_name = ""
        best_score = -1.0
        second_score = -1.0
        for cand in candidates:
            sc = SequenceMatcher(None, str(r.city_norm), str(cand)).ratio()
            if sc > best_score:
                second_score = best_score
                best_score = sc
                best_name = cand
            elif sc > second_score:
                second_score = sc
        if best_score >= args.msa_fuzzy_threshold and (best_score - max(second_score, 0.0)) >= args.msa_fuzzy_gap:
            cbsa = place_lookup.get((best_name, str(r.auditor_state)))
            if cbsa is not None:
                fuzzy_rows.append(
                    {
                        "city_state_norm": r.city_state_norm,
                        "cbsa_code_fuzzy": cbsa[0],
                        "cbsa_title_fuzzy": cbsa[1],
                        "fuzzy_place_norm": best_name,
                        "fuzzy_score": best_score,
                    }
                )

    fuzzy_map = pd.DataFrame(fuzzy_rows).drop_duplicates(["city_state_norm"], keep="first")
    if not fuzzy_map.empty:
        aa = aa.merge(fuzzy_map, on="city_state_norm", how="left")
    else:
        aa["cbsa_code_fuzzy"] = pd.NA
        aa["cbsa_title_fuzzy"] = pd.NA
        aa["fuzzy_place_norm"] = pd.NA
        aa["fuzzy_score"] = np.nan

    aa["cbsa_code_city"] = aa["cbsa_code_exact"].fillna(aa["cbsa_code_fuzzy"])
    aa["cbsa_title_city"] = aa["cbsa_title_exact"].fillna(aa["cbsa_title_fuzzy"])
    aa["msa_code"] = aa["cbsa_code_city"]
    aa["msa_title"] = aa["cbsa_title_city"]

    aa["msa_match_source"] = np.where(
        aa["cbsa_code_exact"].notna(),
        "city_state_exact",
        np.where(aa["cbsa_code_fuzzy"].notna(), "city_state_fuzzy", ""),
    )
    aa["msa_hit_place"] = aa["cbsa_code_exact"].notna()
    aa["msa_hit_fuzzy"] = aa["cbsa_code_fuzzy"].notna()
    aa["msa_hit_any"] = aa["msa_hit_place"] | aa["msa_hit_fuzzy"]
    aa["msa_hit_combined"] = aa["msa_code"].notna()

    # Persist mapping artifacts for auditability and manual QA.
    try:
        place_map.to_csv(processed_dir / "msa_place05_cbsa06_parsed.csv", index=False)
        exact_map.to_csv(processed_dir / "msa_city_state_to_cbsa_exact_used.csv", index=False)
        fuzzy_map.to_csv(processed_dir / "msa_city_state_to_cbsa_fuzzy_used.csv", index=False)
        unmatched_city_state = (
            aa.loc[~aa["msa_hit_combined"], ["auditor_city", "auditor_state", "city_state_norm"]]
            .value_counts()
            .reset_index(name="n")
            .sort_values("n", ascending=False)
        )
        unmatched_city_state.to_csv(processed_dir / "msa_unmatched_city_state_ranked.csv", index=False)
    except Exception:
        pass

    # Base-period universe for variable construction.
    # Build features first, then do sample attrition drops.
    aa = aa[(aa["fiscal_year"] >= args.year_start) & (aa["fiscal_year"] <= args.year_end)].copy()
    aa = aa[(aa["audit_fees"] > 0)].copy()
    counts["aa_in_period"] = len(aa)
    counts["msa_hit_place_rows_pre_core"] = int(aa["msa_hit_place"].sum())
    counts["msa_hit_fuzzy_rows_pre_core"] = int(aa["msa_hit_fuzzy"].sum())
    counts["msa_hit_any_rows_pre_core"] = int(aa["msa_hit_any"].sum())
    counts["msa_hit_combined_rows_pre_core"] = int(aa["msa_hit_combined"].sum())
    counts["msa_match_source_city_rows_pre_core"] = int(aa["msa_match_source"].astype(str).str.startswith("city_state").sum())
    counts["panelA_period_posfee_obs"] = len(aa)
    counts["panelA_nonfinancial_sic_obs"] = int((aa["nonfinancial"] & aa["nonutility"]).sum())
    counts["panelA_domestic_true_obs"] = int(aa["domestic"].fillna(False).sum())
    counts["panelA_domestic_false_obs"] = int((~aa["domestic"].fillna(False)).sum())
    counts["panelA_domestic_filter_mode"] = "fic_only_post_compustat"

    # Keep domestic screening in the post-Compustat stage so panel-A attrition
    # can still report the "not in Compustat" deletion path.
    aa_core = aa[aa["nonfinancial"] & aa["nonutility"]].copy()
    counts["aa_after_core_filters"] = len(aa_core)
    aa = aa_core[aa_core["msa_code"].notna()].copy()
    counts["aa_with_msa"] = len(aa)
    counts["panelA_start_obs"] = len(aa)
    counts["panelA_start_unique_msa"] = int(aa["msa_code"].nunique())

    # Specialist indicators use panel-A start sample (before compustat/city-industry drops).
    spec = build_specialist_indicators(aa)
    aa = aa.merge(spec, on=["company_fkey", "fiscal_year", "sic2", "msa_code", "auditor_fkey"], how="left")
    for dnum in [1, 2]:
        for col in [f"both_d{dnum}", f"nat_only_d{dnum}", f"city_only_d{dnum}", f"nat_spec_d{dnum}", f"city_spec_d{dnum}"]:
            if col not in aa.columns:
                aa[col] = 0
        aa[f"both_d{dnum}"] = pd.to_numeric(aa[f"both_d{dnum}"], errors="coerce").fillna(0).astype(int)
        aa[f"nat_only_d{dnum}"] = pd.to_numeric(aa[f"nat_only_d{dnum}"], errors="coerce").fillna(0).astype(int)
        aa[f"city_only_d{dnum}"] = pd.to_numeric(aa[f"city_only_d{dnum}"], errors="coerce").fillna(0).astype(int)
        # Reconstruct model-1/model-2 indicators from model-3 partitions.
        aa[f"nat_spec_d{dnum}"] = ((aa[f"both_d{dnum}"] == 1) | (aa[f"nat_only_d{dnum}"] == 1)).astype(int)
        aa[f"city_spec_d{dnum}"] = ((aa[f"both_d{dnum}"] == 1) | (aa[f"city_only_d{dnum}"] == 1)).astype(int)

    # Prepare Compustat
    for c in [
        "fyear", "sic", "at", "dltt", "lt", "ppegt", "rect", "ib", "ni", "sale", "oancf", "xidoc",
        "csho", "prcc_f", "prcl_f", "ceq", "act", "lct", "re", "ebit"
    ]:
        if c in comp.columns:
            comp[c] = pd.to_numeric(comp[c], errors="coerce")
    if "datadate" in comp.columns:
        comp["datadate"] = pd.to_datetime(comp["datadate"], errors="coerce")

    counts["comp_unique_cik_pre_filter"] = int(comp["cik"].apply(norm_cik).nunique())
    # Standard Compustat filters when available
    for col, val in [("indfmt", "INDL"), ("datafmt", "STD"), ("consol", "C")]:
        if col in comp.columns:
            comp = comp[comp[col].astype(str).str.upper() == val]
    if args.comp_costat == "A" and "costat" in comp.columns:
        comp = comp[comp["costat"].astype(str).str.upper() == "A"]

    comp["cik_norm"] = comp["cik"].apply(norm_cik)
    comp = comp[(comp["fyear"] >= 1996) & (comp["fyear"] <= args.year_end)].copy()
    counts["comp_unique_cik_post_filter"] = int(comp["cik_norm"].nunique())
    comp = comp.sort_values(["cik_norm", "fyear"])

    # Lags and derived vars
    grp = comp.groupby("cik_norm")
    comp["lag_at"] = grp["at"].shift(1)
    comp["lag2_at"] = grp["at"].shift(2)
    comp["lag_sale"] = grp["sale"].shift(1)
    comp["lag_rect"] = grp["rect"].shift(1)
    comp["lag_ib"] = grp["ib"].shift(1)

    comp["op_cf"] = comp["oancf"] - comp["xidoc"].fillna(0)
    comp["ta"] = (comp["ib"] - comp["op_cf"]) / comp["lag_at"]
    comp["inv_lag_at"] = 1.0 / comp["lag_at"]
    comp["delta_rev"] = (comp["sale"] - comp["lag_sale"]) / comp["lag_at"]
    comp["delta_rec"] = (comp["rect"] - comp["lag_rect"]) / comp["lag_at"]
    comp["ppe_scaled"] = comp["ppegt"] / comp["lag_at"]
    
    # Paper uses average total assets for ROA denominator (Eq 1)
    comp["avg_at_lag"] = (comp["lag_at"] + comp["lag2_at"]) / 2
    comp["roa_lag"] = comp["lag_ib"] / comp["avg_at_lag"]
    
    comp["cfo"] = comp["op_cf"] / comp["lag_at"]
    comp["earn_scaled"] = comp["ib"] / comp["lag_at"]

    # Paper definition: sigma from t-4 to t (5-year window, including current year t).
    comp["sigma_cfo"] = grp["cfo"].transform(lambda s: s.rolling(5, min_periods=5).std())
    comp["sigma_earn"] = grp["earn_scaled"].transform(lambda s: s.rolling(5, min_periods=5).std())

    comp["ta_lag_abs"] = grp["ta"].shift(1).abs()
    comp["accr"] = comp["ta"]
    comp["avg_at"] = (comp["at"] + comp["lag_at"]) / 2
    comp["roa"] = comp["ib"] / comp["avg_at"]
    if "prcl_f" not in comp.columns:
        raise RuntimeError("Compustat extract missing required field 'prcl_f' for SIZE/MB/ALTMAN construction.")
    if "prcc_f" not in comp.columns:
        raise RuntimeError(
            "Compustat extract missing required field 'prcc_f' for MB construction "
            "(MB = PRCC_F * CSHO / CEQ). Please re-download funda with PRCC_F."
        )
    if "ceq" not in comp.columns:
        raise RuntimeError(
            "Compustat extract missing required field 'ceq' for MB and Altman Z''(1983) construction. "
            "Please re-download funda with CEQ."
        )
    comp["mve"] = (comp["csho"].abs()) * (comp["prcl_f"].abs())
    counts["mb_price_col_used"] = "prcc_f"
    comp["mve_prcc"] = (comp["csho"].abs()) * (comp["prcc_f"].abs())
    comp["size"] = np.log(comp["mve"].replace(0, np.nan))
    comp["lev"] = comp["dltt"] / comp["at"]
    comp["loss"] = (comp["ni"] < 0).astype(float)
    comp["mb"] = comp["mve_prcc"] / comp["ceq"]
    comp["mb"] = comp["mb"].replace([np.inf, -np.inf], np.nan)
    lt_nonpositive = comp["lt"].notna() & (comp["lt"] <= 0)
    counts["altman_lt_nonpositive_obs"] = int(lt_nonpositive.sum())

    # Litigation indicator SIC ranges
    sic = comp["sic"]
    comp["lit"] = (
        sic.between(2833, 2836, inclusive="both")
        | sic.between(3570, 3577, inclusive="both")
        | sic.between(3600, 3674, inclusive="both")
        | sic.between(5200, 5961, inclusive="both")
        | sic.between(7370, 7370, inclusive="both")
    ).astype(float)

    # Altman Z'' (1983): 6.56X1 + 3.26X2 + 6.72X3 + 1.05X4
    # where X1=(ACT-LCT)/AT, X2=RE/AT, X3=EBIT/AT, X4=CEQ/LT
    altman_inputs = ["act", "lct", "re", "ebit", "ceq", "lt", "at"]
    missing_altman = [c for c in altman_inputs if c not in comp.columns]
    if missing_altman:
        comp["altman"] = np.nan
    else:
        x4 = np.where(comp["lt"] > 0, comp["ceq"] / comp["lt"], np.nan)
        comp["altman"] = (
            6.56 * ((comp["act"] - comp["lct"]) / comp["at"])
            + 3.26 * (comp["re"] / comp["at"])
            + 6.72 * (comp["ebit"] / comp["at"])
            + 1.05 * x4
        )
        comp["altman"] = comp["altman"].replace([np.inf, -np.inf], np.nan)

    comp_use = comp[[
        "cik_norm", "fyear", "sic", "ta", "inv_lag_at", "delta_rev", "delta_rec", "ppe_scaled", "roa_lag",
        "size", "sigma_cfo", "cfo", "lev", "loss", "mb", "lit", "altman", "ta_lag_abs", "accr", "sigma_earn", "roa", "op_cf"
    ]].copy()

    # Estimate Eq(1) on Compustat full sample (2003-2007)
    eq1 = comp[(comp["fyear"] >= args.year_start) & (comp["fyear"] <= args.year_end)].copy()
    eq1["sic2"] = (eq1["sic"] // 100).astype("Int64")
    eq1 = eq1.dropna(subset=["ta", "inv_lag_at", "delta_rev", "ppe_scaled", "roa_lag", "sic2", "fyear"])
    eq1_vars = ["ta", "inv_lag_at", "delta_rev", "ppe_scaled", "roa_lag"]
    counts["eq1_rows_before_truncate"] = int(len(eq1))
    eq1 = truncate_1pct_rows(eq1, eq1_vars)
    counts["eq1_rows_after_truncate"] = int(len(eq1))

    coefs = []
    for (fy, sic2), g in eq1.groupby(["fyear", "sic2"]):
        if len(g) < 20:
            continue
        X = g[["inv_lag_at", "delta_rev", "ppe_scaled", "roa_lag"]]
        try:
            r = sm.OLS(g["ta"], X).fit()
        except Exception:
            continue
        coefs.append(
            {
                "fyear": fy,
                "sic2": int(sic2),
                "b0": r.params.get("inv_lag_at", np.nan),
                "b1": r.params.get("delta_rev", np.nan),
                "b2": r.params.get("ppe_scaled", np.nan),
                "b3": r.params.get("roa_lag", np.nan),
            }
        )
    coef_df = pd.DataFrame(coefs)

    # Merge AA with Compustat by CIK-aligned company key.
    comp_key_set = set(comp_use["cik_norm"].dropna().astype(str).unique())
    aa["aa_comp_key"] = aa["company_fkey"].fillna(aa["company_key"])
    aa["cik_norm"] = aa["aa_comp_key"].apply(norm_cik)
    counts["aa_cik_direct_matchable"] = int(aa["cik_norm"].isin(comp_key_set).sum())
    aa = aa.merge(comp_use, left_on=["cik_norm", "fiscal_year"], right_on=["cik_norm", "fyear"], how="left")
    counts["aa_comp_rows_with_ta"] = int(aa["ta"].notna().sum())
    aa["sic2"] = aa["sic2"].astype("Int64")
    aa = aa.merge(coef_df, left_on=["fiscal_year", "sic2"], right_on=["fyear", "sic2"], how="left")

    # Paper logic (p.110):
    # 1. Eq(1): estimate coefficients with truncated variables (done above)
    # 2. Winsorize Eq(2) independent variables
    # 3. Eq(2): compute ETA using winsorized variables
    # 4. Winsorize TA and ETA
    # 5. Eq(3): DACC = winsorized_TA - winsorized_ETA
    
    # Winsorize Eq(2) independent variables
    eq2_vars = ["inv_lag_at", "delta_rev", "delta_rec", "ppe_scaled", "roa_lag"]
    aa = winsorize_columns(aa, eq2_vars)
    
    # Eq(2): ETA from winsorized variables
    aa["eta"] = (
        aa["b0"] * aa["inv_lag_at"]
        + aa["b1"] * (aa["delta_rev"] - aa["delta_rec"])
        + aa["b2"] * aa["ppe_scaled"]
        + aa["b3"] * aa["roa_lag"]
    )
    
    # Winsorize TA and ETA (paper p.110)
    aa = winsorize_columns(aa, ["ta", "eta"])
    
    # Eq(3): DACC = winsorized_TA - winsorized_ETA
    aa["dacc"] = aa["ta"] - aa["eta"]
    
    # Winsorize DACC (paper p.110)
    aa = winsorize_columns(aa, ["dacc"])
    
    # Take absolute value and winsorize again
    aa["abs_dacc"] = aa["dacc"].abs()
    aa = winsorize_columns(aa, ["abs_dacc"])

    # Auditor controls
    aa["auditor_name_u"] = aa["auditor_name"].astype(str).str.upper()
    aa["big4"] = aa["auditor_name_u"].str.contains("PRICEWATERHOUSE|PWC|ERNST|KPMG|DELOITTE", regex=True).astype(float)
    aa["sec_tier"] = aa["auditor_name_u"].str.contains("GRANT THORNTON|BDO", regex=True).astype(float)
    aa["gc"] = pd.to_numeric(aa["going_concern"], errors="coerce")

    # Eq(4)-(6): control variables to winsorize within analysis samples
    # (TA, ETA, DACC, abs_dacc already winsorized above)
    eq26_cont_cols = [
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

    # Panel A in paper: perform attrition after variables are constructed.
    aa_full = aa.copy()
    comp_key_set_panel_a = set(comp["cik_norm"].dropna().astype(str).unique())
    aa_full["panelA_in_compustat"] = aa_full["cik_norm"].isin(comp_key_set_panel_a)
    counts["panelA_delete_not_in_compustat"] = int((~aa_full["panelA_in_compustat"]).sum())
    aa_panel_pre_domestic = aa_full[aa_full["panelA_in_compustat"]].copy()
    counts["panelA_after_compustat"] = len(aa_panel_pre_domestic)
    counts["panelA_delete_non_domestic_fic"] = int((~aa_panel_pre_domestic["domestic"].fillna(False)).sum())
    counts["panelA_domestic_kept_obs"] = int(aa_panel_pre_domestic["domestic"].fillna(False).sum())
    aa_panel_pre_city = aa_panel_pre_domestic[aa_panel_pre_domestic["domestic"].fillna(False)].copy()
    counts["panelA_after_domestic"] = len(aa_panel_pre_city)

    cell_n = aa_panel_pre_city.groupby(["fiscal_year", "sic2", "msa_code"])["company_fkey"].transform("size")
    aa_panel_pre_city["panelA_city_industry_n"] = cell_n
    aa_panel_pre_city["panelA_city_industry_ge2"] = cell_n >= 2
    aa_full["panelA_city_industry_n"] = np.nan
    aa_full["panelA_city_industry_ge2"] = pd.Series(pd.NA, index=aa_full.index, dtype="boolean")
    aa_full.loc[aa_panel_pre_city.index, "panelA_city_industry_n"] = aa_panel_pre_city["panelA_city_industry_n"]
    aa_full.loc[aa_panel_pre_city.index, "panelA_city_industry_ge2"] = aa_panel_pre_city["panelA_city_industry_ge2"]
    counts["panelA_delete_city_industry_lt2"] = int((~aa_panel_pre_city["panelA_city_industry_ge2"]).sum())
    aa_panel = aa_panel_pre_city[aa_panel_pre_city["panelA_city_industry_ge2"]].copy()
    counts["aa_after_city_industry_min2"] = len(aa_panel)
    counts["panelA_final_obs"] = len(aa_panel)
    counts["panelA_final_unique_msa"] = int(aa_panel["msa_code"].nunique())
    aa = aa_panel

    panel_cols = [
        "company_fkey", "company_key", "auditor_fkey", "auditor_name", "fiscal_year",
        "sic4", "sic2", "msa_code", "msa_title", "audit_fees", "going_concern", "gc",
        "auditor_city", "auditor_state", "msa_match_source",
        "domestic", "nonfinancial", "tenure_ln",
        "nat_spec_d1", "city_spec_d1", "both_d1", "nat_only_d1", "city_only_d1",
        "nat_spec_d2", "city_spec_d2", "both_d2", "nat_only_d2", "city_only_d2",
        "ta", "eta", "dacc", "abs_dacc",
        "size", "sigma_cfo", "cfo", "lev", "loss", "mb", "lit", "altman",
        "big4", "sec_tier",
        "ta_lag_abs", "accr", "sigma_earn", "roa", "op_cf",
        "panelA_in_compustat", "panelA_city_industry_n", "panelA_city_industry_ge2",
    ]
    panel_cols = [c for c in panel_cols if c in aa.columns]
    panel_path = processed_dir / "panel_with_features_2003_2007.csv"
    panel_export_cols = [c for c in panel_cols if c in aa_full.columns]
    aa_full[panel_export_cols].to_csv(panel_path, index=False)
    if args.stop_after == "panel":
        diagnostics = {
            "counts": counts,
            "missing_altman_inputs": missing_altman,
            "msa_map_source": msa_map_source,
            "msa_match_mode": msa_map_source,
            "sample_drop_after_feature_construction": True,
            "notes": ["Stopped after panel preparation (--stop-after panel)."],
        }
        (paths.output_dir / "replication_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
        print("Done (panel). Outputs:")
        print(panel_path)
        print(paths.output_dir / "replication_diagnostics.json")
        return

    # Optional reduced model without Altman
    base_controls = ["size", "sigma_cfo", "cfo", "lev", "loss", "mb", "lit", "tenure_ln", "ta_lag_abs", "big4", "sec_tier"]
    if aa["altman"].notna().mean() > 0.2:
        controls5 = base_controls[:7] + ["altman"] + base_controls[7:]
    else:
        if not args.allow_reduced_model:
            raise RuntimeError(
                "Altman inputs unavailable in Compustat extract. Re-download funda with ACT, LCT, RE, EBIT or run with --allow-reduced-model."
            )
        controls5 = base_controls

    # Table 5 sample
    counts["panelB_start_obs"] = counts.get("panelA_final_obs", len(aa))
    t5_required = ["abs_dacc"] + controls5 + ["both_d1", "nat_only_d1", "city_only_d1", "both_d2", "nat_only_d2", "city_only_d2"]
    t5_missing_counts = {k: int(aa[k].isna().sum()) for k in t5_required}
    t5 = aa.dropna(subset=t5_required).copy()
    t5_winsor_cols = [c for c in eq26_cont_cols if c in t5.columns]
    t5 = winsorize_columns(t5, t5_winsor_cols)
    counts["panelB_delete_missing"] = int(counts["panelB_start_obs"] - len(t5))
    counts["table5_pre_outlier"] = len(t5)

    # Studentized residual filter from preliminary model (def1, model3)
    prelim_x = controls5 + ["both_d1", "nat_only_d1", "city_only_d1"]
    prelim_res, prelim_use = run_ols_cluster(t5, "abs_dacc", prelim_x, "company_fkey")
    t5_before_outlier = len(t5)
    if prelim_res is not None and not prelim_use.empty:
        infl = OLSInfluence(sm.OLS(prelim_use["abs_dacc"], sm.add_constant(prelim_use[prelim_x], has_constant="add")).fit())
        stud = infl.resid_studentized_external
        keep_mask = np.abs(stud) <= 3
        keep_index = prelim_use.index[keep_mask]
        t5 = t5.loc[keep_index].copy()
    counts["panelB_delete_outlier"] = int(t5_before_outlier - len(t5))
    counts["table5_final_n"] = len(t5)
    counts["panelB_final_unique_msa"] = int(t5["msa_code"].nunique()) if "msa_code" in t5.columns else 0

    # Table 5 regressions
    t5_rows = []
    t5_model_results: Dict[Tuple[int, str], object] = {}
    t5_full_rows = []
    for dnum in [1, 2]:
        m1_x = controls5 + [f"nat_spec_d{dnum}"]
        m2_x = controls5 + [f"city_spec_d{dnum}"]
        m3_x = controls5 + [f"both_d{dnum}", f"nat_only_d{dnum}", f"city_only_d{dnum}"]

        for model_name, xcols, terms in [
            ("model1", m1_x, [f"nat_spec_d{dnum}"]),
            ("model2", m2_x, [f"city_spec_d{dnum}"]),
            ("model3", m3_x, [f"both_d{dnum}", f"nat_only_d{dnum}", f"city_only_d{dnum}"]),
        ]:
            res, use = run_ols_cluster(t5, "abs_dacc", xcols, "company_fkey")
            if res is None:
                for term in terms:
                    t5_rows.append(
                        {
                            "definition": dnum,
                            "model": model_name,
                            "term": term,
                            "coef": float("nan"),
                            "pvalue": float("nan"),
                            "n": 0,
                            "adj_r2": float("nan"),
                        }
                    )
                continue
            t5_model_results[(dnum, model_name)] = res
            for term in res.params.index:
                t5_full_rows.append(
                    {
                        "definition": dnum,
                        "model": model_name,
                        "term": term,
                        "coef": float(res.params.get(term, np.nan)),
                        "pvalue": float(res.pvalues.get(term, np.nan)),
                        "n": int(res.nobs),
                        "adj_r2": float(res.rsquared_adj),
                        "f_value": float(getattr(res, "fvalue", np.nan)),
                        "f_pvalue": float(getattr(res, "f_pvalue", np.nan)),
                    }
                )
            for term in terms:
                t5_rows.append(
                    {
                        "definition": dnum,
                        "model": model_name,
                        "term": term,
                        "coef": float(res.params.get(term, np.nan)),
                        "pvalue": float(res.pvalues.get(term, np.nan)),
                        "n": int(res.nobs),
                        "adj_r2": float(res.rsquared_adj),
                    }
                )

    t5_out = pd.DataFrame(t5_rows)
    t5_out.to_csv(paths.output_dir / "table5_replication_long.csv", index=False)
    pd.DataFrame(t5_full_rows).to_csv(paths.output_dir / "table5_replication_fullcoef.csv", index=False)

    # Table 8 sample: financially distressed (negative operating cash flow)
    controls8 = ["size", "sigma_earn", "lev", "loss", "roa", "mb", "lit", "tenure_ln", "accr", "big4", "sec_tier"]
    if "altman" in controls5:
        controls8 = controls8[:7] + ["altman"] + controls8[7:]

    counts["panelD_start_obs"] = counts.get("panelA_final_obs", len(aa))
    t8 = aa.copy()
    t8_required = ["gc", "op_cf"] + controls8 + [
        "sic2",
        "fiscal_year",
        "both_d1",
        "nat_only_d1",
        "city_only_d1",
        "both_d2",
        "nat_only_d2",
        "city_only_d2",
    ]
    t8_missing_counts = {k: int(t8[k].isna().sum()) for k in t8_required}
    t8 = t8.dropna(subset=t8_required).copy()
    counts["panelD_delete_missing"] = int(counts["panelD_start_obs"] - len(t8))
    counts["panelD_delete_non_distressed"] = int((t8["op_cf"] >= 0).sum())
    t8 = t8[t8["op_cf"] < 0].copy()
    t8_winsor_cols = [c for c in eq26_cont_cols if c in t8.columns]
    t8 = winsorize_columns(t8, t8_winsor_cols)
    t8["gc"] = (t8["gc"] > 0).astype(int)
    t8["sic2_cat"] = t8["sic2"].astype("Int64").astype(str)
    t8["fiscal_year_cat"] = pd.to_numeric(t8["fiscal_year"], errors="coerce").astype("Int64").astype(str)
    counts["table8_pre_outlier"] = len(t8)

    # Deviance residual filter from preliminary logit (def1 model3)
    f_controls = " + ".join(controls8)
    prelim_formula = f"gc ~ {f_controls} + both_d1 + nat_only_d1 + city_only_d1 + C(sic2_cat) + C(fiscal_year_cat)"
    prelim_logit, prelim_use = run_logit_cluster_formula(t8, prelim_formula, "company_fkey")
    t8_before_outlier = len(t8)
    if prelim_logit is not None and not prelim_use.empty:
        dev = prelim_logit.resid_deviance
        t8 = prelim_use.loc[np.abs(dev) <= 3].copy()
    counts["panelD_delete_outlier"] = int(t8_before_outlier - len(t8))
    counts["table8_final_n"] = len(t8)
    counts["panelD_final_unique_msa"] = int(t8["msa_code"].nunique()) if "msa_code" in t8.columns else 0

    table1_attrition = {
        "panelA_start_obs": counts.get("panelA_start_obs"),
        "panelA_delete_not_in_compustat": counts.get("panelA_delete_not_in_compustat"),
        "panelA_delete_city_industry_lt2": counts.get("panelA_delete_city_industry_lt2"),
        "panelA_final_obs": counts.get("panelA_final_obs"),
        "panelB_start_obs": counts.get("panelB_start_obs"),
        "panelB_delete_missing": counts.get("panelB_delete_missing"),
        "panelB_delete_outlier": counts.get("panelB_delete_outlier"),
        "panelB_final_obs": counts.get("table5_final_n"),
        "panelD_start_obs": counts.get("panelD_start_obs"),
        "panelD_delete_missing": counts.get("panelD_delete_missing"),
        "panelD_delete_non_distressed": counts.get("panelD_delete_non_distressed"),
        "panelD_delete_outlier": counts.get("panelD_delete_outlier"),
        "panelD_final_obs": counts.get("table8_final_n"),
    }

    table5_input_path = processed_dir / "table5_input.csv"
    table8_input_path = processed_dir / "table8_input.csv"
    t5.to_csv(table5_input_path, index=False)
    t8.to_csv(table8_input_path, index=False)
    if args.stop_after == "model_data":
        diagnostics = {
            "counts": counts,
            "missing_altman_inputs": missing_altman,
            "tenure_source": args.tenure_source,
            "comp_costat": args.comp_costat,
            "domestic_filter": args.domestic_filter,
            "msa_map_source": msa_map_source,
            "msa_match_mode": msa_map_source,
            "table1_attrition_like": table1_attrition,
            "table5_missing_counts": t5_missing_counts,
            "table8_missing_counts": t8_missing_counts,
            "sample_drop_after_feature_construction": True,
            "notes": ["Stopped after model-data preparation (--stop-after model_data)."],
        }
        (paths.output_dir / "replication_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
        print("Done (model_data). Outputs:")
        print(table5_input_path)
        print(table8_input_path)
        print(paths.output_dir / "replication_diagnostics.json")
        return

    # Table 8 regressions
    t8_rows = []
    t8_model_results: Dict[Tuple[int, str], object] = {}
    t8_full_rows = []
    for dnum in [1, 2]:
        formulas = [
            ("model1", f"gc ~ {f_controls} + nat_spec_d{dnum} + C(sic2_cat) + C(fiscal_year_cat)", [f"nat_spec_d{dnum}"]),
            ("model2", f"gc ~ {f_controls} + city_spec_d{dnum} + C(sic2_cat) + C(fiscal_year_cat)", [f"city_spec_d{dnum}"]),
            (
                "model3",
                f"gc ~ {f_controls} + both_d{dnum} + nat_only_d{dnum} + city_only_d{dnum} + C(sic2_cat) + C(fiscal_year_cat)",
                [f"both_d{dnum}", f"nat_only_d{dnum}", f"city_only_d{dnum}"],
            ),
        ]
        for model_name, formula, terms in formulas:
            res, use = run_logit_cluster_formula(t8, formula, "company_fkey")
            if res is None:
                for term in terms:
                    t8_rows.append(
                        {
                            "definition": dnum,
                            "model": model_name,
                            "term": term,
                            "coef": float("nan"),
                            "pvalue": float("nan"),
                            "n": 0,
                            "pseudo_r2_mcf": float("nan"),
                        }
                    )
                continue
            t8_model_results[(dnum, model_name)] = res
            for term in res.params.index:
                # LR statistic from GLM deviance improvement versus null.
                lr_stat = float(np.nan)
                lr_p = float(np.nan)
                if hasattr(res, "null_deviance") and hasattr(res, "deviance"):
                    nd = getattr(res, "null_deviance", np.nan)
                    dv = getattr(res, "deviance", np.nan)
                    if pd.notna(nd) and pd.notna(dv):
                        lr_stat = float(nd - dv)
                        df_m = getattr(res, "df_model", np.nan)
                        if chi2 is not None and pd.notna(df_m) and float(df_m) > 0:
                            lr_p = float(chi2.sf(lr_stat, float(df_m)))
                pseudo_r2 = float(np.nan)
                llnull = getattr(res, "llnull", np.nan)
                llf = getattr(res, "llf", np.nan)
                if pd.notna(llnull) and pd.notna(llf) and float(llnull) != 0:
                    pseudo_r2 = float(1.0 - (float(llf) / float(llnull)))
                elif hasattr(res, "null_deviance") and hasattr(res, "deviance"):
                    nd = getattr(res, "null_deviance", np.nan)
                    dv = getattr(res, "deviance", np.nan)
                    if pd.notna(nd) and pd.notna(dv) and float(nd) != 0:
                        pseudo_r2 = float(1.0 - (float(dv) / float(nd)))

                t8_full_rows.append(
                    {
                        "definition": dnum,
                        "model": model_name,
                        "term": term,
                        "coef": float(res.params.get(term, np.nan)),
                        "pvalue": float(res.pvalues.get(term, np.nan)),
                        "n": int(res.nobs),
                        "lr_stat": lr_stat,
                        "lr_pvalue": lr_p,
                        "pseudo_r2": pseudo_r2,
                    }
                )
            for term in terms:
                t8_rows.append(
                    {
                        "definition": dnum,
                        "model": model_name,
                        "term": term,
                        "coef": float(res.params.get(term, np.nan)),
                        "pvalue": float(res.pvalues.get(term, np.nan)),
                        "n": int(res.nobs),
                        "pseudo_r2_mcf": float(np.nan),
                    }
                )

    t8_out = pd.DataFrame(t8_rows)
    t8_out.to_csv(paths.output_dir / "table8_replication_long.csv", index=False)
    pd.DataFrame(t8_full_rows).to_csv(paths.output_dir / "table8_replication_fullcoef.csv", index=False)

    # Paper-style tables (economics format) for copy/paste to manuscript.
    table5_rowspec = [
        ("Intercept", "const"),
        ("SIZE", "size"),
        ("sigma(CFO)", "sigma_cfo"),
        ("CFO", "cfo"),
        ("LEV", "lev"),
        ("LOSS", "loss"),
        ("MB", "mb"),
        ("LIT", "lit"),
        ("ALTMAN", "altman"),
        ("TENURE", "tenure_ln"),
        ("ABS_ACCR_LAG", "ta_lag_abs"),
        ("BIG4", "big4"),
        ("SEC_TIER", "sec_tier"),
        ("National Specialist", {1: "nat_spec_d1", 2: "nat_spec_d2"}),
        ("City Specialist", {1: "city_spec_d1", 2: "city_spec_d2"}),
        ("Both National and City Specialist", {1: "both_d1", 2: "both_d2"}),
        ("National Specialist Only", {1: "nat_only_d1", 2: "nat_only_d2"}),
        ("City Specialist Only", {1: "city_only_d1", 2: "city_only_d2"}),
    ]

    def t5_summary_getter(label: str, res):
        if res is None:
            return np.nan, np.nan
        if label == "F-value":
            return getattr(res, "fvalue", np.nan), getattr(res, "f_pvalue", np.nan)
        if label == "Adj. R2":
            return getattr(res, "rsquared_adj", np.nan), np.nan
        return np.nan, np.nan

    t5_paper_df = build_paper_table_df(table5_rowspec, t5_model_results, t5_summary_getter)
    write_paper_table_markdown(
        paths.output_dir / "table5_replication_paperstyle.md",
        "TABLE 5 (Replication)",
        "Dependent variable is the absolute value of abnormal accruals.",
        t5_paper_df,
    )
    write_paper_table_latex(
        paths.output_dir / "table5_replication_paperstyle.tex",
        "TABLE 5 (Replication)",
        "Dependent variable is the absolute value of abnormal accruals.",
        t5_paper_df,
        "Coefficient p-values are two-tailed and based on cluster-robust inference by company_fkey.",
    )

    table8_rowspec = [
        ("Intercept", "Intercept"),
        ("SIZE", "size"),
        ("sigma(EARN)", "sigma_earn"),
        ("LEV", "lev"),
        ("LOSS", "loss"),
        ("ROA", "roa"),
        ("MB", "mb"),
        ("LIT", "lit"),
        ("ALTMAN", "altman"),
        ("TENURE", "tenure_ln"),
        ("ACCR", "accr"),
        ("BIG4", "big4"),
        ("SEC_TIER", "sec_tier"),
        ("National Specialist", {1: "nat_spec_d1", 2: "nat_spec_d2"}),
        ("City Specialist", {1: "city_spec_d1", 2: "city_spec_d2"}),
        ("Both National and City Specialist", {1: "both_d1", 2: "both_d2"}),
        ("National Specialist Only", {1: "nat_only_d1", 2: "nat_only_d2"}),
        ("City Specialist Only", {1: "city_only_d1", 2: "city_only_d2"}),
    ]

    def t8_summary_getter(label: str, res):
        if res is None:
            return np.nan, np.nan
        if label == "Likelihood ratio":
            if hasattr(res, "null_deviance") and hasattr(res, "deviance"):
                nd = getattr(res, "null_deviance", np.nan)
                dv = getattr(res, "deviance", np.nan)
                if pd.notna(nd) and pd.notna(dv):
                    lr_stat = float(nd - dv)
                    df_m = getattr(res, "df_model", np.nan)
                    lr_p = np.nan
                    if chi2 is not None and pd.notna(df_m) and float(df_m) > 0:
                        lr_p = float(chi2.sf(lr_stat, float(df_m)))
                    return lr_stat, lr_p
            return np.nan, np.nan
        if label == "Pseudo-R2":
            llnull = getattr(res, "llnull", np.nan)
            llf = getattr(res, "llf", np.nan)
            if pd.notna(llnull) and pd.notna(llf) and float(llnull) != 0:
                return float(1.0 - (float(llf) / float(llnull))), np.nan
            if hasattr(res, "null_deviance") and hasattr(res, "deviance"):
                nd = getattr(res, "null_deviance", np.nan)
                dv = getattr(res, "deviance", np.nan)
                if pd.notna(nd) and pd.notna(dv) and float(nd) != 0:
                    return float(1.0 - (float(dv) / float(nd))), np.nan
            return np.nan, np.nan
        return np.nan, np.nan

    t8_paper_df = build_paper_table_df(table8_rowspec, t8_model_results, t8_summary_getter)
    write_paper_table_markdown(
        paths.output_dir / "table8_replication_paperstyle.md",
        "TABLE 8 (Replication)",
        "Dependent variable is the probability of issuing a going-concern opinion (GC).",
        t8_paper_df,
    )
    write_paper_table_latex(
        paths.output_dir / "table8_replication_paperstyle.tex",
        "TABLE 8 (Replication)",
        "Dependent variable is the probability of issuing a going-concern opinion (GC).",
        t8_paper_df,
        "Coefficient p-values are two-tailed and based on cluster-robust inference by company_fkey.",
    )

    # Save diagnostics
    diagnostics = {
        "counts": counts,
        "missing_altman_inputs": missing_altman,
        "tenure_source": args.tenure_source,
        "comp_costat": args.comp_costat,
        "domestic_filter": args.domestic_filter,
        "msa_map_source": msa_map_source,
        "msa_match_mode": msa_map_source,
        "table1_attrition_like": table1_attrition,
        "table5_missing_counts": t5_missing_counts,
        "table8_missing_counts": t8_missing_counts,
        "sample_drop_after_feature_construction": True,
        "msa_match_rate": float(aa["msa_code"].notna().mean()) if len(aa) else 0.0,
        "notes": [
            "Office location uses only Audit Opinions AUDITOR_CITY/AUDITOR_STATE.",
            f"City-to-MSA mapping source: {msa_map_source}.",
            "Table 5/8 run with cluster-robust SE by company_fkey.",
            "All variables are constructed on merged panel before Panel-A sample drops.",
            "If Altman inputs missing, reduced models exclude Altman unless strict mode is used.",
        ],
    }
    (paths.output_dir / "replication_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))

    print("Done. Outputs:")
    print(paths.output_dir / "table5_replication_long.csv")
    print(paths.output_dir / "table8_replication_long.csv")
    print(paths.output_dir / "table5_replication_paperstyle.md")
    print(paths.output_dir / "table8_replication_paperstyle.md")
    print(paths.output_dir / "table5_replication_paperstyle.tex")
    print(paths.output_dir / "table8_replication_paperstyle.tex")
    print(paths.output_dir / "replication_diagnostics.json")


if __name__ == "__main__":
    main()
