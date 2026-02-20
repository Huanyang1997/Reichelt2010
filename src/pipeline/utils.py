#!/usr/bin/env python3
"""
utils.py — 共用工具函数
========================
所有步骤共用的标准化、缩尾、截断函数。
变量名与论文变量一一对应，并在 docstring 中标注。
"""
from __future__ import annotations

import re
from typing import List

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════
# 标准化
# ═══════════════════════════════════════════════════════════════════

STATE_FIPS_TO_ABBR = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY", "60": "AS", "66": "GU", "69": "MP",
    "72": "PR", "78": "VI",
}
US_STATE_ABBR = set(STATE_FIPS_TO_ABBR.values())


def norm_cik(v) -> str:
    """CIK 标准化：去非数字、去前导零。"""
    if pd.isna(v):
        return ""
    s = re.sub(r"\D", "", str(v))
    s = s.lstrip("0")
    return s if s else "0"


def norm_text(v) -> str:
    """城市名标准化：大写、替换缩写、去标点。"""
    if pd.isna(v):
        return ""
    s = str(v).upper()
    s = s.replace("&", " AND ")
    s = s.replace("SAINT", "ST")
    s = s.replace("FORT", "FT")
    s = re.sub(r"[^A-Z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ═══════════════════════════════════════════════════════════════════
# 缩尾 / 截断
# ═══════════════════════════════════════════════════════════════════

def winsorize_series(s: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    """
    对连续变量进行缩尾处理。
    论文 p.110: "All continuous variables in equations (2) through (6)
    are winsorized at the 1st and 99th percentiles."
    """
    if s.dropna().empty:
        return s
    q_lo = s.quantile(lo)
    q_hi = s.quantile(hi)
    return s.clip(q_lo, q_hi)


def winsorize_columns(df: pd.DataFrame, cols: List[str],
                      lo: float = 0.01, hi: float = 0.99) -> pd.DataFrame:
    """对 DataFrame 中指定列逐列缩尾。"""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = winsorize_series(
                pd.to_numeric(out[c], errors="coerce"), lo=lo, hi=hi
            )
    return out


def truncate_1pct_rows(df: pd.DataFrame, cols: List[str],
                       pct: float = 0.01) -> pd.DataFrame:
    """
    截断：删除任一指定变量超出 [pct, 1-pct] 分位的行。
    论文 p.110: Eq(1) 中变量做 top/bottom 1% 截断。
    """
    out = df.copy()
    keep = pd.Series(True, index=out.index)
    for c in cols:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        if s.dropna().empty:
            continue
        lo = s.quantile(pct)
        hi = s.quantile(1.0 - pct)
        keep = keep & s.between(lo, hi, inclusive="both")
    return out.loc[keep].copy()


# ═══════════════════════════════════════════════════════════════════
# Compustat 读取
# ═══════════════════════════════════════════════════════════════════

def load_compustat(path) -> pd.DataFrame:
    """读取 Compustat (dta 或 csv)，统一列名小写。"""
    from pathlib import Path
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".dta":
        d = pd.read_stata(path, convert_categoricals=False)
    else:
        d = pd.read_csv(path, dtype={"cik": "string"})
    d.columns = [c.lower() for c in d.columns]
    if "cik" in d.columns:
        d["cik"] = d["cik"].astype("string")
    return d


# ═══════════════════════════════════════════════════════════════════
# 格式化
# ═══════════════════════════════════════════════════════════════════

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
