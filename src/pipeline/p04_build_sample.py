#!/usr/bin/env python3
"""
p04_build_sample.py — 第四步：构建工作样本
============================================
对应论文 Table 1 (Sample Selection) 的完整递减路径：

  Panel A (起始样本)
  ─────────────────────────────────────────
  32,479  AA firm-years: 2003-2007, nonfinancial, positive fees, valid MSA/SIC
  -7,476  Not matched to Compustat by CIK
  -3,420  City-industry-year < 2 observations
  = 21,583 Working sample

本步骤包含：
  1) 读入 AA (含 MSA) + Compustat 变量
  2) 期间/行业/费用基础过滤
  3) **按键匹配** AA × Compustat (CIK + fiscal_year)
  4) 估计 Eq(1) → 计算 ETA (Eq 2), DACC (Eq 3)
  5) 构建审计师行业专家指标 (Definition 1 & 2)
  6) Panel A 递减: Compustat 匹配 → domestic → city-industry ≥ 2
  7) 输出工作样本

合并策略说明:
  论文描述"以 CIK 匹配 Compustat"——按键匹配到了就合并。
  具体实现：先在 AA 上标记哪些 CIK 在 Compustat 中存在；
  再按 (cik_norm, fiscal_year) 键匹配把 Compustat 变量合入。
  未匹配行保留但 Compustat 列为 NaN，显式在递减中删除。

输入:
  data/processed/pipeline/s02_aa_with_msa.csv
  data/processed/pipeline/s03_compustat_vars.csv
输出:
  data/processed/pipeline/s04_working_sample.csv
  data/processed/pipeline/s04_eq1_coefs.csv
  data/processed/pipeline/s04_attrition.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import PipelineConfig, ensure_dirs
from pipeline.utils import norm_cik, truncate_1pct_rows, winsorize_columns


# ═══════════════════════════════════════════════════════════════════
# 审计师行业专家指标
# ═══════════════════════════════════════════════════════════════════

def _specialist_def1(shares: pd.DataFrame,
                     market_cols: List[str]) -> pd.DataFrame:
    """
    Definition 1 (论文 p.109):
      Specialist = 行业(或城市-行业)内审计费最大份额者，
      且份额领先第二名 ≥ 10 个百分点。
    """
    sort_cols = market_cols + ["share"]
    asc = [True] * len(market_cols) + [False]
    s = shares.sort_values(sort_cols, ascending=asc)
    top = s.groupby(market_cols, as_index=False).nth(0).reset_index(drop=True)
    second = s.groupby(market_cols, as_index=False).nth(1).reset_index(drop=True)
    second = second[market_cols + ["share"]].rename(columns={"share": "share2"})
    top = top.merge(second, on=market_cols, how="left")
    top["share2"] = top["share2"].fillna(0.0)
    top["is_specialist"] = (top["share"] - top["share2"]) >= 0.10
    return top.loc[top["is_specialist"], market_cols + ["auditor_fkey"]]


def build_specialist_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    构建 4 个市场份额专家虚拟变量 + 3 个交叉虚拟变量 × 2 个定义。
    论文 Table 2 / p.109:
      Definition 1: largest share and ≥10pp above 2nd
      Definition 2: national > 30%, city > 50%
    """
    d = df.copy()

    # ── National market share (by fiscal_year × sic2) ────────────
    national = (
        d.groupby(["fiscal_year", "sic2", "auditor_fkey"],
                   as_index=False)["audit_fees"].sum()
    )
    national["mkt_total"] = (
        national.groupby(["fiscal_year", "sic2"])["audit_fees"]
        .transform("sum")
    )
    national["share"] = national["audit_fees"] / national["mkt_total"]

    # ── City market share (by fiscal_year × sic2 × msa_code) ────
    city = (
        d.groupby(["fiscal_year", "sic2", "msa_code", "auditor_fkey"],
                   as_index=False)["audit_fees"].sum()
    )
    city["mkt_total"] = (
        city.groupby(["fiscal_year", "sic2", "msa_code"])["audit_fees"]
        .transform("sum")
    )
    city["share"] = city["audit_fees"] / city["mkt_total"]

    # ── Definition 1 ─────────────────────────────────────────────
    nat_d1 = _specialist_def1(national, ["fiscal_year", "sic2"])
    nat_d1["nat_spec_d1"] = 1
    city_d1 = _specialist_def1(city, ["fiscal_year", "sic2", "msa_code"])
    city_d1["city_spec_d1"] = 1

    # ── Definition 2 (论文 p.109) ────────────────────────────────
    nat_d2 = national.loc[national["share"] > 0.30,
                          ["fiscal_year", "sic2", "auditor_fkey"]].copy()
    nat_d2["nat_spec_d2"] = 1
    city_d2 = city.loc[city["share"] > 0.50,
                       ["fiscal_year", "sic2", "msa_code",
                        "auditor_fkey"]].copy()
    city_d2["city_spec_d2"] = 1

    # ── 合并到观测层面 ──────────────────────────────────────────
    out = d[["company_fkey", "fiscal_year", "sic2",
             "msa_code", "auditor_fkey"]].copy()
    out = out.drop_duplicates(
        ["company_fkey", "fiscal_year", "auditor_fkey"]
    )

    out = out.merge(nat_d1, on=["fiscal_year", "sic2", "auditor_fkey"],
                    how="left")
    out = out.merge(city_d1, on=["fiscal_year", "sic2", "msa_code",
                                  "auditor_fkey"], how="left")
    out = out.merge(nat_d2, on=["fiscal_year", "sic2", "auditor_fkey"],
                    how="left")
    out = out.merge(city_d2, on=["fiscal_year", "sic2", "msa_code",
                                  "auditor_fkey"], how="left")

    for c in ["nat_spec_d1", "city_spec_d1", "nat_spec_d2", "city_spec_d2"]:
        out[c] = out[c].fillna(0).astype(int)

    # 交叉虚拟变量
    for dnum in [1, 2]:
        n = f"nat_spec_d{dnum}"
        c = f"city_spec_d{dnum}"
        out[f"both_d{dnum}"] = ((out[n] == 1) & (out[c] == 1)).astype(int)
        out[f"nat_only_d{dnum}"] = ((out[n] == 1) & (out[c] == 0)).astype(int)
        out[f"city_only_d{dnum}"] = ((out[n] == 0) & (out[c] == 1)).astype(int)

    return out


# ═══════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════

def run(cfg: PipelineConfig) -> Path:
    ensure_dirs(cfg)
    pipe = Path(cfg.pipeline_dir)
    counts = {}

    # ── 4.1 读入前步骤输出 ──────────────────────────────────────
    print("[p04] 读取 AA (含 MSA) 和 Compustat 变量 ...")
    aa = pd.read_csv(pipe / "s02_aa_with_msa.csv", dtype=str)
    comp = pd.read_csv(pipe / "s03_compustat_vars.csv", dtype=str)

    # 恢复数值列
    aa_num = ["fiscal_year", "audit_fees", "sic_code_fkey",
              "going_concern", "tenure_years", "tenure_ln"]
    for c in aa_num:
        if c in aa.columns:
            aa[c] = pd.to_numeric(aa[c], errors="coerce")

    comp_num = [
        "fyear", "sic", "ta", "inv_lag_at", "delta_rev", "delta_rec",
        "ppe_scaled", "roa_lag", "size", "sigma_cfo", "cfo", "lev",
        "loss", "mb", "lit", "altman", "ta_lag_abs", "accr",
        "sigma_earn", "roa", "op_cf",
    ]
    for c in comp_num:
        if c in comp.columns:
            comp[c] = pd.to_numeric(comp[c], errors="coerce")

    # ── 4.2 基础过滤 ────────────────────────────────────────────
    #  论文 Table 1, Panel A 起始条件:
    #  - fiscal_year ∈ [2003, 2007]
    #  - audit_fees > 0
    #  - nonfinancial (SIC 不在 6000-6999)
    #  - 有效 MSA
    #  - 有效 SIC

    aa["fiscal_year"] = pd.to_numeric(aa["fiscal_year"], errors="coerce")
    aa = aa[(aa["fiscal_year"] >= cfg.year_start) &
            (aa["fiscal_year"] <= cfg.year_end)].copy()
    aa = aa[aa["audit_fees"] > 0].copy()

    aa["sic4"] = pd.to_numeric(aa["sic_code_fkey"], errors="coerce")
    aa["sic2"] = (aa["sic4"] // 100).astype("Int64")
    aa["nonfinancial"] = aa["sic4"].notna() & (
        ~aa["sic4"].between(6000, 6999, inclusive="both")
    )
    if cfg.exclude_utility:
        aa["nonutility"] = ~aa["sic4"].between(4900, 4999, inclusive="both")
    else:
        aa["nonutility"] = True

    aa_core = aa[aa["nonfinancial"] & aa["nonutility"]].copy()
    aa_core = aa_core[aa_core["msa_code"].notna()].copy()
    counts["panelA_start"] = len(aa_core)
    counts["panelA_start_unique_msa"] = int(aa_core["msa_code"].nunique())

    # ── 4.3 构建审计师专家指标 ──────────────────────────────────
    #  论文 p.109: 基于 Panel A 起始样本（Compustat 匹配前）
    #  的审计费市场份额计算
    print("[p04] 计算审计师行业专家指标 ...")
    spec = build_specialist_indicators(aa_core)
    aa_core = aa_core.merge(
        spec,
        on=["company_fkey", "fiscal_year", "sic2",
            "msa_code", "auditor_fkey"],
        how="left",
    )
    for dnum in [1, 2]:
        for col in [f"both_d{dnum}", f"nat_only_d{dnum}",
                    f"city_only_d{dnum}", f"nat_spec_d{dnum}",
                    f"city_spec_d{dnum}"]:
            if col not in aa_core.columns:
                aa_core[col] = 0
            aa_core[col] = (
                pd.to_numeric(aa_core[col], errors="coerce").fillna(0).astype(int)
            )
        # 从 model-3 交叉指标重构 model-1/2 使用的单一指标
        aa_core[f"nat_spec_d{dnum}"] = (
            (aa_core[f"both_d{dnum}"] == 1) |
            (aa_core[f"nat_only_d{dnum}"] == 1)
        ).astype(int)
        aa_core[f"city_spec_d{dnum}"] = (
            (aa_core[f"both_d{dnum}"] == 1) |
            (aa_core[f"city_only_d{dnum}"] == 1)
        ).astype(int)

    # ── 4.4 按键匹配 AA × Compustat ────────────────────────────
    #  论文: "matched to Compustat by CIK"
    #  步骤:
    #    a) 标记 AA 中哪些 CIK 在 Compustat 中存在
    #    b) 按 (cik_norm, fiscal_year) 合并 Compustat 变量
    #    c) 未匹配行在递减中显式删除
    print("[p04] 按 CIK+fiscal_year 匹配 Compustat ...")
    if "cik_norm" not in aa_core.columns:
        aa_core["cik_norm"] = (
            aa_core["company_fkey"].fillna(aa_core.get("company_key", pd.NA))
            .apply(norm_cik)
        )
    comp_cik_set = set(comp["cik_norm"].dropna().unique())
    aa_core["in_compustat"] = aa_core["cik_norm"].isin(comp_cik_set)
    counts["panelA_delete_not_in_compustat"] = int(
        (~aa_core["in_compustat"]).sum()
    )

    # 合并 Compustat 变量 (只对匹配到的行生效)
    aa_core = aa_core.merge(
        comp,
        left_on=["cik_norm", "fiscal_year"],
        right_on=["cik_norm", "fyear"],
        how="left",
    )
    # 合并后 fyear 列冗余，删除
    aa_core.drop(columns=["fyear"], inplace=True, errors="ignore")

    # ── 4.5 Domestic 过滤 ───────────────────────────────────────
    #  论文: "U.S. domestic firms" → FIC == "USA"
    aa_core["domestic"] = aa_core["fic"].astype(str).str.upper().eq("USA")
    # 先按 Compustat 匹配过滤
    aa_panel = aa_core[aa_core["in_compustat"]].copy()
    counts["panelA_after_compustat"] = len(aa_panel)
    counts["panelA_delete_non_domestic"] = int(
        (~aa_panel["domestic"]).sum()
    )
    aa_panel = aa_panel[aa_panel["domestic"]].copy()
    counts["panelA_after_domestic"] = len(aa_panel)

    # ── 4.6 City-industry-year ≥ 2 ─────────────────────────────
    #  论文 Table 1: remove city-industry-year cells < 2 obs
    cell_n = aa_panel.groupby(
        ["fiscal_year", "sic2", "msa_code"]
    )["company_fkey"].transform("size")
    aa_panel["city_ind_n"] = cell_n
    counts["panelA_delete_city_ind_lt2"] = int((cell_n < 2).sum())
    aa_panel = aa_panel[cell_n >= 2].copy()
    counts["panelA_final"] = len(aa_panel)

    # ── 4.7 Eq(1): 截面回归估计 ────────────────────────────────
    #  论文 p.110: "estimate equation (1) cross-sectionally for each
    #  two-digit SIC industry in each year using all available
    #  observations in Compustat"
    #  → 注意: Eq(1) 在全部 Compustat 上估计，不限于 AA 样本
    print("[p04] 估计 Eq(1) (截面 Modified Jones + ROA) ...")
    eq1 = comp[(comp["fyear"] >= cfg.year_start) &
               (comp["fyear"] <= cfg.year_end)].copy()
    eq1["sic2"] = (eq1["sic"] // 100).astype("Int64")
    eq1_vars = ["ta", "inv_lag_at", "delta_rev", "ppe_scaled", "roa_lag"]
    eq1 = eq1.dropna(subset=eq1_vars + ["sic2", "fyear"])

    # 截断 Eq(1) 变量 top/bottom 1% (论文 p.110)
    counts["eq1_before_truncate"] = len(eq1)
    eq1 = truncate_1pct_rows(eq1, eq1_vars)
    counts["eq1_after_truncate"] = len(eq1)

    coefs = []
    for (fy, sic2), g in eq1.groupby(["fyear", "sic2"]):
        if len(g) < cfg.eq1_min_obs:
            continue
        # 论文 Eq(1): 无截距项, 1/A_{t-1} 充当比例截距
        X = g[["inv_lag_at", "delta_rev", "ppe_scaled", "roa_lag"]]
        try:
            r = sm.OLS(g["ta"], X).fit()
        except Exception:
            continue
        coefs.append({
            "fyear": fy, "sic2": int(sic2),
            "b0": r.params.get("inv_lag_at", np.nan),
            "b1": r.params.get("delta_rev", np.nan),
            "b2": r.params.get("ppe_scaled", np.nan),
            "b3": r.params.get("roa_lag", np.nan),
        })
    coef_df = pd.DataFrame(coefs)
    counts["eq1_industry_year_cells"] = len(coef_df)

    # 保存 Eq(1) 系数
    coef_path = pipe / "s04_eq1_coefs.csv"
    coef_df.to_csv(coef_path, index=False)

    # ── 4.8 计算 ETA (Eq 2) 和 DACC (Eq 3) ─────────────────────
    # 先用未 winsorized 的变量按 Eq(2)/(3) 计算 ETA 与 DACC。
    # Eq(1) 的稳健性来自截面估计时的 1% 截断（见上一步）。
    aa_panel["sic2_int"] = aa_panel["sic2"].astype("Int64")
    aa_panel = aa_panel.merge(
        coef_df,
        left_on=["fiscal_year", "sic2_int"],
        right_on=["fyear", "sic2"],
        how="left",
        suffixes=("", "_eq1"),
    )
    # 清理合并列
    aa_panel.drop(columns=["fyear", "sic2_eq1"],
                  inplace=True, errors="ignore")

    # Eq(2): 用原始自变量代入 Eq(1) 系数 → ETA
    aa_panel["eta"] = (
        aa_panel["b0"] * aa_panel["inv_lag_at"]
        + aa_panel["b1"] * (aa_panel["delta_rev"] - aa_panel["delta_rec"])
        + aa_panel["b2"] * aa_panel["ppe_scaled"]
        + aa_panel["b3"] * aa_panel["roa_lag"]
    )

    # Eq(3): DACC = TA - ETA（此时仍是原始值）
    aa_panel["dacc"] = aa_panel["ta"] - aa_panel["eta"]
    aa_panel["abs_dacc"] = aa_panel["dacc"].abs()

    # ── 4.9 审计师控制变量 ──────────────────────────────────────
    # 按用户指定口径：
    #   BIG4 = 1 if auditor_fkey in {1,2,3,4}
    #   SEC_TIER = 1 if auditor_fkey in {5,6}
    aud_key = pd.to_numeric(aa_panel["auditor_fkey"], errors="coerce")
    aa_panel["big4"] = aud_key.isin([1, 2, 3, 4]).astype(float)
    aa_panel["sec_tier"] = aud_key.isin([5, 6]).astype(float)
    aa_panel["gc"] = pd.to_numeric(
        aa_panel["going_concern"], errors="coerce"
    )

    # ── 4.10 统一一次 winsorize（Eq 4 / Eq 6 连续变量）──────────
    # 按“准备好回归 panel 后统一 winsorize 一次”的流程，
    # 避免在后续步骤重复 winsorize。
    eq46_cont_vars = [
        "dacc", "abs_dacc",
        "size", "sigma_cfo", "cfo", "lev", "mb", "altman",
        "tenure_ln", "ta_lag_abs",
        "sigma_earn", "roa", "accr",
    ]
    aa_panel = winsorize_columns(
        aa_panel,
        [c for c in eq46_cont_vars if c in aa_panel.columns],
        lo=cfg.winsor_lo,
        hi=cfg.winsor_hi,
    )

    # ── 输出 ─────────────────────────────────────────────────────
    out_path = pipe / "s04_working_sample.csv"
    aa_panel.to_csv(out_path, index=False)

    attrition = {
        "panelA_start_obs": counts.get("panelA_start"),
        "panelA_delete_not_in_compustat": counts.get(
            "panelA_delete_not_in_compustat"
        ),
        "panelA_after_compustat": counts.get("panelA_after_compustat"),
        "panelA_delete_non_domestic": counts.get("panelA_delete_non_domestic"),
        "panelA_after_domestic": counts.get("panelA_after_domestic"),
        "panelA_delete_city_ind_lt2": counts.get("panelA_delete_city_ind_lt2"),
        "panelA_final": counts.get("panelA_final"),
        "eq1_industry_year_cells": counts.get("eq1_industry_year_cells"),
    }
    attrition_path = pipe / "s04_attrition.json"
    attrition_path.write_text(
        json.dumps(attrition, indent=2, ensure_ascii=False)
    )

    print(f"[p04] 完成 → {out_path}  ({len(aa_panel)} rows)")
    print(f"[p04] 样本递减:")
    for k, v in attrition.items():
        print(f"       {k}: {v}")
    return out_path


if __name__ == "__main__":
    cfg = PipelineConfig.from_cli()
    run(cfg)
