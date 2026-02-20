#!/usr/bin/env python3
"""
p05_table5.py — 第五步：Table 5 样本准备与 OLS 回归
=====================================================
对应论文 Table 5 — Eq(4):
  |DACC| = α + β SIZE + β σ(CFO) + β CFO + β LEV + β LOSS
           + β MB + β LIT + β ALTMAN + β TENURE + β |ACCR_{t-1}|
           + β BIG4 + β SEC_TIER
           + β SPECIALIST_VARS + ε

样本构建 (Table 1 Panel B):
  21,583  Working sample (来自 p04)
  -7,443  Missing controls or accrual inputs
  -369    |Studentized residual| > 3
  = 13,771  Table 5 final sample

关键操作顺序:
  1) 删除缺失值
  2) 预回归 (def1, model3) 计算 studentized residuals
  3) 删除 |studentized residual| > 3 — 论文 p.110
  4) 正式回归 6 个模型 (2 definitions × 3 models)

推断方式:
  论文 p.110: "Rogers (1993) robust to heteroskedasticity and
  time-series correlation" → firm-level cluster robust SE

注意:
  Table 5 不含行业/年份固定效应（论文 footnote 39: 因为异常应计
  已按行业-年截面估计，行业/年效应已被吸收）。

输入: data/processed/pipeline/s04_working_sample.csv
输出:
  data/processed/pipeline/s05_table5_sample.csv
  outputs/table5_replication_long.csv
  outputs/table5_replication_fullcoef.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import PipelineConfig, ensure_dirs
from pipeline.utils import winsorize_series


# ═══════════════════════════════════════════════════════════════════
# OLS + Cluster Robust SE
# ═══════════════════════════════════════════════════════════════════

def run_ols_cluster(df: pd.DataFrame, y: str,
                    x_cols: List[str], group_col: str):
    """
    OLS 回归 + Rogers (1993) 式 firm-cluster robust SE.
    论文 p.110: 按 firm (company_fkey) 聚类。
    """
    use = df[[y] + x_cols + [group_col]].dropna().copy()
    if use.empty:
        return None, use
    X = sm.add_constant(use[x_cols], has_constant="add")
    model = sm.OLS(use[y], X)
    if use[group_col].nunique() >= 2:
        res = model.fit(cov_type="cluster",
                        cov_kwds={"groups": use[group_col]})
    else:
        res = model.fit(cov_type="HC1")
    return res, use


# ═══════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════

def run(cfg: PipelineConfig) -> Path:
    ensure_dirs(cfg)
    pipe = Path(cfg.pipeline_dir)
    out_dir = Path(cfg.output_dir)
    counts = {}

    # ── 5.1 读入工作样本 ────────────────────────────────────────
    print("[p05] 读取工作样本 ...")
    df = pd.read_csv(pipe / "s04_working_sample.csv", dtype=str)
    # 恢复数值列
    num_cols = [
        "fiscal_year", "audit_fees", "abs_dacc", "dacc",
        "size", "sigma_cfo", "cfo", "lev", "loss", "mb", "lit",
        "altman", "tenure_ln", "ta_lag_abs", "big4", "sec_tier",
        "both_d1", "nat_only_d1", "city_only_d1",
        "both_d2", "nat_only_d2", "city_only_d2",
        "nat_spec_d1", "city_spec_d1", "nat_spec_d2", "city_spec_d2",
        "accr", "sigma_earn", "roa", "op_cf", "gc",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    counts["working_sample_n"] = len(df)

    # ── 5.2 确定控制变量列表 ────────────────────────────────────
    #  论文 Eq(4) 控制变量:
    #  SIZE, σ(CFO), CFO, LEV, LOSS, MB, LIT, ALTMAN, TENURE,
    #  |ACCR_{t-1}|, BIG4, SEC_TIER
    base_controls = [
        "size", "sigma_cfo", "cfo", "lev", "loss", "mb", "lit",
        "tenure_ln", "ta_lag_abs", "big4", "sec_tier",
    ]
    if df["altman"].notna().mean() > 0.2:
        controls = base_controls[:7] + ["altman"] + base_controls[7:]
    else:
        if not cfg.allow_reduced_model:
            raise RuntimeError(
                "Altman 输入缺失严重，请补充 ACT/LCT/RE/EBIT"
                " 或使用 --allow-reduced-model"
            )
        controls = base_controls

    # ── 5.3 删除缺失值 ─────────────────────────────────────────
    spec_cols = [
        "both_d1", "nat_only_d1", "city_only_d1",
        "both_d2", "nat_only_d2", "city_only_d2",
    ]
    required = ["abs_dacc"] + controls + spec_cols
    t5 = df.dropna(subset=required).copy()
    counts["delete_missing"] = int(len(df) - len(t5))
    counts["after_dropna"] = len(t5)
    # 连续变量已在 p04 统一 winsorize，这里不重复处理。

    # 按需求：回归前对 ALTMAN 再做一次 winsorize（1%/99%）。
    if "altman" in controls and t5["altman"].notna().any():
        q_lo = float(t5["altman"].quantile(cfg.winsor_lo))
        q_hi = float(t5["altman"].quantile(cfg.winsor_hi))
        t5["altman"] = winsorize_series(
            t5["altman"], lo=cfg.winsor_lo, hi=cfg.winsor_hi
        )
        counts["altman_rewinsor_lo"] = q_lo
        counts["altman_rewinsor_hi"] = q_hi

    # ── 5.4 异常值剔除 (Studentized Residual) ──────────────────
    #  论文 p.110: "|studentized residual| > 3 的观测被剔除"
    #  用 def1 model3 做预回归
    prelim_x = controls + ["both_d1", "nat_only_d1", "city_only_d1"]
    prelim_res, prelim_use = run_ols_cluster(
        t5, "abs_dacc", prelim_x, "company_fkey"
    )
    t5_before = len(t5)
    if prelim_res is not None and not prelim_use.empty:
        # 标准 OLS 估计（非 cluster）用于 influence diagnostics
        X_p = sm.add_constant(prelim_use[prelim_x], has_constant="add")
        infl = OLSInfluence(sm.OLS(prelim_use["abs_dacc"], X_p).fit())
        stud = infl.resid_studentized_external
        keep = np.abs(stud) <= cfg.stud_resid_threshold
        t5 = t5.loc[prelim_use.index[keep]].copy()

    counts["delete_outlier"] = int(t5_before - len(t5))
    counts["table5_final_n"] = len(t5)

    # 保存 Table 5 分析样本
    t5.to_csv(pipe / "s05_table5_sample.csv", index=False)

    # ── 5.5 正式回归：6 个模型 ─────────────────────────────────
    #  2 definitions × 3 models
    #  Model 1: NAT_SPEC  (论文 Table 5 第一列)
    #  Model 2: CITY_SPEC (论文 Table 5 第二列)
    #  Model 3: BOTH + NAT_ONLY + CITY_ONLY (论文 Table 5 第三列)
    print("[p05] 运行 Table 5 OLS 回归 ...")
    rows_long = []
    rows_full = []
    model_results: Dict[Tuple[int, str], object] = {}

    for dnum in [1, 2]:
        models = [
            ("model1",
             controls + [f"nat_spec_d{dnum}"],
             [f"nat_spec_d{dnum}"]),
            ("model2",
             controls + [f"city_spec_d{dnum}"],
             [f"city_spec_d{dnum}"]),
            ("model3",
             controls + [f"both_d{dnum}", f"nat_only_d{dnum}",
                         f"city_only_d{dnum}"],
             [f"both_d{dnum}", f"nat_only_d{dnum}",
              f"city_only_d{dnum}"]),
        ]
        for mname, xcols, terms in models:
            res, use = run_ols_cluster(
                t5, "abs_dacc", xcols, "company_fkey"
            )
            if res is None:
                for t in terms:
                    rows_long.append({
                        "definition": dnum, "model": mname, "term": t,
                        "coef": np.nan, "pvalue": np.nan,
                        "n": 0, "adj_r2": np.nan,
                    })
                continue

            model_results[(dnum, mname)] = res

            # 完整系数表
            for t in res.params.index:
                rows_full.append({
                    "definition": dnum, "model": mname, "term": t,
                    "coef": float(res.params[t]),
                    "pvalue": float(res.pvalues[t]),
                    "n": int(res.nobs),
                    "adj_r2": float(res.rsquared_adj),
                    "f_value": float(getattr(res, "fvalue", np.nan)),
                    "f_pvalue": float(getattr(res, "f_pvalue", np.nan)),
                })

            # 关键变量概要
            for t in terms:
                rows_long.append({
                    "definition": dnum, "model": mname, "term": t,
                    "coef": float(res.params.get(t, np.nan)),
                    "pvalue": float(res.pvalues.get(t, np.nan)),
                    "n": int(res.nobs),
                    "adj_r2": float(res.rsquared_adj),
                })

    # ── 输出 ─────────────────────────────────────────────────────
    pd.DataFrame(rows_long).to_csv(
        out_dir / "table5_replication_long.csv", index=False
    )
    pd.DataFrame(rows_full).to_csv(
        out_dir / "table5_replication_fullcoef.csv", index=False
    )

    meta = pipe / "s05_counts.json"
    meta.write_text(json.dumps(counts, indent=2, ensure_ascii=False))

    print(f"[p05] 完成 → N={counts['table5_final_n']}")
    for dnum in [1, 2]:
        key = (dnum, "model3")
        if key in model_results:
            r = model_results[key]
            for t in [f"both_d{dnum}", f"nat_only_d{dnum}",
                      f"city_only_d{dnum}"]:
                if t in r.params.index:
                    print(f"       D{dnum} {t}: "
                          f"coef={r.params[t]:.4f}, "
                          f"p={r.pvalues[t]:.4f}")

    return pipe / "s05_table5_sample.csv"


if __name__ == "__main__":
    cfg = PipelineConfig.from_cli()
    run(cfg)
