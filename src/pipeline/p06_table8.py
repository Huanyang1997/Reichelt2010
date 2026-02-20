#!/usr/bin/env python3
"""
p06_table8.py — 第六步：Table 8 样本准备与 Logit 回归
======================================================
对应论文 Table 8 — Eq(6):
  Pr(GC=1) = Logit(α + β SIZE + β σ(EARN) + β LEV + β LOSS
                    + β ROA + β MB + β LIT + β ALTMAN + β TENURE
                    + β ACCR + β BIG4 + β SEC_TIER
                    + β SPECIALIST_VARS
                    + industry FE + year FE)

样本构建 (Table 1 Panel D):
  21,583  Working sample
  -6,136  Missing controls
  -10,472 Non-distressed (OCF ≥ 0)
  -6      |Deviance residual| > 3
  = 4,969  Table 8 final sample

关键操作顺序:
  1) 删除缺失值
  2) 仅保留 financially distressed (OCF < 0) — 论文 p.112
  3) 预回归 logit (def1 model3) 计算 deviance residuals
  4) 删除 |deviance residual| > 3
  5) 正式回归 6 个模型

推断方式:
  Logit + firm-level cluster robust SE (Rogers 1993)

固定效应:
  论文 p.112: "industry indicators (two-digit SIC dummies) and
  year fixed effects" → C(sic2) + C(fiscal_year)

输入: data/processed/pipeline/s04_working_sample.csv
输出:
  data/processed/pipeline/s06_table8_sample.csv
  outputs/table8_replication_long.csv
  outputs/table8_replication_fullcoef.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

try:
    from scipy.stats import chi2
except Exception:
    chi2 = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import PipelineConfig, ensure_dirs
from pipeline.utils import winsorize_series


# ═══════════════════════════════════════════════════════════════════
# Logit + Cluster Robust SE
# ═══════════════════════════════════════════════════════════════════

def run_logit_cluster(df: pd.DataFrame, formula: str, group_col: str):
    """
    GLM (Binomial/Logit) + firm-cluster robust SE.
    论文 p.112: 按 firm 聚类。
    """
    use = df.dropna(subset=[group_col]).copy()
    if use.empty:
        return None, use
    model = smf.glm(formula=formula, data=use,
                    family=sm.families.Binomial())
    try:
        if use[group_col].nunique() >= 2:
            res = model.fit(cov_type="cluster",
                            cov_kwds={"groups": use[group_col]})
        else:
            res = model.fit()
        # 防御 singular robust covariance
        if hasattr(res, "pvalues") and np.isnan(
            np.asarray(res.pvalues)
        ).all():
            res = model.fit()
    except Exception:
        return None, use
    return res, use


# ═══════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════

def run(cfg: PipelineConfig) -> Path:
    ensure_dirs(cfg)
    pipe = Path(cfg.pipeline_dir)
    out_dir = Path(cfg.output_dir)
    counts = {}

    # ── 6.1 读入工作样本 ────────────────────────────────────────
    print("[p06] 读取工作样本 ...")
    df = pd.read_csv(pipe / "s04_working_sample.csv", dtype=str)
    num_cols = [
        "fiscal_year", "gc", "op_cf",
        "size", "sigma_earn", "lev", "loss", "roa", "mb", "lit",
        "altman", "tenure_ln", "accr", "big4", "sec_tier",
        "both_d1", "nat_only_d1", "city_only_d1",
        "both_d2", "nat_only_d2", "city_only_d2",
        "nat_spec_d1", "city_spec_d1", "nat_spec_d2", "city_spec_d2",
        "sic2",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    counts["working_sample_n"] = len(df)

    # ── 6.2 确定控制变量 ───────────────────────────────────────
    #  论文 Eq(6):
    #  SIZE, σ(EARN), LEV, LOSS, ROA, MB, LIT, ALTMAN,
    #  TENURE, ACCR, BIG4, SEC_TIER
    base_controls = [
        "size", "sigma_earn", "lev", "loss", "roa", "mb", "lit",
        "tenure_ln", "accr", "big4", "sec_tier",
    ]
    if df["altman"].notna().mean() > 0.2:
        controls = base_controls[:7] + ["altman"] + base_controls[7:]
    else:
        if not cfg.allow_reduced_model:
            raise RuntimeError("Altman 缺失，请使用 --allow-reduced-model")
        controls = base_controls

    # ── 6.3 删除缺失值 ─────────────────────────────────────────
    spec_cols = [
        "both_d1", "nat_only_d1", "city_only_d1",
        "both_d2", "nat_only_d2", "city_only_d2",
    ]
    required = ["gc", "op_cf"] + controls + spec_cols + [
        "sic2", "fiscal_year",
    ]
    t8 = df.dropna(subset=required).copy()
    counts["delete_missing"] = int(len(df) - len(t8))
    # 连续变量已在 p04 统一 winsorize，这里不重复处理。

    # ── 6.4 保留 financially distressed (OCF < 0) ──────────────
    #  论文 p.112: "we consider as financially distressed only those
    #  firms with negative operating cash flows"
    counts["delete_non_distressed"] = int((t8["op_cf"] >= 0).sum())
    t8 = t8[t8["op_cf"] < 0].copy()

    # 按需求：回归前对 ALTMAN 再做一次 winsorize（1%/99%）。
    if "altman" in controls and t8["altman"].notna().any():
        q_lo = float(t8["altman"].quantile(cfg.winsor_lo))
        q_hi = float(t8["altman"].quantile(cfg.winsor_hi))
        t8["altman"] = winsorize_series(
            t8["altman"], lo=cfg.winsor_lo, hi=cfg.winsor_hi
        )
        counts["altman_rewinsor_lo"] = q_lo
        counts["altman_rewinsor_hi"] = q_hi

    # ── 6.5 GC 二值化 + FE 分类变量 ────────────────────────────
    t8["gc"] = (t8["gc"] > 0).astype(int)
    t8["sic2_cat"] = t8["sic2"].astype("Int64").astype(str)
    t8["fiscal_year_cat"] = (
        pd.to_numeric(t8["fiscal_year"], errors="coerce")
        .astype("Int64").astype(str)
    )
    counts["before_outlier"] = len(t8)

    # ── 6.6 异常值剔除 (Deviance Residual) ─────────────────────
    #  论文: |deviance residual| > 3 → 用 def1 model3 预回归
    f_controls = " + ".join(controls)
    prelim_formula = (
        f"gc ~ {f_controls} + both_d1 + nat_only_d1 + city_only_d1"
        f" + C(sic2_cat) + C(fiscal_year_cat)"
    )
    prelim_res, prelim_use = run_logit_cluster(
        t8, prelim_formula, "company_fkey"
    )
    t8_before = len(t8)
    if prelim_res is not None and not prelim_use.empty:
        dev = prelim_res.resid_deviance
        t8 = prelim_use.loc[np.abs(dev) <= cfg.dev_resid_threshold].copy()

    counts["delete_outlier"] = int(t8_before - len(t8))
    counts["table8_final_n"] = len(t8)

    # 保存 Table 8 分析样本
    t8.to_csv(pipe / "s06_table8_sample.csv", index=False)

    # ── 6.7 正式回归：6 个模型 ─────────────────────────────────
    print("[p06] 运行 Table 8 Logit 回归 ...")
    rows_long = []
    rows_full = []
    model_results: Dict[Tuple[int, str], object] = {}

    for dnum in [1, 2]:
        formulas = [
            ("model1",
             f"gc ~ {f_controls} + nat_spec_d{dnum}"
             f" + C(sic2_cat) + C(fiscal_year_cat)",
             [f"nat_spec_d{dnum}"]),
            ("model2",
             f"gc ~ {f_controls} + city_spec_d{dnum}"
             f" + C(sic2_cat) + C(fiscal_year_cat)",
             [f"city_spec_d{dnum}"]),
            ("model3",
             f"gc ~ {f_controls} + both_d{dnum} + nat_only_d{dnum}"
             f" + city_only_d{dnum}"
             f" + C(sic2_cat) + C(fiscal_year_cat)",
             [f"both_d{dnum}", f"nat_only_d{dnum}",
              f"city_only_d{dnum}"]),
        ]
        for mname, formula, terms in formulas:
            res, use = run_logit_cluster(t8, formula, "company_fkey")
            if res is None:
                for t in terms:
                    rows_long.append({
                        "definition": dnum, "model": mname, "term": t,
                        "coef": np.nan, "pvalue": np.nan,
                        "n": 0, "pseudo_r2": np.nan,
                    })
                continue

            model_results[(dnum, mname)] = res

            # 计算拟合优度指标
            lr_stat, lr_p = np.nan, np.nan
            if hasattr(res, "null_deviance") and hasattr(res, "deviance"):
                nd = getattr(res, "null_deviance", np.nan)
                dv = getattr(res, "deviance", np.nan)
                if pd.notna(nd) and pd.notna(dv):
                    lr_stat = float(nd - dv)
                    df_m = getattr(res, "df_model", np.nan)
                    if (chi2 is not None and pd.notna(df_m)
                            and float(df_m) > 0):
                        lr_p = float(chi2.sf(lr_stat, float(df_m)))

            pseudo_r2 = np.nan
            llnull = getattr(res, "llnull", np.nan)
            llf = getattr(res, "llf", np.nan)
            if (pd.notna(llnull) and pd.notna(llf)
                    and float(llnull) != 0):
                pseudo_r2 = float(1.0 - (float(llf) / float(llnull)))
            elif hasattr(res, "null_deviance") and hasattr(res, "deviance"):
                nd = getattr(res, "null_deviance", np.nan)
                dv = getattr(res, "deviance", np.nan)
                if pd.notna(nd) and pd.notna(dv) and float(nd) != 0:
                    pseudo_r2 = float(1.0 - (float(dv) / float(nd)))

            # 完整系数表
            for t in res.params.index:
                rows_full.append({
                    "definition": dnum, "model": mname, "term": t,
                    "coef": float(res.params[t]),
                    "pvalue": float(res.pvalues[t]),
                    "n": int(res.nobs),
                    "lr_stat": lr_stat, "lr_pvalue": lr_p,
                    "pseudo_r2": pseudo_r2,
                })

            # 关键变量概要
            for t in terms:
                rows_long.append({
                    "definition": dnum, "model": mname, "term": t,
                    "coef": float(res.params.get(t, np.nan)),
                    "pvalue": float(res.pvalues.get(t, np.nan)),
                    "n": int(res.nobs),
                    "pseudo_r2": pseudo_r2,
                })

    # ── 输出 ─────────────────────────────────────────────────────
    pd.DataFrame(rows_long).to_csv(
        out_dir / "table8_replication_long.csv", index=False
    )
    pd.DataFrame(rows_full).to_csv(
        out_dir / "table8_replication_fullcoef.csv", index=False
    )

    meta = pipe / "s06_counts.json"
    meta.write_text(json.dumps(counts, indent=2, ensure_ascii=False))

    print(f"[p06] 完成 → N={counts['table8_final_n']}")
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

    return pipe / "s06_table8_sample.csv"


if __name__ == "__main__":
    cfg = PipelineConfig.from_cli()
    run(cfg)
