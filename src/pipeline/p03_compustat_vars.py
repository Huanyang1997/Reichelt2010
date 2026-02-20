#!/usr/bin/env python3
"""
p03_compustat_vars.py — 第三步：Compustat 财务变量构造
======================================================
严格按论文变量定义构造，每个变量均标注论文方程号与 Compustat 字段。

变量对照表 (与论文 p.110-112 一一对应):
───────────────────────────────────────────────────────────
 论文变量     | 公式                          | Compustat 字段
───────────────────────────────────────────────────────────
 TA           | (IB - OCF) / A_{t-1}          | ib, oancf, xidoc, at
 1/A_{t-1}    | 1 / A_{t-1}                   | at
 ΔREV         | (SALE_t - SALE_{t-1}) / A_{t-1}| sale, at
 ΔREC         | (RECT_t - RECT_{t-1}) / A_{t-1}| rect, at
 PPE          | PPEGT / A_{t-1}               | ppegt, at
 ROA_{t-1}    | NI_{t-1} / avg(AT_{t-1},AT_{t-2}) | ni, at
 SIZE         | ln(CSHO × PRCC_F)             | csho, prcc_f
 LEV          | DLTT / AT                     | dltt, at
 LOSS         | 1{NI < 0}                     | ni
 MB           | (CSHO × PRCC_F) / CEQ         | csho, prcc_f, ceq
 LIT          | 诉讼风险行业虚拟变量          | sic
 ALTMAN       | Z(1968) = 1.2X1+1.4X2+3.3X3+0.6X4+1.0X5 | act,lct,re,ebit,csho,prcc_f,lt,sale,at
 CFO          | OCF / A_{t-1}                 | oancf, xidoc, at
 σ(CFO)       | 5-year rolling std of CFO     | —
 σ(EARN)      | 5-year rolling std of IB/A_{t-1} | ib, at
 ACCR         | TA (= (IB-OCF)/A_{t-1})       | —
 ROA          | NI / avg(AT_t, AT_{t-1})      | ni, at
 OCF_raw      | OANCF - XIDOC (未缩放)        | oancf, xidoc

注:
  按当前项目设定，SIZE 使用 PRCC_F（年度收盘价）口径。

输入: Compustat funda (原始 dta/csv)
输出: data/processed/pipeline/s03_compustat_vars.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import PipelineConfig, ensure_dirs
from pipeline.utils import norm_cik, load_compustat, winsorize_series


def run(cfg: PipelineConfig) -> Path:
    ensure_dirs(cfg)
    pipe = Path(cfg.pipeline_dir)
    counts = {}

    # ── 3.1 读取 Compustat ──────────────────────────────────────
    print("[p03] 读取 Compustat funda ...")
    comp = load_compustat(cfg.comp_funda_path)
    counts["comp_raw_rows"] = len(comp)

    # 数值化
    num_cols = [
        "fyear", "sic", "at", "dltt", "lt", "ppegt", "rect", "ib", "ni",
        "sale", "oancf", "xidoc", "csho", "prcc_f", "ceq",
        "act", "lct", "re", "ebit",
    ]
    for c in num_cols:
        if c in comp.columns:
            comp[c] = pd.to_numeric(comp[c], errors="coerce")
    if "datadate" in comp.columns:
        comp["datadate"] = pd.to_datetime(comp["datadate"], errors="coerce")

    # ── 3.2 标准 Compustat 过滤 ─────────────────────────────────
    for col, val in [("indfmt", "INDL"), ("datafmt", "STD"), ("consol", "C")]:
        if col in comp.columns:
            comp = comp[comp[col].astype(str).str.upper() == val]
    if cfg.comp_costat == "A" and "costat" in comp.columns:
        comp = comp[comp["costat"].astype(str).str.upper() == "A"]

    comp["cik_norm"] = comp["cik"].apply(norm_cik)
    # Compustat 1996 起（为计算 5 年滚动 sigma 预留前期数据）
    comp = comp[(comp["fyear"] >= 1996) &
                (comp["fyear"] <= cfg.year_end)].copy()
    comp = comp.sort_values(["cik_norm", "fyear"])
    counts["comp_after_filter"] = len(comp)
    counts["comp_unique_cik"] = int(comp["cik_norm"].nunique())

    # ── 3.3 滞后变量 ────────────────────────────────────────────
    grp = comp.groupby("cik_norm")
    comp["lag_at"] = grp["at"].shift(1)
    comp["lag2_at"] = grp["at"].shift(2)
    comp["lag_sale"] = grp["sale"].shift(1)
    comp["lag_rect"] = grp["rect"].shift(1)
    comp["lag_ib"] = grp["ib"].shift(1)
    comp["lag_ni"] = grp["ni"].shift(1)

    # ── 3.4 Eq(1) 变量（全部按 A_{t-1} 缩放） ──────────────────
    #  论文 Eq(1): TA = β0(1/A_{t-1}) + β1 ΔREV + β2 PPE + β3 ROA_{t-1} + ε
    # 不做“分母正值保护”，保持与原文口径一致。

    # OCF 按论文口径使用 OANCF - XIDOC；XIDOC 缺失按 0 处理。
    # 这样不会因为 XIDOC 缺失而丢观测，只在 OANCF/分母缺失时丢。
    xidoc_adj = comp["xidoc"].fillna(0) if "xidoc" in comp.columns else 0.0
    comp["op_cf"] = comp["oancf"] - xidoc_adj

    comp["ta"] = (comp["ib"] - comp["op_cf"]) / comp["lag_at"]
    comp["inv_lag_at"] = 1.0 / comp["lag_at"]
    comp["delta_rev"] = (comp["sale"] - comp["lag_sale"]) / comp["lag_at"]
    comp["delta_rec"] = (comp["rect"] - comp["lag_rect"]) / comp["lag_at"]
    comp["ppe_scaled"] = comp["ppegt"] / comp["lag_at"]

    # ROA_{t-1} = NI_{t-1} / avg(AT_{t-1}, AT_{t-2})
    #  论文 p.110: "average total assets in year t-1"
    comp["avg_at_lag"] = (comp["lag_at"] + comp["lag2_at"]) / 2
    comp["roa_lag"] = comp["lag_ni"] / comp["avg_at_lag"]

    # ── 3.5 Table 5/8 控制变量 ──────────────────────────────────

    # CFO = OCF / A_{t-1}  (论文 p.111)
    comp["cfo"] = comp["op_cf"] / comp["lag_at"]
    # Earnings scaled by lagged assets (for sigma)
    comp["earn_scaled"] = comp["ib"] / comp["lag_at"]

    # σ(CFO): rolling std from t-4 to t (含当期, 5年窗口)
    #  论文 p.111: "from t − 4 to t"
    comp["sigma_cfo"] = grp["cfo"].transform(
        lambda s: s.rolling(5, min_periods=5).std()
    )
    # σ(EARN): rolling std from t-4 to t (含当期, 5年窗口)
    #  论文 p.112: 同 σ(CFO) 定义
    comp["sigma_earn"] = grp["earn_scaled"].transform(
        lambda s: s.rolling(5, min_periods=5).std()
    )

    # |ACCR_{t-1}| (论文 p.111: absolute value of accruals in year t-1)
    comp["ta_lag_abs"] = grp["ta"].shift(1).abs()
    # ACCR for Table 8 (总应计 / A_{t-1})
    comp["accr"] = comp["ta"]

    # ROA for Table 8 = NI / avg(AT_t, AT_{t-1})
    comp["avg_at"] = (comp["at"] + comp["lag_at"]) / 2
    comp["roa"] = comp["ni"] / comp["avg_at"]

    # ── 3.6 SIZE = ln(CSHO × PRCC_F) ────────────────────────────
    if "prcc_f" not in comp.columns:
        raise RuntimeError("Compustat 缺少 PRCC_F，无法构造 SIZE/MB")
    if "ceq" not in comp.columns:
        raise RuntimeError("Compustat 缺少 CEQ，无法构造 MB/ALTMAN")

    comp["mve"] = comp["csho"] * comp["prcc_f"]
    # SIZE 仅在 MVE>0 时有定义；MVE<=0 的观测记为缺失并在后续样本筛选中删除。
    comp.loc[comp["mve"] <= 0, "mve"] = np.nan
    comp["size"] = np.log(comp["mve"])

    # ── 3.7 LEV, LOSS, MB ──────────────────────────────────────
    comp["lev"] = comp["dltt"] / comp["at"]
    comp["loss"] = (comp["ni"] < 0).astype(float)
    # MB = MVE / BVE, 论文 p.111: BVE = total assets − total liabilities (LT)
    comp["bve"] = comp["at"] - comp["lt"]
    comp["mb"] = comp["mve"] / comp["bve"]
    comp["mb"] = comp["mb"].replace([np.inf, -np.inf], np.nan)

    # ── 3.8 LIT (诉讼风险行业, Shu 2000) ────────────────────────
    sic = comp["sic"]
    comp["lit"] = (
        sic.between(2833, 2836) | sic.between(3570, 3577)
        | sic.between(3600, 3674) | sic.between(5200, 5961)
        | sic.between(7370, 7370)
    ).astype(float)

    # ── 3.9 ALTMAN Z (1968, public firms) ───────────────────────
    #  Z = 1.2 X1 + 1.4 X2 + 3.3 X3 + 0.6 X4 + 1.0 X5
    #  X1 = (ACT-LCT)/AT, X2 = RE/AT, X3 = EBIT/AT,
    #  X4 = MVE/TL where MVE = CSHO*PRCC_F, X5 = SALE/AT
    altman_inputs = ["act", "lct", "re", "ebit", "csho", "prcc_f", "lt", "sale", "at"]
    missing_altman = [c for c in altman_inputs if c not in comp.columns]
    if missing_altman:
        comp["altman"] = np.nan
        counts["altman_status"] = f"missing cols: {missing_altman}"
    else:
        # 先构造 Altman 组成项，再对组成项做 winsorize（1%/99%）。
        # 按 fiscal year 截面 winsorize，降低跨期分布漂移影响。
        comp["alt_x1"] = (comp["act"] - comp["lct"]) / comp["at"]
        comp["alt_x2"] = comp["re"] / comp["at"]
        comp["alt_x3"] = comp["ebit"] / comp["at"]
        alt_mve = comp["csho"] * comp["prcc_f"]
        comp["alt_x4"] = alt_mve / comp["lt"]
        comp["alt_x5"] = comp["sale"] / comp["at"]

        for x in ["alt_x1", "alt_x2", "alt_x3", "alt_x4", "alt_x5"]:
            comp[x] = (
                comp.groupby("fyear")[x]
                .transform(
                    lambda s: winsorize_series(
                        s, lo=cfg.winsor_lo, hi=cfg.winsor_hi
                    )
                )
            )

        comp["altman"] = (
            1.2 * comp["alt_x1"]
            + 1.4 * comp["alt_x2"]
            + 3.3 * comp["alt_x3"]
            + 0.6 * comp["alt_x4"]
            + 1.0 * comp["alt_x5"]
        )
        comp["altman"] = comp["altman"].replace([np.inf, -np.inf], np.nan)
        counts["altman_status"] = "OK"
        counts["altman_valid_pct"] = round(
            float(comp["altman"].notna().mean()) * 100, 1
        )

    # 比例变量统一清理 inf，避免后续回归报错
    ratio_cols = [
        "ta", "inv_lag_at", "delta_rev", "delta_rec", "ppe_scaled",
        "roa_lag", "cfo", "earn_scaled", "roa", "lev", "mb",
        "alt_x1", "alt_x2", "alt_x3", "alt_x4", "alt_x5",
    ]
    for c in ratio_cols:
        if c in comp.columns:
            comp[c] = comp[c].replace([np.inf, -np.inf], np.nan)

    # ── 3.10 FIC (国内标志, 用于后续 domestic 过滤) ─────────────
    if "fic" in comp.columns:
        comp["fic"] = comp["fic"].astype(str).str.upper().str.strip()
    else:
        comp["fic"] = pd.NA

    # ── 输出 ─────────────────────────────────────────────────────
    out_cols = [
        "cik_norm", "fyear", "sic", "fic",
        # Eq(1) 变量
        "ta", "inv_lag_at", "delta_rev", "delta_rec", "ppe_scaled", "roa_lag",
        # Table 5/8 控制变量
        "size", "sigma_cfo", "cfo", "lev", "loss", "mb", "lit", "altman",
        "ta_lag_abs", "accr", "sigma_earn", "roa",
        # 未缩放 OCF (用于 Table 8 distress 筛选)
        "op_cf",
    ]
    out_cols = [c for c in out_cols if c in comp.columns]
    comp_out = comp[out_cols].copy()

    out_path = pipe / "s03_compustat_vars.csv"
    comp_out.to_csv(out_path, index=False)

    meta = pipe / "s03_counts.json"
    counts["output_rows"] = len(comp_out)
    (meta).write_text(json.dumps(counts, indent=2, ensure_ascii=False))
    print(f"[p03] 完成 → {out_path}  ({len(comp_out)} rows)")
    return out_path


if __name__ == "__main__":
    cfg = PipelineConfig.from_cli()
    run(cfg)
