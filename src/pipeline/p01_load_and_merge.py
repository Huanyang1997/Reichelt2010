#!/usr/bin/env python3
"""
p01_load_and_merge.py — 第一步：原始数据加载与 AA 表合并
=========================================================
对应论文数据来源：
  - Audit Analytics: audit fees, revised audit opinions, auditor_during
  - 合并键: (company_fkey, auditor_fkey, fiscal_year)
  - 保留每个 company-year 审计费最高的记录（唯一行）
  - 从 auditor_during 计算 TENURE（论文 p.111）

合并策略：
  论文描述为 "按键匹配"——key 匹配到则合并两表列，未匹配则标记缺失。
  此处用 pd.merge(how="left") 以 fee 表为基表、opinion 表为补充表，
  再显式统计匹配/未匹配行数，复现论文 Table 1 样本递减逻辑。

输出: data/processed/pipeline/s01_aa_merged.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 允许直接运行或作为模块导入
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import PipelineConfig, ensure_dirs
from pipeline.utils import norm_cik


# ═══════════════════════════════════════════════════════════════════
# Tenure 计算
# ═══════════════════════════════════════════════════════════════════

def compute_tenure_from_during(
    during: pd.DataFrame, year_start: int, year_end: int
) -> pd.DataFrame:
    """
    从 auditor_during 表推算每个 company-auditor-year 的任期年数。
    论文 p.111: "TENURE = the natural log of auditor tenure."
    tenure_years = fiscal_year − begin_year + 1 (至少为 1)。
    """
    need = ["company_fkey", "auditor_fkey",
            "event_date", "deduced_begin_date", "deduced_end_date"]
    d = during[[c for c in need if c in during.columns]].copy()
    if not {"company_fkey", "auditor_fkey"}.issubset(d.columns):
        return pd.DataFrame(
            columns=["company_fkey", "auditor_fkey",
                     "fiscal_year", "tenure_years", "tenure_ln"]
        )

    for c in ["event_date", "deduced_begin_date", "deduced_end_date"]:
        if c in d.columns:
            d[c] = pd.to_datetime(d[c], errors="coerce")
        else:
            d[c] = pd.NaT

    d["begin_date"] = d["deduced_begin_date"].fillna(d["event_date"])
    d["end_date"] = d["deduced_end_date"].fillna(d["event_date"])
    d = d.dropna(subset=["company_fkey", "auditor_fkey",
                         "begin_date", "end_date"]).copy()
    if d.empty:
        return pd.DataFrame(
            columns=["company_fkey", "auditor_fkey",
                     "fiscal_year", "tenure_years", "tenure_ln"]
        )

    # 修正 begin/end 倒置
    swap = d["end_date"] < d["begin_date"]
    if swap.any():
        tmp = d.loc[swap, "begin_date"].copy()
        d.loc[swap, "begin_date"] = d.loc[swap, "end_date"]
        d.loc[swap, "end_date"] = tmp

    d["begin_year"] = d["begin_date"].dt.year
    d["end_year"] = d["end_date"].dt.year
    d = d[(d["end_year"] >= year_start) &
          (d["begin_year"] <= year_end)].copy()
    if d.empty:
        return pd.DataFrame(
            columns=["company_fkey", "auditor_fkey",
                     "fiscal_year", "tenure_years", "tenure_ln"]
        )

    d["begin_clip"] = d["begin_year"].clip(lower=year_start, upper=year_end)
    d["end_clip"] = d["end_year"].clip(lower=year_start, upper=year_end)
    d = d[d["begin_clip"] <= d["end_clip"]].copy()
    if d.empty:
        return pd.DataFrame(
            columns=["company_fkey", "auditor_fkey",
                     "fiscal_year", "tenure_years", "tenure_ln"]
        )

    # 展开为每个 fiscal_year 一行
    d["fiscal_year"] = d.apply(
        lambda r: list(range(int(r["begin_clip"]), int(r["end_clip"]) + 1)),
        axis=1,
    )
    d = d.explode("fiscal_year")
    d["fiscal_year"] = pd.to_numeric(d["fiscal_year"], errors="coerce").astype("Int64")
    d = d.dropna(subset=["fiscal_year"]).copy()

    # 同一 company-auditor-year 保留最早 begin_year（最保守估计）
    d = d.sort_values(["company_fkey", "auditor_fkey",
                       "fiscal_year", "begin_year"])
    d = d.drop_duplicates(["company_fkey", "auditor_fkey", "fiscal_year"],
                          keep="first")

    d["tenure_years"] = (
        d["fiscal_year"].astype(int) - d["begin_year"].astype(int) + 1
    ).clip(lower=1)
    d["tenure_ln"] = np.log(d["tenure_years"].astype(float))
    return d[["company_fkey", "auditor_fkey",
              "fiscal_year", "tenure_years", "tenure_ln"]]


def compute_consecutive_tenure(df: pd.DataFrame) -> pd.DataFrame:
    """备选 tenure 计算方式：基于 fee/op 表推断连续审计年数。"""
    d = df[["company_fkey", "fiscal_year", "auditor_fkey"]].dropna().copy()
    d = d.sort_values(["company_fkey", "fiscal_year"])

    def _streak(g: pd.DataFrame) -> pd.DataFrame:
        prev_aud, prev_year, streak = None, None, 0
        out = []
        for _, r in g.iterrows():
            aud = r["auditor_fkey"]
            yr = int(r["fiscal_year"])
            if prev_aud == aud and prev_year is not None and yr == prev_year + 1:
                streak += 1
            else:
                streak = 1
            out.append(streak)
            prev_aud, prev_year = aud, yr
        g = g.copy()
        g["tenure_years"] = out
        g["tenure_ln"] = np.log(g["tenure_years"].astype(float))
        return g

    d = d.groupby("company_fkey", group_keys=False).apply(_streak)
    return d[["company_fkey", "fiscal_year", "tenure_years", "tenure_ln"]]


# ═══════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════

def run(cfg: PipelineConfig) -> Path:
    """执行第一步，返回输出文件路径。"""
    ensure_dirs(cfg)
    out_dir = Path(cfg.pipeline_dir)
    counts = {}

    # ── 1.1 读取原始表 ───────────────────────────────────────────
    print("[p01] 读取 Audit Analytics 原始表 ...")
    fee = pd.read_csv(
        cfg.aa_fee_path,
        dtype={"COMPANY_FKEY": "string", "COMPANY_KEY": "string",
               "AUDITOR_FKEY": "string"},
    )
    op = pd.read_csv(
        cfg.aa_op_path,
        dtype={"COMPANY_FKEY": "string", "COMPANY_KEY": "string",
               "AUDITOR_FKEY": "string"},
    )
    during = pd.read_csv(
        cfg.aa_during_path,
        dtype={"COMPANY_FKEY": "string", "COMPANY_KEY": "string"},
    )

    # 统一列名小写
    for df in [fee, op, during]:
        df.columns = [c.lower() for c in df.columns]

    # 保持键为 string 避免浮点漂移
    for df in [fee, op, during]:
        for k in ["company_fkey", "company_key", "auditor_fkey"]:
            if k in df.columns:
                df[k] = df[k].astype("string").str.strip()

    counts["fee_raw_rows"] = len(fee)
    counts["op_raw_rows"] = len(op)
    counts["during_raw_rows"] = len(during)

    # ── 1.2 Harmonize AA 列名 ───────────────────────────────────
    op = op.rename(columns={
        "fiscal_year_of_op": "fiscal_year",
        "fiscal_year_end_op": "fiscal_year_ended",
    })
    for col in ["fiscal_year", "audit_fees", "sic_code_fkey"]:
        if col in fee.columns:
            fee[col] = pd.to_numeric(fee[col], errors="coerce")
    for col in ["fiscal_year", "going_concern", "sic_code_fkey"]:
        if col in op.columns:
            op[col] = pd.to_numeric(op[col], errors="coerce")

    # 去空必要键
    fee = fee.dropna(subset=["company_fkey", "auditor_fkey",
                             "fiscal_year", "audit_fees"])
    op = op.dropna(subset=["company_fkey", "auditor_fkey", "fiscal_year"])

    # Opinion 去重：同一 (company_fkey, auditor_fkey, fiscal_year) 保留最新 filing
    if "file_date" in op.columns:
        op["file_date"] = pd.to_datetime(op["file_date"], errors="coerce")
        op = op.sort_values("file_date")
    op = op.drop_duplicates(
        ["company_fkey", "auditor_fkey", "fiscal_year"], keep="last"
    )

    counts["fee_after_dropna"] = len(fee)
    counts["op_after_dedup"] = len(op)

    # ── 1.3 合并 Fee + Opinion ───────────────────────────────────
    #  合并策略：以 fee 为基表，按 (company_fkey, auditor_fkey, fiscal_year)
    #  匹配 opinion 表中的 going_concern、auditor_city/state 等列。
    #  匹配不上的 fee 行保留，opinion 相关列为 NaN。
    op_cols = [
        "company_fkey", "auditor_fkey", "fiscal_year",
        "going_concern", "auditor_city", "auditor_state",
    ]
    # 若有更多位置列也一起带入
    for extra in ["loc_state_country", "bus_state_country", "mail_state_country"]:
        if extra in op.columns:
            op_cols.append(extra)
    op_cols = [c for c in op_cols if c in op.columns]

    merge_key = ["company_fkey", "auditor_fkey", "fiscal_year"]
    aa = fee.merge(op[op_cols], on=merge_key, how="left")
    counts["aa_merged_rows"] = len(aa)
    counts["aa_opinion_matched"] = int(aa["going_concern"].notna().sum())
    counts["aa_opinion_unmatched"] = int(aa["going_concern"].isna().sum())

    # 清洗 auditor 地址字段（仅来自 Opinion 表）
    aa["auditor_city"] = aa["auditor_city"].astype("string").str.strip()
    aa["auditor_state"] = (
        aa["auditor_state"].astype("string").str.upper().str.strip()
    )

    # ── 1.4 去重至每个 company-year 唯一行 ───────────────────────
    #  同一 company-year 可能对应多个 auditor，保留审计费最高的记录
    aa = aa.sort_values(
        ["company_fkey", "fiscal_year", "audit_fees"],
        ascending=[True, True, False],
    )
    aa = aa.drop_duplicates(["company_fkey", "fiscal_year"], keep="first")
    counts["aa_company_year_unique"] = len(aa)

    # ── 1.5 Tenure 计算与合并 ────────────────────────────────────
    tenure_during = compute_tenure_from_during(
        during, cfg.year_start, cfg.year_end
    )
    counts["tenure_during_rows"] = len(tenure_during)

    if cfg.tenure_source == "during":
        aa = aa.merge(
            tenure_during,
            on=["company_fkey", "auditor_fkey", "fiscal_year"],
            how="left",
        )
    elif cfg.tenure_source == "consecutive":
        tenure_consec = compute_consecutive_tenure(aa)
        aa = aa.merge(
            tenure_consec, on=["company_fkey", "fiscal_year"], how="left"
        )
    else:  # hybrid: during 优先, consecutive 补缺
        aa = aa.merge(
            tenure_during,
            on=["company_fkey", "auditor_fkey", "fiscal_year"],
            how="left",
        )
        tenure_consec = compute_consecutive_tenure(aa)
        tc = tenure_consec[["company_fkey", "fiscal_year",
                            "tenure_years", "tenure_ln"]].rename(
            columns={"tenure_years": "tenure_years_c",
                     "tenure_ln": "tenure_ln_c"}
        )
        aa = aa.merge(tc, on=["company_fkey", "fiscal_year"], how="left")
        aa["tenure_years"] = aa["tenure_years"].fillna(aa["tenure_years_c"])
        aa["tenure_ln"] = aa["tenure_ln"].fillna(aa["tenure_ln_c"])
        aa.drop(columns=["tenure_years_c", "tenure_ln_c"],
                inplace=True, errors="ignore")

    counts["tenure_nonmissing"] = int(aa["tenure_ln"].notna().sum())

    # ── 1.6 构建 CIK 标准化键（用于后续与 Compustat 匹配） ──────
    aa["cik_norm"] = (
        aa["company_fkey"].fillna(aa.get("company_key", pd.NA))
        .apply(norm_cik)
    )

    # ── 输出 ─────────────────────────────────────────────────────
    out_path = out_dir / "s01_aa_merged.csv"
    aa.to_csv(out_path, index=False)

    meta_path = out_dir / "s01_counts.json"
    meta_path.write_text(json.dumps(counts, indent=2, ensure_ascii=False))
    print(f"[p01] 完成 → {out_path}  ({len(aa)} rows)")
    print(f"[p01] 诊断 → {meta_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    cfg = PipelineConfig.from_cli()
    run(cfg)
