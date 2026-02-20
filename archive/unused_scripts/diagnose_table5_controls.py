#!/usr/bin/env python3
"""
diagnose_table5_controls.py
============================
系统诊断：排列组合控制变量，找出哪些控制变量导致核心自变量
(Specialist) 系数方向翻转或显著性消失。

测试方案:
  1. Bivariate: 仅 specialist 变量，无控制变量
  2. Plus-one: 每次只加一个控制变量 + specialist
  3. Forward stepwise: 逐步累积加入控制变量
  4. Leave-one-out: 全模型去掉一个控制变量
  5. All-subsets (2^12 = 4096): 所有控制变量组合

输出: outputs/diagnose_table5_controls.csv  (所有组合的系数和p值)
      outputs/diagnose_table5_summary.csv   (问题控制变量汇总)
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ── 配置 ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "data" / "processed" / "pipeline" / "s05_table5_sample.csv"
OUT_DIR = ROOT / "outputs"

ALL_CONTROLS = [
    "size", "sigma_cfo", "cfo", "lev", "loss", "mb", "lit",
    "altman", "tenure_ln", "ta_lag_abs", "big4", "sec_tier",
]

# 核心 specialist 模型定义
SPECIALIST_MODELS = {
    # (definition, model_name): (specialist_vars, paper_expected_signs)
    # paper_expected_signs: dict of {var: (expected_coef_sign, paper_coef, paper_pval)}
    (1, "model1"): {
        "vars": ["nat_spec_d1"],
        "paper": {"nat_spec_d1": (-1, -0.006, 0.012)},
    },
    (1, "model2"): {
        "vars": ["city_spec_d1"],
        "paper": {"city_spec_d1": (-1, -0.005, 0.006)},
    },
    (1, "model3"): {
        "vars": ["both_d1", "nat_only_d1", "city_only_d1"],
        "paper": {
            "both_d1": (-1, -0.009, 0.001),
            "nat_only_d1": (-1, -0.005, 0.227),
            "city_only_d1": (-1, -0.004, 0.027),
        },
    },
    (2, "model1"): {
        "vars": ["nat_spec_d2"],
        "paper": {"nat_spec_d2": (-1, -0.007, "<0.001")},
    },
    (2, "model2"): {
        "vars": ["city_spec_d2"],
        "paper": {"city_spec_d2": (-1, -0.005, 0.006)},
    },
    (2, "model3"): {
        "vars": ["both_d2", "nat_only_d2", "city_only_d2"],
        "paper": {
            "both_d2": (-1, -0.011, "<0.001"),
            "nat_only_d2": (-1, -0.004, 0.128),
            "city_only_d2": (-1, -0.003, 0.178),
        },
    },
}


def run_ols(df, y, x_cols, cluster_col="company_fkey"):
    """OLS + firm-cluster robust SE."""
    use = df[[y] + x_cols + [cluster_col]].dropna()
    if len(use) < len(x_cols) + 5:
        return None
    X = sm.add_constant(use[x_cols], has_constant="add")
    model = sm.OLS(use[y], X)
    if use[cluster_col].nunique() >= 2:
        res = model.fit(cov_type="cluster",
                        cov_kwds={"groups": use[cluster_col]})
    else:
        res = model.fit(cov_type="HC1")
    return res


def main():
    print("=" * 70)
    print("Table 5 控制变量诊断")
    print("=" * 70)

    # ── 读取数据 ──────────────────────────────────────────────────
    df = pd.read_csv(SAMPLE, dtype=str)
    num_cols = [
        "abs_dacc", "size", "sigma_cfo", "cfo", "lev", "loss", "mb",
        "lit", "altman", "tenure_ln", "ta_lag_abs", "big4", "sec_tier",
        "nat_spec_d1", "city_spec_d1", "both_d1", "nat_only_d1", "city_only_d1",
        "nat_spec_d2", "city_spec_d2", "both_d2", "nat_only_d2", "city_only_d2",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    print(f"样本量: {len(df)}")

    all_rows = []  # 存储所有结果

    # ── 针对每个 specialist 模型做诊断 ────────────────────────────
    for (defn, mname), spec_info in SPECIALIST_MODELS.items():
        spec_vars = spec_info["vars"]
        paper = spec_info["paper"]
        label = f"D{defn}_{mname}"
        print(f"\n{'─' * 60}")
        print(f"诊断: {label}  specialist vars = {spec_vars}")
        print(f"{'─' * 60}")

        # ── 1) Bivariate (无控制变量) ────────────────────────────
        res = run_ols(df, "abs_dacc", spec_vars)
        if res is not None:
            for sv in spec_vars:
                coef = float(res.params.get(sv, np.nan))
                pval = float(res.pvalues.get(sv, np.nan))
                all_rows.append({
                    "definition": defn, "model": mname,
                    "test_type": "bivariate", "controls_used": "none",
                    "controls_omitted": ",".join(ALL_CONTROLS),
                    "n_controls": 0,
                    "spec_var": sv, "coef": coef, "pvalue": pval,
                    "n": int(res.nobs), "adj_r2": float(res.rsquared_adj),
                    "paper_coef": paper[sv][1],
                })

        # ── 2) Plus-one: 单个控制变量 ────────────────────────────
        for ctrl in ALL_CONTROLS:
            res = run_ols(df, "abs_dacc", [ctrl] + spec_vars)
            if res is not None:
                for sv in spec_vars:
                    coef = float(res.params.get(sv, np.nan))
                    pval = float(res.pvalues.get(sv, np.nan))
                    all_rows.append({
                        "definition": defn, "model": mname,
                        "test_type": "plus_one", "controls_used": ctrl,
                        "controls_omitted": ",".join(
                            c for c in ALL_CONTROLS if c != ctrl),
                        "n_controls": 1,
                        "spec_var": sv, "coef": coef, "pvalue": pval,
                        "n": int(res.nobs), "adj_r2": float(res.rsquared_adj),
                        "paper_coef": paper[sv][1],
                    })

        # ── 3) Forward stepwise: 逐步加入 ────────────────────────
        cum_controls = []
        for ctrl in ALL_CONTROLS:
            cum_controls.append(ctrl)
            res = run_ols(df, "abs_dacc", cum_controls + spec_vars)
            if res is not None:
                for sv in spec_vars:
                    coef = float(res.params.get(sv, np.nan))
                    pval = float(res.pvalues.get(sv, np.nan))
                    all_rows.append({
                        "definition": defn, "model": mname,
                        "test_type": "forward_step",
                        "controls_used": ",".join(cum_controls),
                        "controls_omitted": ",".join(
                            c for c in ALL_CONTROLS if c not in cum_controls),
                        "n_controls": len(cum_controls),
                        "spec_var": sv, "coef": coef, "pvalue": pval,
                        "n": int(res.nobs), "adj_r2": float(res.rsquared_adj),
                        "paper_coef": paper[sv][1],
                    })

        # ── 4) Leave-one-out: 全模型去掉一个 ────────────────────
        for ctrl_out in ALL_CONTROLS:
            loo_controls = [c for c in ALL_CONTROLS if c != ctrl_out]
            res = run_ols(df, "abs_dacc", loo_controls + spec_vars)
            if res is not None:
                for sv in spec_vars:
                    coef = float(res.params.get(sv, np.nan))
                    pval = float(res.pvalues.get(sv, np.nan))
                    all_rows.append({
                        "definition": defn, "model": mname,
                        "test_type": "leave_one_out",
                        "controls_used": ",".join(loo_controls),
                        "controls_omitted": ctrl_out,
                        "n_controls": len(loo_controls),
                        "spec_var": sv, "coef": coef, "pvalue": pval,
                        "n": int(res.nobs), "adj_r2": float(res.rsquared_adj),
                        "paper_coef": paper[sv][1],
                    })

        # ── 5) Full model (全部控制变量) ─────────────────────────
        res = run_ols(df, "abs_dacc", ALL_CONTROLS + spec_vars)
        if res is not None:
            for sv in spec_vars:
                coef = float(res.params.get(sv, np.nan))
                pval = float(res.pvalues.get(sv, np.nan))
                all_rows.append({
                    "definition": defn, "model": mname,
                    "test_type": "full_model", "controls_used": ",".join(ALL_CONTROLS),
                    "controls_omitted": "none",
                    "n_controls": len(ALL_CONTROLS),
                    "spec_var": sv, "coef": coef, "pvalue": pval,
                    "n": int(res.nobs), "adj_r2": float(res.rsquared_adj),
                    "paper_coef": paper[sv][1],
                })

    # ── 保存完整结果 ──────────────────────────────────────────────
    result_df = pd.DataFrame(all_rows)
    result_df["sign_match"] = np.sign(result_df["coef"]) == np.sign(result_df["paper_coef"])
    result_df["sig_10"] = result_df["pvalue"] < 0.10
    result_df["sig_05"] = result_df["pvalue"] < 0.05
    result_df["sig_01"] = result_df["pvalue"] < 0.01

    out_path = OUT_DIR / "diagnose_table5_controls.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\n完整结果已保存: {out_path}")
    print(f"总行数: {len(result_df)}")

    # ── 生成汇总报告 ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("汇总报告")
    print("=" * 70)

    # 关注问题变量: city_spec_d1, city_only_d1, city_only_d2
    problem_vars = ["city_spec_d1", "city_only_d1", "city_only_d2",
                    "city_spec_d2", "nat_only_d2"]

    summary_rows = []

    for sv in result_df["spec_var"].unique():
        sv_df = result_df[result_df["spec_var"] == sv]

        # Bivariate result
        biv = sv_df[sv_df["test_type"] == "bivariate"]
        if not biv.empty:
            biv_coef = biv.iloc[0]["coef"]
            biv_p = biv.iloc[0]["pvalue"]
            print(f"\n{'━' * 50}")
            print(f"变量: {sv}")
            print(f"  Paper 系数: {biv.iloc[0]['paper_coef']}")
            print(f"  Bivariate (无控制): coef={biv_coef:.6f}, p={biv_p:.4f}, "
                  f"符号{'✓' if biv_coef < 0 else '✗ 正数!'}")

        # Full model result
        full = sv_df[sv_df["test_type"] == "full_model"]
        if not full.empty:
            full_coef = full.iloc[0]["coef"]
            full_p = full.iloc[0]["pvalue"]
            print(f"  Full model (全部控制): coef={full_coef:.6f}, p={full_p:.4f}, "
                  f"符号{'✓' if full_coef < 0 else '✗ 正数!'}")

        # Forward stepwise: 找到符号翻转点
        fwd = sv_df[sv_df["test_type"] == "forward_step"].sort_values("n_controls")
        if not fwd.empty:
            paper_sign = -1  # 期望为负
            prev_sign = None
            flip_controls = []
            for _, row in fwd.iterrows():
                cur_sign = np.sign(row["coef"])
                if prev_sign is not None and cur_sign != prev_sign and cur_sign > 0:
                    # 从负翻正
                    added = row["controls_used"].split(",")
                    last_added = added[-1]
                    flip_controls.append(last_added)
                    print(f"  ⚠️  Forward step: 加入 '{last_added}' 后系数从负变正! "
                          f"coef={row['coef']:.6f}, p={row['pvalue']:.4f}")
                prev_sign = cur_sign

            if not flip_controls:
                print(f"  Forward step: 符号未翻转")

        # Leave-one-out: 哪个控制去掉后符号恢复
        loo = sv_df[sv_df["test_type"] == "leave_one_out"]
        if not loo.empty and not full.empty and full.iloc[0]["coef"] >= 0:
            print(f"  Leave-one-out (去掉哪个恢复负号):")
            for _, row in loo.iterrows():
                if row["coef"] < 0:
                    print(f"    去掉 '{row['controls_omitted']}': "
                          f"coef={row['coef']:.6f}, p={row['pvalue']:.4f} ✓ 恢复负号")
                    summary_rows.append({
                        "spec_var": sv,
                        "problematic_control": row["controls_omitted"],
                        "effect": "removing_restores_negative_sign",
                        "coef_without": row["coef"],
                        "pvalue_without": row["pvalue"],
                        "coef_full": full.iloc[0]["coef"],
                        "pvalue_full": full.iloc[0]["pvalue"],
                    })

        # Leave-one-out: 对所有变量都报告去掉后的系数变化
        if not loo.empty and not full.empty:
            print(f"  Leave-one-out 全部结果:")
            for _, row in loo.iterrows():
                delta = row["coef"] - full.iloc[0]["coef"]
                mark = ""
                if abs(delta) > 0.001:
                    mark = " ← 影响较大"
                if row["coef"] < 0 and full.iloc[0]["coef"] >= 0:
                    mark = " ← ⚠️ 恢复负号!"
                print(f"    去掉 '{row['controls_omitted']:12s}': "
                      f"coef={row['coef']:+.6f}, p={row['pvalue']:.4f}, "
                      f"Δcoef={delta:+.6f}{mark}")

        # Plus-one: 哪个单独影响最大
        po = sv_df[sv_df["test_type"] == "plus_one"]
        if not po.empty:
            print(f"  Plus-one (各单独加入一个控制后的系数):")
            for _, row in po.iterrows():
                biv_c = biv.iloc[0]["coef"] if not biv.empty else np.nan
                delta = row["coef"] - biv_c if not np.isnan(biv_c) else np.nan
                mark = ""
                if not np.isnan(delta) and abs(delta) > 0.001:
                    mark = " ← 影响较大"
                if row["coef"] >= 0 and biv_c < 0:
                    mark = " ← ⚠️ 翻正!"
                print(f"    加入 '{row['controls_used']:12s}': "
                      f"coef={row['coef']:+.6f}, p={row['pvalue']:.4f}, "
                      f"Δ={delta:+.6f}{mark}" if not np.isnan(delta)
                      else f"    加入 '{row['controls_used']:12s}': "
                           f"coef={row['coef']:+.6f}, p={row['pvalue']:.4f}")

    # 保存汇总
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = OUT_DIR / "diagnose_table5_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n问题控制变量汇总已保存: {summary_path}")

    # ── 额外: All-subsets 全组合测试 (仅针对问题变量) ─────────────
    # 对问题最大的变量做全组合测试:
    # city_spec_d1 (Def1 Model2), city_only_d1 (Def1 Model3)
    print("\n" + "=" * 70)
    print("全组合测试 (2^12 = 4096 种控制变量组合)")
    print("=" * 70)

    subset_rows = []
    target_models = [
        (1, "model2", ["city_spec_d1"]),
        (1, "model3", ["both_d1", "nat_only_d1", "city_only_d1"]),
        (2, "model2", ["city_spec_d2"]),
        (2, "model3", ["both_d2", "nat_only_d2", "city_only_d2"]),
    ]

    total_combos = 2 ** len(ALL_CONTROLS)
    for defn, mname, spec_vars in target_models:
        print(f"\n测试 D{defn}_{mname} ({spec_vars}) ...")
        count = 0
        for r in range(len(ALL_CONTROLS) + 1):
            for combo in itertools.combinations(ALL_CONTROLS, r):
                combo_list = list(combo)
                res = run_ols(df, "abs_dacc", combo_list + spec_vars)
                if res is not None:
                    for sv in spec_vars:
                        coef = float(res.params.get(sv, np.nan))
                        pval = float(res.pvalues.get(sv, np.nan))
                        subset_rows.append({
                            "definition": defn, "model": mname,
                            "controls_used": ",".join(combo_list) if combo_list else "none",
                            "n_controls": len(combo_list),
                            "spec_var": sv, "coef": coef, "pvalue": pval,
                            "sign_negative": coef < 0,
                            "sig_05": pval < 0.05,
                            "sig_10": pval < 0.10,
                            "n": int(res.nobs),
                            "adj_r2": float(res.rsquared_adj),
                        })
                count += 1
                if count % 500 == 0:
                    print(f"  ... {count}/{total_combos} 组合已测试")

    subset_df = pd.DataFrame(subset_rows)
    subset_path = OUT_DIR / "diagnose_table5_allsubsets.csv"
    subset_df.to_csv(subset_path, index=False)
    print(f"\n全组合结果已保存: {subset_path}")
    print(f"总行数: {len(subset_df)}")

    # ── 全组合汇总: 哪些控制变量频繁出现在"好"的组合中 ────────────
    print("\n" + "=" * 70)
    print("全组合分析: 什么控制变量组合使核心变量符合预期?")
    print("=" * 70)

    for sv in subset_df["spec_var"].unique():
        sv_sub = subset_df[subset_df["spec_var"] == sv]
        total = len(sv_sub)
        neg = sv_sub[sv_sub["sign_negative"]].copy()
        neg_sig = neg[neg["sig_05"]].copy()

        print(f"\n变量: {sv}")
        print(f"  总组合数: {total}")
        print(f"  系数为负: {len(neg)} ({100*len(neg)/total:.1f}%)")
        print(f"  系数为负且 p<0.05: {len(neg_sig)} ({100*len(neg_sig)/total:.1f}%)")

        # 统计每个控制变量在"好"组合(负且显著)中出现的频率 vs 在"坏"组合中
        if len(neg_sig) > 0 and len(neg_sig) < total:
            print(f"\n  各控制变量在'好'组合(负且p<0.05)中的出现率:")
            for ctrl in ALL_CONTROLS:
                in_good = neg_sig["controls_used"].str.contains(ctrl, na=False).mean()
                in_bad = sv_sub[~(sv_sub["sign_negative"] & sv_sub["sig_05"])][
                    "controls_used"].str.contains(ctrl, na=False).mean()
                diff = in_good - in_bad
                mark = ""
                if diff < -0.10:
                    mark = " ← 出现越多越差"
                elif diff > 0.10:
                    mark = " ← 出现越多越好"
                print(f"    {ctrl:12s}: 好组合 {in_good:.1%}, "
                      f"差组合 {in_bad:.1%}, 差异 {diff:+.1%}{mark}")

        # 最多控制变量下仍为负且显著的组合
        if len(neg_sig) > 0:
            best = neg_sig.nlargest(5, "n_controls")
            print(f"\n  控制变量最多且仍为负/显著的 Top-5 组合:")
            for _, row in best.iterrows():
                print(f"    [{row['n_controls']}个控制] coef={row['coef']:.6f}, "
                      f"p={row['pvalue']:.4f}")
                print(f"      控制变量: {row['controls_used']}")
                omitted = set(ALL_CONTROLS) - set(
                    row['controls_used'].split(',')) if row['controls_used'] != 'none' else set(ALL_CONTROLS)
                if omitted:
                    print(f"      省略的: {','.join(omitted)}")

    print("\n" + "=" * 70)
    print("诊断结束")
    print("=" * 70)


if __name__ == "__main__":
    main()
