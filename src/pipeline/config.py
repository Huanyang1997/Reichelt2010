#!/usr/bin/env python3
"""
config.py — 全局配置
====================
集中管理所有路径、参数、开关。
与原文 Reichelt & Wang (2010) 对应的参数均标注论文出处。
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class PipelineConfig:
    """全部可配置参数，均附论文/数据依据。"""

    # ── 原始数据路径 ──────────────────────────────────────────────
    aa_fee_path: str = "data/raw/audit_analytics/audit_fee_manual_2026-02-03.csv"
    aa_op_path: str = "data/raw/audit_analytics/revised_audit_opinions_manual_2026-02-02.csv"
    aa_during_path: str = "data/raw/audit_analytics/auditor_during_manual_2026-01-30.csv"
    comp_funda_path: str = "data/raw/compustat/funda_1996_2009_manual.dta"
    msa_place_map_path: str = "data/raw/reference/msa/geocorr2022.xlsx"
    msa_list2_path: str = "data/raw/reference/msa/_check/list2_2023.xlsx"

    # ── 输出路径 ──────────────────────────────────────────────────
    output_dir: str = "outputs"
    processed_dir: str = "data/processed"
    pipeline_dir: str = "data/processed/pipeline"

    # ── 样本区间（论文 p.107: "our sample spans 2003–2007"） ──────
    year_start: int = 2003
    year_end: int = 2007

    # ── Tenure 来源 ──────────────────────────────────────────────
    #  "during"  = 仅用 auditor_during 表
    #  "consecutive" = 仅由 fee/op 表推断连续年数
    #  "hybrid"  = during 优先，缺失用 consecutive 补
    tenure_source: str = "during"

    # ── SIC 行业过滤 ─────────────────────────────────────────────
    #  论文明确排除金融业 (SIC 6000-6999)
    #  公用事业 (SIC 4900-4999) 作为可选开关
    exclude_utility: bool = False

    # ── Compustat 补充过滤 ───────────────────────────────────────
    #  all = 不过滤 COSTAT; A = 仅保留 active
    comp_costat: str = "all"

    # ── MSA 模糊匹配参数 ────────────────────────────────────────
    msa_fuzzy_threshold: float = 0.90
    msa_fuzzy_gap: float = 0.03

    # ── Eq(1) 截面回归最低观测数（论文 p.110） ──────────────────
    eq1_min_obs: int = 20

    # ── 截断/缩尾（论文 p.110） ─────────────────────────────────
    #  Eq(1) 变量：截断 top/bottom 1%
    #  Eq(2)-(6) 连续变量：缩尾 1%/99%
    truncate_pct: float = 0.01
    winsor_lo: float = 0.01
    winsor_hi: float = 0.99

    # ── 异常值剔除阈值 ──────────────────────────────────────────
    #  Table 5: |studentized residual| > 3
    #  Table 8: |deviance residual| > 3
    stud_resid_threshold: float = 3.0
    dev_resid_threshold: float = 3.0

    # ── 允许缺少 Altman 时退化模型 ─────────────────────────────
    allow_reduced_model: bool = False

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "PipelineConfig":
        d = json.loads(path.read_text())
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_cli(cls) -> "PipelineConfig":
        p = argparse.ArgumentParser(
            description="Reichelt & Wang (2010) Table 5/8 模块化复刻管道"
        )
        p.add_argument("--aa-fee", dest="aa_fee_path",
                        default=cls.aa_fee_path)
        p.add_argument("--aa-op", dest="aa_op_path",
                        default=cls.aa_op_path)
        p.add_argument("--aa-during", dest="aa_during_path",
                        default=cls.aa_during_path)
        p.add_argument("--comp-funda", dest="comp_funda_path",
                        default=cls.comp_funda_path)
        p.add_argument("--msa-place-map", dest="msa_place_map_path",
                        default=cls.msa_place_map_path)
        p.add_argument("--output-dir", dest="output_dir",
                        default=cls.output_dir)
        p.add_argument("--processed-dir", dest="processed_dir",
                        default=cls.processed_dir)
        p.add_argument("--pipeline-dir", dest="pipeline_dir",
                        default=cls.pipeline_dir)
        p.add_argument("--year-start", dest="year_start", type=int,
                        default=cls.year_start)
        p.add_argument("--year-end", dest="year_end", type=int,
                        default=cls.year_end)
        p.add_argument("--tenure-source", dest="tenure_source",
                        choices=["during", "consecutive", "hybrid"],
                        default=cls.tenure_source)
        p.add_argument("--exclude-utility", dest="exclude_utility",
                        action="store_true")
        p.add_argument("--comp-costat", dest="comp_costat",
                        choices=["all", "A"], default=cls.comp_costat)
        p.add_argument("--allow-reduced-model", dest="allow_reduced_model",
                        action="store_true")
        args = p.parse_args()
        return cls(**{k: v for k, v in vars(args).items()
                      if k in cls.__dataclass_fields__})


def ensure_dirs(cfg: PipelineConfig) -> None:
    """创建所有输出目录。"""
    for d in [cfg.output_dir, cfg.processed_dir, cfg.pipeline_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)
