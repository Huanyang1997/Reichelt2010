#!/usr/bin/env python3
"""
run_all.py — 完整管道入口
===========================
按顺序执行 p01 → p07，复刻 Reichelt & Wang (2010) Table 5 和 Table 8。

使用方法:
  cd Reichelt2010
  python src/pipeline/run_all.py [--选项]

管道步骤:
  p01  数据加载与审计分析表合并
  p02  审计师办公室→MSA 地理映射
  p03  Compustat 财务变量构造
  p04  工作样本构建（AA+Compustat 匹配、Eq(1)→DACC、专家指标、Panel A 筛选）
  p05  Table 5 样本准备与 OLS 回归
  p06  Table 8 样本准备与 Logit 回归
  p07  输出论文格式表格

所有中间数据保存在 data/processed/pipeline/，可逐步调试。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import PipelineConfig, ensure_dirs
from pipeline import (
    p01_load_and_merge,
    p02_msa_mapping,
    p03_compustat_vars,
    p04_build_sample,
    p05_table5,
    p06_table8,
    p07_export,
)


def main() -> None:
    cfg = PipelineConfig.from_cli()
    ensure_dirs(cfg)

    # 保存本次运行配置
    cfg.save(Path(cfg.pipeline_dir) / "run_config.json")

    steps = [
        ("p01 数据加载与合并",     p01_load_and_merge.run),
        ("p02 MSA 地理映射",       p02_msa_mapping.run),
        ("p03 Compustat 变量构造", p03_compustat_vars.run),
        ("p04 工作样本构建",       p04_build_sample.run),
        ("p05 Table 5 回归",       p05_table5.run),
        ("p06 Table 8 回归",       p06_table8.run),
        ("p07 输出表格",           p07_export.run),
    ]

    t0 = time.time()
    print("=" * 60)
    print("Reichelt & Wang (2010) 模块化复刻管道")
    print("=" * 60)

    for i, (name, func) in enumerate(steps, 1):
        print(f"\n{'─' * 60}")
        print(f"[{i}/7] {name}")
        print(f"{'─' * 60}")
        ts = time.time()
        func(cfg)
        print(f"  ⏱ {time.time() - ts:.1f}s")

    print(f"\n{'=' * 60}")
    print(f"全部完成  总耗时 {time.time() - t0:.1f}s")
    print(f"中间数据: {cfg.pipeline_dir}/")
    print(f"最终输出: {cfg.output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
