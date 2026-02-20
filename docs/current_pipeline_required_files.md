# Current Fixed Version: Required Files

This repository now uses the **pipeline workflow only** (`src/pipeline/run_all.py`).

## 1) Required code
- `src/pipeline/config.py`
- `src/pipeline/run_all.py`
- `src/pipeline/p01_load_and_merge.py`
- `src/pipeline/p02_msa_mapping.py`
- `src/pipeline/p03_compustat_vars.py`
- `src/pipeline/p04_build_sample.py`
- `src/pipeline/p05_table5.py`
- `src/pipeline/p06_table8.py`
- `src/pipeline/p07_export.py`
- `src/pipeline/export_table2_table3_to_word.py`
- `src/pipeline/utils.py`

## 2) Required local raw inputs (not tracked by git)
Place these files under `data/raw/`:

### Audit Analytics
- `data/raw/audit_analytics/audit_fee_manual_2026-02-03.csv`
- `data/raw/audit_analytics/revised_audit_opinions_manual_2026-02-02.csv`
- `data/raw/audit_analytics/auditor_during_manual_2026-01-30.csv`

### Compustat
- `data/raw/compustat/funda_1996_2009_manual.dta`

### MSA mapping
- Default: `data/raw/reference/msa/geocorr2022.xlsx`
- Optional override: `data/raw/reference/msa/_check/place05-cbsa06.xls`

## 3) Run commands
- Main pipeline:
  - `python src/pipeline/run_all.py`
- Export Table 2 / Table 3 comparison doc:
  - `python src/pipeline/export_table2_table3_to_word.py`

## 4) Archived (not used by current pipeline)
All previous standalone / diagnostic scripts are moved to:
- `archive/unused_scripts/`

They are kept for reference only.
