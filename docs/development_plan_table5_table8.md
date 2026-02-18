# Development Plan: Replication of Reichelt and Wang (2010) Table 5 & Table 8

## 1. Planning Inputs
- Requirements source: `docs/requirements_table5_table8.md`
- Architecture source: `docs/technical_architecture_table5_table8.md`

## 2. Goal and Success Criteria
- Goal: deliver reproducible replication outputs for Table 5 and Table 8.
- Success criteria:
1. Table 5 model 3: `BOTH` negative and significant under Definition 1 and 2.
2. Table 8 model 3: `BOTH` positive and significant under Definition 1 and 2.
3. Full pipeline rerunnable from raw source snapshots to final tables.
4. Sample attrition and assumption switches fully logged.

## 3. Work Breakdown Structure (WBS)

### Phase 0: Project Bootstrap
- Tasks:
1. Create repo structure (`src/01_extract` ... `src/05_tables`, `logs`, `outputs`).
2. Add config file for paths, date range, filter switches (including `exclude_utility`).
3. Add run entrypoint script.
- Deliverables:
1. Runnable scaffold and folder conventions.
2. Initial config template and environment instructions.
- Definition of done:
1. `python -m src.run_all --help` (or equivalent) works.

### Phase 1: Data Ingestion Layer
- Tasks:
1. Build extract adapters for Audit Analytics, Compustat, I/B/E/S.
2. Standardize IDs and fiscal-year keys.
3. Persist raw snapshots with timestamped metadata.
- Deliverables:
1. Raw-zone datasets and extraction log.
- Definition of done:
1. Row counts and key coverage report generated.

### Phase 2: Sample Construction Layer
- Tasks:
1. Implement baseline sample filters (nonfinancial, domestic, positive audit fees, valid MSA/SIC, 2003-2007).
2. Apply Compustat link filter and city-industry-year minimum observation rule.
3. Build Table 5 and Table 8 analysis subsamples exactly per Table 1 logic.
- Deliverables:
1. `sample_attrition_report` with each deletion step count.
- Definition of done:
1. Counts reasonably close to paper benchmarks (document deviations).

### Phase 3: Feature Engineering Layer
- Tasks:
1. Compute national/city audit-fee market shares by two-digit SIC.
2. Build specialist flags for Definition 1 and Definition 2.
3. Construct equations (1)-(3) accrual outputs, controls, and dependent variables.
4. Apply truncation/winsorization rules and outlier logic.
- Deliverables:
1. Model-ready panel datasets for Table 5 and Table 8.
- Definition of done:
1. Variable dictionary and summary stats output.

### Phase 4: Econometric Modeling Layer
- Tasks:
1. Estimate Table 5 models (1/2/3; def1/def2).
2. Estimate Table 8 logit models (1/2/3; def1/def2; industry/year FE).
3. Apply Rogers-style robust covariance; implement cluster setting in config.
- Deliverables:
1. Regression outputs with coefficients, p-values, fit stats.
2. Replicated table artifacts (`csv` and `md`).
- Definition of done:
1. Direction/significance pattern aligned with target criteria.

### Phase 5: Validation and Reproducibility
- Tasks:
1. Compare sample and coefficients against paper tables.
2. Run robustness checks: specialist definitions, filter switch impact, residual diagnostics.
3. Write final replication note (data version, assumptions, residual differences).
- Deliverables:
1. `outputs/validation_report.md`.
2. Final run log and checklist.
- Definition of done:
1. Another user can rerun and obtain same outputs from same inputs.

## 4. Milestones and Timeline (Suggested)
1. M1 (Day 1): Bootstrap + ingestion scripts complete.
2. M2 (Day 2): Sample construction and attrition report complete.
3. M3 (Day 3): Features and specialist flags complete.
4. M4 (Day 4): Table 5/Table 8 regressions complete.
5. M5 (Day 5): Validation package and replication note complete.

## 5. Task Dependency Graph
1. Phase 0 -> Phase 1
2. Phase 1 -> Phase 2
3. Phase 2 -> Phase 3
4. Phase 3 -> Phase 4
5. Phase 4 -> Phase 5

## 6. Git Workflow
- Branching:
1. `main`: stable state.
2. `feat/bootstrap`, `feat/sample`, `feat/features`, `feat/models`, `feat/validation`.
- Commit policy:
1. One logical change per commit.
2. Commit message format: `<type>: <scope> - <summary>`.
- Suggested tags:
1. `v0.1-bootstrap`
2. `v0.2-sample-ready`
3. `v0.3-model-ready`
4. `v1.0-replication-report`

## 7. Risk Register
1. Licensed data refresh causes sample drift: freeze extract date and snapshot IDs.
2. Cluster dimension ambiguity affects p-values: expose cluster setting and log choice.
3. Utility exclusion ambiguity: configurable switch + explicit delta report.

## 8. Immediate Next Actions
1. Implement Phase 0 scaffold and config file.
2. Implement Phase 1 extraction interfaces with row-count checks.
3. Generate first sample attrition report draft.
