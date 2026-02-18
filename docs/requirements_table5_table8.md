# Reichelt and Wang (2010) Table 5 & Table 8 Replication Requirements

## 1. Scope
- Goal: replicate Table 5 and Table 8 in `JAR - 2010 - REICHELT - National and Office‐Specific Measures of Auditor Industry Expertise and.pdf`.
- Coverage:
1. Table 5: OLS on `|DACC|` (abnormal accrual magnitude).
2. Table 8: Logit on `GC` (going-concern opinion).
- Target period: fiscal years 2003-2007.
- Core sample anchor in paper: nonfinancial U.S. domestic firms with valid audit fees and MSA/SIC in Audit Analytics.

## 2. Research Hypotheses (for formal testing)

### 2.1 Table 5 (audit quality via abnormal accruals)
- H1 (primary): `Both National and City Specialist` coefficient < 0.
- H2: `National Specialist Only` coefficient < 0 (weaker/possibly insignificant per paper).
- H3: `City Specialist Only` coefficient < 0 (definition-dependent in paper).
- Null form:
1. H0_1: beta(BOTH) >= 0
2. H0_2: beta(NAT_ONLY) >= 0
3. H0_3: beta(CITY_ONLY) >= 0

### 2.2 Table 8 (audit quality via going-concern conservatism)
- H4 (primary): `Both National and City Specialist` coefficient > 0.
- H5: `National Specialist Only` coefficient > 0 (paper reports insignificant).
- H6: `City Specialist Only` coefficient > 0 (paper reports insignificant in model 3).
- Null form:
1. H0_4: beta(BOTH) <= 0
2. H0_5: beta(NAT_ONLY) <= 0
3. H0_6: beta(CITY_ONLY) <= 0

## 3. Hypothesis-Testing Prompts

### 3.1 Prompt template (generic)
```text
你是会计实证研究助理。基于输入的回归结果，按以下规则判断假设是否被支持：
1) 报告每个检验变量的系数符号、p 值、显著性水平(10%/5%/1%)。
2) 严格按原假设 H0 与备择假设 H1 给出“拒绝/不拒绝 H0”。
3) 明确区分统计显著性与经济显著性。
4) 给出一句中文结论，说明是否与 Reichelt and Wang (2010) Table [X] 一致。
输入字段：变量名、系数、标准误、p 值、模型编号、样本量、pseudo/adj R2。
输出格式：Markdown 表格 + 3 条要点结论。
```

### 3.2 Prompt for Table 5
```text
请检验 Table 5 的 H1-H3：
- H1: BOTH < 0
- H2: NAT_ONLY < 0
- H3: CITY_ONLY < 0
并额外报告：
- 两个 specialist 定义(Definition 1/2)下，model 3 是否稳定；
- H1 是否是最稳健结果；
- 与论文方向是否一致。
```

### 3.3 Prompt for Table 8
```text
请检验 Table 8 的 H4-H6：
- H4: BOTH > 0
- H5: NAT_ONLY > 0
- H6: CITY_ONLY > 0
并额外报告：
- Definition 1/2 下 model 3 的方向与显著性；
- 是否支持“joint specialist 审计质量更高”；
- 与论文方向是否一致。
```

## 4. Deliverables
- D1: clean, reproducible code pipeline (data pull -> sample build -> variable build -> regressions -> tables).
- D2: Table 5 replication outputs (Definition 1 and 2; model 1/2/3).
- D3: Table 8 replication outputs (Definition 1 and 2; model 1/2/3 with FE).
- D4: replication log (sample attrition, assumptions, deviations, data version dates).

## 5. Acceptance Criteria
- Must reproduce core direction/significance patterns:
1. Table 5 model 3: `BOTH` negative and significant under both definitions.
2. Table 8 model 3: `BOTH` positive and significant under both definitions.
- Must include robust inference consistent with paper note (Rogers-style robust to heteroskedasticity and serial correlation).
- Must document winsorization/truncation and sample exclusions.
- Must disclose all deviations from paper wording (example: utility exclusion rule if user-mandated).

## 6. Constraints and Decisions
- Data are licensed sources (Audit Analytics, Compustat, I/B/E/S); replication depends on local subscription access.
- Paper explicitly states nonfinancial screening; utility exclusion is included as a user requirement and should be logged as a configurable switch if it changes sample counts.
