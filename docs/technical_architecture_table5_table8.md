# Technical Architecture: Reichelt and Wang (2010) Table 5 & Table 8 Replication

## 1. Objective
Build a reproducible pipeline from raw databases to final regressions for Table 5 and Table 8.

## 2. Data Sources and Keys
- Audit Analytics:
1. Audit fees.
2. Auditor office city info (mapped to MSA).
3. Going-concern indicator (`GC`).
4. CIK, SIC, fiscal year.
- Compustat Annual:
1. Accounting and market variables used in equations (1)-(6).
2. CIK for match (paper removes records without Compustat CIK match).
- I/B/E/S Unadjusted Detail:
1. Analyst forecast variables for MEET tests (not required for Table 5/8 coefficients, but needed for full-paper compatibility).
- Geographic mapping:
1. City -> MSA mapping (U.S. Census MSA cross-map).

Primary merge keys:
- `CIK + fiscal_year` between Audit Analytics and Compustat.
- I/B/E/S merge key should be documented per local mapping table; keep this layer modular.

## 3. Sample Construction (paper-aligned baseline)
Baseline flow from paper Table 1:
1. Start: 32,479 firm-years, domestic nonfinancial from Audit Analytics (2003-2007), positive audit fees, valid MSA and SIC.
2. Remove no-Compustat CIK match: -7,476.
3. Remove city-industry-fiscal-year cells with <2 obs: -3,420.
4. Working sample: 21,583.
5. Table 5 sample (`|DACC|`): remove missing controls/accrual inputs (-7,443), remove `|studentized residual|>3` (-369), final N=13,771.
6. Table 8 sample (`GC`): remove missing (-6,136), keep financially distressed only (negative OCF, remove 10,472), remove `|logit deviance residual|>3` (-6), final N=4,969.

User-required filter switch:
- `exclude_utility = True/False`.
- Paper text explicitly states nonfinancial; utility exclusion is not clearly stated in extracted text, so implement as configurable and log impact on counts.

## 4. Industry Specialist Construction
Industry unit: two-digit SIC.
Market shares based on audit fees.

Definition 1:
- National(city) specialist = largest share in industry (industry-city) and at least 10 percentage points above second-largest competitor.

Definition 2:
- National specialist if share > 30%.
- City specialist if share > 50%.

Derived dummies:
1. `BOTH = 1` if national=1 and city=1.
2. `NAT_ONLY = 1` if national=1 and city=0.
3. `CITY_ONLY = 1` if national=0 and city=1.
4. Omitted category: neither specialist.

## 5. Variable Construction

### 5.1 Accrual model (for Table 5 dependent variable)
Equation (1), estimated cross-sectionally by two-digit SIC x year:
- `TA = beta0*(1/A_{t-1}) + beta1*DeltaREV + beta2*PPE + beta3*ROA_{t-1} + e`
- Paper details:
1. Use all available Compustat annual observations for years ending 2003-2007 to estimate eq(1).
2. Truncate top/bottom 1% of variables in eq(1).
3. Require >=20 observations per industry-year cell.

Equation (2): expected accruals with receivables adjustment.
Equation (3): `DACC = TA - ETA`.
Dependent variable in Table 5: `|DACC|`.

### 5.2 Controls for Table 5 (equation 4)
- `SIZE, sigma(CFO), CFO, LEV, LOSS, MB, LIT, ALTMAN, TENURE, |ACCR_{t-1}|, BIG4, SEC_TIER, BOTH, NAT_ONLY, CITY_ONLY`.
- Note from paper: industry/year dummies are not included in eq(4) because abnormal accruals are already estimated by industry-year.

### 5.3 Controls for Table 8 (equation 6)
- Dependent: `GC=1` if auditor issues going-concern opinion.
- Regressors:
`SIZE, sigma(EARN), LEV, LOSS, MB, LIT, ALTMAN, TENURE, ROA, ACCR, BIG4, SEC_TIER, BOTH, NAT_ONLY, CITY_ONLY + industry FE + year FE`.

## 6. Outlier Handling and Scaling
- All continuous variables in equations (2)-(6): winsorize at 1% and 99%.
- Remove high-influence observations:
1. OLS sample: `|studentized residual| > 3`.
2. Logit sample: `|logit deviance residual| > 3`.

## 7. Estimation and Inference
- Table 5: OLS with robust inference reported as asymptotic t-stats robust to heteroskedasticity and time-series correlation (Rogers, 1993).
- Table 8: Logit with robust Wald tests using same Rogers-style robust covariance.
- Fixed effects:
1. Table 5: no explicit FE in final regression (industry-year handled upstream in accrual estimation).
2. Table 8: include two-digit SIC FE and fiscal-year FE.

Implementation note:
- Paper does not explicitly state cluster dimension in extracted text; operational default should be firm-level clustering (`cluster = firm_id`) and validated by matching reported p-values/significance patterns.

## 8. Suggested Code Modules
- `src/01_extract/`: source-specific pulls and raw snapshots.
- `src/02_build/`: merge and sample-selection pipeline with row-count logs.
- `src/03_features/`: specialist flags and variable constructors.
- `src/04_models/`: eq(1)-(6) estimation scripts.
- `src/05_tables/`: Table 5/8 formatter and export.
- `logs/`: run metadata, data versions, assumption switches.

## 9. Validation Plan
- Structural checks:
1. Reproduce sample attrition counts near Table 1 benchmarks.
2. Reproduce means for key variables near Table 3 ranges.
- Coefficient checks:
1. Table 5 model 3: `BOTH` negative and significant in both definitions.
2. Table 8 model 3: `BOTH` positive and significant in both definitions.
- Diagnostics:
1. Distribution checks before/after winsorization.
2. Specialist share sanity by year and industry.
3. FE collinearity checks in Table 8 logit.

## 10. Known Risks
- Licensed data refresh may shift counts and coefficients.
- Utility exclusion can change sample materially; keep it configurable and explicitly reported.
- Different cluster implementation may shift p-values; document exact covariance settings in run log.
