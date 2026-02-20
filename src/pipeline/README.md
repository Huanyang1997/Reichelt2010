# Reichelt & Wang (2010) 模块化复刻管道

## 运行方法

```bash
cd Reichelt2010
python src/pipeline/run_all.py
```

可逐步运行单个模块（每步读取上一步输出）：

```bash
python src/pipeline/p01_load_and_merge.py
python src/pipeline/p02_msa_mapping.py
python src/pipeline/p03_compustat_vars.py
python src/pipeline/p04_build_sample.py
python src/pipeline/p05_table5.py
python src/pipeline/p06_table8.py
python src/pipeline/p07_export.py
```

## 管道结构

| 步骤 | 文件 | 功能 | 论文对应 |
|------|------|------|----------|
| 1 | `p01_load_and_merge.py` | 加载 AA 原始表（Fee + Opinion + During）, 合并, 计算 tenure | Data sources |
| 2 | `p02_msa_mapping.py` | 审计师办公室城市/州 → MSA/CBSA 映射 | p.109 office location |
| 3 | `p03_compustat_vars.py` | Compustat 财务变量构造（所有控制变量） | p.110-112 变量定义 |
| 4 | `p04_build_sample.py` | 合并 AA×Compustat, Eq(1)→DACC, 专家指标, Panel A 递减 | Table 1, Eq(1)-(3) |
| 5 | `p05_table5.py` | Table 5 样本准备 (缩尾→异常值) + OLS 回归 | Table 5, Eq(4) |
| 6 | `p06_table8.py` | Table 8 样本准备 (缩尾→异常值) + Logit 回归 | Table 8, Eq(6) |
| 7 | `p07_export.py` | 论文格式表格输出 (Markdown + LaTeX) | Table 5/8 格式 |

## 中间数据

所有中间产物保存在 `data/processed/pipeline/`，可逐步检查和调试：

```
s01_aa_merged.csv          # AA 合并后面板
s02_aa_with_msa.csv        # 添加 MSA 代码
s03_compustat_vars.csv     # Compustat 变量
s04_working_sample.csv     # Panel A 工作样本 (≈21,583)
s04_eq1_coefs.csv          # Eq(1) 截面回归系数
s04_attrition.json         # 样本递减统计
s05_table5_sample.csv      # Table 5 分析样本
s06_table8_sample.csv      # Table 8 分析样本
```

## 与原文对照：变量构造

### Eq(1) — 修正 Jones 模型 + ROA (Kothari 2005)

```
TA_it = β₀(1/A_{i,t-1}) + β₁ ΔREV_it + β₂ PPE_it + β₃ ROA_{i,t-1} + ε_it
```

- 无截距项，`1/A_{t-1}` 充当比例截距
- 按 2-digit SIC × year 截面估计
- 最少 20 个观测 (论文 p.110)
- **估计前截断** Eq(1) 各变量 top/bottom 1% (论文 p.110)
- 在 **全部 Compustat** 上估计（不限于 AA 样本）

### Eq(2)-(3) — 异常应计

```
ETA = β̂₀(1/A) + β̂₁(ΔREV - ΔREC) + β̂₂ PPE + β̂₃ ROA_{t-1}
DACC = TA - ETA
```

### 变量定义 (论文 p.111-112)

| 变量 | 定义 | Compustat 字段 |
|------|------|----------------|
| SIZE | ln(市值) = ln(CSHO × **PRCC_F**) | csho, prcc_f |
| LEV | DLTT / AT | dltt, at |
| LOSS | 1{NI < 0} | ni |
| MB | (CSHO × PRCC_F) / CEQ | csho, prcc_f, ceq |
| LIT | 诉讼风险行业虚拟变量 (Shu 2000) | sic |
| ALTMAN | Z (1968) = 1.2X₁ + 1.4X₂ + 3.3X₃ + 0.6X₄ + 1.0X₅（X₄=MVE/TL） | act,lct,re,ebit,csho,prcc_f,lt,sale,at |
| CFO | (OANCF - XIDOC) / A_{t-1} | oancf, xidoc, at |
| σ(CFO) | 5 年滚动标准差 of CFO | — |
| σ(EARN) | 5 年滚动标准差 of IB/A_{t-1} | ib, at |
| TENURE | ln(任期年数) | auditor_during |
| ACCR | TA = (IB - OCF)/A_{t-1} | — |
| ROA | IB / avg(AT_t, AT_{t-1}) | ib, at |

### SIZE 变量口径

当前管道使用 `SIZE = ln(CSHO × PRCC_F)`（按本项目设定）。

## 数据合并策略

论文描述为"匹配"——按键匹配到了就合并：

1. **Fee + Opinion**: 按 `(company_fkey, auditor_fkey, fiscal_year)` 匹配
2. **AA + Tenure**: 按 `(company_fkey, auditor_fkey, fiscal_year)` 匹配
3. **AA + Compustat**: 按 `(cik_norm, fiscal_year)` 匹配
4. 未匹配的行保留但标记，在 Panel A 递减中**显式删除**并计数

## 缩尾/截断时机

| 场景 | 方法 | 时机 |
|------|------|------|
| Eq(1) 自变量 | **截断** top/bottom 1% (删行) | 估计 Eq(1) **之前** |
| Eq(2)-(6) 连续变量 | **缩尾** 1%/99% (clip) | 删除缺失值**之后**、异常值剔除**之前** |

## 回归模型

### Table 5 (OLS)

- 因变量: |DACC|
- **无**行业/年份固定效应 (论文脚注 39: 已被 Eq(1) 截面估计吸收)
- Firm-cluster robust SE (Rogers 1993)

### Table 8 (Logit)

- 因变量: GC (going-concern opinion)
- 子样本: financially distressed (OCF < 0)
- 含 2-digit SIC FE + fiscal year FE
- Firm-cluster robust SE (Rogers 1993)

### 专家定义

- **Definition 1**: 最大份额 + 领先第二名 ≥ 10pp
- **Definition 2**: National > 30%, City > 50%

### 每个定义 3 个模型:

- Model 1: `NAT_SPEC` (单一国家专家指标)
- Model 2: `CITY_SPEC` (单一城市专家指标)
- Model 3: `BOTH` + `NAT_ONLY` + `CITY_ONLY` (交叉分解)
