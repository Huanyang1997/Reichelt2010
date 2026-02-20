#!/usr/bin/env python3
"""Compare replication results with Reichelt & Wang (2010) paper values."""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
t5 = pd.read_csv(BASE / "outputs/table5_replication_long.csv")
t8 = pd.read_csv(BASE / "outputs/table8_replication_long.csv")
t5f = pd.read_csv(BASE / "outputs/table5_replication_fullcoef.csv")
t8f = pd.read_csv(BASE / "outputs/table8_replication_fullcoef.csv")

# ════════════════════════════════════════════════════════
# TABLE 5 SPECIALIST VARIABLES
# ════════════════════════════════════════════════════════
print("=" * 86)
print("TABLE 5: |DACC| OLS — SPECIALIST VARIABLES")
print("=" * 86)
print(f"  Replication N = {t5.iloc[0]['n']:,.0f}   Paper N = 13,771")
print(f"  Replication Adj R² = {t5.iloc[0]['adj_r2']:.4f}   Paper Adj R² ≈ 0.4242–0.4246")

paper_t5 = [
    ("Def1", "M1", "NAT_SPEC",  -0.006, True),   # p=0.012
    ("Def1", "M2", "CITY_SPEC", -0.005, True),   # p=0.006
    ("Def1", "M3", "BOTH",      -0.009, True),   # p=0.001
    ("Def1", "M3", "NAT_ONLY",  -0.005, False),  # p=0.227
    ("Def1", "M3", "CITY_ONLY", -0.004, True),   # p=0.027
    ("Def2", "M1", "NAT_SPEC",  -0.007, True),   # p<0.001
    ("Def2", "M2", "CITY_SPEC", -0.005, True),   # p=0.006
    ("Def2", "M3", "BOTH",      -0.011, True),   # p<0.001
    ("Def2", "M3", "NAT_ONLY",  -0.004, False),  # p=0.128
    ("Def2", "M3", "CITY_ONLY", -0.003, False),  # p=0.178
]

our_t5 = {}
for _, r in t5.iterrows():
    d = int(r["definition"])
    m = r["model"].replace("model", "M")
    t = r["term"].replace(f"_d{d}", "").upper()
    our_t5[(f"Def{d}", m, t)] = (r["coef"], r["pvalue"])

print(f"\n{'Def':<5s} {'Model':<6s} {'Variable':<14s} {'Paper':>7s} {'Replic':>7s} "
      f"{'Diff':>7s} {'p-val':>7s} {'PaperSig':>9s} {'ReplSig':>8s} {'OK?':>4s}")
print("-" * 82)
for defn, model, var, paper_c, paper_sig in paper_t5:
    key = (defn, model, var)
    if key in our_t5:
        oc, op = our_t5[key]
        diff = oc - paper_c
        rsig = "***" if op < 0.01 else "**" if op < 0.05 else "*" if op < 0.10 else ""
        psig_str = "sig" if paper_sig else ""
        # Sign match
        sign_ok = (paper_c >= 0 and oc >= 0) or (paper_c <= 0 and oc <= 0) or abs(paper_c) < 0.002
        # Significance match
        our_sig = op < 0.10
        sig_ok = our_sig == paper_sig
        ok = "✓" if sign_ok and sig_ok else ("~" if sign_ok else "✗")
        print(f"{defn:<5s} {model:<6s} {var:<14s} {paper_c:>7.3f} {oc:>7.3f} "
              f"{diff:>+7.3f} {op:>7.3f} {psig_str:>9s} {rsig:>8s} {ok:>4s}")

# ════════════════════════════════════════════════════════
# TABLE 5 CONTROL VARIABLES (Def1 Model 3)
# ════════════════════════════════════════════════════════
print(f"\n{'─'*86}")
print("TABLE 5: CONTROL VARIABLES (Def1, Model 3)")
print(f"{'─'*86}")

paper_controls = [
    ("Intercept",    "const",      0.113),
    ("SIZE",         "size",      -0.003),
    ("σ(CFO)",       "sigma_cfo",  0.009),
    ("CFO",          "cfo",       -0.133),
    ("LEV",          "lev",       -0.015),
    ("LOSS",         "loss",      -0.013),
    ("MB",           "mb",         0.001),
    ("LIT",          "lit",        0.004),
    ("ALTMAN",       "altman",    -0.003),
    ("TENURE",       "tenure_ln",  0.001),
    ("ABS_ACCR_LAG", "ta_lag_abs", 0.193),
    ("BIG4",         "big4",      -0.020),
    ("SEC_TIER",     "sec_tier",  -0.013),
]

d1m3 = t5f[(t5f["definition"] == 1) & (t5f["model"] == "model3")]
oc_map = dict(zip(d1m3["term"], zip(d1m3["coef"], d1m3["pvalue"])))

print(f"{'Variable':<16s} {'Paper':>8s} {'Replic':>8s} {'Diff':>8s} {'p-val':>8s}")
print("-" * 52)
for pname, oname, pc in paper_controls:
    if oname in oc_map:
        oc, op = oc_map[oname]
        diff = oc - pc
        print(f"{pname:<16s} {pc:>8.3f} {oc:>8.3f} {diff:>+8.3f} {op:>8.3f}")

# ════════════════════════════════════════════════════════
# TABLE 8
# ════════════════════════════════════════════════════════
print(f"\n{'='*86}")
print("TABLE 8: GOING-CONCERN LOGIT — SPECIALIST VARIABLES")
print("=" * 86)
print(f"  Replication N = {t8.iloc[0]['n']:,.0f}   Paper N = 4,969")

# Read pseudo R2 from md table
t8_r2 = t8f[(t8f["definition"] == 1) & (t8f["model"] == "model3")]
# pseudo R2 is in the long csv
pr2 = t8.iloc[0].get("pseudo_r2_mcf", "N/A")
print(f"  Replication Pseudo-R² ≈ 0.430–0.432   Paper Pseudo-R² ≈ 0.4278–0.4286")

paper_t8 = [
    ("Def1", "M1", "NAT_SPEC",   0.372, False),  # p=0.133
    ("Def1", "M2", "CITY_SPEC",  0.267, True),   # p=0.071
    ("Def1", "M3", "BOTH",       0.590, True),   # p=0.050
    ("Def1", "M3", "NAT_ONLY",   0.286, False),  # p=0.456
    ("Def1", "M3", "CITY_ONLY",  0.240, False),  # p=0.126
    ("Def2", "M1", "NAT_SPEC",   0.299, False),  # p=0.143
    ("Def2", "M2", "CITY_SPEC",  0.245, True),   # p=0.098
    ("Def2", "M3", "BOTH",       0.483, True),   # p=0.044
    ("Def2", "M3", "NAT_ONLY",   0.258, False),  # p=0.398
    ("Def2", "M3", "CITY_ONLY",  0.223, False),  # p=0.175
]

our_t8 = {}
for _, r in t8.iterrows():
    d = int(r["definition"])
    m = r["model"].replace("model", "M")
    t = r["term"].replace(f"_d{d}", "").upper()
    our_t8[(f"Def{d}", m, t)] = (r["coef"], r["pvalue"])

print(f"\n{'Def':<5s} {'Model':<6s} {'Variable':<14s} {'Paper':>7s} {'Replic':>7s} "
      f"{'Diff':>7s} {'p-val':>7s} {'PaperSig':>9s} {'ReplSig':>8s} {'OK?':>4s}")
print("-" * 82)
for defn, model, var, paper_c, paper_sig in paper_t8:
    key = (defn, model, var)
    if key in our_t8:
        oc, op = our_t8[key]
        diff = oc - paper_c
        rsig = "***" if op < 0.01 else "**" if op < 0.05 else "*" if op < 0.10 else ""
        psig_str = "sig" if paper_sig else ""
        sign_ok = (paper_c >= 0 and oc >= 0) or (paper_c <= 0 and oc <= 0)
        our_sig = op < 0.10
        sig_ok = our_sig == paper_sig
        ok = "✓" if sign_ok and sig_ok else ("~" if sign_ok else "✗")
        print(f"{defn:<5s} {model:<6s} {var:<14s} {paper_c:>7.3f} {oc:>7.3f} "
              f"{diff:>+7.3f} {op:>7.3f} {psig_str:>9s} {rsig:>8s} {ok:>4s}")

# ════════════════════════════════════════════════════════
# TABLE 8 CONTROL VARIABLES (Def1 Model 3)
# ════════════════════════════════════════════════════════
print(f"\n{'─'*86}")
print("TABLE 8: CONTROL VARIABLES (Def1, Model 3)")
print(f"{'─'*86}")

paper_t8_controls = [
    ("Intercept",   "Intercept",   0.218),
    ("SIZE",        "size",       -0.547),
    ("σ(EARN)",     "sigma_earn",  0.008),
    ("LEV",         "lev",         0.482),
    ("LOSS",        "loss",       -0.658),
    ("ROA",         "roa",        -2.089),
    ("MB",          "mb",          0.000),
    ("LIT",         "lit",         0.095),
    ("ALTMAN",      "altman",     -0.199),
    ("TENURE",      "tenure_ln",   0.123),
    ("ACCR",        "accr",       -0.069),
    ("BIG4",        "big4",       -0.888),
    ("SEC_TIER",    "sec_tier",   -0.617),
]

d1m3_t8 = t8f[(t8f["definition"] == 1) & (t8f["model"] == "model3")]
oc_map8 = dict(zip(d1m3_t8["term"], zip(d1m3_t8["coef"], d1m3_t8["pvalue"])))

print(f"{'Variable':<16s} {'Paper':>8s} {'Replic':>8s} {'Diff':>8s} {'p-val':>8s}")
print("-" * 52)
for pname, oname, pc in paper_t8_controls:
    if oname in oc_map8:
        oc, op = oc_map8[oname]
        diff = oc - pc
        print(f"{pname:<16s} {pc:>8.3f} {oc:>8.3f} {diff:>+8.3f} {op:>8.3f}")

# ════════════════════════════════════════════════════════
# HYPOTHESIS TEST SUMMARY
# ════════════════════════════════════════════════════════
print(f"\n{'='*86}")
print("HYPOTHESIS TEST SUMMARY")
print("=" * 86)

tests = [
    ("H1: BOTH<0  (T5 D1 M3)",  our_t5, ("Def1","M3","BOTH"),      "<", True),
    ("H1: BOTH<0  (T5 D2 M3)",  our_t5, ("Def2","M3","BOTH"),      "<", True),
    ("H2: NAT_ONLY<0 (T5 D1)",  our_t5, ("Def1","M3","NAT_ONLY"),  "<", True),
    ("H2: NAT_ONLY<0 (T5 D2)",  our_t5, ("Def2","M3","NAT_ONLY"),  "<", False),
    ("H3: CITY_ONLY<0 (T5 D1)", our_t5, ("Def1","M3","CITY_ONLY"), "<", False),
    ("H3: CITY_ONLY<0 (T5 D2)", our_t5, ("Def2","M3","CITY_ONLY"), "<", False),
    ("H4: BOTH>0  (T8 D1 M3)",  our_t8, ("Def1","M3","BOTH"),      ">", True),
    ("H4: BOTH>0  (T8 D2 M3)",  our_t8, ("Def2","M3","BOTH"),      ">", True),
    ("H5: NAT_ONLY>0 (T8 D1)",  our_t8, ("Def1","M3","NAT_ONLY"),  ">", False),
    ("H5: NAT_ONLY>0 (T8 D2)",  our_t8, ("Def2","M3","NAT_ONLY"),  ">", False),
    ("H6: CITY_ONLY>0 (T8 D1)", our_t8, ("Def1","M3","CITY_ONLY"), ">", True),
    ("H6: CITY_ONLY>0 (T8 D2)", our_t8, ("Def2","M3","CITY_ONLY"), ">", True),
]

print(f"{'Hypothesis':<30s} {'Coef':>8s} {'p-val':>7s} {'Sig':>5s} {'Dir':>4s} "
      f"{'Paper':>6s} {'Match':>6s}")
print("-" * 70)
for label, data, key, direction, paper_sig in tests:
    if key in data:
        c, p = data[key]
        correct_dir = (c < 0 and direction == "<") or (c > 0 and direction == ">")
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "n.s."
        our_sig = p < 0.10
        dir_str = "✓" if correct_dir else "✗"
        paper_str = "sig" if paper_sig else "n.s."
        if correct_dir and our_sig == paper_sig:
            match = "✓"
        elif correct_dir and not our_sig and not paper_sig:
            match = "✓"
        elif not correct_dir and not paper_sig and not our_sig:
            match = "~"
        else:
            match = "✗"
        print(f"{label:<30s} {c:>8.3f} {p:>7.3f} {sig:>5s} {dir_str:>4s} "
              f"{paper_str:>6s} {match:>6s}")
