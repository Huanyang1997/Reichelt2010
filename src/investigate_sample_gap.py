#!/usr/bin/env python3
"""
Comprehensive investigation of the replication sample-size gap.

Reichelt & Wang (2010) paper reports:
  - Table 5 N = 13,771
  - Table 8 N =  4,969
Our replication currently produces:
  - Table 5 N = 10,809
  - Table 8 N =  3,533

This script investigates every stage of the pipeline to determine
where observations are lost and how many could be recovered.

It does NOT modify the main pipeline.  Creates new output only.
"""
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict

BASE = Path(__file__).resolve().parent.parent

# ────────────────────────────────────────────────────────────────
# Helper functions (copied from main pipeline to keep standalone)
# ────────────────────────────────────────────────────────────────
def norm_text(v: str) -> str:
    if pd.isna(v):
        return ""
    s = str(v).upper()
    s = s.replace("&", " AND ")
    s = s.replace("SAINT", "ST")
    s = s.replace("FORT", "FT")
    s = re.sub(r"[^A-Z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def norm_cik(v) -> str:
    if pd.isna(v):
        return ""
    s = re.sub(r"\D", "", str(v))
    s = s.lstrip("0")
    return s if s else "0"

# ════════════════════════════════════════════════════════════════
# PART 1:  MANUAL MSA LOOKUP TABLE FOR UNMATCHED US CITIES
# ════════════════════════════════════════════════════════════════
# These are US cities that appear in Audit Analytics AUDITOR_CITY
# but are NOT matched by the Census place05-cbsa06 or 03cbsa files.
# Hand-mapped to the correct CBSA code & title.
# Key assumption: if a neighbourhood / sub-municipality is inside
#   a Metropolitan Statistical Area, it maps to that MSA's CBSA code.

MANUAL_MSA_MAP = {
    # city_state_norm → (cbsa_code, cbsa_title)
    # ── NJ suburbs (mostly Newark / New York MSA) ──────────
    "PARSIPPANY|NJ":        ("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    "SHORT HILLS|NJ":       ("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    "METROPARK|NJ":         ("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    "SKILLMAN|NJ":          ("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    "EAST HANOVER|NJ":      ("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    "BAYVILLE|NJ":          ("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    "BRIDGEWATER|NJ":       ("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    "MIDDLETOWN|NJ":        ("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    "WHIPPANY|NJ":          ("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    "PINE BROOK|NJ":        ("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    "OCEAN|NJ":             ("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    "TOTOWA BORO|NJ":       ("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    "BASKING RIDGE|NJ":     ("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    "CHERRY HILL|NJ":       ("37980", "Philadelphia-Camden-Wilmington, PA-NJ-DE-MD"),
    "HOLMDEL|NJ":           ("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    "BRANCHBURG|NJ":        ("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    "MONTVILLE|NJ":         ("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    "BEDMINSTER|NJ":        ("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    "NORTH BRUNSWICK|NJ":   ("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    "MOORESTOWN|NJ":        ("37980", "Philadelphia-Camden-Wilmington, PA-NJ-DE-MD"),
    "HAMILTON|NJ":          ("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    "MAPLE SHADE|NJ":       ("37980", "Philadelphia-Camden-Wilmington, PA-NJ-DE-MD"),
    "MOUNT LAUREL|NJ":      ("37980", "Philadelphia-Camden-Wilmington, PA-NJ-DE-MD"),
    "MERCERVILLE|NJ":       ("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    # ── NYC boroughs / neighbourhoods ──────────────────────
    "FLUSHING|NY":          ("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    "BROOKLYN|NY":          ("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    "OAKLAND GARDENS|NY":   ("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    "FOREST HILLS|NY":      ("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    "BAYSIDE|NY":           ("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    "ASTORIA|NY":           ("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    "PURCHASE|NY":          ("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    "AMHERST|NY":           ("15380", "Buffalo-Cheektowaga, NY"),
    "LONG ISLAND CITY|NY":  ("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    "ELMHURST|NY":          ("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    "REGO PARK|NY":         ("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    "STATEN ISLAND|NY":     ("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    "NORTH WHITE PLAINS|NY":("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    "JACKSON HEIGHTS|NY":   ("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    "HYDE PARK|NY":         ("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    "HUDSON|NY":            ("35620", "New York-Newark-Jersey City, NY-NJ-PA"),
    # ── Los Angeles area (CA) ──────────────────────────────
    "ORANGE COUNTY|CA":     ("31080", "Los Angeles-Long Beach-Anaheim, CA"),
    "ENCINO|CA":            ("31080", "Los Angeles-Long Beach-Anaheim, CA"),
    "SHERMAN OAKS|CA":      ("31080", "Los Angeles-Long Beach-Anaheim, CA"),
    "WOODLAND HILLS|CA":    ("31080", "Los Angeles-Long Beach-Anaheim, CA"),
    "GRANADA HILLS|CA":     ("31080", "Los Angeles-Long Beach-Anaheim, CA"),
    "CHATSWORTH|CA":        ("31080", "Los Angeles-Long Beach-Anaheim, CA"),
    "NEWHALL|CA":           ("31080", "Los Angeles-Long Beach-Anaheim, CA"),
    "CENTURY CITY|CA":      ("31080", "Los Angeles-Long Beach-Anaheim, CA"),
    "CITY OF INDUSTRY|CA":  ("31080", "Los Angeles-Long Beach-Anaheim, CA"),
    "VAN NUYS|CA":          ("31080", "Los Angeles-Long Beach-Anaheim, CA"),
    "NORTHRIDGE|CA":        ("31080", "Los Angeles-Long Beach-Anaheim, CA"),
    "TARZANA|CA":           ("31080", "Los Angeles-Long Beach-Anaheim, CA"),
    "TOPANGA|CA":           ("31080", "Los Angeles-Long Beach-Anaheim, CA"),
    "REDWOOD SHORES|CA":    ("41860", "San Francisco-Oakland-Berkeley, CA"),
    "LA JOLLA|CA":          ("41740", "San Diego-Chula Vista-Carlsbad, CA"),
    # ── Virginia ───────────────────────────────────────────
    "TYSONS|VA":            ("47900", "Washington-Arlington-Alexandria, DC-VA-MD-WV"),
    # Galax, VA is in a micropolitan area—not an MSA
    # ── Pennsylvania (Philadelphia / Pittsburgh) ───────────
    "ABINGTON|PA":          ("37980", "Philadelphia-Camden-Wilmington, PA-NJ-DE-MD"),
    "BALA CYNWYD|PA":       ("37980", "Philadelphia-Camden-Wilmington, PA-NJ-DE-MD"),
    "HUNTINGDON VALLEY|PA": ("37980", "Philadelphia-Camden-Wilmington, PA-NJ-DE-MD"),
    "RADNOR|PA":            ("37980", "Philadelphia-Camden-Wilmington, PA-NJ-DE-MD"),
    "WAYNE|PA":             ("37980", "Philadelphia-Camden-Wilmington, PA-NJ-DE-MD"),
    "INDIANA|PA":           ("26860", "Indiana, PA"),  # micropolitan—small
    "WEXFORD|PA":           ("38300", "Pittsburgh, PA"),
    "CRANBERRY TOWNSHIP|PA":("38300", "Pittsburgh, PA"),
    # ── Tennessee ──────────────────────────────────────────
    "NASHVILLE|TN":         ("34980", "Nashville-Davidson--Murfreesboro--Franklin, TN"),
    # ── Connecticut ────────────────────────────────────────
    "GLASTONBURY|CT":       ("25540", "Hartford-East Hartford-Middletown, CT"),
    "WOODBRIDGE|CT":        ("35300", "New Haven-Milford, CT"),
    "FARMINGTON|CT":        ("25540", "Hartford-East Hartford-Middletown, CT"),
    # ── Massachusetts ──────────────────────────────────────
    "WEST PEABODY|MA":      ("14460", "Boston-Cambridge-Newton, MA-NH"),
    "TEWKSBURY|MA":         ("14460", "Boston-Cambridge-Newton, MA-NH"),
    "NORWELL|MA":           ("14460", "Boston-Cambridge-Newton, MA-NH"),
    # ── Idaho ──────────────────────────────────────────────
    "BOISE|ID":             ("14260", "Boise City, ID"),
    # ── Kentucky ───────────────────────────────────────────
    "LEXINGTON|KY":         ("30460", "Lexington-Fayette, KY"),
    # ── Maryland ───────────────────────────────────────────
    "HUNT VALLEY|MD":       ("12580", "Baltimore-Columbia-Towson, MD"),
    # ── Georgia ────────────────────────────────────────────
    "AUGUSTA|GA":           ("12260", "Augusta-Richmond County, GA-SC"),
    "DUBLIN|GA":            (None, None),  # micropolitan
    "STATESBORO|GA":        (None, None),  # micropolitan
    # ── Texas ──────────────────────────────────────────────
    "CYPRESS|TX":           ("26420", "Houston-The Woodlands-Sugar Land, TX"),
    "WHARTON|TX":           ("26420", "Houston-The Woodlands-Sugar Land, TX"),
    "HUDSON|TX":            (None, None),  # small, not in MSA
    # ── Michigan ───────────────────────────────────────────
    "CLINTON TOWNSHIP|MI":  ("19820", "Detroit-Warren-Dearborn, MI"),
    # ── Florida ────────────────────────────────────────────
    "ALTAMONTE|FL":         ("36740", "Orlando-Kissimmee-Sanford, FL"),
    "PONTE VEDRA BEACH|FL": ("27260", "Jacksonville, FL"),
    "DELAND|FL":            ("19660", "Deltona-Daytona Beach-Ormond Beach, FL"),
    "SUNNY ISLES|FL":       ("33100", "Miami-Fort Lauderdale-Pompano Beach, FL"),
    # ── Illinois ───────────────────────────────────────────
    "OAK BROOK TERRACE|IL": ("16980", "Chicago-Naperville-Elgin, IL-IN-WI"),
    # ── Ohio ───────────────────────────────────────────────
    "MAYFIELD VILLAGE|OH":  ("17460", "Cleveland-Elyria, OH"),
    # ── North Carolina ─────────────────────────────────────
    "SANFORD|NC":           (None, None),  # micropolitan only
    "LEXINGTON|NC":         (None, None),  # part of Winston-Salem? Actually small
    "SOUTHERN PINES|NC":    (None, None),  # micropolitan
    # ── Mississippi ────────────────────────────────────────
    "COLUMBUS|MS":          (None, None),  # micropolitan
    # ── Alabama ────────────────────────────────────────────
    "ENTERPRISE|AL":        (None, None),  # micropolitan
}


# ════════════════════════════════════════════════════════════════
# PART 2:  LOAD DATA AND RUN PIPELINE DIAGNOSTICS
# ════════════════════════════════════════════════════════════════
print("=" * 80)
print("LOADING RAW DATA")
print("=" * 80)

fee = pd.read_csv(
    BASE / "data/raw/audit_analytics/audit_fee_manual_2026-02-03.csv",
    dtype={"COMPANY_FKEY": "string", "COMPANY_KEY": "string", "AUDITOR_FKEY": "string"},
)
op = pd.read_csv(
    BASE / "data/raw/audit_analytics/revised_audit_opinions_manual_2026-02-02.csv",
    dtype={"COMPANY_FKEY": "string", "COMPANY_KEY": "string", "AUDITOR_FKEY": "string"},
)
comp = pd.read_csv(
    BASE / "data/raw/compustat/funda_1997_2025_manual.csv",
    dtype={"cik": "string"},
)
fee.columns = [c.lower() for c in fee.columns]
op.columns = [c.lower() for c in op.columns]
comp.columns = [c.lower() for c in comp.columns]

for df in [fee, op]:
    for k in ["company_fkey", "company_key", "auditor_fkey"]:
        if k in df.columns:
            df[k] = df[k].astype("string").str.strip()

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
fee = fee.dropna(subset=["company_fkey", "auditor_fkey", "fiscal_year", "audit_fees"])
op = op.dropna(subset=["company_fkey", "auditor_fkey", "fiscal_year"])
if "file_date" in op.columns:
    op["file_date"] = pd.to_datetime(op["file_date"], errors="coerce")
    op = op.sort_values("file_date")
op = op.drop_duplicates(["company_fkey", "auditor_fkey", "fiscal_year"], keep="last")

# Merge fee + opinion
aa = fee.merge(
    op[["company_fkey", "auditor_fkey", "fiscal_year", "going_concern",
        "auditor_city", "auditor_state",
        "loc_state_country", "bus_state_country", "mail_state_country"]],
    on=["company_fkey", "auditor_fkey", "fiscal_year"],
    how="left",
)
# Dedupe to one row per company-year (largest fee)
aa = aa.sort_values(["company_fkey", "fiscal_year", "audit_fees"], ascending=[True, True, False])
aa = aa.drop_duplicates(["company_fkey", "fiscal_year"], keep="first")

print(f"Total company-year rows: {len(aa):,}")
print(f"Fee-Opinion merge: {aa['auditor_city'].notna().sum():,} have city "
      f"({aa['auditor_city'].isna().sum():,} missing = {aa['auditor_city'].isna().mean():.2%})")

# Filter to study period + positive fees
aa = aa[(aa["fiscal_year"] >= 2003) & (aa["fiscal_year"] <= 2007) & (aa["audit_fees"] > 0)].copy()
print(f"After period 2003-2007 & positive fees: {len(aa):,}")

# SIC filter
aa["sic4"] = pd.to_numeric(aa["sic_code_fkey"], errors="coerce")
aa["sic2"] = (aa["sic4"] // 100).astype("Int64")
aa["nonfinancial"] = aa["sic4"].notna() & (~aa["sic4"].between(6000, 6999, inclusive="both"))
aa_nonfin = aa[aa["nonfinancial"]].copy()
print(f"Nonfinancial SIC: {len(aa_nonfin):,}")


# ════════════════════════════════════════════════════════════════
# PART 3:  MSA MAPPING — EXISTING vs ENHANCED
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PART 3: MSA MAPPING ANALYSIS")
print("=" * 80)

# Load existing MSA map
msa_used = pd.read_csv(BASE / "data/processed/msa_city_state_to_cbsa_used.csv", dtype=str)
msa_existing_set = set(msa_used["city_state_norm"].dropna().unique())
print(f"Existing MSA map entries: {len(msa_existing_set):,}")

# Build city_state_norm for AA
aa_nonfin["city_state_norm"] = aa_nonfin.apply(
    lambda r: f"{norm_text(r.get('auditor_city', ''))}|{str(r.get('auditor_state', '')).upper().strip()}",
    axis=1,
)

# Check how many match existing map
aa_nonfin["msa_existing"] = aa_nonfin["city_state_norm"].isin(msa_existing_set)
print(f"  Matched by existing map: {aa_nonfin['msa_existing'].sum():,}")
print(f"  Not matched: {(~aa_nonfin['msa_existing']).sum():,}")

# Check how many of the unmatched are US (2-letter state code)
us_states = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL",
    "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME",
    "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
    "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "PR",
    "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY",
}
unmatched = aa_nonfin[~aa_nonfin["msa_existing"]].copy()
unmatched["state_part"] = unmatched["city_state_norm"].str.split("|").str[1]
unmatched["is_us_state"] = unmatched["state_part"].isin(us_states)
n_unmatched_us = unmatched["is_us_state"].sum()
n_unmatched_foreign = (~unmatched["is_us_state"]).sum()
print(f"  Unmatched with US state code: {n_unmatched_us:,}")
print(f"  Unmatched with non-US code: {n_unmatched_foreign:,}")

# Apply manual MSA map
manual_map_df = pd.DataFrame([
    {"city_state_norm": k, "manual_cbsa": v[0], "manual_title": v[1]}
    for k, v in MANUAL_MSA_MAP.items()
    if v[0] is not None  # skip micropolitan entries
])
manual_metro_set = set(manual_map_df["city_state_norm"])

# Count how many unmatched US rows would be recovered
unmatched_us = unmatched[unmatched["is_us_state"]].copy()
unmatched_us["manual_hit"] = unmatched_us["city_state_norm"].isin(manual_metro_set)
recovered_by_manual = unmatched_us["manual_hit"].sum()
still_unmatched_us = (~unmatched_us["manual_hit"]).sum()
print(f"\n  Manual MSA lookup would recover: {recovered_by_manual:,} US observations")
print(f"  Still unmatched US (micropolitan/rural/unmapped): {still_unmatched_us:,}")

# Show top still-unmatched US cities
still_un = unmatched_us[~unmatched_us["manual_hit"]]
if len(still_un) > 0:
    top_still = still_un["city_state_norm"].value_counts().head(20)
    print(f"\n  Top 20 still-unmatched US cities:")
    for cs, n in top_still.items():
        print(f"    {cs}: {n}")

# TOTAL MSA recoverable impact
total_nonfin = len(aa_nonfin)
existing_match = aa_nonfin["msa_existing"].sum()
enhanced_match = existing_match + recovered_by_manual
print(f"\n  MSA coverage summary (nonfinancial, 2003-2007):")
print(f"    Total obs:       {total_nonfin:,}")
print(f"    Existing map:    {existing_match:,} ({existing_match/total_nonfin:.1%})")
print(f"    Enhanced map:    {enhanced_match:,} ({enhanced_match/total_nonfin:.1%})")
print(f"    Gain:            {recovered_by_manual:,} ({recovered_by_manual/total_nonfin:.1%})")


# ════════════════════════════════════════════════════════════════
# PART 4:  COMPUSTAT MERGE — CIK COVERAGE ANALYSIS
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PART 4: COMPUSTAT CIK COVERAGE")
print("=" * 80)

# Standard Compustat filters
for col, val in [("indfmt", "INDL"), ("datafmt", "STD"), ("consol", "C")]:
    if col in comp.columns:
        comp = comp[comp[col].astype(str).str.upper() == val]
comp["cik_norm"] = comp["cik"].apply(norm_cik)
comp = comp[(comp["fyear"] >= 1999) & (comp["fyear"] <= 2007)].copy()
comp_cik_set = set(comp["cik_norm"].dropna().unique()) - {"", "0"}

# AA CIK
aa_nonfin["cik_norm"] = aa_nonfin["company_fkey"].fillna(aa_nonfin.get("company_key", "")).apply(norm_cik)
aa_nonfin["in_compustat"] = aa_nonfin["cik_norm"].isin(comp_cik_set) & (aa_nonfin["cik_norm"] != "") & (aa_nonfin["cik_norm"] != "0")

n_in_comp = aa_nonfin["in_compustat"].sum()
n_not_in_comp = (~aa_nonfin["in_compustat"]).sum()
print(f"AA nonfinancial 2003-2007: {len(aa_nonfin):,}")
print(f"  In Compustat: {n_in_comp:,} ({n_in_comp/len(aa_nonfin):.1%})")
print(f"  Not in Compustat: {n_not_in_comp:,} ({n_not_in_comp/len(aa_nonfin):.1%})")
print(f"  Unique AA CIKs: {aa_nonfin['cik_norm'].nunique():,}")
print(f"  Unique Compustat CIKs: {len(comp_cik_set):,}")

# CIK match rate for rows with MSA
aa_with_msa = aa_nonfin[aa_nonfin["msa_existing"]].copy()
n_msa_comp = aa_with_msa["in_compustat"].sum()
n_msa_nocomp = (~aa_with_msa["in_compustat"]).sum()
print(f"\n  Among MSA-matched rows ({len(aa_with_msa):,}):")
print(f"    In Compustat:     {n_msa_comp:,} ({n_msa_comp/len(aa_with_msa):.1%})")
print(f"    Not in Compustat: {n_msa_nocomp:,} ({n_msa_nocomp/len(aa_with_msa):.1%})")


# ════════════════════════════════════════════════════════════════
# PART 5:  VARIABLE MISSINGNESS (sigma_cfo, altman, mb)
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PART 5: VARIABLE MISSINGNESS IN COMPUSTAT")
print("=" * 80)

# Reconstruct key variables for the Compustat subsample
comp_sub = comp[comp["cik_norm"].isin(comp_cik_set)].copy()
comp_sub = comp_sub.sort_values(["cik_norm", "fyear"])
grp = comp_sub.groupby("cik_norm")

for c in ["at", "dltt", "lt", "ppegt", "rect", "ib", "ni", "sale", "oancf", "xidoc", "csho", "prcc_f"]:
    if c in comp_sub.columns:
        comp_sub[c] = pd.to_numeric(comp_sub[c], errors="coerce")

comp_sub["lag_at"] = grp["at"].shift(1)
comp_sub["lag_ib"] = grp["ib"].shift(1)
comp_sub["op_cf"] = comp_sub["oancf"] - comp_sub["xidoc"].fillna(0)
comp_sub["cfo"] = comp_sub["op_cf"] / comp_sub["lag_at"]
comp_sub["earn_scaled"] = comp_sub["ib"] / comp_sub["lag_at"]

# sigma_cfo: rolling 4-year std of lagged cfo
comp_sub["sigma_cfo"] = grp["cfo"].transform(lambda s: s.shift(1).rolling(4, min_periods=4).std())

# Alternative: rolling with min_periods=3
comp_sub["sigma_cfo_mp3"] = grp["cfo"].transform(lambda s: s.shift(1).rolling(4, min_periods=3).std())

# Alternative: rolling with min_periods=2
comp_sub["sigma_cfo_mp2"] = grp["cfo"].transform(lambda s: s.shift(1).rolling(4, min_periods=2).std())

# MVE and MB
comp_sub["mve"] = comp_sub["csho"].abs() * comp_sub["prcc_f"].abs()
comp_sub["size"] = np.log(comp_sub["mve"].replace(0, np.nan))
comp_sub["mb"] = comp_sub["mve"] / (comp_sub["at"] - comp_sub["lt"])
comp_sub["lev"] = comp_sub["dltt"] / comp_sub["at"]
comp_sub["loss"] = (comp_sub["ni"] < 0).astype(float)

# Altman
altman_cols = ["act", "lct", "re", "ebit"]
missing_altman = [c for c in altman_cols if c not in comp_sub.columns]
if not missing_altman:
    for c in altman_cols:
        comp_sub[c] = pd.to_numeric(comp_sub[c], errors="coerce")
    comp_sub["altman"] = (
        1.2 * ((comp_sub["act"] - comp_sub["lct"]) / comp_sub["at"])
        + 1.4 * (comp_sub["re"] / comp_sub["at"])
        + 3.3 * (comp_sub["ebit"] / comp_sub["at"])
        + 0.6 * (comp_sub["mve"] / comp_sub["lt"])
        + 1.0 * (comp_sub["sale"] / comp_sub["at"])
    )
else:
    comp_sub["altman"] = np.nan
    print(f"  WARNING: Altman inputs missing from Compustat: {missing_altman}")

# Focus on study period
comp_period = comp_sub[(comp_sub["fyear"] >= 2003) & (comp_sub["fyear"] <= 2007)].copy()
print(f"Compustat rows in 2003-2007 (after standard filters): {len(comp_period):,}")

key_vars = {
    "size": comp_period["size"],
    "sigma_cfo (strict 4yr)": comp_period["sigma_cfo"],
    "sigma_cfo (min3)": comp_period["sigma_cfo_mp3"],
    "sigma_cfo (min2)": comp_period["sigma_cfo_mp2"],
    "cfo": comp_period["cfo"],
    "lev": comp_period["lev"],
    "loss": comp_period["loss"],
    "mb": comp_period["mb"],
    "altman": comp_period["altman"],
}
print(f"\nVariable missingness rates in Compustat study-period sample:")
for name, series in key_vars.items():
    nmiss = series.isna().sum()
    rate = nmiss / len(comp_period) if len(comp_period) > 0 else 0
    print(f"  {name:25s}: {nmiss:>6,} missing ({rate:.2%})")

# Show how sigma_cfo missingness evolves by year
print(f"\n  sigma_cfo missingness by fiscal year:")
for yr in range(2003, 2008):
    sub = comp_sub[comp_sub["fyear"] == yr]
    n_total = len(sub)
    n_miss_strict = sub["sigma_cfo"].isna().sum()
    n_miss_mp3 = sub["sigma_cfo_mp3"].isna().sum()
    print(f"    {yr}: total={n_total:,}  strict_miss={n_miss_strict:,} ({n_miss_strict/n_total:.1%})  "
          f"min3_miss={n_miss_mp3:,} ({n_miss_mp3/n_total:.1%})")


# ════════════════════════════════════════════════════════════════
# PART 6:  FULL ATTRITION DECOMPOSITION — CURRENT vs PAPER
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PART 6: ATTRITION DECOMPOSITION")
print("=" * 80)

diag = json.loads((BASE / "outputs/replication_diagnostics.json").read_text())
c = diag["counts"]

print("""
Paper Table 1 reports (approximate from the paper):
  Panel A start (company-years):   ~26,548   (ours: {pA_start})
  Delete not in Compustat:         ~7,476    (ours: {del_comp})
  Delete city-industry < 2:        ~5,301    (ours: {del_ci})
  Panel A final (Table 2):         ~13,771   (ours: {pA_final})
  
  Panel B final (Table 5):          13,771    (ours: {t5_final})
  Panel D final (Table 8):           4,969    (ours: {t8_final})
""".format(
    pA_start=f"{c['panelA_start_obs']:,}",
    del_comp=f"{c['panelA_delete_not_in_compustat']:,}",
    del_ci=f"{c['panelA_delete_city_industry_lt2']:,}",
    pA_final=f"{c['panelA_final_obs']:,}",
    t5_final=f"{c['table5_final_n']:,}",
    t8_final=f"{c['table8_final_n']:,}",
))

# Detailed gap accounting
print("DETAILED GAP ANALYSIS:")
print("-" * 60)

# Stage 1: Panel A start (= MSA-matched, nonfinancial, period observations)
# Paper: ~26,548.   Ours: 32,357.
# Our number is LARGER because we use domestic_filter=none.
# If the paper uses domestic_filter=strict:
aa_domestic = aa_nonfin[aa_nonfin.apply(
    lambda r: any(str(r.get(c, '')).upper() == 'USA' for c in
                  ['loc_state_country', 'bus_state_country', 'mail_state_country']),
    axis=1,
)]
aa_domestic_with_msa = aa_domestic[aa_domestic["msa_existing"]]
print(f"If domestic_filter=strict AND existing MSA map:")
print(f"  Domestic nonfinancial 2003-2007:  {len(aa_domestic):,}")
print(f"  With existing MSA:                {len(aa_domestic_with_msa):,}")

# With enhanced MSA
aa_domestic["manual_msa_hit"] = aa_domestic["city_state_norm"].isin(manual_metro_set)
aa_domestic["enhanced_msa"] = aa_domestic["msa_existing"] | aa_domestic["manual_msa_hit"]
aa_domestic_enhanced_msa = aa_domestic[aa_domestic["enhanced_msa"]]
print(f"  With enhanced MSA:                {len(aa_domestic_enhanced_msa):,}")
print(f"  Paper Panel A start:              ~26,548")

# Stage 2: missing observations between Panel A start and Table 5
# Paper: Panel A = 13,771 (same as Table 5, meaning no missing-variable drop)
# Ours: panelA_final=20,143 but table5_final=10,809 (drop 9,334 due to missing vars + outliers)

t5_miss = c.get("table5_missing_counts", {})
print(f"\n  Table 5 variable missingness (from diagnostics):")
for var, nmiss in sorted(t5_miss.items(), key=lambda x: -x[1]):
    if nmiss > 0:
        print(f"    {var:20s}: {nmiss:>6,}")

# PAPER has N=13,771 for BOTH Table 2 (Panel A) AND Table 5 (Panel B).
# This means the paper has NO missing variables after Panel A attrition.
# Our biggest missingness: sigma_cfo=7,801, altman=2,983, mb=2,389, size=2,240
# The union of these will be roughly 9,147 (matching panelB_delete_missing).
print(f"\n  panelB_delete_missing (total vars missing): {c['panelB_delete_missing']:,}")
print(f"  panelB_delete_outlier: {c['panelB_delete_outlier']:,}")

# The paper likely does NOT lose 7,801 to sigma_cfo, meaning their sigma_cfo
# calculation is different or they have broader Compustat history.

# Check if sigma_cfo is the main driver
# Our sigma_cfo requires 4 consecutive prior years of cfo.
# For 2003, that means we need 1999, 2000, 2001, 2002 cfo values.
# If the paper uses min_periods=3 or a different window, fewer would be lost.
print(f"\n  KEY INSIGHT: sigma_cfo accounts for {t5_miss.get('sigma_cfo', 0):,} of the {c['panelB_delete_missing']:,} missing obs")
print(f"  This is {t5_miss.get('sigma_cfo', 0)/max(c['panelB_delete_missing'],1):.1%} of missing-variable drops")


# ════════════════════════════════════════════════════════════════
# PART 7:  SENSITIVITY — sigma_cfo WINDOW RELAXATION
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PART 7: SIGMA_CFO WINDOW SENSITIVITY")
print("=" * 80)

# How many Table-5 observations would we have if sigma_cfo used min_periods=3?
# We need to match AA to Compustat first.
# Use the already-filtered comp_period
comp_vars = comp_period[["cik_norm", "fyear", "sigma_cfo", "sigma_cfo_mp3", "sigma_cfo_mp2",
                          "size", "cfo", "lev", "loss", "mb", "altman"]].copy()

# Merge to AA nonfinancial with MSA
aa_panel = aa_nonfin[aa_nonfin["msa_existing"]].copy()
aa_panel["cik_norm"] = aa_panel["company_fkey"].fillna(aa_panel.get("company_key", "")).apply(norm_cik)
aa_panel = aa_panel.merge(comp_vars, left_on=["cik_norm", "fiscal_year"], right_on=["cik_norm", "fyear"], how="inner")

# City-industry >= 2 filter
cell_n = aa_panel.groupby(["fiscal_year", "sic2", "city_state_norm"])["company_fkey"].transform("size")
aa_panel = aa_panel[cell_n >= 2].copy()

print(f"Panel A equivalent (with MSA, in Compustat, city-ind>=2): {len(aa_panel):,}")

# Table 5 all-vars required
t5_vars_strict = ["size", "sigma_cfo", "cfo", "lev", "loss", "mb", "altman"]
t5_vars_mp3    = ["size", "sigma_cfo_mp3", "cfo", "lev", "loss", "mb", "altman"]
t5_vars_mp2    = ["size", "sigma_cfo_mp2", "cfo", "lev", "loss", "mb", "altman"]

n_strict = aa_panel.dropna(subset=t5_vars_strict).shape[0]
n_mp3    = aa_panel.dropna(subset=t5_vars_mp3).shape[0]
n_mp2    = aa_panel.dropna(subset=t5_vars_mp2).shape[0]

print(f"Table 5 obs with all vars (sigma_cfo strict 4yr):   {n_strict:,}")
print(f"Table 5 obs with all vars (sigma_cfo min_periods=3): {n_mp3:,}  (+{n_mp3-n_strict:,})")
print(f"Table 5 obs with all vars (sigma_cfo min_periods=2): {n_mp2:,}  (+{n_mp2-n_strict:,})")
print(f"Paper Table 5 N: 13,771")


# ════════════════════════════════════════════════════════════════
# PART 8:  COMBINED EFFECT — MSA + sigma_cfo RELAXATION
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PART 8: COMBINED IMPROVEMENTS SIMULATION")
print("=" * 80)

# Also test with enhanced MSA map
enhanced_msa_set = msa_existing_set | manual_metro_set
aa_nonfin["msa_enhanced"] = aa_nonfin["city_state_norm"].isin(enhanced_msa_set)
aa_panel_e = aa_nonfin[aa_nonfin["msa_enhanced"]].copy()
aa_panel_e["cik_norm"] = aa_panel_e["company_fkey"].fillna(aa_panel_e.get("company_key", "")).apply(norm_cik)
aa_panel_e = aa_panel_e.merge(comp_vars, left_on=["cik_norm", "fiscal_year"], right_on=["cik_norm", "fyear"], how="inner")
cell_n_e = aa_panel_e.groupby(["fiscal_year", "sic2", "city_state_norm"])["company_fkey"].transform("size")
aa_panel_e = aa_panel_e[cell_n_e >= 2].copy()

n_e_strict = aa_panel_e.dropna(subset=t5_vars_strict).shape[0]
n_e_mp3    = aa_panel_e.dropna(subset=t5_vars_mp3).shape[0]
n_e_mp2    = aa_panel_e.dropna(subset=t5_vars_mp2).shape[0]

print(f"{'Scenario':<55s} {'N':>7s}  {'Gap':>7s}")
print("-" * 72)
print(f"{'Paper Table 5':<55s} {'13,771':>7s}  {'---':>7s}")
print(f"{'Current replication':<55s} {10809:>7,}  {13771-10809:>+7,}")
print(f"{'+ Enhanced MSA,  sigma_cfo strict':<55s} {n_e_strict:>7,}  {13771-n_e_strict:>+7,}")
print(f"{'+ Enhanced MSA,  sigma_cfo min_periods=3':<55s} {n_e_mp3:>7,}  {13771-n_e_mp3:>+7,}")
print(f"{'+ Enhanced MSA,  sigma_cfo min_periods=2':<55s} {n_e_mp2:>7,}  {13771-n_e_mp2:>+7,}")
print(f"{'  Existing MSA,  sigma_cfo min_periods=3':<55s} {n_mp3:>7,}  {13771-n_mp3:>+7,}")

# Test domestic=strict with existing MSA
aa_dom_msa = aa_domestic_with_msa.copy()
aa_dom_msa["cik_norm"] = aa_dom_msa["company_fkey"].fillna(aa_dom_msa.get("company_key", "")).apply(norm_cik)
aa_dom_msa = aa_dom_msa.merge(comp_vars, left_on=["cik_norm", "fiscal_year"], right_on=["cik_norm", "fyear"], how="inner")
cell_n_d = aa_dom_msa.groupby(["fiscal_year", "sic2", "city_state_norm"])["company_fkey"].transform("size")
aa_dom_msa = aa_dom_msa[cell_n_d >= 2].copy()
n_dom_strict = aa_dom_msa.dropna(subset=t5_vars_strict).shape[0]
n_dom_mp3 = aa_dom_msa.dropna(subset=t5_vars_mp3).shape[0]
print(f"{'  Domestic=strict, existing MSA, sigma strict':<55s} {n_dom_strict:>7,}  {13771-n_dom_strict:>+7,}")
print(f"{'  Domestic=strict, existing MSA, sigma min3':<55s} {n_dom_mp3:>7,}  {13771-n_dom_mp3:>+7,}")

# Domestic + enhanced MSA
aa_dom_emsa = aa_domestic[aa_domestic["enhanced_msa"]].copy()
aa_dom_emsa["cik_norm"] = aa_dom_emsa["company_fkey"].fillna(aa_dom_emsa.get("company_key", "")).apply(norm_cik)
aa_dom_emsa = aa_dom_emsa.merge(comp_vars, left_on=["cik_norm", "fiscal_year"], right_on=["cik_norm", "fyear"], how="inner")
cell_n_de = aa_dom_emsa.groupby(["fiscal_year", "sic2", "city_state_norm"])["company_fkey"].transform("size")
aa_dom_emsa = aa_dom_emsa[cell_n_de >= 2].copy()
n_de_strict = aa_dom_emsa.dropna(subset=t5_vars_strict).shape[0]
n_de_mp3 = aa_dom_emsa.dropna(subset=t5_vars_mp3).shape[0]
print(f"{'  Domestic=strict, enhanced MSA, sigma strict':<55s} {n_de_strict:>7,}  {13771-n_de_strict:>+7,}")
print(f"{'  Domestic=strict, enhanced MSA, sigma min3':<55s} {n_de_mp3:>7,}  {13771-n_de_mp3:>+7,}")


# ════════════════════════════════════════════════════════════════
# PART 9:  INDIVIDUAL VARIABLE CONTRIBUTION TO MISSING DROP
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PART 9: MARGINAL VARIABLE CONTRIBUTION TO MISSING DROP")
print("=" * 80)

# Using the existing pipeline's Panel A equivalent
all_t5_vars = ["size", "sigma_cfo", "cfo", "lev", "loss", "mb", "altman"]
base_n = len(aa_panel)
print(f"Panel A rows (existing MSA, in Compustat, city-ind>=2): {base_n:,}")
print(f"\nIf we exclude ONLY that variable, how many rows survive?")
for exclude_var in all_t5_vars:
    remaining = [v for v in all_t5_vars if v != exclude_var]
    n_without = aa_panel.dropna(subset=remaining).shape[0]
    n_with = aa_panel.dropna(subset=all_t5_vars).shape[0]
    marginal_loss = n_without - n_with
    print(f"  Exclude {exclude_var:15s}: N={n_without:>7,}  (adds {marginal_loss:>5,} vs full model)")

# Also show sequential drops
print(f"\nSequential variable filtering (worst-missing first):")
ordered_vars = sorted(all_t5_vars, key=lambda v: -aa_panel[v].isna().sum())
remaining = aa_panel.copy()
for v in ordered_vars:
    before = len(remaining)
    remaining = remaining.dropna(subset=[v])
    after = len(remaining)
    print(f"  After requiring {v:15s}: {after:>7,} (dropped {before-after:>5,})")


# ════════════════════════════════════════════════════════════════
# PART 10:  FEE-OPINION MERGE LOSS ANALYSIS
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PART 10: FEE-OPINION MERGE LOSS")
print("=" * 80)

# Reload to check the merge properly
fee2 = pd.read_csv(
    BASE / "data/raw/audit_analytics/audit_fee_manual_2026-02-03.csv",
    dtype={"COMPANY_FKEY": "string", "AUDITOR_FKEY": "string"},
)
fee2.columns = [c.lower() for c in fee2.columns]
fee2["fiscal_year"] = pd.to_numeric(fee2["fiscal_year"], errors="coerce")
fee2 = fee2[(fee2["fiscal_year"] >= 2003) & (fee2["fiscal_year"] <= 2007)]
fee2["audit_fees"] = pd.to_numeric(fee2["audit_fees"], errors="coerce")
fee2 = fee2[fee2["audit_fees"] > 0]

op2 = pd.read_csv(
    BASE / "data/raw/audit_analytics/revised_audit_opinions_manual_2026-02-02.csv",
    dtype={"COMPANY_FKEY": "string", "AUDITOR_FKEY": "string"},
)
op2.columns = [c.lower() for c in op2.columns]
op2 = op2.rename(columns={"fiscal_year_of_op": "fiscal_year"})
op2["fiscal_year"] = pd.to_numeric(op2["fiscal_year"], errors="coerce")
if "file_date" in op2.columns:
    op2["file_date"] = pd.to_datetime(op2["file_date"], errors="coerce")
    op2 = op2.sort_values("file_date")
op2 = op2.drop_duplicates(["company_fkey", "auditor_fkey", "fiscal_year"], keep="last")
op2["has_city"] = op2["auditor_city"].notna()

merged = fee2.merge(
    op2[["company_fkey", "auditor_fkey", "fiscal_year", "has_city"]],
    on=["company_fkey", "auditor_fkey", "fiscal_year"],
    how="left",
)
merged["has_city"] = merged["has_city"].fillna(False)

# dedupe to company-year
merged = merged.sort_values(["company_fkey", "fiscal_year", "audit_fees"], ascending=[True, True, False])
merged = merged.drop_duplicates(["company_fkey", "fiscal_year"], keep="first")

n_total = len(merged)
n_has_city = merged["has_city"].sum()
n_no_city = n_total - n_has_city
print(f"Fee company-year rows (2003-2007, pos fee): {n_total:,}")
print(f"  With city from opinion merge: {n_has_city:,} ({n_has_city/n_total:.1%})")
print(f"  Without city (no opinion):    {n_no_city:,} ({n_no_city/n_total:.1%})")

# Could the paper have used a different merge key or city source?
# Some possibilities: the paper may have merged on (company_fkey, fiscal_year) only,
# not requiring auditor_fkey match.
merged_lax = fee2.merge(
    op2[["company_fkey", "fiscal_year", "has_city"]].drop_duplicates(["company_fkey", "fiscal_year"]),
    on=["company_fkey", "fiscal_year"],
    how="left",
)
merged_lax["has_city"] = merged_lax["has_city"].fillna(False)
merged_lax = merged_lax.sort_values(["company_fkey", "fiscal_year", "audit_fees"], ascending=[True, True, False])
merged_lax = merged_lax.drop_duplicates(["company_fkey", "fiscal_year"], keep="first")
n_lax_city = merged_lax["has_city"].sum()
print(f"\n  If merging on (company_fkey, fiscal_year) ONLY (no auditor_fkey):")
print(f"    With city: {n_lax_city:,} ({n_lax_city/len(merged_lax):.1%})")
print(f"    Gain: {n_lax_city - n_has_city:,}")


# ════════════════════════════════════════════════════════════════
# PART 11:  SUMMARY OF FINDINGS
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SUMMARY OF FINDINGS AND RECOMMENDATIONS")
print("=" * 80)

print("""
Current replication: Table 5 N = 10,809  (paper: 13,771, gap: -2,962)
                     Table 8 N =  3,533  (paper:  4,969, gap: -1,436)

ROOT CAUSES OF SAMPLE-SIZE GAP (in order of impact):

1. SIGMA_CFO WINDOW (strict min_periods=4)
   - Causes ~7,801 missing values in the working sample
   - This is the SINGLE LARGEST driver of variable-missingness drops
   - The paper likely uses a less strict rolling window or broader history
   - Relaxing to min_periods=3 recovers significant observations

2. COMPUSTAT COVERAGE
   - {del_comp:,} observations deleted (not in Compustat)
   - Our Compustat extract may not cover all firms the authors had access to
   - The authors may have used GVKEY-based matching instead of CIK

3. MSA MAPPING GAPS
   - {msa_gap:,} nonfinancial observations lost due to no MSA match
   - Manual lookup table recovers {manual_rec:,} observations
   - Key unmatched US cities: Nashville TN, Parsippany NJ, Short Hills NJ, etc.

4. DOMESTIC FILTER
   - Current: domestic_filter=none (keeps all)
   - If paper uses domestic=strict, Panel A start would decrease but gap pattern changes

5. FEE-OPINION MERGE
   - ~{no_city:,} company-year rows have no matching opinion → no city
   - Relaxing merge key to (company_fkey, fiscal_year) only recovers some

RECOMMENDED ACTIONS (create new pipeline variant):
  a. Add manual MSA lookup table for ~60 unmatched US cities
  b. Test sigma_cfo with min_periods=3 (paper may allow partial windows)
  c. Consider GVKEY-based Compustat matching if available
  d. Ensure Compustat extract includes all historical years needed for rolling windows
""".format(
    del_comp=c["panelA_delete_not_in_compustat"],
    msa_gap=int((~aa_nonfin["msa_existing"]).sum()),
    manual_rec=recovered_by_manual,
    no_city=n_no_city,
))

# Save manual MSA map for potential use
manual_map_df.to_csv(BASE / "data/processed/manual_msa_lookup.csv", index=False)
print(f"Saved manual MSA lookup to data/processed/manual_msa_lookup.csv ({len(manual_map_df)} entries)")
