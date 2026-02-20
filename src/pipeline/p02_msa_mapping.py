#!/usr/bin/env python3
"""
p02_msa_mapping.py — 第二步：审计师办公室 → MSA/CBSA 地理映射
===============================================================
论文 p.109:
  "We identify auditor office location using the city and state
   reported on Audit Analytics... We map each city-state pair
   to a Metropolitan Statistical Area (MSA) using the Census
   Bureau's place-to-CBSA crosswalk."

匹配策略 (当前: geocorr2022-only):
  1) Exact match: 标准化的 city|state 精确匹配 Census geocorr2022 表
  2) token unique: 同州下 city tokens ⊆ place tokens，且唯一 CBSA
  3) Fuzzy match: 同州下 SequenceMatcher 模糊匹配
     (score ≥ threshold, best-second gap ≥ gap)

输入: data/processed/pipeline/s01_aa_merged.csv
输出: data/processed/pipeline/s02_aa_with_msa.csv
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import PipelineConfig, ensure_dirs
from pipeline.utils import norm_text, STATE_FIPS_TO_ABBR


PLACE_SUFFIX_TOKENS = {
    "CITY", "TOWN", "TOWNSHIP", "VILLAGE", "BOROUGH", "MUNICIPALITY",
    "CDP", "CENSUS", "DESIGNATED", "PLACE", "COUNTY", "TWP",
}
US50DC_STATE_ABBR = {
    abbr for fips, abbr in STATE_FIPS_TO_ABBR.items()
    if str(fips).isdigit() and int(str(fips)) <= 56
}
AUTO_MSA_BLOCKLIST = {"CEDAR CITY|UT"}


# ═══════════════════════════════════════════════════════════════════
# Census place→CBSA 解析
# ═══════════════════════════════════════════════════════════════════

def parse_place_to_cbsa(path: Path) -> pd.DataFrame:
    """
    解析 place→CBSA 映射，返回 city_state_norm → cbsa 映射。

    支持两类输入：
    1) geocorr2022.xlsx（推荐）:
       - cbsa20 / cbsatype20 / stab / PlaceName / CBSAName20
       - PlaceName 形如 "Abilene city, TX"
    2) place05-cbsa06.xls（兼容旧版）
    """
    d = pd.read_excel(path)
    d.columns = [str(c).strip().upper() for c in d.columns]

    # ── geocorr2022 分支 ───────────────────────────────────────
    geocorr_need = {"PLACENAME", "STAB", "CBSA20"}
    if geocorr_need.issubset(set(d.columns)):
        col_place = "PLACENAME"
        col_state = "STAB"
        col_cbsa = "CBSA20"
        col_title = "CBSANAME20"
        col_type = "CBSATYPE20"

        cols = [col_place, col_state, col_cbsa]
        if col_title in d.columns:
            cols.append(col_title)
        if col_type in d.columns:
            cols.append(col_type)
        m = d[cols].copy()

        # MSA: 仅保留 Metropolitan（排除 Micropolitan）
        if col_type in m.columns:
            m = m[m[col_type].astype(str).str.contains("METRO", case=False, na=False)]

        m = m.dropna(subset=[col_place, col_state, col_cbsa]).copy()
        m[col_state] = m[col_state].astype(str).str.upper().str.strip()

        # PlaceName 清洗规则：
        # 1) 删除逗号后州缩写（", TX"）
        # 2) 删除残余逗号
        # 3) 删除后缀 city/town/village/CDP（及常见等价词）
        m["place_clean"] = (
            m[col_place]
            .astype(str)
            .str.replace(r",\s*[A-Z]{2}\s*$", "", regex=True)
            .str.replace(",", " ", regex=False)
            .str.replace(
                r"\s+(CITY|TOWN|TOWNSHIP|VILLAGE|CDP"
                r"|CENSUS DESIGNATED PLACE)$",
                "",
                case=False,
                regex=True,
            )
            .str.strip()
        )
        # 仅保留含字母的 place 名称，剔除纯编码类噪声值
        m = m[m["place_clean"].astype(str).str.contains(r"[A-Z]", case=False, regex=True)].copy()

        m["cbsa_code"] = pd.to_numeric(m[col_cbsa], errors="coerce").astype("Int64")
        m = m.dropna(subset=["cbsa_code"]).copy()
        m["cbsa_code"] = m["cbsa_code"].astype(int).astype(str).str.zfill(5)
        if col_title in m.columns:
            m["cbsa_title"] = m[col_title].astype(str).str.strip()
        else:
            m["cbsa_title"] = ""

        m["place_norm"] = m["place_clean"].apply(norm_text)
        m = m[m["place_norm"].astype(str).ne("")].copy()
        m["state"] = m[col_state]
        m["city_state_norm"] = m["place_norm"] + "|" + m["state"]

        g = (
            m.groupby(
                ["city_state_norm", "place_norm", "state", "cbsa_code", "cbsa_title"],
                as_index=False,
            )
            .size()
            .sort_values(["city_state_norm", "size"], ascending=[True, False])
        )
        g = g.drop_duplicates(["city_state_norm"], keep="first")
        return g[["city_state_norm", "place_norm", "state", "cbsa_code", "cbsa_title"]]

    # ── place05 兼容分支 ───────────────────────────────────────
    col_place = "PLACE"
    col_state = "STATE"
    col_cbsa = "CBSA CODE"
    col_title = "CBSA TITLE"
    col_lsad = "CBSA LSAD"

    need = [col_place, col_state, col_cbsa, col_title]
    if not set(need).issubset(d.columns):
        return pd.DataFrame(
            columns=["city_state_norm", "place_norm",
                     "state", "cbsa_code", "cbsa_title"]
        )

    cols = need + ([col_lsad] if col_lsad in d.columns else [])
    m = d[cols].copy()

    # 仅保留 Metropolitan 类（排除 Micropolitan）
    if col_lsad in m.columns:
        m = m[m[col_lsad].astype(str).str.contains(
            "Metropolitan", case=False, na=False
        )]

    m = m.dropna(subset=[col_place, col_state, col_cbsa])
    m[col_state] = m[col_state].astype(str).str.upper().str.strip()
    m[col_place] = (
        m[col_place]
        .astype(str)
        .str.replace(r"\s*\(.*?\)\s*", " ", regex=True)
        .str.strip()
        .str.replace(
            r"\s+(CITY|TOWN|VILLAGE|BOROUGH|MUNICIPALITY|CDP"
            r"|CENSUS DESIGNATED PLACE)$",
            "", case=False, regex=True,
        )
        .str.strip()
    )
    m["cbsa_code"] = pd.to_numeric(m[col_cbsa], errors="coerce").astype("Int64")
    m = m.dropna(subset=["cbsa_code"]).copy()
    m["cbsa_code"] = m["cbsa_code"].astype(int).astype(str).str.zfill(5)
    m["cbsa_title"] = m[col_title].astype(str).str.strip()
    m["place_norm"] = m[col_place].apply(norm_text)
    m["state"] = m[col_state]
    m["city_state_norm"] = m.apply(
        lambda r: f"{r['place_norm']}|{r['state']}", axis=1
    )

    # 同一 city-state 映射到多个 CBSA → 取出现次数最多的
    g = (
        m.groupby(["city_state_norm", "place_norm", "state",
                    "cbsa_code", "cbsa_title"], as_index=False)
        .size()
        .sort_values(["city_state_norm", "size"], ascending=[True, False])
    )
    g = g.drop_duplicates(["city_state_norm"], keep="first")
    return g[["city_state_norm", "place_norm",
              "state", "cbsa_code", "cbsa_title"]]


def _tokenize_place_name(v: str) -> set[str]:
    """用于 token 匹配的城市名分词（剔除通用后缀词）。"""
    if pd.isna(v):
        return set()
    tokens = [t for t in str(v).split() if t and t not in PLACE_SUFFIX_TOKENS]
    return set(tokens)


def build_place05_token_unique_map(
    unmatched_pairs: pd.DataFrame,
    place_map: pd.DataFrame,
) -> pd.DataFrame:
    """
    同州唯一 token 匹配：
      city_tokens ⊆ place_tokens，且候选 CBSA 唯一。
    """
    if unmatched_pairs.empty or place_map.empty:
        return pd.DataFrame(
            columns=[
                "city_state_norm", "cbsa_token", "cbsa_title_token",
                "token_matched_place", "token_candidate_n",
            ]
        )

    pm = place_map.copy()
    pm["place_tokens"] = pm["place_norm"].apply(_tokenize_place_name)
    state_rows = {
        st: g[["place_norm", "place_tokens", "cbsa_code", "cbsa_title"]].to_dict("records")
        for st, g in pm.groupby("state")
    }

    rows = []
    seen = set()
    for r in unmatched_pairs.itertuples(index=False):
        city_state_norm = str(r.city_state_norm)
        city_norm = str(r.city_norm)
        state = str(r.auditor_state)
        if city_state_norm in seen or not city_norm or not state:
            continue
        seen.add(city_state_norm)

        city_tokens = _tokenize_place_name(city_norm)
        if not city_tokens:
            continue

        candidates = []
        for rec in state_rows.get(state, []):
            if city_tokens.issubset(rec["place_tokens"]):
                candidates.append(rec)
        if not candidates:
            continue

        cbsa_codes = sorted({str(c["cbsa_code"]) for c in candidates if pd.notna(c["cbsa_code"])})
        if len(cbsa_codes) != 1:
            continue

        chosen_cbsa = cbsa_codes[0]
        chosen = next((c for c in candidates if str(c["cbsa_code"]) == chosen_cbsa), candidates[0])
        rows.append(
            {
                "city_state_norm": city_state_norm,
                "cbsa_token": str(chosen["cbsa_code"]),
                "cbsa_title_token": str(chosen["cbsa_title"]),
                "token_matched_place": str(chosen["place_norm"]),
                "token_candidate_n": int(len(candidates)),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "city_state_norm", "cbsa_token", "cbsa_title_token",
                "token_matched_place", "token_candidate_n",
            ]
        )
    return out.drop_duplicates(["city_state_norm"], keep="first")


def parse_list2_principal(path: Path) -> pd.DataFrame:
    """解析 list2_2023 Principal City → CBSA（仅 Metropolitan）。"""
    d = pd.read_excel(path)
    d.columns = [str(c).strip().upper() for c in d.columns]

    col_cbsa = "CBSA CODE"
    col_title = "CBSA TITLE"
    col_type = "METROPOLITAN/MICROPOLITAN STATISTICAL AREA"
    col_city = "PRINCIPAL CITY NAME"
    col_state_fips = "FIPS STATE CODE"

    need = [col_cbsa, col_title, col_city, col_state_fips]
    if not set(need).issubset(d.columns):
        return pd.DataFrame(
            columns=["city_state_norm", "city_norm",
                     "state", "cbsa_code", "cbsa_title"]
        )

    m = d[need + ([col_type] if col_type in d.columns else [])].copy()
    if col_type in m.columns:
        m = m[m[col_type].astype(str).str.contains(
            "Metropolitan", case=False, na=False
        )]

    m = m.dropna(subset=[col_cbsa, col_title, col_city, col_state_fips]).copy()
    m["state_fips"] = pd.to_numeric(m[col_state_fips], errors="coerce").astype("Int64")
    m = m.dropna(subset=["state_fips"]).copy()
    m["state_fips"] = m["state_fips"].astype(int).astype(str).str.zfill(2)
    m["state"] = m["state_fips"].map(STATE_FIPS_TO_ABBR)
    m = m.dropna(subset=["state"]).copy()
    m = m[m["state"].isin(US50DC_STATE_ABBR)].copy()

    m["cbsa_code"] = pd.to_numeric(m[col_cbsa], errors="coerce").astype("Int64")
    m = m.dropna(subset=["cbsa_code"]).copy()
    m["cbsa_code"] = m["cbsa_code"].astype(int).astype(str).str.zfill(5)
    m["cbsa_title"] = m[col_title].astype(str).str.strip()
    m["city_norm"] = m[col_city].apply(norm_text)
    m = m[m["city_norm"].astype(str).ne("")].copy()
    m["city_state_norm"] = m["city_norm"] + "|" + m["state"].astype(str)

    # 同一 city-state 若出现多个 CBSA，视为歧义并剔除。
    amb_n = m.groupby("city_state_norm")["cbsa_code"].nunique().rename("n_cbsa")
    m = m.merge(amb_n, on="city_state_norm", how="left")
    m = m[m["n_cbsa"] == 1].copy()
    m = m.drop(columns=["n_cbsa"])
    m = m.drop_duplicates(["city_state_norm"], keep="first")
    return m[["city_state_norm", "city_norm", "state", "cbsa_code", "cbsa_title"]]


def parse_candidate_high_map(path: Path) -> pd.DataFrame:
    """
    从 unmatched candidates 中读取 high confidence 映射。
    规则：confidence=high，且排除 CEDAR CITY|UT。
    """
    if not path.exists():
        return pd.DataFrame(
            columns=["city_state_norm", "cbsa_code", "cbsa_title"]
        )
    d = pd.read_csv(path, dtype=str)
    if d.empty:
        return pd.DataFrame(
            columns=["city_state_norm", "cbsa_code", "cbsa_title"]
        )
    for c in ["city_norm", "state", "confidence", "candidate_cbsa", "candidate_title"]:
        if c not in d.columns:
            return pd.DataFrame(
                columns=["city_state_norm", "cbsa_code", "cbsa_title"]
            )
    d["city_norm"] = d["city_norm"].astype(str).str.upper().str.strip()
    d["state"] = d["state"].astype(str).str.upper().str.strip()
    d["city_state_norm"] = d["city_norm"] + "|" + d["state"]
    d = d[d["confidence"].astype(str).str.lower().eq("high")].copy()
    d = d[~d["city_state_norm"].isin(AUTO_MSA_BLOCKLIST)].copy()
    d["candidate_cbsa"] = d["candidate_cbsa"].astype(str).str.strip()
    d = d[
        d["candidate_cbsa"].astype(str).ne("")
        & ~d["candidate_cbsa"].astype(str).str.contains(r"\|", regex=True)
    ].copy()
    d["cbsa_code"] = pd.to_numeric(d["candidate_cbsa"], errors="coerce").astype("Int64")
    d = d.dropna(subset=["cbsa_code"]).copy()
    d["cbsa_code"] = d["cbsa_code"].astype(int).astype(str).str.zfill(5)
    d["cbsa_title"] = d["candidate_title"].astype(str).str.strip()
    d = d.drop_duplicates(["city_state_norm"], keep="first")
    return d[["city_state_norm", "cbsa_code", "cbsa_title"]]


def _norm_county_name(v: str) -> str:
    s = str(v).upper()
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(
        r"\b(COUNTY|PARISH|BOROUGH|CENSUS AREA|MUNICIPALITY|CITY AND BOROUGH)\b",
        "", s,
    )
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_county_to_cbsa(path: Path) -> pd.DataFrame:
    """
    从 place05-cbsa06 提取 county-state -> CBSA 映射（Metropolitan only）。
    """
    d = pd.read_excel(path)
    d.columns = [str(c).strip().upper() for c in d.columns]
    need = ["STATE", "COUNTY NAME", "CBSA CODE", "CBSA TITLE", "CBSA LSAD"]
    if not set(need).issubset(d.columns):
        return pd.DataFrame(
            columns=["state", "county_norm", "cbsa_code", "cbsa_title"]
        )
    m = d[need].copy()
    m = m[m["CBSA LSAD"].astype(str).str.contains("Metropolitan", case=False, na=False)]
    m = m.dropna(subset=["STATE", "COUNTY NAME", "CBSA CODE"]).copy()
    m["state"] = m["STATE"].astype(str).str.upper().str.strip()
    m["county_norm"] = m["COUNTY NAME"].apply(_norm_county_name)
    m["cbsa_code"] = pd.to_numeric(m["CBSA CODE"], errors="coerce").astype("Int64")
    m = m.dropna(subset=["cbsa_code"]).copy()
    m["cbsa_code"] = m["cbsa_code"].astype(int).astype(str).str.zfill(5)
    m["cbsa_title"] = m["CBSA TITLE"].astype(str).str.strip()
    g = (
        m.groupby(["state", "county_norm", "cbsa_code", "cbsa_title"], as_index=False)
        .size()
        .sort_values(["state", "county_norm", "size"], ascending=[True, True, False])
    )
    g = g.drop_duplicates(["state", "county_norm"], keep="first")
    return g[["state", "county_norm", "cbsa_code", "cbsa_title"]]


def _geocode_county_nominatim(city: str, state: str) -> tuple[str, str]:
    """
    使用 Nominatim 获取 county。
    返回 (county_norm, display_name)。失败时 county_norm 为空字符串。
    """
    q = f"{city}, {state}, USA"
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": q, "format": "jsonv2", "addressdetails": 1, "limit": 1}
    )
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ReicheltReplicationBot/1.0"},
    )
    with urllib.request.urlopen(req, timeout=25) as resp:
        data = json.load(resp)
    if not data:
        return "", ""
    rec = data[0]
    addr = rec.get("address", {})
    county = (
        addr.get("county")
        or addr.get("borough")
        or addr.get("municipality")
        or ""
    )
    return _norm_county_name(county) if county else "", str(rec.get("display_name", ""))


def build_external_geocode_topn_map(
    candidate_path: Path,
    county_cbsa_map: pd.DataFrame,
    cache_path: Path,
    top_n: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    对 unmatched candidates 中 note 含 'need external geocode' 的前 top_n 高频城市：
      city-state -> geocode county -> county-state -> CBSA。
    返回:
      - map_df: city_state_norm -> cbsa_code/cbsa_title
      - log_df: 每个城市的 geocode + county 映射日志
    """
    empty_map = pd.DataFrame(
        columns=["city_state_norm", "cbsa_code", "cbsa_title", "county_norm"]
    )
    empty_log = pd.DataFrame(
        columns=[
            "city_state_norm", "city_norm", "state", "obs_n", "status",
            "county_norm", "cbsa_code", "cbsa_title", "display_name",
        ]
    )
    if (not candidate_path.exists()) or county_cbsa_map.empty:
        return empty_map, empty_log

    cand = pd.read_csv(candidate_path, dtype=str)
    need = ["city_norm", "state", "obs_n", "note"]
    if not set(need).issubset(cand.columns):
        return empty_map, empty_log
    cand["city_norm"] = cand["city_norm"].astype(str).str.upper().str.strip()
    cand["state"] = cand["state"].astype(str).str.upper().str.strip()
    cand["obs_n"] = pd.to_numeric(cand["obs_n"], errors="coerce")
    cand["note"] = cand["note"].astype(str)
    cand["city_state_norm"] = cand["city_norm"] + "|" + cand["state"]
    pick = cand[
        cand["note"].str.contains("need external geocode", case=False, na=False)
        & cand["city_norm"].astype(str).ne("")
        & cand["state"].isin(US50DC_STATE_ABBR)
    ].copy()
    if pick.empty:
        return empty_map, empty_log
    pick = pick.sort_values("obs_n", ascending=False).head(int(top_n)).copy()
    pick = pick[~pick["city_state_norm"].isin(AUTO_MSA_BLOCKLIST)].copy()

    if cache_path.exists():
        cache = pd.read_csv(cache_path, dtype=str)
    else:
        cache = pd.DataFrame(columns=["city_state_norm", "county_norm", "display_name"])
    cache = cache.drop_duplicates(["city_state_norm"], keep="last")
    cache_lookup = cache.set_index("city_state_norm").to_dict(orient="index")

    county_lookup = {
        (str(r.state), str(r.county_norm)): (str(r.cbsa_code), str(r.cbsa_title))
        for r in county_cbsa_map.itertuples(index=False)
    }

    logs = []
    cache_rows = []
    for r in pick.itertuples(index=False):
        city_state_norm = str(r.city_state_norm)
        city_norm = str(r.city_norm)
        state = str(r.state)
        obs_n = int(r.obs_n) if pd.notna(r.obs_n) else 0

        county_norm = ""
        display_name = ""
        from_cache = False
        if city_state_norm in cache_lookup:
            county_norm = str(cache_lookup[city_state_norm].get("county_norm", "") or "")
            display_name = str(cache_lookup[city_state_norm].get("display_name", "") or "")
            from_cache = True
        else:
            try:
                county_norm, display_name = _geocode_county_nominatim(city_norm, state)
            except Exception:
                county_norm, display_name = "", ""
            cache_rows.append(
                {
                    "city_state_norm": city_state_norm,
                    "county_norm": county_norm,
                    "display_name": display_name,
                }
            )
            time.sleep(1.0)

        cbsa = county_lookup.get((state, county_norm))
        if cbsa is None and county_norm:
            status = "geocoded_no_county_cbsa"
            cbsa_code = ""
            cbsa_title = ""
        elif cbsa is None:
            status = "geocode_failed"
            cbsa_code = ""
            cbsa_title = ""
        else:
            status = "mapped_from_county"
            cbsa_code, cbsa_title = cbsa

        logs.append(
            {
                "city_state_norm": city_state_norm,
                "city_norm": city_norm,
                "state": state,
                "obs_n": obs_n,
                "status": status if not from_cache else f"{status}_cache",
                "county_norm": county_norm,
                "cbsa_code": cbsa_code,
                "cbsa_title": cbsa_title,
                "display_name": display_name,
            }
        )

    if cache_rows:
        cache_new = pd.concat([cache, pd.DataFrame(cache_rows)], ignore_index=True)
        cache_new = cache_new.drop_duplicates(["city_state_norm"], keep="last")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_new.to_csv(cache_path, index=False)

    log_df = pd.DataFrame(logs)
    if log_df.empty:
        return empty_map, empty_log
    map_df = log_df[
        log_df["status"].astype(str).str.startswith("mapped_from_county")
    ].copy()
    if map_df.empty:
        return empty_map, log_df
    map_df = map_df.drop_duplicates(["city_state_norm"], keep="first")
    return (
        map_df[["city_state_norm", "cbsa_code", "cbsa_title", "county_norm"]],
        log_df,
    )


# ═══════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════

def run(cfg: PipelineConfig) -> Path:
    ensure_dirs(cfg)
    pipe = Path(cfg.pipeline_dir)
    counts = {}
    place05_only = True

    # ── 2.1 读入前一步输出 ───────────────────────────────────────
    print("[p02] 读取 s01_aa_merged.csv ...")
    aa = pd.read_csv(pipe / "s01_aa_merged.csv", dtype=str)
    # 恢复数值列
    for c in ["fiscal_year", "audit_fees", "sic_code_fkey",
              "going_concern", "tenure_years", "tenure_ln"]:
        if c in aa.columns:
            aa[c] = pd.to_numeric(aa[c], errors="coerce")
    counts["input_rows"] = len(aa)

    # ── 2.2 解析 Census place→CBSA 映射 ─────────────────────────
    msa_path = Path(cfg.msa_place_map_path)
    if not msa_path.exists():
        raise FileNotFoundError(f"MSA 映射文件不存在: {msa_path}")

    place_map = parse_place_to_cbsa(msa_path)
    if place_map.empty:
        raise RuntimeError(f"未能解析到 Metropolitan 行: {msa_path}")
    counts["place_map_rows"] = len(place_map)
    if place05_only:
        list2_map = pd.DataFrame(
            columns=["city_state_norm", "city_norm",
                     "state", "cbsa_code", "cbsa_title"]
        )
        high_candidate_map = pd.DataFrame(
            columns=["city_state_norm", "cbsa_code", "cbsa_title"]
        )
        county_cbsa_map = pd.DataFrame(
            columns=["state", "county_norm", "cbsa_code", "cbsa_title"]
        )
    else:
        list2_path = Path(cfg.msa_list2_path)
        if list2_path.exists():
            list2_map = parse_list2_principal(list2_path)
        else:
            list2_map = pd.DataFrame(
                columns=["city_state_norm", "city_norm",
                         "state", "cbsa_code", "cbsa_title"]
            )
        # 可选：读取 unmatched candidates high-confidence 自动并入
        candidate_path = Path(cfg.output_dir) / "unmatched_city_state_msa_candidates.csv"
        high_candidate_map = parse_candidate_high_map(candidate_path)
        # county -> CBSA 映射（用于 external geocode 手工补全）
        county_cbsa_map = parse_county_to_cbsa(msa_path)

    counts["list2_principal_rows"] = len(list2_map)
    counts["candidate_high_rows"] = len(high_candidate_map)
    counts["county_cbsa_rows"] = len(county_cbsa_map)

    # ── 2.3 标准化 AA 审计师城市/州 ─────────────────────────────
    aa["auditor_state"] = (
        aa["auditor_state"].astype("string").str.upper().str.strip()
    )
    aa["city_norm"] = aa["auditor_city"].apply(norm_text)
    aa["city_state_norm"] = aa.apply(
        lambda r: f"{r['city_norm']}|{r['auditor_state']}"
        if (pd.notna(r["city_norm"]) and r["city_norm"]
            and pd.notna(r["auditor_state"]) and r["auditor_state"])
        else "",
        axis=1,
    )

    # ── 2.4 精确匹配 ────────────────────────────────────────────
    exact_map = (
        place_map[["city_state_norm", "cbsa_code", "cbsa_title"]]
        .drop_duplicates(["city_state_norm"], keep="first")
    )
    aa = aa.merge(
        exact_map.rename(columns={
            "cbsa_code": "cbsa_exact", "cbsa_title": "cbsa_title_exact"
        }),
        on="city_state_norm",
        how="left",
    )
    counts["exact_matched"] = int(aa["cbsa_exact"].notna().sum())

    # ── 2.5 High-confidence candidates 直连 ────────────────────
    if not high_candidate_map.empty:
        aa = aa.merge(
            high_candidate_map.rename(
                columns={"cbsa_code": "cbsa_high", "cbsa_title": "cbsa_title_high"}
            ),
            on="city_state_norm",
            how="left",
        )
    else:
        aa["cbsa_high"] = pd.NA
        aa["cbsa_title_high"] = pd.NA
    # exact 已命中则不使用 high 覆盖
    aa.loc[aa["cbsa_exact"].notna(), ["cbsa_high", "cbsa_title_high"]] = pd.NA
    counts["candidate_high_matched"] = int(aa["cbsa_high"].notna().sum())

    # ── 2.6 Top30 external geocode + county -> CBSA ───────────
    if place05_only:
        geocode_map = pd.DataFrame(
            columns=["city_state_norm", "cbsa_code", "cbsa_title", "county_norm"]
        )
        geocode_log = pd.DataFrame(
            columns=[
                "city_state_norm", "city_norm", "state", "obs_n", "status",
                "county_norm", "cbsa_code", "cbsa_title", "display_name",
            ]
        )
    else:
        geocode_cache_path = pipe / "s02_external_geocode_cache.csv"
        geocode_map, geocode_log = build_external_geocode_topn_map(
            candidate_path=Path(cfg.output_dir) / "unmatched_city_state_msa_candidates.csv",
            county_cbsa_map=county_cbsa_map,
            cache_path=geocode_cache_path,
            top_n=30,
        )
    counts["external_geocode_top30_mapped_pairs"] = len(geocode_map)
    if not geocode_map.empty:
        aa = aa.merge(
            geocode_map.rename(
                columns={
                    "cbsa_code": "cbsa_geocode",
                    "cbsa_title": "cbsa_title_geocode",
                    "county_norm": "geocode_county_norm",
                }
            ),
            on="city_state_norm",
            how="left",
        )
    else:
        aa["cbsa_geocode"] = pd.NA
        aa["cbsa_title_geocode"] = pd.NA
        aa["geocode_county_norm"] = pd.NA
    # 仅在 exact/high 均未命中时使用 geocode 映射
    aa.loc[
        aa["cbsa_exact"].notna() | aa["cbsa_high"].notna(),
        ["cbsa_geocode", "cbsa_title_geocode", "geocode_county_norm"],
    ] = pd.NA
    counts["external_geocode_top30_matched"] = int(aa["cbsa_geocode"].notna().sum())

    # ── 2.7 place05 同州唯一 token 匹配 ─────────────────────────
    place_states = set(place_map["state"].dropna().astype(str).unique())
    token_base = (
        aa["cbsa_exact"].isna()
        & aa["cbsa_high"].isna()
        & aa["cbsa_geocode"].isna()
        & aa["city_norm"].astype(str).ne("")
        & aa["auditor_state"].isin(place_states)
        & aa["auditor_state"].isin(US50DC_STATE_ABBR)
    )
    token_pairs = aa.loc[token_base, ["city_state_norm", "city_norm", "auditor_state"]].drop_duplicates()
    token_map = build_place05_token_unique_map(token_pairs, place_map)
    if not token_map.empty:
        token_map = token_map[~token_map["city_state_norm"].isin(AUTO_MSA_BLOCKLIST)].copy()
    if not token_map.empty:
        aa = aa.merge(token_map, on="city_state_norm", how="left")
    else:
        aa["cbsa_token"] = pd.NA
        aa["cbsa_title_token"] = pd.NA
        aa["token_matched_place"] = pd.NA
        aa["token_candidate_n"] = np.nan
    token_hit_n = int(aa["cbsa_token"].notna().sum())
    # 向后兼容保留旧 key（place05_token_matched）
    counts["token_unique_matched"] = token_hit_n
    counts["place05_token_matched"] = token_hit_n

    # ── 2.8 list2 principal city 精确补 ────────────────────────
    list2_base = (
        aa["cbsa_exact"].isna()
        & aa["cbsa_high"].isna()
        & aa["cbsa_geocode"].isna()
        & aa["cbsa_token"].isna()
        & aa["city_norm"].astype(str).ne("")
        & aa["auditor_state"].isin(US50DC_STATE_ABBR)
    )
    if not list2_map.empty:
        list2_exact_map = (
            list2_map[["city_state_norm", "cbsa_code", "cbsa_title"]]
            .drop_duplicates(["city_state_norm"], keep="first")
            .rename(columns={"cbsa_code": "cbsa_list2", "cbsa_title": "cbsa_title_list2"})
        )
        aa = aa.merge(list2_exact_map, on="city_state_norm", how="left")
        aa.loc[~list2_base, ["cbsa_list2", "cbsa_title_list2"]] = pd.NA
    else:
        aa["cbsa_list2"] = pd.NA
        aa["cbsa_title_list2"] = pd.NA
        list2_exact_map = pd.DataFrame(
            columns=["city_state_norm", "cbsa_list2", "cbsa_title_list2"]
        )
    counts["list2_principal_matched"] = int(aa["cbsa_list2"].notna().sum())

    # ── 2.9 模糊匹配（同州约束） ────────────────────────────────
    state_to_places = (
        place_map.dropna(subset=["state", "place_norm"])
        .groupby("state")["place_norm"]
        .apply(lambda s: sorted(set(s.astype(str))))
        .to_dict()
    )
    place_lookup = {
        (str(r.place_norm), str(r.state)): (str(r.cbsa_code), str(r.cbsa_title))
        for r in place_map[["place_norm", "state",
                            "cbsa_code", "cbsa_title"]].itertuples(index=False)
    }

    pre_hit = (
        aa["cbsa_exact"].notna()
        | aa["cbsa_high"].notna()
        | aa["cbsa_geocode"].notna()
        | aa["cbsa_token"].notna()
        | aa["cbsa_list2"].notna()
    )
    fuzzy_base = (
        (~pre_hit)
        & aa["city_norm"].astype(str).ne("")
        & aa["auditor_state"].isin(place_states)
        & aa["auditor_state"].isin(US50DC_STATE_ABBR)
    )

    fuzzy_rows = []
    seen = set()
    for r in (aa.loc[fuzzy_base, ["city_state_norm", "city_norm",
                                   "auditor_state"]]
              .drop_duplicates().itertuples(index=False)):
        if r.city_state_norm in seen:
            continue
        seen.add(r.city_state_norm)
        cands = state_to_places.get(str(r.auditor_state), [])
        if not cands:
            continue
        best_name, best_sc, second_sc = "", -1.0, -1.0
        for cand in cands:
            sc = SequenceMatcher(None, str(r.city_norm), str(cand)).ratio()
            if sc > best_sc:
                second_sc = best_sc
                best_sc = sc
                best_name = cand
            elif sc > second_sc:
                second_sc = sc
        if (best_sc >= cfg.msa_fuzzy_threshold
                and (best_sc - max(second_sc, 0.0)) >= cfg.msa_fuzzy_gap):
            cbsa = place_lookup.get((best_name, str(r.auditor_state)))
            if cbsa is not None:
                fuzzy_rows.append({
                    "city_state_norm": r.city_state_norm,
                    "cbsa_fuzzy": cbsa[0],
                    "cbsa_title_fuzzy": cbsa[1],
                    "fuzzy_place_norm": best_name,
                    "fuzzy_score": best_sc,
                })

    fuzzy_map = pd.DataFrame(fuzzy_rows).drop_duplicates(
        ["city_state_norm"], keep="first"
    )
    if not fuzzy_map.empty:
        aa = aa.merge(fuzzy_map, on="city_state_norm", how="left")
    else:
        aa["cbsa_fuzzy"] = pd.NA
        aa["cbsa_title_fuzzy"] = pd.NA
        aa["fuzzy_place_norm"] = pd.NA
        aa["fuzzy_score"] = np.nan

    counts["fuzzy_matched"] = int(aa["cbsa_fuzzy"].notna().sum())

    # ── 2.10 合并来源 → msa_code ───────────────────────────────
    aa["msa_code"] = (
        aa["cbsa_exact"]
        .fillna(aa["cbsa_high"])
        .fillna(aa["cbsa_geocode"])
        .fillna(aa["cbsa_token"])
        .fillna(aa["cbsa_list2"])
        .fillna(aa["cbsa_fuzzy"])
    )
    aa["msa_title"] = (
        aa["cbsa_title_exact"]
        .fillna(aa["cbsa_title_high"])
        .fillna(aa["cbsa_title_geocode"])
        .fillna(aa["cbsa_title_token"])
        .fillna(aa["cbsa_title_list2"])
        .fillna(aa["cbsa_title_fuzzy"])
    )
    aa["msa_match_source"] = np.select(
        [
            aa["cbsa_exact"].notna(),
            aa["cbsa_high"].notna(),
            aa["cbsa_geocode"].notna(),
            aa["cbsa_token"].notna(),
            aa["cbsa_list2"].notna(),
            aa["cbsa_fuzzy"].notna(),
        ],
        ["exact", "candidate_high", "geocode_county", "place05_token", "list2_principal", "fuzzy"],
        default="",
    )
    counts["msa_hit_total"] = int(aa["msa_code"].notna().sum())
    counts["msa_miss_total"] = int(aa["msa_code"].isna().sum())

    # ── 输出 ─────────────────────────────────────────────────────
    # 清理中间列，保留必要列
    drop_cols = [
        "cbsa_exact", "cbsa_title_exact",
        "cbsa_high", "cbsa_title_high",
        "cbsa_geocode", "cbsa_title_geocode", "geocode_county_norm",
        "cbsa_token", "cbsa_title_token", "token_matched_place", "token_candidate_n",
        "cbsa_list2", "cbsa_title_list2",
        "cbsa_fuzzy", "cbsa_title_fuzzy", "fuzzy_place_norm", "fuzzy_score",
        "city_norm", "city_state_norm",
    ]
    aa.drop(columns=[c for c in drop_cols if c in aa.columns],
            inplace=True, errors="ignore")

    out_path = pipe / "s02_aa_with_msa.csv"
    aa.to_csv(out_path, index=False)

    # 保存映射审计记录
    place_map.to_csv(pipe / "s02_msa_place_map_parsed.csv", index=False)
    exact_map.to_csv(pipe / "s02_exact_map_used.csv", index=False)
    if not high_candidate_map.empty:
        high_candidate_map.to_csv(pipe / "s02_candidate_high_map_used.csv", index=False)
    if not geocode_map.empty:
        geocode_map.to_csv(pipe / "s02_external_geocode_county_map_used.csv", index=False)
    if not geocode_log.empty:
        geocode_log.to_csv(pipe / "s02_external_geocode_top30_log.csv", index=False)
    if not token_map.empty:
        token_map.to_csv(pipe / "s02_place05_token_map_used.csv", index=False)
    if not list2_map.empty:
        list2_map.to_csv(pipe / "s02_list2_principal_map_parsed.csv", index=False)
    if not list2_exact_map.empty:
        list2_exact_map.to_csv(pipe / "s02_list2_principal_map_used.csv", index=False)
    if not fuzzy_map.empty:
        fuzzy_map.to_csv(pipe / "s02_fuzzy_map_used.csv", index=False)

    meta = pipe / "s02_counts.json"
    meta.write_text(json.dumps(counts, indent=2, ensure_ascii=False))
    print(f"[p02] 完成 → {out_path}  ({len(aa)} rows)")
    print(f"[p02] MSA 命中: exact={counts['exact_matched']}, "
          f"candidate_high={counts['candidate_high_matched']}, "
          f"geocode_county={counts['external_geocode_top30_matched']}, "
          f"token_unique={counts['token_unique_matched']}, "
          f"list2_principal={counts['list2_principal_matched']}, "
          f"fuzzy={counts['fuzzy_matched']}, "
          f"total={counts['msa_hit_total']}")
    return out_path


if __name__ == "__main__":
    cfg = PipelineConfig.from_cli()
    run(cfg)
