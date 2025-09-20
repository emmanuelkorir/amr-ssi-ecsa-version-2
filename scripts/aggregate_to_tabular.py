#!/usr/bin/env python3
import os, sys, json, argparse, pathlib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    yaml = None  # type: ignore
    _HAS_YAML = False

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for k in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k, None)
        else:
            return default
    return default if cur is None else cur

def to_int(x):
    try:
        return int(x)
    except:
        return None

def to_float(x):
    try:
        return float(x)
    except:
        return None

def collect_rows(ex: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    study = ex.get("study", {})
    design = ex.get("design", {})
    setting = ex.get("setting", {})
    timeframe = ex.get("timeframe", {})
    outcomes = ex.get("outcomes", {})
    qual = ex.get("qualitative_thematic", {})

    study_id = study.get("study_id")
    countries = study.get("country_list", [])
    country = countries[0] if countries else None
    facility_level = setting.get("facility_level")
    care_setting = setting.get("care_setting")
    specialties = study.get("surgical_specialty", [])
    year_start = timeframe.get("data_collection_start_year")
    year_end = timeframe.get("data_collection_end_year")

    # SSI incidence (burden_incidence)
    ssi_rows = []
    for item in outcomes.get("ssi_incidence", []) or []:
        n_events = to_int(item.get("n_ssi_events"))
        n_total = to_int(item.get("n_total"))
        if n_events is None or n_total is None:
            continue
        ssi_rows.append({
            "study_id": study_id,
            "outcome": "ssi_incidence",
            "theme": "burden_incidence",
            "events_x": n_events,
            "sample_size_n": n_total,
            "ssi_type": item.get("ssi_type"),
            "country": item.get("subgroup_labels", {}).get("country", country),
            "facility_level": item.get("subgroup_labels", {}).get("facility_level", facility_level),
            "setting": item.get("subgroup_labels", {}).get("setting", care_setting),
            "surgical_specialty": item.get("surgical_specialty") or (specialties[0] if specialties else None),
            "denominator_type": item.get("denominator_type"),
            "ascertainment_window_days": item.get("ascertainment_window_days"),
            "readmission_window_days": item.get("readmission_window_days"),
            "year_start": safe_get(item, ["data_period", "start_year"], year_start),
            "year_end": safe_get(item, ["data_period", "end_year"], year_end),
            "reoperation_definition_text": item.get("reoperation_definition_text"),
        })

    # AMR proportions (amr_prevalence)
    amr_rows = []
    for item in outcomes.get("amr_proportions", []) or []:
        n_res = to_int(item.get("n_resistant"))
        n_tested = to_int(item.get("n_tested"))
        if n_res is None or n_tested is None:
            continue
        amr_rows.append({
            "study_id": study_id,
            "outcome": "amr_prevalence",
            "theme": "amr_prevalence",
            "events_x": n_res,
            "sample_size_n": n_tested,
            "country": country,
            "facility_level": facility_level,
            "setting": care_setting,
            "surgical_specialty": None,
            "denominator_type": "isolates",
            "ascertainment_window_days": None,
            "year_start": safe_get(item, ["data_period", "start_year"], year_start),
            "year_end": safe_get(item, ["data_period", "end_year"], year_end),
            "pathogen": item.get("pathogen"),
            "antibiotic": item.get("antibiotic"),
            "who_class": item.get("who_class"),
            "ast_method": item.get("ast_method"),
            "breakpoint_standard": item.get("breakpoint_standard"),
            "breakpoint_version": item.get("breakpoint_version"),
            "specimen_type": item.get("specimen_type"),
            "mdr_definition_text": item.get("mdr_definition_text"),
            "n_mdr": to_int(item.get("n_mdr")),
            "n_xdr": to_int(item.get("n_xdr")),
            "esbl_test": item.get("esbl_test"),
            "n_esbl_positive": to_int(item.get("n_esbl_positive")),
            "carbapenemase_test": item.get("carbapenemase_test"),
            "n_carbapenemase_positive": to_int(item.get("n_carbapenemase_positive")),
            "carbapenemase_genes": item.get("carbapenemase_genes"),
            "mrsa_detection_method": item.get("mrsa_detection_method"),
            "resistance_gene_markers": item.get("resistance_gene_markers"),
            "mic_unit": item.get("mic_unit"),
            "mic_breakpoint_s": item.get("mic_breakpoint_s"),
            "mic_breakpoint_r": item.get("mic_breakpoint_r"),
            "zone_diameter_unit": item.get("zone_diameter_unit"),
        })

    # Qualitative codes
    code_rows = []
    for code in (qual.get("codes") or []):
        quotes = code.get("evidence_quotes") or []
        if quotes:
            for q in quotes:
                code_rows.append({
                    "study_id": study_id,
                    "theme": code.get("theme"),
                    "subtheme": code.get("subtheme"),
                    "summary": code.get("summary"),
                    "quote": q.get("quote"),
                    "quote_page": q.get("page"),
                    "quote_section": q.get("section"),
                    "location_context": code.get("location_context"),
                    "confidence": code.get("confidence"),
                })
        else:
            code_rows.append({
                "study_id": study_id,
                "theme": code.get("theme"),
                "subtheme": code.get("subtheme"),
                "summary": code.get("summary"),
                "quote": None,
                "quote_page": None,
                "quote_section": None,
                "location_context": code.get("location_context"),
                "confidence": code.get("confidence"),
            })

    # Mortality
    mortality_rows = []
    for item in outcomes.get("mortality", []) or []:
        mortality_rows.append({
            "study_id": study_id,
            "outcome": "mortality",
            "measure": item.get("measure"),
            "n_deaths": to_int(item.get("n_deaths")),
            "n_total": to_int(item.get("n_total")),
            "rate_value": to_float(safe_get(item, ["rate_as_reported", "value"], None)),
            "rate_unit": safe_get(item, ["rate_as_reported", "unit"], None),
            "definition_text": item.get("definition_text"),
            "country": safe_get(item, ["subgroup_labels", "country"], country),
            "facility_level": safe_get(item, ["subgroup_labels", "facility_level"], facility_level),
            "setting": safe_get(item, ["subgroup_labels", "setting"], care_setting),
            "year_start": safe_get(item, ["data_period", "start_year"], year_start),
            "year_end": safe_get(item, ["data_period", "end_year"], year_end),
        })

    # Readmissions
    readm_rows = []
    for item in outcomes.get("readmissions", []) or []:
        readm_rows.append({
            "study_id": study_id,
            "outcome": "readmissions",
            "n_events": to_int(item.get("n_events")),
            "n_total": to_int(item.get("n_total")),
            "window_days": to_int(item.get("window_days")),
            "definition_text": item.get("definition_text"),
            "country": safe_get(item, ["subgroup_labels", "country"], country),
            "facility_level": safe_get(item, ["subgroup_labels", "facility_level"], facility_level),
            "setting": safe_get(item, ["subgroup_labels", "setting"], care_setting),
            "year_start": safe_get(item, ["data_period", "start_year"], year_start),
            "year_end": safe_get(item, ["data_period", "end_year"], year_end),
        })

    # Length of stay
    los_rows = []
    for item in outcomes.get("length_of_stay", []) or []:
        los_rows.append({
            "study_id": study_id,
            "outcome": "length_of_stay",
            "group": item.get("group"),
            "mean_days": to_float(item.get("mean_days")),
            "sd_days": to_float(item.get("sd_days")),
            "median_days": to_float(item.get("median_days")),
            "iqr_days": item.get("iqr_days"),
            "n": to_int(item.get("n")),
            "attributable_days": to_float(item.get("attributable_days")),
            "country": country,
            "facility_level": facility_level,
            "setting": care_setting,
            "year_start": year_start,
            "year_end": year_end,
        })

    # Costs
    cost_rows = []
    for item in outcomes.get("costs", []) or []:
        cost_rows.append({
            "study_id": study_id,
            "outcome": "costs",
            "cost_type": item.get("cost_type"),
            "perspective": item.get("perspective"),
            "component": item.get("component"),
            "currency_code": item.get("currency_code"),
            "price_year": to_int(item.get("price_year")),
            "amount": to_float(item.get("amount")),
            "n": to_int(item.get("n")),
            "per_patient": item.get("per_patient"),
            "exchange_rate_source": item.get("exchange_rate_source"),
            "ppp_adjusted": item.get("ppp_adjusted"),
            "country": country,
            "facility_level": facility_level,
            "setting": care_setting,
            "year_start": year_start,
            "year_end": year_end,
        })

    dfs = {}
    dfs["ssi"] = pd.DataFrame(ssi_rows) if ssi_rows else pd.DataFrame(columns=[
        "study_id","outcome","theme","events_x","sample_size_n","ssi_type","country","facility_level",
        "setting","surgical_specialty","denominator_type","ascertainment_window_days","readmission_window_days","year_start","year_end","reoperation_definition_text"
    ])
    dfs["amr"] = pd.DataFrame(amr_rows) if amr_rows else pd.DataFrame(columns=[
        "study_id","outcome","theme","events_x","sample_size_n","country","facility_level","setting",
        "surgical_specialty","denominator_type","ascertainment_window_days","year_start","year_end",
        "pathogen","antibiotic","who_class","ast_method","breakpoint_standard","breakpoint_version","specimen_type",
        "mdr_definition_text","n_mdr","n_xdr","esbl_test","n_esbl_positive","carbapenemase_test","n_carbapenemase_positive","carbapenemase_genes","mrsa_detection_method","resistance_gene_markers","mic_unit","mic_breakpoint_s","mic_breakpoint_r","zone_diameter_unit"
    ])
    dfs["codes"] = pd.DataFrame(code_rows) if code_rows else pd.DataFrame(columns=[
        "study_id","theme","subtheme","summary","quote","quote_page","quote_section","location_context","confidence"
    ])
    dfs["mortality"] = pd.DataFrame(mortality_rows) if mortality_rows else pd.DataFrame(columns=[
        "study_id","outcome","measure","n_deaths","n_total","rate_value","rate_unit","definition_text","country","facility_level","setting","year_start","year_end"
    ])
    dfs["readmissions"] = pd.DataFrame(readm_rows) if readm_rows else pd.DataFrame(columns=[
        "study_id","outcome","n_events","n_total","window_days","definition_text","country","facility_level","setting","year_start","year_end"
    ])
    dfs["length_of_stay"] = pd.DataFrame(los_rows) if los_rows else pd.DataFrame(columns=[
        "study_id","outcome","group","mean_days","sd_days","median_days","iqr_days","n","attributable_days","country","facility_level","setting","year_start","year_end"
    ])
    dfs["costs"] = pd.DataFrame(cost_rows) if cost_rows else pd.DataFrame(columns=[
        "study_id","outcome","cost_type","perspective","component","currency_code","price_year","amount","n","per_patient","exchange_rate_source","ppp_adjusted","country","facility_level","setting","year_start","year_end"
    ])
    return dfs

def _load_theme_taxonomy(root: pathlib.Path) -> Dict[str, Any]:
    path = root / "data_schemas" / "theme_taxonomy.yaml"
    if not (_HAS_YAML and yaml is not None):
        return {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}
    return {}

def _taxonomy_subthemes(taxonomy: Dict[str, Any]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    try:
        themes = taxonomy.get("themes", []) if isinstance(taxonomy, dict) else []
        for t in themes:
            code = (t or {}).get("code")
            subs = (t or {}).get("subthemes", []) or []
            if code:
                out[str(code)] = [str(s) for s in subs]
    except Exception:
        return {}
    return out

def _load_stratification_vars(root: pathlib.Path) -> List[str]:
    path = root / "analysis" / "stratification_variables.yaml"
    if not (_HAS_YAML and yaml is not None):
        return []
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                return list(data.get("core_subgroups", []) or [])
        except Exception:
            return []
    return []

def _percent(numer: Optional[float], denom: Optional[float]) -> Optional[float]:
    try:
        if numer is None or denom in (None, 0, 0.0):
            return None
        return round((float(numer) / float(denom)) * 100.0, 2)
    except Exception:
        return None

def _wilson_ci(numer: Optional[float], denom: Optional[float], z: float = 1.96) -> Tuple[Optional[float], Optional[float]]:
    try:
        if numer is None or denom in (None, 0, 0.0):
            return (None, None)
        n = float(denom)
        x = float(numer)
        p = x / n
        z2 = z * z
        denom_adj = 1.0 + z2 / n
        center = (p + z2 / (2.0 * n)) / denom_adj
        margin = z / denom_adj * ((p * (1 - p) / n + z2 / (4.0 * n * n)) ** 0.5)
        low = max(0.0, center - margin)
        high = min(1.0, center + margin)
        return (round(low * 100.0, 2), round(high * 100.0, 2))
    except Exception:
        return (None, None)

def build_quant_thematic(ssi_df: pd.DataFrame, amr_df: pd.DataFrame, mortality_df: pd.DataFrame, readm_df: pd.DataFrame, los_df: pd.DataFrame, costs_df: pd.DataFrame, taxonomy_map: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    taxonomy_map = taxonomy_map or {}

    # SSI incidence -> theme: burden_incidence
    if not ssi_df.empty:
        for _, r in ssi_df.iterrows():
            events = r.get("events_x")
            total = r.get("sample_size_n")
            pct = _percent(events, total)
            ci_lo, ci_hi = _wilson_ci(events, total)
            context_bits = []
            if pd.notna(r.get("country")):
                context_bits.append(str(r.get("country")))
            if pd.notna(r.get("surgical_specialty")):
                context_bits.append(str(r.get("surgical_specialty")))
            years = None
            if pd.notna(r.get("year_start")) or pd.notna(r.get("year_end")):
                ys = str(r.get("year_start")) if pd.notna(r.get("year_start")) else ""
                ye = str(r.get("year_end")) if pd.notna(r.get("year_end")) else ""
                years = f"{ys}-{ye}" if ys or ye else None
            if years:
                context_bits.append(years)
            context = ", ".join([c for c in context_bits if c]) if context_bits else None
            summary = f"SSI incidence {events}/{total}" + (f" ({pct}%)" if pct is not None else "")
            if context:
                summary += f" in {context}"

            # Decide subthemes
            subthemes = ["overall_incidence"]
            if pd.notna(r.get("surgical_specialty")):
                subthemes.append("specialty_specific")
            if pd.notna(r.get("country")):
                subthemes.append("geographic_variation")
            if pd.notna(r.get("year_start")) and pd.notna(r.get("year_end")) and r.get("year_start") != r.get("year_end"):
                subthemes.append("temporal_trends")

            # keep only subthemes defined in taxonomy if provided
            allowed = taxonomy_map.get("burden_incidence")
            if allowed:
                subthemes = [s for s in subthemes if s in allowed]
                if not subthemes:
                    subthemes = [allowed[0]]
            for st in subthemes:
                rows.append({
                    "study_id": r.get("study_id"),
                    "theme": "burden_incidence",
                    "subtheme": st,
                    "summary": summary,
                    "quote": None,
                    "quote_page": None,
                    "quote_section": None,
                    "location_context": None,
                    "confidence": None,
                    "events": events,
                    "total": total,
                    "pct": pct,
                    "ci95_low": ci_lo,
                    "ci95_high": ci_hi,
                })

    # AMR proportions -> theme: amr_prevalence
    if not amr_df.empty:
        for _, r in amr_df.iterrows():
            events = r.get("events_x")
            total = r.get("sample_size_n")
            pct = _percent(events, total)
            ci_lo, ci_hi = _wilson_ci(events, total)
            patho = r.get("pathogen")
            abx = r.get("antibiotic")
            summary = "AMR prevalence"
            parts = []
            if pd.notna(patho):
                parts.append(str(patho))
            if pd.notna(abx):
                parts.append(str(abx))
            if parts:
                summary += f" for {' / '.join(parts)}"
            summary += f": {events}/{total}" + (f" ({pct}%)" if pct is not None else "")

            subthemes = ["pathogen_specific"] if pd.notna(patho) else []
            if pd.notna(r.get("who_class")) or pd.notna(abx):
                subthemes.append("antibiotic_class_trends")
            if pd.notna(r.get("n_mdr")) or pd.notna(r.get("n_xdr")):
                subthemes.append("mdr_xdr_pandr")
            if pd.notna(r.get("specimen_type")):
                subthemes.append("specimen_site")
            if not subthemes:
                subthemes = ["pathogen_specific"]

            allowed = taxonomy_map.get("amr_prevalence")
            if allowed:
                subthemes = [s for s in subthemes if s in allowed]
                if not subthemes:
                    subthemes = [allowed[0]]
            for st in subthemes:
                rows.append({
                    "study_id": r.get("study_id"),
                    "theme": "amr_prevalence",
                    "subtheme": st,
                    "summary": summary,
                    "quote": None,
                    "quote_page": None,
                    "quote_section": None,
                    "location_context": None,
                    "confidence": None,
                    "events": events,
                    "total": total,
                    "pct": pct,
                    "ci95_low": ci_lo,
                    "ci95_high": ci_hi,
                })

    # Clinical outcomes -> outcomes_clinical
    if not mortality_df.empty:
        for _, r in mortality_df.iterrows():
            events = r.get('n_deaths')
            total = r.get('n_total')
            pct = _percent(events, total)
            ci_lo, ci_hi = _wilson_ci(events, total)
            summary = f"Mortality: {events}/{total}" + (f" ({pct}%)" if pct is not None else "")
            allowed = taxonomy_map.get("outcomes_clinical")
            st = "mortality"
            if allowed and st not in allowed:
                st = allowed[0]
            rows.append({
                "study_id": r.get("study_id"),
                "theme": "outcomes_clinical",
                "subtheme": st,
                "summary": summary,
                "quote": None,
                "quote_page": None,
                "quote_section": None,
                "location_context": None,
                "confidence": None,
                "events": events,
                "total": total,
                "pct": pct,
                "ci95_low": ci_lo,
                "ci95_high": ci_hi,
            })
    if not readm_df.empty:
        for _, r in readm_df.iterrows():
            events = r.get('n_events')
            total = r.get('n_total')
            pct = _percent(events, total)
            ci_lo, ci_hi = _wilson_ci(events, total)
            summary = f"Readmissions: {events}/{total} in {r.get('window_days')} days" + (f" ({pct}%)" if pct is not None else "")
            allowed = taxonomy_map.get("outcomes_clinical")
            st = "readmissions"
            if allowed and st not in allowed:
                st = allowed[0]
            rows.append({
                "study_id": r.get("study_id"),
                "theme": "outcomes_clinical",
                "subtheme": st,
                "summary": summary,
                "quote": None,
                "quote_page": None,
                "quote_section": None,
                "location_context": None,
                "confidence": None,
                "events": events,
                "total": total,
                "pct": pct,
                "ci95_low": ci_lo,
                "ci95_high": ci_hi,
            })
    if not los_df.empty:
        for _, r in los_df.iterrows():
            days = r.get("mean_days") or r.get("median_days")
            summary = f"Length of stay: {days} days"
            allowed = taxonomy_map.get("outcomes_clinical")
            st = "prolonged_los"
            if allowed and st not in allowed:
                st = allowed[0]
            rows.append({
                "study_id": r.get("study_id"),
                "theme": "outcomes_clinical",
                "subtheme": st,
                "summary": summary,
                "quote": None,
                "quote_page": None,
                "quote_section": None,
                "location_context": None,
                "confidence": None,
                "events": None,
                "total": None,
                "pct": None,
                "ci95_low": None,
                "ci95_high": None,
            })

    # Economic -> economic_impact
    if not costs_df.empty:
        for _, r in costs_df.iterrows():
            amt = r.get("amount")
            cur = r.get("currency_code")
            summary = f"Costs ({r.get('cost_type') or 'costs'}): {amt} {cur}"
            allowed = taxonomy_map.get("economic_impact")
            st = "direct_costs"
            if allowed and st not in allowed:
                st = allowed[0]
            rows.append({
                "study_id": r.get("study_id"),
                "theme": "economic_impact",
                "subtheme": st,
                "summary": summary,
                "quote": None,
                "quote_page": None,
                "quote_section": None,
                "location_context": None,
                "confidence": None,
                "events": None,
                "total": None,
                "pct": None,
                "ci95_low": None,
                "ci95_high": None,
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "study_id","theme","subtheme","summary","quote","quote_page","quote_section","location_context","confidence","events","total","pct","ci95_low","ci95_high"
    ])

def _continuity_correction(e: float, n: float) -> Tuple[float, float]:
    # Add 0.5 to each cell to avoid 0 counts in logit transform
    e_adj = float(e) + 0.5
    ne_adj = float(n - e) + 0.5
    return e_adj, ne_adj

def _logit_and_se(events: Optional[float], total: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    try:
        if events is None or total in (None, 0, 0.0):
            return (None, None)
        e = float(events)
        n = float(total)
        if e <= 0.0 or e >= n:
            e_adj, ne_adj = _continuity_correction(e, n)
        else:
            e_adj, ne_adj = e, n - e
        logit = float(np.log(e_adj / ne_adj))
        se = (1.0 / e_adj + 1.0 / ne_adj) ** 0.5
        return (round(logit, 6), round(se, 6))
    except Exception:
        return (None, None)

def _compute_rollup(df: pd.DataFrame, events_col: str, total_col: str, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        cols = group_cols + ["n_studies","events","total","pct","ci95_low","ci95_high","logit","se_logit"]
        return pd.DataFrame(columns=cols)
    # Ensure columns exist
    for c in group_cols:
        if c not in df.columns:
            df[c] = pd.NA
    grouped = df.groupby(group_cols, dropna=False).agg({
        "study_id": pd.Series.nunique,
        events_col: "sum",
        total_col: "sum",
    }).reset_index()
    grouped = grouped.rename(columns={"study_id": "n_studies", events_col: "events", total_col: "total"})
    # Derived metrics
    def _row_metrics(r):
        pct = _percent(r["events"], r["total"])
        lo, hi = _wilson_ci(r["events"], r["total"]) if r["total"] else (None, None)
        lg, se = _logit_and_se(r["events"], r["total"]) if r["total"] else (None, None)
        return pd.Series({"pct": pct, "ci95_low": lo, "ci95_high": hi, "logit": lg, "se_logit": se})
    metrics = grouped.apply(_row_metrics, axis=1)
    out = pd.concat([grouped, metrics], axis=1)
    return out

def align_to_stratification(df: pd.DataFrame, subgroup_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        # Create empty with expected columns
        base_cols = [c for c in df.columns] if len(df.columns) > 0 else []
        for c in subgroup_cols:
            if c not in base_cols:
                base_cols.append(c)
        return pd.DataFrame(columns=base_cols)
    out = df.copy()
    for c in subgroup_cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out

def main():
    ap = argparse.ArgumentParser(description="Aggregate extracted JSONs into analyzable CSVs.")
    ap.add_argument("--in-dir", required=True, help="Directory with *.extracted.json files")
    ap.add_argument("--out-processed", required=True, help="Path for combined long-format processed.csv")
    ap.add_argument("--out-ssi", required=True, help="Path for ssi_incidence_long.csv")
    ap.add_argument("--out-amr", required=True, help="Path for amr_proportions_long.csv")
    ap.add_argument("--out-codes", required=True, help="Path for qualitative codes.csv")
    ap.add_argument("--out-mortality", required=True, help="Path for mortality_long.csv")
    ap.add_argument("--out-readmissions", required=True, help="Path for readmissions_long.csv")
    ap.add_argument("--out-los", required=True, help="Path for length_of_stay_long.csv")
    ap.add_argument("--out-costs", required=True, help="Path for costs_long.csv")
    # New optional thematic and stratified outputs
    ap.add_argument("--out-quant-thematic", required=False, default=None, help="Optional path for quantitative thematic rows (quant_thematic.csv)")
    ap.add_argument("--out-thematic-all", required=False, default=None, help="Optional path for combined thematic rows (qual + quant)")
    ap.add_argument("--out-ssi-stratified", required=False, default=None, help="Optional path for SSI stratified table aligned to core subgroups")
    ap.add_argument("--out-amr-stratified", required=False, default=None, help="Optional path for AMR stratified table aligned to core subgroups")
    # Rollups and meta-analysis ready summaries
    ap.add_argument("--out-ssi-rollup", required=False, default=None, help="Optional path for SSI subgroup rollups (events/total/pct/CI/logit/SE)")
    ap.add_argument("--out-amr-rollup", required=False, default=None, help="Optional path for AMR subgroup rollups (events/total/pct/CI/logit/SE)")
    ap.add_argument("--rollup-ssi-by", required=False, default=None, help="Comma-separated columns to group SSI rollups by (overrides config)")
    ap.add_argument("--rollup-amr-by", required=False, default=None, help="Comma-separated columns to group AMR rollups by (overrides config)")
    ap.add_argument("--rollup-config", required=False, default=None, help="Optional JSON file defining default grouping lists under keys 'ssi' and 'amr'")
    args = ap.parse_args()

    in_dir = pathlib.Path(args.in_dir)
    files = sorted([p for p in in_dir.glob("*.extracted.json")])
    if not files:
        print(f"No extracted JSONs found in {in_dir}", file=sys.stderr)
        sys.exit(1)

    all_ssi = []
    all_amr = []
    all_codes = []
    all_mortality = []
    all_readm = []
    all_los = []
    all_costs = []

    for p in tqdm(files, desc="Aggregating"):
        ex = load_json(str(p))
        dfs = collect_rows(ex)
        if not dfs["ssi"].empty:
            all_ssi.append(dfs["ssi"])
        if not dfs["amr"].empty:
            all_amr.append(dfs["amr"])
        if not dfs["codes"].empty:
            all_codes.append(dfs["codes"])
        if not dfs["mortality"].empty:
            all_mortality.append(dfs["mortality"])
        if not dfs["readmissions"].empty:
            all_readm.append(dfs["readmissions"])
        if not dfs["length_of_stay"].empty:
            all_los.append(dfs["length_of_stay"])
        if not dfs["costs"].empty:
            all_costs.append(dfs["costs"])

    ssi_df = pd.concat(all_ssi, ignore_index=True) if all_ssi else pd.DataFrame()
    amr_df = pd.concat(all_amr, ignore_index=True) if all_amr else pd.DataFrame()
    codes_df = pd.concat(all_codes, ignore_index=True) if all_codes else pd.DataFrame()
    mortality_df = pd.concat(all_mortality, ignore_index=True) if all_mortality else pd.DataFrame()
    readm_df = pd.concat(all_readm, ignore_index=True) if all_readm else pd.DataFrame()
    los_df = pd.concat(all_los, ignore_index=True) if all_los else pd.DataFrame()
    costs_df = pd.concat(all_costs, ignore_index=True) if all_costs else pd.DataFrame()

    # Build quantitative thematic rows using taxonomy where available
    project_root = pathlib.Path(args.in_dir).resolve().parents[0]
    taxonomy = _load_theme_taxonomy(project_root)
    taxonomy_map = _taxonomy_subthemes(taxonomy)
    quant_thematic_df = build_quant_thematic(ssi_df, amr_df, mortality_df, readm_df, los_df, costs_df, taxonomy_map)

    # Unified processed.csv
    common_cols = [
        "study_id","outcome","theme","events_x","sample_size_n","country","facility_level","setting",
        "surgical_specialty","denominator_type","ascertainment_window_days","year_start","year_end",
        "ssi_type","readmission_window_days","reoperation_definition_text"
    ]
    # AMR has extra columns; keep them in union
    union_cols = list({*common_cols, *amr_df.columns.tolist()})
    # Build processed by row-binding ssi and amr and aligning columns
    def align_cols(df, cols):
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA
        return df[cols]

    ssi_aligned = align_cols(ssi_df, union_cols) if not ssi_df.empty else pd.DataFrame(columns=union_cols)
    amr_aligned = align_cols(amr_df, union_cols) if not amr_df.empty else pd.DataFrame(columns=union_cols)
    processed = pd.concat([ssi_aligned, amr_aligned], ignore_index=True)

    # Ensure numeric dtypes
    for c in ["events_x","sample_size_n","year_start","year_end","ascertainment_window_days"]:
        if c in processed.columns:
            processed[c] = pd.to_numeric(processed[c], errors="coerce")

    # Write outputs
    os.makedirs(os.path.dirname(args.out_processed), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_ssi), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_amr), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_codes), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_mortality), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_readmissions), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_los), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_costs), exist_ok=True)

    processed.to_csv(args.out_processed, index=False)
    if not ssi_df.empty:
        ssi_df.to_csv(args.out_ssi, index=False)
    if not amr_df.empty:
        amr_df.to_csv(args.out_amr, index=False)
    if not codes_df.empty:
        codes_df.to_csv(args.out_codes, index=False)
    if not mortality_df.empty:
        mortality_df.to_csv(args.out_mortality, index=False)
    if not readm_df.empty:
        readm_df.to_csv(args.out_readmissions, index=False)
    if not los_df.empty:
        los_df.to_csv(args.out_los, index=False)
    if not costs_df.empty:
        costs_df.to_csv(args.out_costs, index=False)

    # Optional: quantitative thematic
    if args.out_quant_thematic:
        os.makedirs(os.path.dirname(args.out_quant_thematic), exist_ok=True)
        if not quant_thematic_df.empty:
            quant_thematic_df.to_csv(args.out_quant_thematic, index=False)
        else:
            pd.DataFrame(columns=["study_id","theme","subtheme","summary","quote","quote_page","quote_section","location_context","confidence"]).to_csv(args.out_quant_thematic, index=False)

    # Optional: combined thematic (qual + quant)
    if args.out_thematic_all:
        os.makedirs(os.path.dirname(args.out_thematic_all), exist_ok=True)
        if not codes_df.empty or not quant_thematic_df.empty:
            # Align columns to codes_df schema
            def align_codes_cols(df: pd.DataFrame) -> pd.DataFrame:
                cols = ["study_id","theme","subtheme","summary","quote","quote_page","quote_section","location_context","confidence"]
                for c in cols:
                    if c not in df.columns:
                        df[c] = pd.NA
                return df[cols]
            combined = pd.concat([align_codes_cols(codes_df), align_codes_cols(quant_thematic_df)], ignore_index=True)
            combined.to_csv(args.out_thematic_all, index=False)
        else:
            pd.DataFrame(columns=["study_id","theme","subtheme","summary","quote","quote_page","quote_section","location_context","confidence"]).to_csv(args.out_thematic_all, index=False)

    # Optional stratified tables aligned to core subgroups
    subgroup_cols = _load_stratification_vars(project_root)
    if subgroup_cols:
        if args.out_ssi_stratified:
            os.makedirs(os.path.dirname(args.out_ssi_stratified), exist_ok=True)
            ssi_strat = align_to_stratification(ssi_df, subgroup_cols)
            ssi_strat.to_csv(args.out_ssi_stratified, index=False)
        if args.out_amr_stratified:
            os.makedirs(os.path.dirname(args.out_amr_stratified), exist_ok=True)
            amr_strat = align_to_stratification(amr_df, subgroup_cols)
            amr_strat.to_csv(args.out_amr_stratified, index=False)

    # Load rollup config if provided
    rollup_cfg_ssi: Optional[List[str]] = None
    rollup_cfg_amr: Optional[List[str]] = None
    if args.rollup_config and os.path.exists(args.rollup_config):
        try:
            with open(args.rollup_config, "r", encoding="utf-8") as f:
                cfg = json.load(f) or {}
                ssi_cfg = cfg.get("ssi")
                if isinstance(ssi_cfg, list) and len(ssi_cfg) > 0:
                    rollup_cfg_ssi = [str(c) for c in ssi_cfg]
                amr_cfg = cfg.get("amr")
                if isinstance(amr_cfg, list) and len(amr_cfg) > 0:
                    rollup_cfg_amr = [str(c) for c in amr_cfg]
        except Exception:
            pass

    # Optional subgroup rollups with meta-analysis-ready columns
    if args.out_ssi_rollup:
        os.makedirs(os.path.dirname(args.out_ssi_rollup), exist_ok=True)
        if args.rollup_ssi_by:
            ssi_groups = [c.strip() for c in args.rollup_ssi_by.split(",") if c.strip()]
        else:
            ssi_groups = rollup_cfg_ssi or ["country","facility_level","setting","surgical_specialty","year_start","year_end"]
        ssi_roll = _compute_rollup(ssi_df.rename(columns={"events_x":"events","sample_size_n":"total"}), "events", "total", ssi_groups)
        ssi_roll.to_csv(args.out_ssi_rollup, index=False)
    if args.out_amr_rollup:
        os.makedirs(os.path.dirname(args.out_amr_rollup), exist_ok=True)
        if args.rollup_amr_by:
            amr_groups = [c.strip() for c in args.rollup_amr_by.split(",") if c.strip()]
        else:
            amr_groups = rollup_cfg_amr or ["country","facility_level","setting","pathogen","antibiotic","specimen_type","year_start","year_end"]
        amr_roll = _compute_rollup(amr_df.rename(columns={"events_x":"events","sample_size_n":"total"}), "events", "total", amr_groups)
        amr_roll.to_csv(args.out_amr_rollup, index=False)

    print("Aggregation complete.")
    print(f"processed: {args.out_processed}")
    if not ssi_df.empty:
        print(f"ssi: {args.out_ssi}")
    if not amr_df.empty:
        print(f"amr: {args.out_amr}")
    if not codes_df.empty:
        print(f"codes: {args.out_codes}")
    if not mortality_df.empty:
        print(f"mortality: {args.out_mortality}")
    if not readm_df.empty:
        print(f"readmissions: {args.out_readmissions}")
    if not los_df.empty:
        print(f"length_of_stay: {args.out_los}")
    if not costs_df.empty:
        print(f"costs: {args.out_costs}")
    if args.out_quant_thematic:
        print(f"quant_thematic: {args.out_quant_thematic}")
    if args.out_thematic_all:
        print(f"thematic_all: {args.out_thematic_all}")
    if args.out_ssi_stratified:
        print(f"ssi_stratified: {args.out_ssi_stratified}")
    if args.out_amr_stratified:
        print(f"amr_stratified: {args.out_amr_stratified}")
    if args.out_ssi_rollup:
        print(f"ssi_rollup: {args.out_ssi_rollup}")
    if args.out_amr_rollup:
        print(f"amr_rollup: {args.out_amr_rollup}")

if __name__ == "__main__":
    main()