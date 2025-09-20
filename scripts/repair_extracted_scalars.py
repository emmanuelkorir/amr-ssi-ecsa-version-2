import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Iterable


def _get_in(d: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _set_in(d: Dict[str, Any], path: List[str], value: Any) -> None:
    cur: Any = d
    for k in path[:-1]:
        if not isinstance(cur.get(k), dict):
            cur[k] = {}
        cur = cur[k]
    cur[path[-1]] = value


def _flatten_scalar_enums(payload: Dict[str, Any]) -> Dict[str, Any]:
    scalar_paths = [
        ["study", "ownership_type"],
        ["setting", "facility_level"],
        ["setting", "urban_rural"],
        ["setting", "care_setting"],
    ]
    for path in scalar_paths:
        v = _get_in(payload, path)
        if isinstance(v, list):
            v = v[0] if len(v) > 0 else None
        if v is not None and not isinstance(v, str):
            v = None
        _set_in(payload, path, v)
    return payload


def _strip_and_alias(payload: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload, dict) and "provenance" in payload:
        payload.pop("provenance", None)
    for key in ["study", "design", "setting", "timeframe", "qualitative_thematic"]:
        v = payload.get(key)
        if isinstance(v, dict) and "provenance" in v:
            v.pop("provenance", None)
    d = payload.get("design")
    if isinstance(d, dict) and "rationale" in d:
        if "duplicate_rationale" not in d or not d.get("duplicate_rationale"):
            d["duplicate_rationale"] = d["rationale"]
        d.pop("rationale", None)
    return payload


def _prune_nulls(o: Any) -> Any:
    """Recursively drop keys whose value is None to satisfy non-nullable schema fields."""
    if isinstance(o, dict):
        return {k: _prune_nulls(v) for k, v in o.items() if v is not None}
    if isinstance(o, list):
        return [_prune_nulls(v) for v in o]
    return o


def _normalize_item_provenance(item: Any) -> Any:
    if isinstance(item, dict) and "provenance" in item:
        v = item.get("provenance")
        if isinstance(v, list):
            v = v[0] if len(v) > 0 else None
        elif isinstance(v, dict):
            pass
        elif isinstance(v, str):
            v = {"note": v}
        elif isinstance(v, (int, float)):
            v = {"note": str(v)}
        else:
            v = None
        item["provenance"] = v
    return item


def _normalize_provenance_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    outcomes = payload.get("outcomes")
    if isinstance(outcomes, dict):
        for _, arr in outcomes.items():
            if isinstance(arr, list):
                for i, itm in enumerate(arr):
                    if isinstance(itm, dict):
                        arr[i] = _normalize_item_provenance(itm)
    qt = payload.get("qualitative_thematic")
    if isinstance(qt, dict):
        codes = qt.get("codes")
        if isinstance(codes, list):
            for i, itm in enumerate(codes):
                if isinstance(itm, dict):
                    codes[i] = _normalize_item_provenance(itm)
    for key in ["interventions", "stewardship_guidelines", "policy_capacity", "notes_flags"]:
        arr = payload.get(key)
        if isinstance(arr, list):
            for i, itm in enumerate(arr):
                if isinstance(itm, dict):
                    arr[i] = _normalize_item_provenance(itm)
    return payload


def _collect_item_level_provenance(o: Any, acc: set) -> None:
    if isinstance(o, dict):
        prov = o.get("provenance")
        if isinstance(prov, dict):
            try:
                acc.add(json.dumps(prov, sort_keys=True))
            except Exception:
                pass
        for v in o.values():
            _collect_item_level_provenance(v, acc)
    elif isinstance(o, list):
        for v in o:
            _collect_item_level_provenance(v, acc)


def _ensure_provenance_index(payload: Dict[str, Any]) -> Dict[str, Any]:
    cur = payload.get("provenance_index")
    need = not isinstance(cur, list) or len(cur) == 0
    if need:
        s: set = set()
        _collect_item_level_provenance(payload.get("outcomes", {}), s)
        _collect_item_level_provenance(payload.get("qualitative_thematic", {}), s)
        _collect_item_level_provenance(payload.get("interventions", []), s)
        _collect_item_level_provenance(payload.get("stewardship_guidelines", []), s)
        _collect_item_level_provenance(payload.get("policy_capacity", []), s)
        _collect_item_level_provenance(payload.get("notes_flags", []), s)
        items = []
        for j in s:
            try:
                items.append(json.loads(j))
            except Exception:
                continue
        if not items:
            items = [{"note": "auto-generated provenance index placeholder; per-item provenance missing or empty"}]
        payload["provenance_index"] = items
    return payload


def _normalize_provenance_deep(o: Any) -> Any:
    """Recursively coerce any 'provenance' field to an object or null.
    - list -> first item coerced recursively, else null
    - str -> {note: str}
    - int/float/bool -> {note: str(value)}
    - dict -> unchanged
    - other -> null
    """
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            if k == "provenance":
                pv = v
                if isinstance(pv, list):
                    pv = pv[0] if len(pv) > 0 else None
                if isinstance(pv, dict):
                    out[k] = _normalize_provenance_deep(pv)
                elif isinstance(pv, str):
                    out[k] = {"note": pv}
                elif isinstance(pv, (int, float, bool)):
                    out[k] = {"note": str(pv)}
                else:
                    out[k] = None
            else:
                out[k] = _normalize_provenance_deep(v)
        return out
    if isinstance(o, list):
        return [_normalize_provenance_deep(v) for v in o]
    return o


def _coerce_int(x: Any) -> Any:
    if isinstance(x, int):
        return x
    if isinstance(x, float) and x.is_integer():
        return int(x)
    if isinstance(x, str):
        s = x.strip().replace(",", "")
        if s.isdigit():
            return int(s)
        try:
            f = float(s)
            if f.is_integer():
                return int(f)
        except Exception:
            return x
    return x


def _repair_amr_rows(payload: Dict[str, Any]) -> Dict[str, Any]:
    outcomes = payload.get("outcomes")
    if not isinstance(outcomes, dict):
        return payload
    amr = outcomes.get("amr_proportions")
    if not isinstance(amr, list):
        return payload
    def _parse_percent(val: Any):
        try:
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                s = val.strip().replace("%", "")
                return float(s)
        except Exception:
            return None
        return None

    keep = []
    for row in amr:
        if not isinstance(row, dict):
            continue
        if not row.get("pathogen") or not row.get("antibiotic"):
            continue
        n_tested = _coerce_int(row.get("n_tested"))
        n_res = _coerce_int(row.get("n_resistant"))
        n_int = _coerce_int(row.get("n_intermediate"))
        n_susc = _coerce_int(row.get("n_susceptible"))

        perc = None
        for k in ["percent_resistant", "percent", "resistant_percent", "resistance_percent"]:
            if k in row:
                perc = _parse_percent(row.get(k))
                if perc is not None:
                    break
        rar = row.get("rate_as_reported")
        if perc is None and isinstance(rar, dict):
            rv = rar.get("value")
            ru = (rar.get("unit") or "").lower()
            if isinstance(rv, (int, float)):
                if ru in ("percent", "%"):
                    perc = float(rv)
                elif ru in ("per100", "per 100"):
                    perc = float(rv)
                elif ru in ("per1000", "per 1000") and isinstance(n_tested, int) and n_tested > 0:
                    perc = float(rv) / 10.0

        if (n_tested is None or (isinstance(n_tested, int) and n_tested <= 0)) and isinstance(n_res, int) and perc is not None and perc > 0:
            denom = max(n_res, int(round(100.0 * n_res / max(1e-9, perc))))
            n_tested = denom
        if (n_res is None or (isinstance(n_res, int) and n_res < 0)) and isinstance(n_tested, int) and n_tested > 0 and perc is not None and perc >= 0:
            events = int(round(perc / 100.0 * n_tested))
            if events < 0:
                events = 0
            if events > n_tested:
                events = n_tested
            n_res = events
        if (n_tested is None or (isinstance(n_tested, int) and n_tested <= 0)):
            parts = [x for x in [n_res, n_int, n_susc] if isinstance(x, int) and x >= 0]
            if parts:
                n_tested = sum(parts)

        if isinstance(n_tested, int) and isinstance(n_res, int):
            if n_tested < 1:
                n_tested = None
            else:
                if n_res < 0:
                    n_res = 0
                if n_res > n_tested:
                    n_res = n_tested

        if isinstance(n_tested, int) and n_tested >= 1 and isinstance(n_res, int) and n_res >= 0:
            row["n_tested"] = n_tested
            row["n_resistant"] = n_res
            keep.append(row)
        else:
            continue
    outcomes["amr_proportions"] = keep
    return payload


def repair_file(p: Path, write: bool = True) -> bool:
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    before = json.dumps(data, sort_keys=True)
    data = _flatten_scalar_enums(data)
    data = _strip_and_alias(data)
    data = _normalize_provenance_fields(data)
    data = _repair_amr_rows(data)
    data = _normalize_provenance_deep(data)
    data = _prune_nulls(data)
    data = _ensure_provenance_index(data)
    after = json.dumps(data, sort_keys=True)
    if before != after and write:
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Repair extracted JSON: flatten scalar enum arrays to strings")
    ap.add_argument("directory", type=str, help="Path to folder with *.extracted.json files (e.g., outputs/extracted)")
    ap.add_argument("--dry-run", action="store_true", help="Show what would change without writing")
    args = ap.parse_args()

    root = Path(args.directory)
    changed = 0
    total = 0
    for p in root.rglob("*.extracted.json"):
        total += 1
        if args.dry_run:
            # Simulate repair to see if it would change
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    continue
                before = json.dumps(data, sort_keys=True)
                data = _flatten_scalar_enums(data)
                after = json.dumps(data, sort_keys=True)
                if before != after:
                    changed += 1
                    print(f"Would repair: {p}")
            except Exception:
                continue
        else:
            if repair_file(p, write=True):
                changed += 1
                print(f"Repaired: {p}")

    print(f"Processed {total} files; {'would repair' if args.dry_run else 'repaired'} {changed}.")


if __name__ == "__main__":
    main()
