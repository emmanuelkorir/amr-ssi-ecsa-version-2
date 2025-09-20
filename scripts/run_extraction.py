#!/usr/bin/env python3
import os, sys, json, argparse, time, pathlib
from typing import Dict, Any, Optional, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt
from jsonschema import validate, Draft202012Validator
from jsonschema.exceptions import ValidationError

# Optional OpenAI imports; script works in dry-run mode without them
try:
    # New-style OpenAI SDK
    from openai import OpenAI as OpenAIClientClass
    _HAS_OPENAI = True
except Exception:
    OpenAIClientClass = None  # type: ignore[assignment]
    _HAS_OPENAI = False

try:
    from openai import AzureOpenAI as AzureClient
    _HAS_AZURE = True
except Exception:
    AzureClient = None  # type: ignore[assignment]
    _HAS_AZURE = False

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _json_path(path: Iterable[Any]) -> str:
    parts = []
    for p in path:
        if isinstance(p, int):
            parts.append(f"[{p}]")
        else:
            parts.append(f".{p}")
    return "$" + "".join(parts)


def validate_json_schema(obj: Any, schema: Dict[str, Any], file_label: str) -> Optional[str]:
    try:
        Draft202012Validator(schema).validate(obj)
        return None
    except ValidationError as e:
        return f"Validation error for {file_label} at {_json_path(e.path)}: {e.message}"


def _prune_nulls(o: Any) -> Any:
    """Recursively remove keys whose value is None; keep arrays and items intact."""
    if isinstance(o, dict):
        return {k: _prune_nulls(v) for k, v in o.items() if v is not None}
    if isinstance(o, list):
        return [_prune_nulls(v) for v in o]
    return o


def _get_in(d: Dict[str, Any], path: list) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _set_in(d: Dict[str, Any], path: list, value: Any) -> None:
    cur: Any = d
    for k in path[:-1]:
        if not isinstance(cur.get(k), dict):
            cur[k] = {}
        cur = cur[k]
    cur[path[-1]] = value


def _sanitize_scalar_enums(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce fields that should be scalar strings (or null) but sometimes arrive as single-item arrays.
    This helps satisfy JSON Schema when models emit ["public"] instead of "public".
    """
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


def _strip_disallowed_provenance(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Remove 'provenance' from objects that disallow additionalProperties (e.g., study, design, setting, timeframe, qualitative_thematic)."""
    # Remove stray top-level provenance if present
    if isinstance(payload, dict) and "provenance" in payload:
        payload.pop("provenance", None)
    for key in ["study", "design", "setting", "timeframe", "qualitative_thematic"]:
        v = payload.get(key)
        if isinstance(v, dict) and "provenance" in v:
            v.pop("provenance", None)
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
        # Collect from arrays and nested objects that carry per-item provenance
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


def _normalize_item_provenance(item: Any) -> Any:
    if isinstance(item, dict) and "provenance" in item:
        v = item.get("provenance")
        if isinstance(v, list):
            v = v[0] if len(v) > 0 else None
        elif isinstance(v, dict):
            pass
        elif isinstance(v, str):
            v = {"note": v}
        else:
            v = None
        item["provenance"] = v
    return item


def _normalize_provenance_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Outcomes arrays: normalize each item's provenance to be an object or null
    outcomes = payload.get("outcomes")
    if isinstance(outcomes, dict):
        for k, arr in outcomes.items():
            if isinstance(arr, list):
                for i, itm in enumerate(arr):
                    if isinstance(itm, dict):
                        arr[i] = _normalize_item_provenance(itm)
    # Qual codes
    qt = payload.get("qualitative_thematic")
    if isinstance(qt, dict):
        codes = qt.get("codes")
        if isinstance(codes, list):
            for i, itm in enumerate(codes):
                if isinstance(itm, dict):
                    codes[i] = _normalize_item_provenance(itm)
    # Other arrays with potential per-item provenance
    for key in ["interventions", "stewardship_guidelines", "policy_capacity", "notes_flags"]:
        arr = payload.get(key)
        if isinstance(arr, list):
            for i, itm in enumerate(arr):
                if isinstance(itm, dict):
                    arr[i] = _normalize_item_provenance(itm)
    return payload


def _append_note(payload: Dict[str, Any], note_type: str, detail: str) -> None:
    arr = payload.get("notes_flags")
    if not isinstance(arr, list):
        arr = []
        payload["notes_flags"] = arr
    arr.append({"type": note_type, "detail": detail})


def _parse_percent(val: Any) -> Optional[float]:
    try:
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            s = val.strip().replace("%", "")
            return float(s)
    except Exception:
        return None
    return None


def _normalize_amr_counts(payload: Dict[str, Any]) -> Dict[str, Any]:
    outcomes = payload.get("outcomes")
    if not isinstance(outcomes, dict):
        return payload
    amr = outcomes.get("amr_proportions")
    if not isinstance(amr, list):
        return payload
    keep: list = []
    dropped = 0
    for row in amr:
        if not isinstance(row, dict):
            continue
        # Drop if essential identifiers are missing
        if not row.get("pathogen") or not row.get("antibiotic"):
            dropped += 1
            _append_note(payload, "dropped_amr_rows", "Dropped AMR row missing pathogen or antibiotic")
            continue
        n_tested = row.get("n_tested")
        n_res = row.get("n_resistant")
        n_int = row.get("n_intermediate")
        n_susc = row.get("n_susceptible")
        # Try to coerce to ints if numeric strings
        def to_int(x: Any) -> Optional[int]:
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
                    return None
            return None
        n_tested = to_int(n_tested)
        n_res = to_int(n_res)
        n_int = to_int(n_int)
        n_susc = to_int(n_susc)
        # Attempt to infer from percent fields when possible
        percent_keys = [
            "percent_resistant", "percent", "resistant_percent", "resistance_percent"
        ]
        perc: Optional[float] = None
        for k in percent_keys:
            if k in row:
                perc = _parse_percent(row.get(k))
                if perc is not None:
                    break
        # Try nested rate_as_reported object with value + unit (percent/per100)
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
                    # Convert per1000 to percent if denominator is known; otherwise skip
                    perc = float(rv) / 10.0
        # If n_tested missing but have n_res and percent
        if (n_tested is None or (isinstance(n_tested, int) and n_tested <= 0)) and isinstance(n_res, int) and perc is not None and perc > 0:
            denom = max(n_res, int(round(100.0 * n_res / max(1e-9, perc))))
            if denom >= n_res:
                n_tested = denom
                _append_note(payload, "derived_amr_counts", f"Computed n_tested from percent ({perc}%) and n_resistant={n_res} for {row.get('pathogen')} / {row.get('antibiotic')}")
        # If n_res missing but have n_tested and percent
        if (n_res is None or (isinstance(n_res, int) and n_res < 0)) and isinstance(n_tested, int) and n_tested > 0 and perc is not None and perc >= 0:
            events = int(round(perc / 100.0 * n_tested))
            events = min(max(0, events), n_tested)
            n_res = events
            _append_note(payload, "derived_amr_counts", f"Computed n_resistant from percent ({perc}%) and n_tested={n_tested} for {row.get('pathogen')} / {row.get('antibiotic')}")
        # If no percent, derive n_tested from components
        if (n_tested is None or (isinstance(n_tested, int) and n_tested <= 0)):
            parts = [x for x in [n_res, n_int, n_susc] if isinstance(x, int) and x >= 0]
            if parts:
                n_tested = sum(parts)
                _append_note(payload, "derived_amr_counts", f"Computed n_tested as sum of components for {row.get('pathogen')} / {row.get('antibiotic')}")
        # Clamp logic
        if isinstance(n_tested, int) and isinstance(n_res, int):
            if n_tested < 1:
                n_tested = None
            else:
                if n_res < 0:
                    n_res = 0
                if n_res > n_tested:
                    n_res = n_tested
        # Assign back
        if isinstance(n_tested, int):
            row["n_tested"] = n_tested
        if isinstance(n_res, int):
            row["n_resistant"] = n_res
        # Keep only valid rows with required counts
        if isinstance(row.get("n_tested"), int) and row["n_tested"] >= 1 and isinstance(row.get("n_resistant"), int) and row["n_resistant"] >= 0:
            keep.append(row)
        else:
            dropped += 1
    if dropped:
        _append_note(payload, "dropped_amr_rows", f"Dropped {dropped} AMR rows missing required counts (n_tested and/or n_resistant)")
    outcomes["amr_proportions"] = keep
    return payload


def _normalize_ssi_counts(payload: Dict[str, Any]) -> Dict[str, Any]:
    outcomes = payload.get("outcomes")
    if not isinstance(outcomes, dict):
        return payload
    ssi = outcomes.get("ssi_incidence")
    if not isinstance(ssi, list):
        return payload
    allowed_types = {"overall", "superficial", "deep", "organ_space"}
    keep: list = []
    dropped = 0
    for row in ssi:
        if not isinstance(row, dict):
            continue
        events = row.get("n_ssi_events")
        total = row.get("n_total")
        ssi_type = row.get("ssi_type")
        # Coerce ints
        try:
            if isinstance(events, str) and events.isdigit():
                events = int(events)
        except Exception:
            pass
        try:
            if isinstance(total, str) and total.isdigit():
                total = int(total)
        except Exception:
            pass
        # Normalize ssi_type
        if ssi_type not in allowed_types and ssi_type is not None:
            ssi_type = None
            row["ssi_type"] = None
        # Try to use rate_as_reported percent when provided
        rar = row.get("rate_as_reported")
        perc = None
        if isinstance(rar, dict):
            rv = rar.get("value")
            ru = (rar.get("unit") or "").lower()
            if isinstance(rv, (int, float)):
                if ru in ("percent", "%"):
                    perc = float(rv)
                elif ru in ("per100", "per 100") and isinstance(total, int) and total > 0:
                    perc = float(rv)
                elif ru in ("per1000", "per 1000") and isinstance(total, int) and total > 0:
                    perc = float(rv) / 10.0
        # Derive missing counts
        if (events is None or (isinstance(events, int) and events < 0)) and isinstance(total, int) and total > 0 and perc is not None:
            ev = int(round(perc / 100.0 * total))
            ev = min(max(0, ev), total)
            events = ev
            _append_note(payload, "derived_ssi_counts", f"Computed n_ssi_events from percent ({perc}%) and n_total={total}")
        if (total is None or (isinstance(total, int) and total <= 0)) and isinstance(events, int) and events >= 0 and perc is not None and perc > 0:
            denom = max(events, int(round(100.0 * events / max(1e-9, perc))))
            total = denom
            _append_note(payload, "derived_ssi_counts", f"Computed n_total from percent ({perc}%) and n_ssi_events={events}")
        # Assign back
        if isinstance(events, int):
            row["n_ssi_events"] = events
        if isinstance(total, int):
            row["n_total"] = total
        # Keep only valid rows
        if isinstance(row.get("n_ssi_events"), int) and row["n_ssi_events"] >= 0 and isinstance(row.get("n_total"), int) and row["n_total"] >= 1 and row["n_ssi_events"] <= row["n_total"]:
            keep.append(row)
        else:
            dropped += 1
    if dropped:
        _append_note(payload, "dropped_ssi_rows", f"Dropped {dropped} SSI rows missing required counts (n_ssi_events and/or n_total)")
    outcomes["ssi_incidence"] = keep
    return payload


def _rename_design_rationale(payload: Dict[str, Any]) -> Dict[str, Any]:
    d = payload.get("design")
    if isinstance(d, dict) and "rationale" in d:
        # Map to duplicate_rationale to satisfy schema
        if "duplicate_rationale" not in d or not d.get("duplicate_rationale"):
            d["duplicate_rationale"] = d["rationale"]
        d.pop("rationale", None)
    return payload


def _compact_ocr(o: Any, remove_keys: Optional[set] = None) -> Any:

    """Remove heavy/non-essential keys from OCR JSON to reduce prompt tokens.
    Default removes keys like bbox, image, raw, tokens, figures.
    """
    if remove_keys is None:
        remove_keys = {"bbox", "image", "images", "figure", "figures", "raw", "tokens", "confidence"}
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            if k in remove_keys:
                continue
            out[k] = _compact_ocr(v, remove_keys)
        return out
    if isinstance(o, list):
        return [_compact_ocr(v, remove_keys) for v in o]
    return o


def _filter_ocr_text_only(o: Any) -> Any:
    """Keep only text-bearing fields (page, headings, text, table structures) for cleaner prompts."""
    allowed_keys = {
        "page", "pages", "section", "section_heading", "heading", "title",
        "text", "table", "table_id", "rows", "cells", "content", "paragraphs"
    }
    if isinstance(o, dict):
        # Special case: if dict has text, keep only text + some context keys
        if "text" in o:
            kept = {k: o[k] for k in ["text", "page", "section_heading", "table_id"] if k in o}
            return kept
        out = {}
        for k, v in o.items():
            if k in allowed_keys:
                out[k] = _filter_ocr_text_only(v)
        return out
    if isinstance(o, list):
        return [_filter_ocr_text_only(v) for v in o]
    return o

def build_messages(prompt_text: str, ocr_json: Any) -> list:
    # The provided prompt already contains ROLE/INPUT/OUTPUT instructions
    user_content = f"{prompt_text}\n\n---\nOCR_JSON:\n{json.dumps(ocr_json, ensure_ascii=False)}"
    messages = [
        {"role": "system", "content": "You are a meticulous, non-hallucinating data extraction agent for a medical systematic review."},
        {"role": "user", "content": user_content}
    ]
    return messages

@retry(wait=wait_exponential(multiplier=2, min=1, max=20), stop=stop_after_attempt(4))
def call_openai_chat(messages: list, model: str, temperature: float, max_tokens: int) -> str:
    # Try new client first
    if _HAS_OPENAI and OpenAIClientClass is not None:
        try:
            client = OpenAIClientClass()
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}  # ask the model to return JSON
            )
            content = resp.choices[0].message.content
            if content is None:
                raise RuntimeError("OpenAI returned empty content")
            return content
        except Exception as e:
            raise RuntimeError(f"OpenAI call failed: {str(e)}")
    else:
        raise RuntimeError("OpenAI SDK not installed. Install `openai` and set OPENAI_API_KEY, or use --dry-run.")


@retry(wait=wait_exponential(multiplier=2, min=1, max=20), stop=stop_after_attempt(4))
def call_azure_chat(messages: list, endpoint: str, api_key: str, deployment: str, api_version: str, temperature: float, max_tokens: int) -> str:
    if not (_HAS_AZURE and AzureClient is not None):
        raise RuntimeError("Azure OpenAI SDK not available. Install `openai`>=1.0 and use Azure settings, or use --dry-run.")
    try:
        client = AzureClient(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)
        resp = client.chat.completions.create(
            model=deployment,  # Azure uses deployment name here
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        content = resp.choices[0].message.content
        if content is None:
            raise RuntimeError("Azure OpenAI returned empty content")
        return content
    except Exception as e:
        raise RuntimeError(f"Azure OpenAI call failed: {str(e)}")

def main():
    ap = argparse.ArgumentParser(description="Batch extract structured data from OCR JSONs using an LLM, with JSON Schema validation.")
    ap.add_argument("--input-dir", required=True, help="Directory with OCR JSON files (one article per *.json)")
    ap.add_argument("--prompt", required=True, help="Path to prompts/ocr_to_dataset_prompt.txt")
    ap.add_argument("--out-dir", required=True, help="Directory to write extracted JSONs")
    ap.add_argument("--quant-schema", required=True, help="Path to data_schemas/quantitative_schema.json")
    ap.add_argument("--qual-schema", required=True, help="Path to data_schemas/qualitative_schema.json")
    ap.add_argument("--model", default="gpt-4o-mini", help="LLM model name (OpenAI example); ignored for Azure where deployment is used")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--dry-run", action="store_true", help="Do not call LLM; write composed prompts to .prompt.txt files")
    # Performance knobs
    ap.add_argument("--compact-ocr", action="store_true", help="Compact OCR JSON by dropping heavy keys (bbox, images, raw, tokens) to reduce tokens")
    ap.add_argument("--max-prompt-chars", type=int, default=None, help="Hard cap on OCR JSON string length included in prompt (last-resort truncation)")
    ap.add_argument("--filter-ocr-text-only", action="store_true", help="Keep only text-bearing fields (text, headings, tables) to clean prompt while preserving evidence")
    # Azure OpenAI options (via CLI or env). If provided, Azure will be used.
    ap.add_argument("--use-azure", action="store_true", help="Use Azure OpenAI (alternatively auto-enabled if Azure env vars are set)")
    ap.add_argument("--azure-endpoint", default=None, help="Azure OpenAI endpoint, e.g., https://<resource>.openai.azure.com")
    ap.add_argument("--azure-api-key", default=None, help="Azure OpenAI API key")
    ap.add_argument("--azure-deployment", default=None, help="Azure OpenAI deployment name (e.g., gpt-4o-mini) for chat.completions")
    ap.add_argument("--azure-api-version", default=None, help="Azure OpenAI API version (default 2024-02-15-preview)")
    ap.add_argument("--disable-azure", action="store_true", help="Ignore Azure settings from CLI/env/.env and do not use Azure OpenAI")
    # Concurrency
    ap.add_argument("--workers", type=int, default=1, help="Number of parallel workers (use small number to avoid rate limits)")
    args = ap.parse_args()

    # Load .env if available (supports lines like KEY=VALUE or export KEY=VALUE)
    project_root = pathlib.Path(__file__).resolve().parents[1]
    env_path = project_root / ".env"
    if env_path.exists():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.lower().startswith("export "):
                        line = line[len("export "):].strip()
                    if "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and v and k not in os.environ:
                        os.environ[k] = v
        except Exception:
            pass

    prompt_text = load_text(args.prompt)
    q_schema = load_json(args.quant_schema)
    l_schema = load_json(args.qual_schema)

    in_dir = pathlib.Path(args.input_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in in_dir.glob("*.json")])
    if not files:
        print(f"No OCR JSON files found in {in_dir}", file=sys.stderr)
        sys.exit(1)

    # Resolve Azure config (env fallbacks)
    use_azure = bool(args.use_azure)
    azure_endpoint = args.azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
    azure_api_key = args.azure_api_key or os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_KEY")
    azure_deployment = args.azure_deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT") or os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
    azure_api_version = args.azure_api_version or os.environ.get("AZURE_OPENAI_API_VERSION") or "2024-02-15-preview"

    # Optionally disable Azure entirely (even if .env/env provide values)
    if args.disable_azure:
        use_azure = False
        azure_endpoint = None
        azure_api_key = None
        azure_deployment = None
    # Auto-enable Azure if all key settings are present from env/CLI and not explicitly disabled
    if not args.disable_azure and not use_azure and azure_endpoint and azure_api_key and azure_deployment:
        use_azure = True

    def _build_messages_from_ocr(ocr_obj: Any) -> list:
        if args.max_prompt_chars is not None and args.max_prompt_chars > 0:
            ocr_str = json.dumps(ocr_obj, ensure_ascii=False)
            if len(ocr_str) > args.max_prompt_chars:
                ocr_str = ocr_str[: args.max_prompt_chars] + "\n... [TRUNCATED]"
            user_content = f"{prompt_text}\n\n---\nOCR_JSON:\n{ocr_str}"
            return [
                {"role": "system", "content": "You are a meticulous, non-hallucinating data extraction agent for a medical systematic review."},
                {"role": "user", "content": user_content}
            ]
        else:
            return build_messages(prompt_text, ocr_obj)

    def process_one(p: pathlib.Path) -> str:
        """Process a single OCR json file and return a status string."""
        ocr = load_json(str(p))
        if args.filter_ocr_text_only:
            ocr = _filter_ocr_text_only(ocr)
        if args.compact_ocr:
            ocr = _compact_ocr(ocr)
        messages = _build_messages_from_ocr(ocr)

        out_base = out_dir / (p.stem + ".extracted.json")
        if args.dry_run:
            prompt_out = out_dir / (p.stem + ".prompt.txt")
            with open(prompt_out, "w", encoding="utf-8") as f:
                for m in messages:
                    role = (m.get("role") or "user").upper()
                    f.write(f"[{role}]\n")
                    f.write(m.get("content") or "")
                    f.write("\n\n")
            return f"DRY {p.name}"

        try:
            if use_azure:
                if not (azure_endpoint and azure_api_key and azure_deployment):
                    raise RuntimeError("Missing Azure settings: --azure-endpoint, --azure-api-key, and --azure-deployment are required (or set env AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT)")
                content = call_azure_chat(messages, azure_endpoint, azure_api_key, azure_deployment, azure_api_version, args.temperature, args.max_tokens)
            else:
                content = call_openai_chat(messages, args.model, args.temperature, args.max_tokens)
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                content_stripped = content.strip().strip("`")
                data = json.loads(content_stripped)
        except Exception as e:
            return f"ERR {p.name}: {e}"

        data = _prune_nulls(data)
        # Flatten single-item arrays for scalar enums (ownership_type, facility_level, etc.)
        if isinstance(data, dict):
            data = _sanitize_scalar_enums(data)
            data = _strip_disallowed_provenance(data)
            data = _normalize_provenance_fields(data)
            data = _normalize_amr_counts(data)
            data = _normalize_ssi_counts(data)
            data = _rename_design_rationale(data)
            data = _ensure_provenance_index(data)
        err_q = validate_json_schema(data, q_schema, p.name)
        err_l = validate_json_schema({"qualitative_thematic": data.get("qualitative_thematic", {"codes": []})}, l_schema, p.name)
        if err_q:
            print(err_q, file=sys.stderr)
        if err_l:
            print(err_l, file=sys.stderr)

        save_json(data, str(out_base))
        return f"OK  {p.name}"

    # Run sequentially or in parallel based on workers
    n_workers = max(1, int(args.workers))
    if n_workers == 1:
        for p in tqdm(files, desc="Extracting"):
            _ = process_one(p)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            with tqdm(total=len(files), desc="Extracting") as bar:
                futures = [pool.submit(process_one, p) for p in files]
                for fut in as_completed(futures):
                    _ = fut.result()
                    bar.update(1)

    print("Extraction complete.")
    if args.dry_run:
        print("Dry-run created *.prompt.txt files. Paste into your LLM UI, save returned JSON as *.extracted.json, then run aggregation.")

if __name__ == "__main__":
    main()