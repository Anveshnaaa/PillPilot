from __future__ import annotations

import datetime as dt
import json
import math
import re
import ssl
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from decision_engine import find_best_transfer_or_reorder
from ml_pipeline import forecast_demand_all_groups, predict_status, validate_input_schema
from rag_pipeline import (
    BasicLocalRetriever,
    answer_with_evidence,
    build_local_corpus,
    collect_store_runtime_facts,
    detect_intent,
)


PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_DIR = PROJECT_ROOT / "static"
MODEL_DIR = PROJECT_ROOT / "models"
FORECAST_COLUMNS = ["store_name", "medicine_name", "forecast_day", "date", "predicted_demand"]

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")


def _is_critical(status: object) -> bool:
    cleaned = str(status).strip().lower()
    return "critical" in cleaned or cleaned in {"low", "very low", "critically low"}


def _format_currency(value: object) -> str:
    if pd.isna(value):
        return "-"
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "-"


def _format_recommendation_row(rec: dict) -> dict:
    decision = rec.get("final_decision", "No Action")
    if decision == "Transfer Stock":
        details = (
            f"Move {int(rec.get('transfer_units', 0))} units from "
            f"{rec.get('from_store', '-')} to {rec.get('to_store', '-')}."
        )
    elif decision == "Reorder Stock":
        details = f"Reorder {int(rec.get('needed_units', 0))} units for this location."
    else:
        details = rec.get("reason", "No action required.")

    return {
        "store_name": rec.get("store_name", "-"),
        "medicine_name": rec.get("medicine_name", "-"),
        "final_decision": decision,
        "needed_units": int(rec.get("needed_units", 0)) if pd.notna(rec.get("needed_units")) else 0,
        "reorder_cost": _format_currency(rec.get("reorder_cost")),
        "transfer_cost": _format_currency(rec.get("transfer_cost")),
        "estimated_savings": _format_currency(rec.get("savings")),
        "action_details": details,
        "reason": rec.get("reason", "-"),
    }


def latest_inventory_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    snapshot = df.copy()
    snapshot["date"] = pd.to_datetime(snapshot["date"])
    snapshot = snapshot.sort_values(["store_name", "medicine_name", "date"])
    snapshot = snapshot.groupby(["store_name", "medicine_name"], as_index=False).tail(1)
    snapshot["date"] = snapshot["date"].dt.date.astype(str)
    return snapshot.sort_values(["store_name", "medicine_name"]).reset_index(drop=True)


def ensure_forecast_schema(forecast_df: pd.DataFrame) -> pd.DataFrame:
    if forecast_df is None or forecast_df.empty:
        return pd.DataFrame(columns=FORECAST_COLUMNS)
    out = forecast_df.copy()
    for col in FORECAST_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[FORECAST_COLUMNS]


def build_recommendations(scored_df: pd.DataFrame) -> pd.DataFrame:
    latest_rows = latest_inventory_snapshot(scored_df)
    critical_rows = latest_rows[latest_rows["predicted_status"].map(_is_critical)]
    if critical_rows.empty:
        return pd.DataFrame(
            columns=[
                "store_name",
                "medicine_name",
                "final_decision",
                "needed_units",
                "reorder_cost",
                "transfer_cost",
                "estimated_savings",
                "action_details",
                "reason",
            ]
        )

    recommendations = []
    for _, row in critical_rows.iterrows():
        raw = find_best_transfer_or_reorder(
            df=scored_df,
            target_store=row["store_name"],
            medicine_name=row["medicine_name"],
        )
        raw.setdefault("store_name", row["store_name"])
        raw.setdefault("medicine_name", row["medicine_name"])
        recommendations.append(_format_recommendation_row(raw))
    return pd.DataFrame(recommendations)


def _load_artifacts():
    # Load pre-trained artifacts once at startup (train-once, infer-many pattern).
    required = {
        "status_model": MODEL_DIR / "status_model.pkl",
        "status_metadata": MODEL_DIR / "status_metadata.pkl",
        "demand_model": MODEL_DIR / "demand_model.pkl",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts. Run `./venv/bin/python train_and_save_models.py` first. "
            f"Missing: {', '.join(missing)}"
        )
    return {
        "status_model": joblib.load(required["status_model"]),
        "status_metadata": joblib.load(required["status_metadata"]),
        "demand_model": joblib.load(required["demand_model"]),
    }


ARTIFACTS = _load_artifacts()
# RAG retriever is initialized once and reused across requests.
RETRIEVER = BasicLocalRetriever(build_local_corpus(PROJECT_ROOT))
ANALYSIS_CACHE: dict[str, dict] = {}
MED_FACTS_PATH = PROJECT_ROOT / "medicine_facts.md"
OPEN_FDA_BASE_URL = "https://api.fda.gov"
OPEN_FDA_HEADERS = {"User-Agent": "PillPilot/1.0 (pharmacy inventory app)"}
RXNORM_BASE_URL = "https://rxnav.nlm.nih.gov/REST"
FALLBACK_CLASS_RISK = {
    "blood thinner": 0.9,
    "antibiotic": 0.7,
    "insulin": 0.8,
    "generic": 0.6,
    "otc": 0.3,
}
DRUG_CLASS_KEYWORDS = {
    "blood thinner": ["warfarin", "heparin", "apixaban", "rivaroxaban", "dabigatran"],
    "antibiotic": ["amoxicillin", "azithromycin", "doxycycline", "ciprofloxacin", "clindamycin"],
    "insulin": ["insulin", "glargine", "lispro", "aspart", "detemir"],
    "generic": ["generic", "hcl", "acetaminophen", "ibuprofen", "metformin"],
    "otc": ["aspirin", "paracetamol", "acetaminophen", "cough", "cold", "allergy"],
}
INTERACTION_FALLBACK_RULES = {
    frozenset({"warfarin", "ibuprofen"}): {
        "severity": "high",
        "description": "Ibuprofen may increase warfarin-associated bleeding risk.",
    },
    frozenset({"warfarin", "naproxen"}): {
        "severity": "high",
        "description": "This combination can significantly increase bleeding risk.",
    },
    frozenset({"sertraline", "ibuprofen"}): {
        "severity": "moderate",
        "description": "Combined use may increase GI bleeding risk.",
    },
    frozenset({"losartan", "ibuprofen"}): {
        "severity": "moderate",
        "description": "NSAIDs may reduce antihypertensive efficacy and affect renal function.",
    },
}
LIVE_FDA_CACHE: dict[str, object] = {
    "class_risk_map": None,
    "class_risk_updated_at": None,
}
COMMON_DRUG_HINTS = [
    "warfarin",
    "ibuprofen",
    "amoxicillin",
    "cetirizine",
    "semaglutide",
    "metformin",
    "insulin",
    "aspirin",
]
CHAT_INTENT_TRAINING = [
    ("is this medicine recalled", "recall"),
    ("check fda recall for ibuprofen", "recall"),
    ("is there any active recall for semaglutide", "recall"),
    ("customer takes warfarin and wants ibuprofen", "interaction"),
    ("check interaction between amoxicillin and warfarin", "interaction"),
    ("can these two drugs be taken together", "interaction"),
    ("do we have amoxicillin in stock", "inventory"),
    ("how many units of cetirizine are available", "inventory"),
    ("inventory check for metformin", "inventory"),
    ("customer wants ibuprofen and takes warfarin and check recall", "mixed"),
    ("is ibuprofen recalled and does it interact with warfarin", "mixed"),
    ("check inventory and recall risk for semaglutide", "mixed"),
]


def _build_chat_intent_model():
    texts = [item[0] for item in CHAT_INTENT_TRAINING]
    labels = [item[1] for item in CHAT_INTENT_TRAINING]
    model = make_pipeline(
        TfidfVectorizer(lowercase=True, ngram_range=(1, 2)),
        LogisticRegression(max_iter=1000),
    )
    model.fit(texts, labels)
    return model


CHAT_INTENT_MODEL = _build_chat_intent_model()


def _load_medicine_facts_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    chunks = text.split("\n## ")
    facts: dict[str, str] = {}
    for idx, chunk in enumerate(chunks):
        if idx == 0:
            continue
        lines = chunk.strip().splitlines()
        if not lines:
            continue
        med = lines[0].strip()
        body = " ".join(line.strip("- ").strip() for line in lines[1:] if line.strip())
        facts[med] = body
    return facts


MED_FACTS = _load_medicine_facts_map(MED_FACTS_PATH)


def _build_ssl_context():
    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        # Development fallback when local cert chain is incomplete.
        return ssl._create_unverified_context()


def _openfda_get(path: str, params: dict | None = None, retries: int = 1) -> dict:
    params = params or {}
    query = urlencode({k: v for k, v in params.items() if v is not None and v != ""})
    url = f"{OPEN_FDA_BASE_URL}{path}"
    if query:
        url = f"{url}?{query}"
    req = Request(url, headers=OPEN_FDA_HEADERS)
    ssl_context = _build_ssl_context()

    last_err = None
    for _ in range(retries + 1):
        try:
            with urlopen(req, timeout=12, context=ssl_context) as resp:
                payload = resp.read().decode("utf-8", errors="ignore")
                return json.loads(payload)
        except HTTPError as exc:
            last_err = exc
            if exc.code == 429:
                continue
            raise
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_err = exc
            continue
    raise RuntimeError(f"openFDA request failed: {last_err}")


def _http_get_json(url: str, headers: dict | None = None, retries: int = 1) -> dict:
    headers = headers or OPEN_FDA_HEADERS
    req = Request(url, headers=headers)
    ssl_context = _build_ssl_context()
    last_err = None
    for _ in range(retries + 1):
        try:
            with urlopen(req, timeout=12, context=ssl_context) as resp:
                payload = resp.read().decode("utf-8", errors="ignore")
                return json.loads(payload)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_err = exc
            continue
    raise RuntimeError(f"HTTP request failed: {last_err}")


def _rxnorm_rxcui(drug_name: str) -> str | None:
    term = str(drug_name or "").strip()
    if not term:
        return None
    query = urlencode({"name": term, "search": 2})
    url = f"{RXNORM_BASE_URL}/rxcui.json?{query}"
    data = _http_get_json(url, retries=1)
    ids = data.get("idGroup", {}).get("rxnormId") or []
    return str(ids[0]) if ids else None


def _rxnorm_interaction_payload(rxcuis: list[str]) -> dict:
    list_value = "+".join([item for item in rxcuis if item])
    query = urlencode({"rxcuis": list_value})
    url = f"{RXNORM_BASE_URL}/interaction/list.json?{query}"
    return _http_get_json(url, retries=1)


def _rxnorm_parse_interaction(payload: dict) -> dict:
    groups = payload.get("fullInteractionTypeGroup") or []
    for group in groups:
        for interaction_type in group.get("fullInteractionType") or []:
            pair = (interaction_type.get("interactionPair") or [None])[0]
            if not pair:
                continue
            description = str(pair.get("description") or "Interaction information available.")
            severity_raw = str(pair.get("severity") or description).lower()
            if any(token in severity_raw for token in ["high", "major", "contraindicated", "serious"]):
                severity = "high"
            elif any(token in severity_raw for token in ["moderate", "significant"]):
                severity = "moderate"
            else:
                severity = "low"
            return {
                "interaction_found": True,
                "severity": severity,
                "description": description,
                "source_name": group.get("sourceName") or "ONCHigh",
            }
    return {
        "interaction_found": False,
        "severity": "none",
        "description": "No major interaction found in RxNorm response.",
        "source_name": "ONCHigh",
    }


def _fallback_interaction_by_names(drug1: str, drug2: str, drug3: str = "") -> dict:
    names = [
        _normalize_drug_name(drug1).split()[0] if _normalize_drug_name(drug1) else "",
        _normalize_drug_name(drug2).split()[0] if _normalize_drug_name(drug2) else "",
        _normalize_drug_name(drug3).split()[0] if _normalize_drug_name(drug3) else "",
    ]
    names = [name for name in names if name]
    pairs = []
    if len(names) >= 2:
        pairs.append(frozenset({names[0], names[1]}))
    if len(names) == 3:
        pairs.append(frozenset({names[0], names[2]}))
        pairs.append(frozenset({names[1], names[2]}))

    for pair in pairs:
        if pair in INTERACTION_FALLBACK_RULES:
            rule = INTERACTION_FALLBACK_RULES[pair]
            return {
                "interaction_found": True,
                "severity": rule["severity"],
                "description": rule["description"],
                "source_name": "ONCHigh (fallback pair rules)",
            }
    return {
        "interaction_found": False,
        "severity": "none",
        "description": "No major interaction found in available interaction datasets.",
        "source_name": "ONCHigh (fallback pair rules)",
    }


def _normalize_drug_name(raw_name: object) -> str:
    if raw_name is None:
        return ""
    text = str(raw_name).lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    noise_tokens = {
        "tablet",
        "tablets",
        "tab",
        "tabs",
        "capsule",
        "capsules",
        "cap",
        "caps",
        "usp",
        "inj",
        "injection",
        "solution",
        "suspension",
        "oral",
        "iv",
        "ml",
    }
    text = " ".join(token for token in text.split() if token not in noise_tokens)
    return re.sub(r"\b(\d+)\s+(mg|mcg|g)\b", r"\1\2", text).strip()


def _predict_chat_intent(question: str) -> tuple[str, float]:
    q = str(question or "").strip().lower()
    if not q:
        return "inventory", 0.25

    has_recall = any(token in q for token in ["recall", "fda", "recalled", "quarantine", "safety"])
    has_interaction = any(token in q for token in ["interaction", "takes", "with", "together", "combine"])
    has_inventory = any(token in q for token in ["stock", "inventory", "available", "units", "reorder"])
    if sum([has_recall, has_interaction, has_inventory]) >= 2:
        return "mixed", 0.92

    pred = str(CHAT_INTENT_MODEL.predict([q])[0])
    proba = float(max(CHAT_INTENT_MODEL.predict_proba([q])[0]))
    return pred, proba


def _extract_drug_candidates(question: str, known_medicines: list[str]) -> list[str]:
    q_norm = _normalize_drug_name(question)
    candidates: list[str] = []

    for med in known_medicines:
        med_name = str(med or "").strip()
        med_norm = _normalize_drug_name(med_name)
        if not med_norm:
            continue
        if med_norm in q_norm:
            candidates.append(med_name)
            continue
        first = med_norm.split()[0]
        if len(first) >= 4 and re.search(rf"\b{re.escape(first)}\b", q_norm):
            candidates.append(first)

    for hint in COMMON_DRUG_HINTS:
        if re.search(rf"\b{re.escape(hint)}\b", q_norm):
            candidates.append(hint)

    out: list[str] = []
    seen = set()
    for item in candidates:
        key = _normalize_drug_name(item)
        if key and key not in seen:
            out.append(item)
            seen.add(key)
    return out[:4]


def _inventory_tool_result(drug_name: str, latest_df: pd.DataFrame, forecast_df: pd.DataFrame) -> dict:
    query = _normalize_drug_name(drug_name)
    if not query:
        return {"ok": False, "found": False, "error": "No drug name provided."}

    latest_rows = latest_df.copy()
    latest_rows["__norm"] = latest_rows["medicine_name"].astype(str).map(_normalize_drug_name)
    latest_match = latest_rows[latest_rows["__norm"].map(lambda value: query in value or value in query)]
    latest_pick = latest_match.iloc[0].to_dict() if not latest_match.empty else None

    if not latest_pick:
        candidates = latest_rows[latest_rows["__norm"].map(lambda value: query in value or value in query)]
        latest_pick = candidates.iloc[0].to_dict() if not candidates.empty else None

    if not latest_pick:
        return {"ok": True, "found": False, "drug_name": drug_name}

    med_name = str(latest_pick.get("medicine_name", drug_name))
    store_name = str(latest_pick.get("store_name", ""))
    stock = float(latest_pick.get("current_stock") or 0)
    daily = float(latest_pick.get("daily_demand") or 0)
    demand3 = max(0.0, daily * 3)
    days = 999 if daily <= 0 else int(max(0, stock // daily))
    reorder_qty = max(0, int(math.ceil(demand3 - stock)))

    return {
        "ok": True,
        "found": True,
        "drug_name": med_name,
        "store_name": store_name,
        "current_stock": int(round(stock)),
        "combined_3_day_demand": round(demand3, 2),
        "days_until_stockout": days,
        "reorder_quantity": reorder_qty,
        "quality": 0.9,
        "match_confidence": 0.88,
    }


def _recall_tool_result(drug_name: str) -> dict:
    live_inventory = _to_live_inventory_rows([{"medicine_name": drug_name, "current_stock": 0}])
    if not live_inventory:
        return {"ok": False, "error": "No valid medicine provided."}
    recalls = _fetch_active_recalls_live()
    item = live_inventory[0]
    matches = _match_recall_to_inventory_item(item, recalls)
    best_match = max(matches, key=lambda x: float(x.get("confidence", 0)), default=None)
    if not best_match:
        return {
            "ok": True,
            "recalled": False,
            "drug_name": drug_name,
            "finding": "No direct recall match found.",
            "quality": 0.72,
            "match_confidence": 0.5,
        }
    rec = best_match.get("recall_item", {})
    confidence = float(best_match.get("confidence", 0))
    return {
        "ok": True,
        "recalled": True,
        "drug_name": drug_name,
        "finding": "High-confidence FDA recall match found." if confidence >= 0.85 else "Possible FDA recall match found.",
        "recall_number": rec.get("recall_number"),
        "recall_class": rec.get("classification"),
        "recall_reason": rec.get("reason_for_recall"),
        "affected_lots": best_match.get("lot_numbers_matched") or _parse_lot_numbers(rec.get("code_info")),
        "confidence": round(confidence, 4),
        "quality": 0.9 if confidence >= 0.85 else 0.78,
        "match_confidence": confidence,
    }


def _interaction_tool_result(drug1: str, drug2: str) -> dict:
    rxcui1 = _rxnorm_rxcui(drug1)
    rxcui2 = _rxnorm_rxcui(drug2)
    if not rxcui1 or not rxcui2:
        return {
            "ok": True,
            "interaction_found": False,
            "severity": "none",
            "description": "Could not resolve one or more drugs in RxNorm.",
            "quality": 0.55,
            "match_confidence": 0.45,
        }
    parsed = None
    try:
        payload = _rxnorm_interaction_payload([rxcui1, rxcui2])
        parsed = _rxnorm_parse_interaction(payload)
    except Exception:
        parsed = None
    if not parsed:
        parsed = _fallback_interaction_by_names(drug1, drug2)

    severity = str(parsed.get("severity", "none")).lower()
    if severity in {"high", "moderate"}:
        quality = 0.9
        match_conf = 0.86 if severity == "high" else 0.75
    else:
        quality = 0.72
        match_conf = 0.55
    return {
        "ok": True,
        "interaction_found": bool(parsed.get("interaction_found")),
        "severity": severity,
        "description": parsed.get("description", "No major interaction found."),
        "quality": quality,
        "match_confidence": match_conf,
    }


def _confidence_band(score: float) -> str:
    if score >= 0.75:
        return "High"
    if score >= 0.5:
        return "Medium"
    return "Low"


def _normalize_ndc(raw: object) -> str:
    return re.sub(r"[^0-9]", "", str(raw or ""))


def _parse_lot_numbers(code_info_text: object) -> list[str]:
    text = str(code_info_text or "")
    patterns = [
        r"\b(?:lot|lot#|lot:)\s*([a-z0-9-]{3,})\b",
        r"\b(?:batch|batch#|batch:)\s*([a-z0-9-]{3,})\b",
        r"\b([a-z]{1,4}\d{3,}[a-z0-9-]*)\b",
    ]
    hits = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = str(match.group(1)).strip().upper()
            if len(value) >= 3:
                hits.add(value)
    return sorted(hits)


def _tokenize(text: object) -> list[str]:
    return [token for token in _normalize_drug_name(text).split() if token]


def _build_tfidf(documents: list[str]) -> list[dict[str, float]]:
    tokenized = [_tokenize(doc) for doc in documents]
    term_counts = []
    for tokens in tokenized:
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        term_counts.append(counts)

    df: dict[str, int] = {}
    for counts in term_counts:
        for term in counts.keys():
            df[term] = df.get(term, 0) + 1

    num_docs = len(documents) or 1
    vectors: list[dict[str, float]] = []
    for counts in term_counts:
        total_terms = sum(counts.values()) or 1
        vector: dict[str, float] = {}
        for term, count in counts.items():
            tf = count / total_terms
            idf = math.log((num_docs + 1) / (df.get(term, 0) + 1)) + 1
            vector[term] = tf * idf
        vectors.append(vector)
    return vectors


def _cosine_similarity(vector_a: dict[str, float], vector_b: dict[str, float]) -> float:
    terms = set(vector_a.keys()) | set(vector_b.keys())
    dot = sum((vector_a.get(term, 0.0) * vector_b.get(term, 0.0)) for term in terms)
    mag_a = math.sqrt(sum((vector_a.get(term, 0.0) ** 2) for term in terms))
    mag_b = math.sqrt(sum((vector_b.get(term, 0.0) ** 2) for term in terms))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _levenshtein_distance(a: object, b: object) -> int:
    left = _normalize_drug_name(a)
    right = _normalize_drug_name(b)
    rows = len(left) + 1
    cols = len(right) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if left[i - 1] == right[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[-1][-1]


def _levenshtein_score(a: object, b: object) -> float:
    left = _normalize_drug_name(a)
    right = _normalize_drug_name(b)
    max_len = max(len(left), len(right), 1)
    dist = _levenshtein_distance(left, right)
    return max(0.0, 1.0 - (dist / max_len))


def _fetch_active_recalls_live() -> list[dict]:
    ongoing = _openfda_get(
        "/drug/enforcement.json",
        {"search": "status:Ongoing", "limit": 50},
        retries=1,
    ).get("results", [])
    class_one = _openfda_get(
        "/drug/enforcement.json",
        {"search": 'classification:"Class I"', "limit": 50},
        retries=1,
    ).get("results", [])
    merged = ongoing + class_one
    by_key: dict[str, dict] = {}
    for rec in merged:
        key = rec.get("recall_number") or rec.get("event_id") or f"{rec.get('recalling_firm','')}-{rec.get('product_description','')}"
        if key not in by_key:
            by_key[key] = {
                "recall_number": rec.get("recall_number"),
                "product_description": rec.get("product_description", ""),
                "reason_for_recall": rec.get("reason_for_recall", ""),
                "classification": rec.get("classification", ""),
                "recalling_firm": rec.get("recalling_firm", ""),
                "recall_initiation_date": rec.get("recall_initiation_date", ""),
                "status": rec.get("status", ""),
                "product_ndc": [
                    _normalize_ndc(item)
                    for item in (
                        rec.get("product_ndc")
                        or rec.get("openfda", {}).get("product_ndc")
                        or rec.get("openfda", {}).get("package_ndc")
                        or []
                    )
                    if _normalize_ndc(item)
                ],
                "code_info": rec.get("code_info", ""),
            }
    return list(by_key.values())


def _fetch_shortages_live() -> list[dict]:
    try:
        results = _openfda_get("/drug/shortage.json", {"limit": 100}, retries=1).get("results", [])
        out = []
        for item in results:
            name = item.get("product_name") or item.get("drug_name") or item.get("name") or item.get("brand_name") or ""
            out.append({"product_name": name, "normalized_product_name": _normalize_drug_name(name)})
        return out
    except Exception:
        return []


def _manufacturer_recall_count_3y(manufacturer: str) -> int:
    if not manufacturer:
        return 0
    now = dt.datetime.utcnow()
    from_date = (now - dt.timedelta(days=365 * 3)).strftime("%Y%m%d")
    search = f'recalling_firm:"{manufacturer}"+AND+recall_initiation_date:[{from_date}+TO+*]'
    try:
        payload = _openfda_get("/drug/enforcement.json", {"search": search, "limit": 1}, retries=1)
        return int(payload.get("meta", {}).get("results", {}).get("total", 0))
    except Exception:
        return 0


def _drug_last_recall_days(drug_name: str) -> int | None:
    term = _normalize_drug_name(drug_name)
    if not term:
        return None
    search = f'product_description:"{term}"'
    try:
        results = _openfda_get("/drug/enforcement.json", {"search": search, "limit": 20}, retries=1).get("results", [])
    except Exception:
        return None
    dates = [str(item.get("recall_initiation_date", "")).strip() for item in results if item.get("recall_initiation_date")]
    if not dates:
        return None
    latest = sorted(dates, reverse=True)[0]
    try:
        dt_latest = dt.datetime.strptime(latest[:8], "%Y%m%d")
    except ValueError:
        return None
    return max(0, (dt.datetime.utcnow() - dt_latest).days)


def _time_on_market_score(drug_name: str) -> float:
    term = _normalize_drug_name(drug_name)
    if not term:
        return 0.2
    try:
        results = _openfda_get("/drug/ndc.json", {"search": f"brand_name:{term}", "limit": 5}, retries=1).get("results", [])
    except Exception:
        return 0.2
    if not results:
        return 0.2
    active = sum(1 for item in results if not item.get("listing_expiration_date")) or len(results)
    starts = []
    for item in results:
        value = str(item.get("marketing_start_date", "")).strip()
        if len(value) == 8 and value.isdigit():
            starts.append(f"{value[:4]}-{value[4:6]}-{value[6:8]}")
        elif value:
            starts.append(value)
    oldest = sorted(starts)[0] if starts else ""
    years_on_market = 0.0
    if oldest:
        try:
            years_on_market = max(0.0, (dt.datetime.utcnow() - dt.datetime.fromisoformat(oldest)).days / 365)
        except ValueError:
            years_on_market = 0.0
    years_score = min(1.0, years_on_market / 20)
    listing_score = min(1.0, active / 25)
    return round((0.5 * years_score) + (0.5 * listing_score), 4)


def _compute_dynamic_class_risk_map() -> dict[str, float]:
    if LIVE_FDA_CACHE.get("class_risk_map"):
        return LIVE_FDA_CACHE["class_risk_map"]  # type: ignore[return-value]

    now = dt.datetime.utcnow()
    from_date = (now - dt.timedelta(days=365 * 3)).strftime("%Y%m%d")
    counts: dict[str, int] = {}
    for drug_class, keywords in DRUG_CLASS_KEYWORDS.items():
        total = 0
        for keyword in keywords:
            try:
                search = f'product_description:"{keyword}"+AND+recall_initiation_date:[{from_date}+TO+*]'
                payload = _openfda_get("/drug/enforcement.json", {"search": search, "limit": 1}, retries=1)
                total += int(payload.get("meta", {}).get("results", {}).get("total", 0))
            except Exception:
                continue
        counts[drug_class] = total

    max_count = max([1] + list(counts.values()))
    if max_count <= 0:
        LIVE_FDA_CACHE["class_risk_map"] = FALLBACK_CLASS_RISK.copy()
    else:
        LIVE_FDA_CACHE["class_risk_map"] = {
            drug_class: round(min(1.0, count / max_count), 4) for drug_class, count in counts.items()
        }
    LIVE_FDA_CACHE["class_risk_updated_at"] = now.isoformat()
    return LIVE_FDA_CACHE["class_risk_map"]  # type: ignore[return-value]


def _pick_drug_class(drug_name: str) -> str:
    normalized = _normalize_drug_name(drug_name)
    for drug_class, keywords in DRUG_CLASS_KEYWORDS.items():
        for keyword in keywords:
            if keyword in normalized:
                return drug_class
    return "generic"


def _lot_intersection(inventory_lots: list[str], recall_lots: list[str]) -> list[str]:
    inv = {str(item).upper().strip() for item in inventory_lots}
    return [lot for lot in recall_lots if str(lot).upper().strip() in inv]


def _extract_strength(text: str) -> str | None:
    normalized = _normalize_drug_name(text)
    match = re.search(r"\b(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml)\b", normalized)
    if match:
        return f"{match.group(1)}{match.group(2)}"
    match = re.search(r"\b(\d+(?:\.\d+)?)(mg|mcg|g|ml)\b", normalized)
    if match:
        return f"{match.group(1)}{match.group(2)}"
    return None


def _match_recall_to_inventory_item(item: dict, recalls: list[dict]) -> list[dict]:
    matches = []
    inv_ndc = _normalize_ndc(item.get("ndc_code"))
    inv_name = str(item.get("drug_name", ""))
    inv_lots = [str(x).upper().strip() for x in (item.get("lot_numbers") or [])]
    inv_norm = _normalize_drug_name(inv_name)
    inv_tokens = [token for token in inv_norm.split() if token]
    inv_first = inv_tokens[0] if inv_tokens else ""
    inv_strength = _extract_strength(inv_name)

    for rec in recalls:
        recall_ndcs = [_normalize_ndc(x) for x in (rec.get("product_ndc") or [])]
        if inv_ndc and inv_ndc in recall_ndcs:
            recall_lots = _parse_lot_numbers(rec.get("code_info"))
            matches.append(
                {
                    "confidence": 1.0,
                    "confidence_label": "HIGH_MATCH",
                    "match_stage": "EXACT_NDC_MATCH",
                    "inventory_item": item,
                    "recall_item": rec,
                    "lot_numbers_matched": _lot_intersection(inv_lots, recall_lots),
                }
            )
            continue

        rec_name = str(rec.get("product_description", ""))
        rec_norm = _normalize_drug_name(rec_name)
        rec_tokens = [token for token in rec_norm.split() if token]
        rec_set = set(rec_tokens)
        rec_first = rec_tokens[0] if rec_tokens else ""
        rec_strength = _extract_strength(rec_name)

        [v1, v2] = _build_tfidf([inv_name, rec_name])
        cosine = _cosine_similarity(v1, v2)
        lev = _levenshtein_score(inv_name, rec_name)
        confidence = (0.7 * cosine) + (0.3 * lev)

        # Boost obvious ingredient/strength matches that TF-IDF + edit distance can under-score
        # when recall descriptions include long packaging/manufacturer text.
        ingredient_overlap = bool(set(inv_tokens) & rec_set)
        first_token_match = bool(inv_first and rec_first and inv_first == rec_first)
        strength_match = bool(inv_strength and rec_strength and inv_strength == rec_strength)
        if ingredient_overlap:
            confidence += 0.12
        if first_token_match:
            confidence += 0.18
        if strength_match:
            confidence += 0.08
        if first_token_match and strength_match:
            confidence = max(confidence, 0.86)

        confidence = min(1.0, confidence)
        if confidence < 0.65:
            continue
        label = "HIGH_MATCH" if confidence >= 0.85 else "POSSIBLE_MATCH"
        recall_lots = _parse_lot_numbers(rec.get("code_info"))
        matches.append(
            {
                "confidence": round(confidence, 4),
                "confidence_label": label,
                "match_stage": "NLP_FUZZY_NAME_MATCH",
                "inventory_item": item,
                "recall_item": rec,
                "lot_numbers_matched": _lot_intersection(inv_lots, recall_lots),
            }
        )
    return matches


def _match_explanation(match: dict) -> str:
    if match.get("match_stage") == "EXACT_NDC_MATCH":
        lots = match.get("lot_numbers_matched") or []
        suffix = f" Lot number {', '.join(lots)} confirmed in inventory." if lots else ""
        return f"Matched via exact NDC code (100% confidence).{suffix}"
    confidence_pct = int(round(float(match.get("confidence", 0)) * 100))
    ingredient = (_normalize_drug_name(match.get("inventory_item", {}).get("drug_name", "")).split() or ["active ingredient"])[0]
    lots = match.get("lot_numbers_matched") or []
    suffix = f" Lot number {', '.join(lots)} confirmed in inventory." if lots else ""
    return f"Matched via name similarity ({confidence_pct}% confidence). Active ingredient '{ingredient}' found in both.{suffix}"


def _to_live_inventory_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        lots = row.get("lot_numbers")
        if isinstance(lots, list):
            lot_numbers = [str(item).upper().strip() for item in lots if str(item).strip()]
        else:
            raw = str(row.get("lot_number") or row.get("batch") or row.get("code_info") or "")
            lot_numbers = [part.strip().upper() for part in re.split(r"[,;\s|/]+", raw) if len(part.strip()) >= 3]
        out.append(
            {
                "drug_name": row.get("medicine_name") or row.get("drug_name") or "",
                "ndc_code": row.get("ndc_code") or row.get("product_ndc"),
                "manufacturer": row.get("manufacturer") or row.get("supplier") or row.get("distributor") or "Unknown Manufacturer",
                "lot_numbers": lot_numbers,
                "current_stock": float(row.get("current_stock") or 0),
            }
        )
    return out


@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/api/health")
def health():
    return jsonify({"ok": True})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "CSV file is required."}), 400
    file = request.files["file"]
    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Please upload a CSV file."}), 400

    try:
        raw_df = pd.read_csv(file)
        # For website inference, target label is optional in uploaded file.
        validate_input_schema(raw_df, require_target=False)
    except Exception as exc:
        return jsonify({"error": f"Validation failed: {exc}"}), 400

    # Model 1 inference: predict inventory status for each uploaded row.
    scored_df = raw_df.copy()
    scored_df["predicted_status"] = predict_status(
        scored_df, ARTIFACTS["status_model"], ARTIFACTS["status_metadata"]
    )
    # Model 2 inference: generate 3-day demand forecast per store+medicine group.
    forecast_df = ensure_forecast_schema(
        forecast_demand_all_groups(scored_df, ARTIFACTS["demand_model"], horizon_days=3)
    )
    latest_df = latest_inventory_snapshot(scored_df)
    low_stock_df = latest_df[latest_df["predicted_status"].map(_is_critical)].copy()
    recommendations_df = build_recommendations(scored_df)

    analysis_id = str(uuid.uuid4())
    ANALYSIS_CACHE[analysis_id] = {
        "scored_df": scored_df,
        "forecast_df": forecast_df,
        "latest_df": latest_df,
        "low_stock_df": low_stock_df,
        "recommendations_df": recommendations_df,
    }

    return jsonify(
        {
            "analysis_id": analysis_id,
            "stores": sorted(latest_df["store_name"].dropna().unique().tolist()),
            "kpis": {
                "stores_monitored": int(latest_df["store_name"].nunique()),
                "skus_latest_snapshot": int(len(latest_df)),
                "low_critical_items": int(len(low_stock_df)),
                "transfer_opportunities": int(
                    (recommendations_df["final_decision"] == "Transfer Stock").sum()
                )
                if not recommendations_df.empty
                else 0,
            },
            "latest_inventory": latest_df.to_dict(orient="records"),
            "forecast": forecast_df.to_dict(orient="records"),
            "low_stock": low_stock_df.to_dict(orient="records"),
            "recommendations": recommendations_df.to_dict(orient="records"),
        }
    )


@app.route("/api/ask-anything-hybrid", methods=["POST"])
def ask_anything_hybrid():
    payload = request.get_json(silent=True) or {}
    analysis_id = payload.get("analysis_id")
    question = str(payload.get("question", "")).strip()
    store_name = str(payload.get("store_name", "")).strip()

    if not analysis_id or analysis_id not in ANALYSIS_CACHE:
        return jsonify({"error": "Invalid or expired analysis id. Run analysis again."}), 400
    if not question:
        return jsonify({"error": "question is required."}), 400

    cached = ANALYSIS_CACHE[analysis_id]
    latest_df = cached["latest_df"]
    forecast_df = cached["forecast_df"]
    if store_name:
        latest_df = latest_df[latest_df["store_name"].astype(str) == store_name].copy()
        forecast_df = forecast_df[forecast_df["store_name"].astype(str) == store_name].copy()

    known_meds = latest_df["medicine_name"].dropna().astype(str).unique().tolist()
    candidates = _extract_drug_candidates(question, known_meds)
    intent, intent_confidence = _predict_chat_intent(question)

    tools_used: list[str] = []
    tool_results: dict[str, dict] = {}

    def run_recall() -> None:
        drug = candidates[0] if candidates else ""
        if not drug:
            return
        tools_used.append("check_fda_recall")
        tool_results["check_fda_recall"] = _recall_tool_result(drug)

    def run_interaction() -> None:
        if len(candidates) < 2:
            return
        tools_used.append("check_drug_interaction")
        tool_results["check_drug_interaction"] = _interaction_tool_result(candidates[0], candidates[1])

    def run_inventory() -> None:
        drug = candidates[0] if candidates else ""
        if not drug:
            return
        tools_used.append("check_inventory")
        tool_results["check_inventory"] = _inventory_tool_result(drug, latest_df, forecast_df)

    if intent == "recall":
        run_recall()
    elif intent == "interaction":
        run_interaction()
    elif intent == "inventory":
        run_inventory()
    else:
        # Mixed intent: attempt all relevant checks.
        run_recall()
        run_interaction()
        run_inventory()

    if not tools_used:
        # Safe fallback for ambiguous text.
        run_inventory()

    sections: list[str] = []
    rec = tool_results.get("check_fda_recall")
    if rec:
        if rec.get("recalled"):
            sections.append(
                f"Recall: DO NOT DISPENSE {rec.get('drug_name','medicine')} ({rec.get('recall_class','Class Unknown')}). "
                f"Reason: {rec.get('recall_reason','No reason provided.')}"
            )
        else:
            sections.append(f"Recall: {rec.get('drug_name','medicine')} has no direct active recall match.")

    interaction = tool_results.get("check_drug_interaction")
    if interaction:
        severity = str(interaction.get("severity", "none")).upper()
        sections.append(
            f"Interaction: {severity} — {interaction.get('description','No major interaction found.')}"
        )

    inventory = tool_results.get("check_inventory")
    if inventory:
        if inventory.get("found"):
            sections.append(
                f"Inventory: {inventory.get('drug_name')} has {inventory.get('current_stock')} units, "
                f"{inventory.get('days_until_stockout')} days until stockout."
            )
        else:
            sections.append(f"Inventory: {inventory.get('drug_name','medicine')} not found in current inventory.")

    rag_query = (
        f"role=store_owner; store={store_name or 'all'}; intent={intent}; "
        f"question={question}; tools={','.join(tools_used)}"
    )
    retrieved = RETRIEVER.search(rag_query, k=3)
    citations = []
    for chunk in retrieved:
        snippet = chunk.text.replace("\n", " ").strip()
        if len(snippet) > 150:
            snippet = snippet[:150] + "..."
        citations.append(f"{chunk.source}: {snippet}")

    quality_values = [float(item.get("quality", 0.6)) for item in tool_results.values()]
    match_values = [float(item.get("match_confidence", 0.55)) for item in tool_results.values()]
    avg_quality = sum(quality_values) / len(quality_values) if quality_values else 0.6
    avg_match = sum(match_values) / len(match_values) if match_values else 0.55
    score = (0.4 * float(intent_confidence)) + (0.35 * avg_quality) + (0.25 * avg_match)
    score = round(max(0.0, min(1.0, score)), 4)

    answer = "\n".join(sections) if sections else "I could not find enough context to run checks."
    answer += "\nAction: Follow the highest-risk result first and escalate to the pharmacist when risk is high."

    return jsonify(
        {
            "intent": intent,
            "intent_confidence": round(float(intent_confidence), 4),
            "tools_used": tools_used,
            "tool_results": tool_results,
            "answer": answer,
            "citations": citations,
            "confidence_score": score,
            "confidence_band": _confidence_band(score),
        }
    )


@app.route("/api/medicine-insights", methods=["POST"])
def medicine_insights():
    payload = request.get_json(silent=True) or {}
    analysis_id = payload.get("analysis_id")
    role = str(payload.get("role", "")).strip().lower()
    store_name = payload.get("store_name")
    medicines = payload.get("medicines") or []
    selected_stores = payload.get("stores") or []

    if not analysis_id or analysis_id not in ANALYSIS_CACHE:
        return jsonify({"error": "Invalid or expired analysis id. Run analysis again."}), 400
    if role not in {"distributor", "store_owner"}:
        return jsonify({"error": "role must be distributor or store_owner"}), 400
    if not isinstance(medicines, list) or not medicines:
        return jsonify({"error": "medicines list is required"}), 400

    cached = ANALYSIS_CACHE[analysis_id]
    latest_df = cached["latest_df"]
    valid_stores = set(latest_df["store_name"].dropna().astype(str).tolist())

    if role == "store_owner":
        if not store_name:
            return jsonify({"error": "store_name is required for store_owner insights"}), 400
        if store_name not in valid_stores:
            return jsonify({"error": f"Unknown store '{store_name}' for this analysis run."}), 400
        med_scope = set(
            latest_df[latest_df["store_name"] == store_name]["medicine_name"].dropna().astype(str).tolist()
        )
    else:
        chosen_stores = [s for s in selected_stores if s in valid_stores]
        if not chosen_stores:
            chosen_stores = sorted(valid_stores)
        med_scope = set(
            latest_df[latest_df["store_name"].isin(chosen_stores)]["medicine_name"]
            .dropna()
            .astype(str)
            .tolist()
        )

    findings = []
    for med in medicines[:8]:
        med_name = str(med).strip()
        if not med_name or med_name not in med_scope:
            continue

        if role == "store_owner":
            query = f"role=store_owner; store={store_name}; medicine={med_name}; recall safety storage transfer guidance"
        else:
            query = (
                f"role=distributor; stores={','.join(selected_stores) if selected_stores else 'all'}; "
                f"medicine={med_name}; recall safety storage transfer guidance"
            )
        retrieved = RETRIEVER.search(query, k=3)
        combined = " ".join(chunk.text.lower() for chunk in retrieved)

        finding = "No active recall signal found in indexed docs."
        action = "Continue standard checks before transfer/dispense."
        if "do not sell" in combined or "recalled" in combined or "quarantine" in combined:
            finding = "Recall/safety alert found for this medicine or related batch."
            action = "Hold sale/transfer, quarantine impacted stock, and escalate to manager."
        elif "hold & verify" in combined:
            finding = "Hold-and-verify guidance found for this medicine."
            action = "Run packaging/seal/temperature checks before release."
        elif "2c to 8c" in combined or "cold-chain" in combined:
            finding = "Temperature-sensitive handling guidance found."
            action = "Use cold-chain handling and avoid non-controlled transfer."

        evidence = []
        for chunk in retrieved:
            snippet = chunk.text.replace("\n", " ").strip()
            if len(snippet) > 150:
                snippet = snippet[:150] + "..."
            evidence.append(f"{chunk.source}: {snippet}")

        findings.append(
            {
                "medicine_name": med_name,
                "finding": finding,
                "action": action,
                "evidence": evidence,
            }
        )

    return jsonify({"role": role, "count": len(findings), "findings": findings})


@app.route("/api/live-fda-insights", methods=["POST"])
def live_fda_insights():
    payload = request.get_json(silent=True) or {}
    analysis_id = payload.get("analysis_id")
    role = str(payload.get("role", "")).strip().lower()
    rows = payload.get("rows") or []

    if not analysis_id or analysis_id not in ANALYSIS_CACHE:
        return jsonify({"error": "Invalid or expired analysis id. Run analysis again."}), 400
    if role not in {"distributor", "store_owner"}:
        return jsonify({"error": "role must be distributor or store_owner"}), 400
    if not isinstance(rows, list) or not rows:
        return jsonify({"error": "rows payload is required"}), 400

    try:
        live_inventory = _to_live_inventory_rows(rows[:120])
        recalls = _fetch_active_recalls_live()
        shortages = _fetch_shortages_live()
        shortage_set = {
            _normalize_drug_name(item.get("product_name") or item.get("drug_name") or "")
            for item in shortages
        }
    except Exception as exc:
        return jsonify({"error": f"Live FDA fetch failed: {exc}"}), 502

    all_matches = []
    findings = []
    for item in live_inventory:
        matches = _match_recall_to_inventory_item(item, recalls)
        all_matches.extend(matches)
        best_match = max(matches, key=lambda x: float(x.get("confidence", 0)), default=None)

        shortage_flag = 1.0 if _normalize_drug_name(item.get("drug_name", "")) in shortage_set else 0.0

        finding = "No direct recall match found."
        action = "Continue monitoring and verify supplier notices."
        evidence = []
        if shortage_flag:
            evidence.append("Drug appears on current FDA shortage list.")

        if best_match:
            confidence = float(best_match.get("confidence", 0))
            if confidence >= 0.85:
                finding = "High-confidence FDA recall match found."
                action = "Hold sale/transfer, quarantine impacted stock, and escalate immediately."
            else:
                finding = "Possible FDA recall match found (needs pharmacist review)."
                action = "Review lot/label details and temporarily hold affected batches."
            rec = best_match.get("recall_item", {})
            recall_number = rec.get("recall_number") or "(no recall #)"
            recall_class = rec.get("classification") or "Class Unknown"
            recall_reason = rec.get("reason_for_recall") or "No reason provided."
            affected_lots = best_match.get("lot_numbers_matched") or _parse_lot_numbers(rec.get("code_info"))
            evidence.extend(
                [
                    f"FDA recall {recall_number} — {recall_class}",
                    f"Reason: {recall_reason}",
                    _match_explanation(best_match),
                ]
            )
        elif shortage_flag:
            finding = "No active recall, but FDA shortage risk detected."
            action = "Increase safety stock and prepare substitute strategy."
            confidence = 0.0
            recall_number = None
            recall_class = None
            recall_reason = None
            affected_lots = []
        else:
            confidence = 0.0
            recall_number = None
            recall_class = None
            recall_reason = None
            affected_lots = []

        findings.append(
            {
                "medicine_name": item.get("drug_name", ""),
                "finding": finding,
                "action": action,
                "evidence": evidence,
                "confidence": round(float(confidence), 4),
                "recall_number": recall_number,
                "recall_class": recall_class,
                "recall_reason": recall_reason,
                "affected_lots": affected_lots,
            }
        )

    findings = [row for row in findings if row.get("medicine_name")][:16]
    findings.sort(key=lambda row: ("High-confidence" not in row["finding"], row["medicine_name"]))

    return jsonify(
        {
            "role": role,
            "source": "live_fda",
            "updated_at": dt.datetime.utcnow().isoformat(),
            "count": len(findings),
            "findings": findings,
            "matches_count": len(all_matches),
        }
    )


@app.route("/api/rxnorm-interaction", methods=["POST"])
def rxnorm_interaction():
    payload = request.get_json(silent=True) or {}
    drug1 = str(payload.get("drug1", "")).strip()
    drug2 = str(payload.get("drug2", "")).strip()
    drug3 = str(payload.get("drug3", "")).strip()

    if not drug1 or not drug2:
        return jsonify({"error": "drug1 and drug2 are required"}), 400

    try:
        rxcui1 = _rxnorm_rxcui(drug1)
        rxcui2 = _rxnorm_rxcui(drug2)
        rxcui3 = _rxnorm_rxcui(drug3) if drug3 else None
        if not rxcui1 or not rxcui2:
            return (
                jsonify(
                    {
                        "interaction_found": False,
                        "severity": "none",
                        "description": "Could not resolve one or more drugs in RxNorm.",
                        "source_name": "ONCHigh",
                        "rxcui": {"drug1": rxcui1, "drug2": rxcui2, "drug3": rxcui3},
                    }
                ),
                200,
            )

        interaction_payload = {}
        parsed = None
        try:
            interaction_payload = _rxnorm_interaction_payload(
                [rxcui1, rxcui2, rxcui3] if rxcui3 else [rxcui1, rxcui2]
            )
            parsed = _rxnorm_parse_interaction(interaction_payload)
        except Exception:
            parsed = None
        if not parsed or (not parsed.get("interaction_found") and parsed.get("severity") == "none"):
            parsed = _fallback_interaction_by_names(drug1, drug2, drug3)
        return jsonify(
            {
                **parsed,
                "rxcui": {"drug1": rxcui1, "drug2": rxcui2, "drug3": rxcui3},
                "raw": interaction_payload,
            }
        )
    except Exception as exc:
        return jsonify({"error": f"RxNorm check failed: {exc}"}), 502


@app.route("/api/distributor-chat", methods=["POST"])
def distributor_chat():
    payload = request.get_json(silent=True) or {}
    analysis_id = payload.get("analysis_id")
    stores = payload.get("stores") or []
    medicine_name = str(payload.get("medicine_name", "All Medicines")).strip() or "All Medicines"
    question = str(payload.get("question", "")).strip()

    if not analysis_id or analysis_id not in ANALYSIS_CACHE:
        return jsonify({"error": "Invalid or expired analysis id. Run analysis again."}), 400
    if not question:
        return jsonify({"error": "question is required."}), 400

    cached = ANALYSIS_CACHE[analysis_id]
    latest_df = cached["latest_df"]
    low_stock_df = cached["low_stock_df"]
    forecast_df = cached["forecast_df"]
    recommendations_df = cached["recommendations_df"]

    valid_stores = sorted(set(latest_df["store_name"].dropna().astype(str).tolist()))
    selected_stores = [s for s in stores if s in valid_stores]
    if not selected_stores:
        selected_stores = valid_stores

    dist_inventory = latest_df[latest_df["store_name"].isin(selected_stores)].copy()
    dist_low = low_stock_df[low_stock_df["store_name"].isin(selected_stores)].copy()
    dist_forecast = forecast_df[forecast_df["store_name"].isin(selected_stores)].copy()
    dist_recs = recommendations_df[recommendations_df["store_name"].isin(selected_stores)].copy()

    if medicine_name != "All Medicines":
        available = set(dist_inventory["medicine_name"].dropna().astype(str).tolist())
        if medicine_name not in available:
            return jsonify({"error": f"Medicine '{medicine_name}' is not present in selected stores."}), 400
        dist_inventory = dist_inventory[dist_inventory["medicine_name"] == medicine_name].copy()
        dist_low = dist_low[dist_low["medicine_name"] == medicine_name].copy()
        dist_forecast = dist_forecast[dist_forecast["medicine_name"] == medicine_name].copy()
        dist_recs = dist_recs[dist_recs["medicine_name"] == medicine_name].copy()

    med_fact = MED_FACTS.get(medicine_name, "") if medicine_name != "All Medicines" else ""
    intent = detect_intent(question)
    runtime_facts = collect_store_runtime_facts(
        store_name=f"Distributor({len(selected_stores)} stores)",
        owner_inventory=dist_inventory,
        owner_low=dist_low,
        owner_forecast=dist_forecast,
        owner_recommendations=dist_recs,
        selected_medicine=medicine_name,
        medicine_fact=med_fact,
    )
    query = (
        f"role=distributor; stores={','.join(selected_stores)}; medicine={medicine_name}; "
        f"intent={intent}; question={question}"
    )
    retrieved = RETRIEVER.search(query, k=3)
    response = answer_with_evidence(
        question=question,
        intent=intent,
        runtime_facts=runtime_facts,
        retrieved=retrieved,
    )
    response["store_scope"] = selected_stores
    response["medicine_scope"] = medicine_name
    return jsonify(response)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
