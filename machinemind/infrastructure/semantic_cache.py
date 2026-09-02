"""Semantic response cache and company knowledge-version infrastructure.

This module contains the existing production V13 cache behavior extracted from
``main.py``.  It never imports the composition root.  Instead, the thin compatibility
wrappers in ``main`` pass their live globals mapping on every call.  That preserves:

* environment-derived configuration;
* request-budget and embedding-cache state;
* current DB/OpenAI/link callbacks and monkeypatch points;
* the historical fail-open behavior of cache reads, writes and invalidation;
* the mutable bootstrap state previously held by ``main``.

No cache threshold, SQL statement, compatibility guard, quality formula or response
shape is intentionally changed by this extraction.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, MutableMapping
from typing import Any, Optional


class _Runtime:
    """Late-bound view over the production composition-root namespace."""

    def __init__(self, values: MutableMapping[str, Any]):
        self.values = values

    def get(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)

    def require(self, name: str) -> Any:
        return self.values[name]

    def set(self, name: str, value: Any) -> None:
        self.values[name] = value

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return self.values[name](*args, **kwargs)


def cache_bootstrap(runtime_globals: MutableMapping[str, Any]) -> bool:
    rt = _Runtime(runtime_globals)

    if not rt.require("V13_SEMANTIC_CACHE_ENABLED"):
        return False
    if rt.get("_V13_CACHE_READY") is True:
        return True
    now = rt.require("time_module").monotonic()
    if rt.get("_V13_CACHE_READY") is False and now < float(rt.get("_V13_CACHE_RETRY_AT") or 0.0):
        return False

    with rt.require("_V13_CACHE_LOCK"):
        now = rt.require("time_module").monotonic()
        if rt.get("_V13_CACHE_READY") is True:
            return True
        if rt.get("_V13_CACHE_READY") is False and now < float(rt.get("_V13_CACHE_RETRY_AT") or 0.0):
            return False
        if rt.get("_V13_CACHE_READY") is False:
            rt.set("_V13_CACHE_READY", None)
        if not rt.require("V13_SEMANTIC_CACHE_AUTO_DDL"):
            rt.set("_V13_CACHE_READY", True)
            return True

        conn = None
        try:
            conn = rt.call("_db_conn")
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS public.mm_v13_knowledge_versions (
                        company_id TEXT PRIMARY KEY,
                        version BIGINT NOT NULL DEFAULT 1,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS public.mm_v13_semantic_cache (
                        id BIGSERIAL PRIMARY KEY,
                        company_id TEXT NOT NULL,
                        machine_id TEXT NOT NULL DEFAULT '',
                        ai_scope TEXT NOT NULL,
                        scope_key TEXT NOT NULL,
                        mode TEXT NOT NULL,
                        language TEXT NOT NULL,
                        engine_key TEXT NOT NULL DEFAULT '',
                        knowledge_version BIGINT NOT NULL,
                        query_hash TEXT,
                        query_text TEXT NOT NULL,
                        query_embedding JSONB NOT NULL,
                        response_json JSONB NOT NULL,
                        quality_score DOUBLE PRECISION NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        expires_at TIMESTAMPTZ NOT NULL
                    );
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE public.mm_v13_semantic_cache
                        ADD COLUMN IF NOT EXISTS engine_key TEXT NOT NULL DEFAULT '',
                        ADD COLUMN IF NOT EXISTS query_hash TEXT;
                    """
                )
                cur.execute(
                    """
                    UPDATE public.mm_v13_semantic_cache
                    SET query_hash = 'legacy:' || md5(
                        COALESCE(engine_key, '') || E'\n' ||
                        COALESCE(query_text, '') || E'\n' || id::text
                    )
                    WHERE query_hash IS NULL OR btrim(query_hash) = '';
                    """
                )
                cur.execute(
                    """
                    DELETE FROM public.mm_v13_semantic_cache older
                    USING public.mm_v13_semantic_cache newer
                    WHERE older.id < newer.id
                      AND older.company_id = newer.company_id
                      AND older.machine_id = newer.machine_id
                      AND older.ai_scope = newer.ai_scope
                      AND older.scope_key = newer.scope_key
                      AND older.mode = newer.mode
                      AND older.language = newer.language
                      AND older.engine_key = newer.engine_key
                      AND older.knowledge_version = newer.knowledge_version
                      AND older.query_hash = newer.query_hash;
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE public.mm_v13_semantic_cache
                        ALTER COLUMN query_hash SET NOT NULL;
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE public.mm_v13_semantic_cache
                        DROP CONSTRAINT IF EXISTS mm_v13_semantic_cache_exact_key;
                    """
                )
                cur.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS uq_mm_v13_semantic_cache_key
                    ON public.mm_v13_semantic_cache (
                        company_id, machine_id, ai_scope, scope_key,
                        mode, language, engine_key, knowledge_version, query_hash
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_mm_v13_semantic_cache_lookup
                    ON public.mm_v13_semantic_cache (
                        company_id, machine_id, ai_scope, scope_key,
                        mode, language, engine_key, knowledge_version, expires_at DESC
                    );
                    """
                )
            conn.commit()
            rt.set("_V13_CACHE_READY", True)
            rt.set("_V13_CACHE_ERROR", "")
            rt.set("_V13_CACHE_RETRY_AT", 0.0)
        except Exception as exc:
            if conn is not None:
                try:
                    conn.rollback()
                except Exception:
                    pass
            rt.set("_V13_CACHE_READY", False)
            rt.set("_V13_CACHE_ERROR", str(exc)[:800])
            rt.set(
                "_V13_CACHE_RETRY_AT",
                rt.require("time_module").monotonic()
                + float(rt.require("V13_SEMANTIC_CACHE_BOOTSTRAP_RETRY_SECONDS")),
            )
            print("V13_CACHE_BOOTSTRAP_RETRY", rt.get("_V13_CACHE_ERROR"))
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    return bool(rt.get("_V13_CACHE_READY"))


def normalize_query(value: str, runtime_globals: Mapping[str, Any]) -> str:
    value = runtime_globals["_normalize_unicode_advanced"](str(value or "")).lower()
    value = re.sub(r"\s+", " ", value).strip()
    return value


def scope_key(scope: dict, runtime_globals: Mapping[str, Any]) -> str:
    doc_ids = runtime_globals["_normalize_document_ids"](scope.get("document_ids")) or []
    bubble_document_id = str(scope.get("bubble_document_id") or "").strip()
    payload = {
        "ai_scope": str(scope.get("ai_scope") or "machine_all"),
        "bubble_document_id": bubble_document_id,
        "document_ids": sorted(doc_ids),
        "top_k": runtime_globals["_safe_int"](scope.get("_v13_top_k"), 0),
        "max_causes": runtime_globals["_safe_int"](scope.get("_v13_max_causes"), 0),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:40]


def get_knowledge_version(
    company_id: str,
    runtime_globals: MutableMapping[str, Any],
) -> int:
    rt = _Runtime(runtime_globals)
    if not rt.call("_v13_cache_bootstrap"):
        return 0
    company_id = str(company_id or "").strip()
    if not company_id:
        return 0

    conn = None
    try:
        conn = rt.call("_db_conn")
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.mm_v13_knowledge_versions(company_id, version, updated_at)
                VALUES (%s, 1, NOW())
                ON CONFLICT (company_id) DO NOTHING;
                """,
                (company_id,),
            )
            cur.execute(
                "SELECT version FROM public.mm_v13_knowledge_versions WHERE company_id=%s;",
                (company_id,),
            )
            row = cur.fetchone()
        conn.commit()
        return int((row or [1])[0] or 1)
    except Exception as exc:
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                pass
        print("V13_CACHE_VERSION_FAIL_OPEN", str(exc)[:500])
        return 0
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def bump_knowledge_version(
    company_id: str,
    runtime_globals: MutableMapping[str, Any],
) -> None:
    rt = _Runtime(runtime_globals)
    company_id = str(company_id or "").strip()
    if not company_id or not rt.call("_v13_cache_bootstrap"):
        return

    conn = rt.call("_db_conn")
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.mm_v13_knowledge_versions(company_id, version, updated_at)
                VALUES (%s, 1, NOW())
                ON CONFLICT (company_id)
                DO UPDATE SET version = public.mm_v13_knowledge_versions.version + 1,
                              updated_at = NOW();
                """,
                (company_id,),
            )
            cur.execute(
                "DELETE FROM public.mm_v13_semantic_cache WHERE company_id=%s;",
                (company_id,),
            )
        conn.commit()
    finally:
        conn.close()


def invalidate_company_knowledge(
    company_id: str,
    runtime_globals: MutableMapping[str, Any],
) -> None:
    rt = _Runtime(runtime_globals)
    try:
        rt.call("_v13_bump_knowledge_version", company_id)
    except Exception as exc:
        # Cache invalidation must never break ingest/delete/indexing.
        print("V13_CACHE_INVALIDATION_SKIPPED", str(exc)[:500])


def cache_code_tokens(q: str, runtime_globals: Mapping[str, Any]) -> list[str]:
    """Return only identifier-like tokens that are safe semantic-cache guards."""
    text = runtime_globals["_normalize_unicode_advanced"](q or "")
    raw = re.findall(
        r"(?<![A-Za-z0-9])[A-Za-z0-9][A-Za-z0-9_.:/\-]{1,63}(?![A-Za-z0-9])",
        text,
    )
    out: list[str] = []
    seen: set[str] = set()
    for token in raw:
        token = str(token or "").strip("._:/-")
        if not token or token.isdigit():
            continue
        letters = "".join(ch for ch in token if ch.isalpha())
        has_digit = any(ch.isdigit() for ch in token)
        has_separator = any(ch in "_./:-" for ch in token)
        is_acronym = bool(letters) and len(letters) >= 2 and letters.upper() == letters
        if not (has_digit or has_separator or is_acronym):
            continue
        key = token.upper()
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return sorted(out)


def query_number_tokens(q: str, runtime_globals: Mapping[str, Any]) -> list[str]:
    """Extract numeric constraints without swallowing ordinary following words."""
    text = runtime_globals["_normalize_unicode_advanced"](q or "")
    number_rx = re.compile(r"(?<![A-Za-z0-9])[-+]?\d+(?:[.,]\d+)?(?![A-Za-z0-9])")
    multi_units = [
        "mm/min", "mm/sec", "mm/s", "m/min", "m/sec", "m/s", "l/min", "ml/min",
        "mm2", "mm²", "cm2", "cm²", "m2", "m²", "mm3", "mm³", "cm3", "cm³", "m3", "m³",
        "khz", "mhz", "hz", "rpm", "mbar", "kpa", "mpa", "psi", "bar", "pa",
        "kwh", "kw", "mw", "ma", "ka", "mv", "kv", "nm", "kn",
        "kg", "mg", "ml", "min", "ms", "µs", "us", "mm", "cm", "km",
        "deg", "sec",
    ]
    multi_unit_rx = re.compile(
        r"^\s*(" + "|".join(re.escape(x) for x in multi_units) + r")(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )
    symbol_unit_rx = re.compile(r"^\s*(%|°\s*[CFcf]?)(?![A-Za-z0-9])")
    uppercase_single_rx = re.compile(r"^\s*([AVWNF])(?=$|[^A-Za-z0-9])")
    lowercase_single_rx = re.compile(r"^\s*([smghl])(?=$|[^A-Za-z0-9])")

    values: set[str] = set()
    for match in number_rx.finditer(text):
        number = str(match.group(0) or "").replace(",", ".")
        tail = text[match.end(): match.end() + 24]
        unit = ""
        unit_match = symbol_unit_rx.match(tail)
        if unit_match:
            unit = re.sub(r"\s+", "", unit_match.group(1)).lower()
        else:
            unit_match = multi_unit_rx.match(tail)
            if unit_match:
                unit = re.sub(r"\s+", "", unit_match.group(1)).lower()
            else:
                unit_match = uppercase_single_rx.match(tail)
                if unit_match:
                    unit = unit_match.group(1).lower()
                else:
                    unit_match = lowercase_single_rx.match(tail)
                    if unit_match:
                        unit = unit_match.group(1).lower()
        values.add(number.lower() + unit)
    return sorted(values)


def query_polarity_signature(q: str, runtime_globals: Mapping[str, Any]) -> tuple[str, ...]:
    low = f" {runtime_globals['_v13_normalize_query'](q)} "
    groups = {
        "negation": [" non ", " not ", " no ", " senza ", " without ", " never ", " mai "],
        "exclusive": [" solo ", " soltanto ", " only ", " esclusivamente ", " exclusively "],
        "exception": [" tranne ", " eccetto ", " except ", " excluding ", " escluso "],
        "before": [" prima ", " before ", " preventiv", " preliminar"],
        "after": [" dopo ", " after ", " al termine ", " completed", " complet"],
    }
    return tuple(sorted(name for name, markers in groups.items() if any(marker in low for marker in markers)))


def query_source_signature(q: str, runtime_globals: Mapping[str, Any]) -> tuple[str, str]:
    try:
        profile = runtime_globals["_ask_source_preference_profile"](q)
        return (
            str(profile.get("preferred_source") or ""),
            str(profile.get("strength") or "none"),
        )
    except Exception:
        return ("", "none")


def semantic_cache_compatible(
    mode: str,
    current_q: str,
    cached_q: str,
    runtime_globals: Mapping[str, Any],
) -> bool:
    current_norm = runtime_globals["_v13_normalize_query"](current_q)
    cached_norm = runtime_globals["_v13_normalize_query"](cached_q)
    if not current_norm or not cached_norm:
        return False

    ratio = len(current_norm) / max(1, len(cached_norm))
    if ratio < 0.55 or ratio > 1.80:
        return False

    current_codes = set(runtime_globals["_v13_cache_code_tokens"](current_q))
    cached_codes = set(runtime_globals["_v13_cache_code_tokens"](cached_q))
    if current_codes != cached_codes:
        return False

    if runtime_globals["_v13_query_number_tokens"](current_q) != runtime_globals["_v13_query_number_tokens"](cached_q):
        return False

    if runtime_globals["_v13_query_polarity_signature"](current_q) != runtime_globals["_v13_query_polarity_signature"](cached_q):
        return False

    if runtime_globals["_v13_query_source_signature"](current_q) != runtime_globals["_v13_query_source_signature"](cached_q):
        return False

    current_terms = runtime_globals["_content_term_set"](current_q, limit=60)
    cached_terms = runtime_globals["_content_term_set"](cached_q, limit=60)
    if current_norm != cached_norm:
        if min(len(current_terms), len(cached_terms)) <= 2:
            return False
        anchor_ratio = len(current_terms & cached_terms) / max(1, min(len(current_terms), len(cached_terms)))
        minimum_anchor_ratio = 0.70 if str(mode or "") == "root_cause" else 0.55
        if anchor_ratio < minimum_anchor_ratio:
            return False

    if str(mode or "") == "root_cause":
        current_profile = runtime_globals["_query_symptom_profile"](current_q)
        cached_profile = runtime_globals["_query_symptom_profile"](cached_q)
        if set(current_profile.get("classes") or []) != set(cached_profile.get("classes") or []):
            return False
        if bool(current_profile.get("automatic_mode")) != bool(cached_profile.get("automatic_mode")):
            return False

    return True


def jsonb_to_python(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return fallback


def cache_lookup(
    *,
    mode: str,
    q: str,
    company_id: str,
    machine_id: str,
    scope: dict,
    language: str,
    debug: bool,
    runtime_globals: MutableMapping[str, Any],
) -> Optional[dict]:
    rt = _Runtime(runtime_globals)
    budget = rt.call("_v13_current_budget")
    if budget is not None:
        budget.semantic_cache = "bypass_debug" if debug else "miss"
    if debug or not rt.require("V13_SEMANTIC_CACHE_ENABLED") or not rt.call("_v13_cache_bootstrap"):
        return None

    knowledge_version = rt.call("_v13_get_knowledge_version", company_id)
    if knowledge_version <= 0:
        return None

    scope_key_value = rt.call("_v13_scope_key", scope)
    ai_scope = str(scope.get("ai_scope") or "machine_all")
    machine_key = str(machine_id or "")

    conn = None
    try:
        conn = rt.call("_db_conn")
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, query_text, response_json, quality_score, created_at
                FROM public.mm_v13_semantic_cache
                WHERE company_id=%s
                  AND machine_id=%s
                  AND ai_scope=%s
                  AND scope_key=%s
                  AND mode=%s
                  AND language=%s
                  AND engine_key=%s
                  AND knowledge_version=%s
                  AND expires_at > NOW()
                  AND quality_score >= %s
                ORDER BY created_at DESC
                LIMIT %s;
                """,
                (
                    company_id,
                    machine_key,
                    ai_scope,
                    scope_key_value,
                    mode,
                    language,
                    rt.require("V13_ENGINE_KEY"),
                    knowledge_version,
                    rt.require("V13_SEMANTIC_CACHE_MIN_QUALITY"),
                    rt.require("V13_SEMANTIC_CACHE_SCAN_LIMIT"),
                ),
            )
            rows = cur.fetchall()
    except Exception as exc:
        print("V13_CACHE_LOOKUP_FAIL_OPEN", str(exc)[:500])
        return None
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    if not rows:
        return None

    threshold = (
        rt.require("V13_SEMANTIC_CACHE_THRESHOLD_ROOT_CAUSE")
        if mode == "root_cause"
        else rt.require("V13_SEMANTIC_CACHE_THRESHOLD_ASK")
    )
    best: Optional[tuple[float, dict, float, Any]] = None
    current_norm = rt.call("_v13_normalize_query", q)

    for _row_id, cached_q, response_json, quality_score, created_at in rows:
        if rt.call("_v13_normalize_query", str(cached_q or "")) != current_norm:
            continue
        response = rt.call("_v13_jsonb_to_python", response_json, {})
        if isinstance(response, dict) and response.get("ok") is True:
            best = (1.0, response, float(quality_score or 0.0), created_at)
            break

    if best is None:
        compatible_rows = [
            row for row in rows
            if rt.call("_v13_semantic_cache_compatible", mode, q, str(row[1] or ""))
        ]
        if not compatible_rows:
            return None

        try:
            q_vec = rt.call("_openai_embed_texts", [q], timeout=10)[0]
        except Exception as exc:
            print("V13_CACHE_EMBED_LOOKUP_FAIL", str(exc)[:500])
            return None

        ids = [int(row[0]) for row in compatible_rows]
        embedding_by_id: dict[int, list] = {}
        conn = None
        try:
            conn = rt.call("_db_conn")
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, query_embedding
                    FROM public.mm_v13_semantic_cache
                    WHERE id = ANY(%s);
                    """,
                    (ids,),
                )
                for row_id, embedding_json in cur.fetchall():
                    embedding = rt.call("_v13_jsonb_to_python", embedding_json, [])
                    if isinstance(embedding, list) and embedding:
                        embedding_by_id[int(row_id)] = embedding
        except Exception as exc:
            print("V13_CACHE_VECTOR_FETCH_FAIL_OPEN", str(exc)[:500])
            return None
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

        for row_id, _cached_q, response_json, quality_score, created_at in compatible_rows:
            embedding = embedding_by_id.get(int(row_id)) or []
            if not embedding:
                continue
            try:
                similarity = float(rt.call("_cosine_sim", q_vec, [float(x) for x in embedding]))
            except Exception:
                continue
            if similarity < threshold:
                continue
            response = rt.call("_v13_jsonb_to_python", response_json, {})
            if not isinstance(response, dict) or response.get("ok") is not True:
                continue
            candidate = (similarity, response, float(quality_score or 0.0), created_at)
            if best is None or candidate[0] > best[0]:
                best = candidate

    if best is None:
        return None

    similarity, response, quality_score, created_at = best
    out = dict(response)
    citations = list(out.get("citations") or [])
    try:
        out["rg_links"] = rt.call("_build_rg_links", company_id, citations) if citations else []
    except Exception as exc:
        print("V13_CACHE_LINK_REFRESH_FAIL", str(exc)[:500])
        out["rg_links"] = []

    meta = dict(out.get("meta") or {})
    meta["v13_semantic_cache"] = {
        "hit": True,
        "similarity": round(similarity, 6),
        "quality_score": round(quality_score, 4),
        "created_at": str(created_at or ""),
        "knowledge_version": knowledge_version,
    }
    out["meta"] = meta
    out["chat_model"] = "v13_semantic_cache"

    if budget is not None:
        budget.semantic_cache = "hit"
        budget.route = "semantic_cache"
    return out


def response_quality(
    mode: str,
    response: dict,
    runtime_globals: Mapping[str, Any],
) -> float:
    if not isinstance(response, dict) or response.get("ok") is not True:
        return 0.0
    if str(response.get("status") or "").lower() != "answered":
        return 0.0

    citations = [
        c for c in (response.get("citations") or [])
        if isinstance(c, dict) and c.get("citation_id")
    ]
    if not citations:
        return 0.0

    if mode == "root_cause":
        causes = [c for c in (response.get("possible_causes") or []) if isinstance(c, dict)]
        if not causes:
            return 0.0
        grounded = sum(1 for c in causes if c.get("cause") and c.get("why") and c.get("citations"))
        checks = sum(len(c.get("checks") or []) for c in causes)
        quality = 0.68 + min(0.16, 0.06 * grounded) + min(0.10, 0.025 * checks)
        family_key = runtime_globals["_root_cause_evidence_family_key"]
        if len({family_key(cit) for cit in citations}) >= 2:
            quality += 0.04
        return max(0.0, min(0.98, quality))

    answer = str(response.get("answer") or "").strip()
    if len(answer) < 32:
        return 0.0
    quality = 0.72 + min(0.12, len(citations) * 0.025)
    if any(bool(c.get("exact_machine_scope")) for c in citations):
        quality += 0.04
    if any(str(c.get("evidence_role") or "") in {"procedure", "step", "manual_support"} for c in citations):
        quality += 0.05
    return max(0.0, min(0.98, quality))


def assistant_core_cache_certified(mode: str, response: dict) -> bool:
    """Only fully validated canonical responses may enter any response cache."""
    if not isinstance(response, dict) or response.get("ok") is not True:
        return False
    if str(response.get("status") or "").strip().lower() != "answered":
        return False
    meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
    if not bool(meta.get("canonical_final_answer")):
        return False
    citations = [c for c in (response.get("citations") or []) if isinstance(c, dict)]
    if not citations and str(response.get("grounding") or "") != "general_technical_knowledge":
        return False
    mode_key = str(mode or response.get("effective_mode") or "ask").strip().lower()
    if mode_key == "root_cause":
        causes = [
            c for c in (response.get("possible_causes") or [])
            if isinstance(c, dict) and str(c.get("cause") or "").strip()
        ]
        if not causes:
            return False
        valid_ids = {
            str(c.get("citation_id") or "").strip()
            for c in citations
            if str(c.get("citation_id") or "").strip()
        }
        for cause in causes:
            own = {
                str(x or "").strip()
                for x in (cause.get("citation_ids") or [])
                if str(x or "").strip()
            }
            if own and not (own & valid_ids):
                return False
            if not str(cause.get("why") or "").strip() or not list(cause.get("checks") or []):
                return False
        return True
    validation = meta.get("assistant_core_validation") if isinstance(meta.get("assistant_core_validation"), dict) else {}
    contract = validation.get("answer_contract") if isinstance(validation.get("answer_contract"), dict) else {}
    if not bool(contract.get("passed")):
        return False
    if contract.get("missing_answer_facets") or contract.get("missing_evidence_facets") or contract.get("missing_list_items"):
        return False
    if not str(response.get("answer") or "").strip() or not str(response.get("answer_html") or "").strip():
        return False
    return True


def cache_store(
    *,
    mode: str,
    q: str,
    company_id: str,
    machine_id: str,
    scope: dict,
    language: str,
    response: dict,
    debug: bool,
    runtime_globals: MutableMapping[str, Any],
) -> None:
    rt = _Runtime(runtime_globals)
    if debug or not rt.require("V13_SEMANTIC_CACHE_ENABLED") or not rt.call("_v13_cache_bootstrap"):
        return
    if not rt.call("_assistant_core_cache_certified", mode, response):
        return
    response_meta = response.get("meta") or {}
    if response_meta.get("cacheable") is False or response_meta.get("semantic_cacheable") is False:
        return

    quality = rt.call("_v13_response_quality", mode, response)
    if quality < rt.require("V13_SEMANTIC_CACHE_MIN_QUALITY"):
        return

    budget = rt.call("_v13_current_budget")
    if budget is not None and (rt.require("OPENAI_EMBED_MODEL"), str(q or "")) not in budget.embedding_cache:
        return

    try:
        embedding = rt.call("_openai_embed_texts", [q], timeout=10)[0]
    except Exception as exc:
        print("V13_CACHE_EMBED_STORE_FAIL", str(exc)[:500])
        return

    knowledge_version = rt.call("_v13_get_knowledge_version", company_id)
    if knowledge_version <= 0:
        return

    normalized_q = rt.call("_v13_normalize_query", q)
    query_hash = hashlib.sha256(
        (rt.require("V13_ENGINE_KEY") + "\n" + normalized_q).encode("utf-8")
    ).hexdigest()
    ai_scope = str(scope.get("ai_scope") or "machine_all")
    scope_key_value = rt.call("_v13_scope_key", scope)
    machine_key = str(machine_id or "")

    stored_response = dict(response)
    stored_response.pop("debug", None)
    stored_response["rg_links"] = []
    meta = dict(stored_response.get("meta") or {})
    meta.pop("v13_runtime", None)
    meta.pop("v13_semantic_cache", None)
    stored_response["meta"] = meta

    conn = None
    try:
        conn = rt.call("_db_conn")
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.mm_v13_semantic_cache(
                    company_id, machine_id, ai_scope, scope_key,
                    mode, language, engine_key, knowledge_version,
                    query_hash, query_text, query_embedding,
                    response_json, quality_score, created_at, expires_at
                )
                VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s::jsonb,
                    %s::jsonb, %s, NOW(), NOW() + (%s * INTERVAL '1 second')
                )
                ON CONFLICT (
                    company_id, machine_id, ai_scope, scope_key,
                    mode, language, engine_key, knowledge_version, query_hash
                )
                DO UPDATE SET
                    engine_key=EXCLUDED.engine_key,
                    query_text=EXCLUDED.query_text,
                    query_embedding=EXCLUDED.query_embedding,
                    response_json=EXCLUDED.response_json,
                    quality_score=EXCLUDED.quality_score,
                    created_at=NOW(),
                    expires_at=EXCLUDED.expires_at;
                """,
                (
                    company_id,
                    machine_key,
                    ai_scope,
                    scope_key_value,
                    mode,
                    language,
                    rt.require("V13_ENGINE_KEY"),
                    knowledge_version,
                    query_hash,
                    q,
                    json.dumps(embedding),
                    json.dumps(stored_response, ensure_ascii=False),
                    quality,
                    rt.require("V13_SEMANTIC_CACHE_TTL_SECONDS"),
                ),
            )
            cur.execute(
                "DELETE FROM public.mm_v13_semantic_cache WHERE expires_at <= NOW();"
            )
            cur.execute(
                """
                DELETE FROM public.mm_v13_semantic_cache
                WHERE id IN (
                    SELECT id
                    FROM public.mm_v13_semantic_cache
                    WHERE company_id=%s
                    ORDER BY created_at DESC
                    OFFSET %s
                );
                """,
                (company_id, rt.require("V13_SEMANTIC_CACHE_MAX_ROWS_PER_COMPANY")),
            )
        conn.commit()
    except Exception as exc:
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                pass
        print("V13_CACHE_STORE_FAIL_OPEN", str(exc)[:500])
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
