"""Offline source selection regressions; no network, model or database imports."""
from copy import deepcopy
from dataclasses import fields, replace
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import assistant_core_v2 as core
from machinemind.retrieval import smart_sources as sources, smart_evidence
from machinemind.retrieval import evidence_orchestration as orchestration
from machinemind.retrieval import candidate_ranking as ranking, retrieval_policy

SCOPE = dict(company_id="tenant-fixture", machine_id="machine-fixture", ai_scope="machine_all")


def terms(text, limit=4096):
    return retrieval_policy.content_term_set(text, limit,
        runtime=retrieval_policy.ContentTermSetRuntime(str, re))


def row(cid, text, *, owner=None, kind="document", **values):
    return {"citation_id": cid, "bubble_document_id": owner or cid, "source_type": kind,
            "company_id": SCOPE["company_id"], "machine_id": SCOPE["machine_id"],
            "chunk_full": text, "snippet": text[:100], "page_from": 1, "page_to": 1,
            "semantic_similarity": 0.5, **values}


def request(query="ZX400 pump pressure alarm after service"):
    return core.AssistantCoreRequest(query=query, requested_mode="smart_diagnostic",
        response_language="en", top_k=8, **SCOPE)


def decision(**overrides):
    return replace(core.AssistantCoreDecision(request_kind="fault_diagnostic",
        effective_mode="smart_diagnostic", confidence=.9, requested_mode_fit=True,
        evidence_state="supported", evidence_policy="machine_sources_required",
        diagnostic_subsystems=("pump pressure",),
        facet_queries=(core.AssistantCoreFacetQuery("pressure indication",
            dense_queries=("pressione pompa segnale sensore",), exact_terms=("ZX400",)),)), **overrides)


def select(rows, req=None, dec=None, **kwargs):
    return sources.select(rows, request=req or request(), decision=dec or decision(),
        term_set=terms, candidate_text=lambda c: c.get("snippet", ""), **kwargs)


def bomb(*args, **kwargs):
    raise AssertionError("legacy causal filtering / external IO must not run")


def scoring_runtime(dedup=bomb):
    return ranking.V13ScoreCandidatesRuntime(.4, lambda *a, **k: (0, {}), lambda *a: 0,
        terms, lambda q: len(q.split()), dedup, lambda q: (), str, lambda key: "document",
        lambda q, t: len(q & t) / max(1, len(q)), lambda r: r["chunk_full"],
        lambda r: r.get("semantic_similarity", 0))


def initial_runtime(rows, calls):
    values = {f.name: bomb if f.name.startswith("_") else 10
              for f in fields(orchestration.V13InitialRetrievalRuntime)}
    def score(q, data, **kwargs):
        calls.append(("score", bool(kwargs.get("preserve_complete"))))
        return ranking.v13_score_candidates(q, data,
            runtime=scoring_runtime(lambda rows, **kw: rows[:1]), **kwargs)
    values.update(_v13_current_budget=lambda: None, _v13_fallback_plan=lambda q: {},
        _dedup_text_values=lambda rows, limit: list(dict.fromkeys(rows))[:limit],
        _openai_embed_texts=lambda queries, **kw: [[1] for _ in queries],
        _vector_literal=str, _fetch_dense_chunk_candidates=lambda **kw: (len(rows), rows),
        _raw_rows_to_dense_candidates=lambda rows, **kw: rows,
        _rrf_merge_candidates=lambda groups, **kw: ranking.v13_merge_candidates(groups),
        _fts_search_chunks_prefix=lambda **kw: [], _fts_search_chunks_multi=lambda **kw: [],
        _v13_exact_identifier_candidates=lambda **kw: [],
        _ask_source_preference_profile=lambda q: {"strength": "none"},
        _count_query_tokens=lambda q: len(q.split()), _v13_build_profile_from_plan=lambda *a: {},
        _v13_fetch_scored_pages=lambda **kw: [], _v13_fetch_structured_dense_candidates=lambda **kw: [],
        _structured_rescue_query_intent=lambda *a: False,
        _v13_merge_candidates=ranking.v13_merge_candidates, _v13_score_candidates=score,
        _v13_evidence_metrics=lambda rows: {})
    return orchestration.V13InitialRetrievalRuntime(**values)


class SmartSourceSelectionTests(unittest.TestCase):
    def test_late_check_sources_survive_cover_flood_and_facet_metadata(self):
        covers = [row("cover"+str(n), "ZX400 general manual cover contents conformity declaration",
            owner="manual-cover", v13_score=999, assistant_core_facet_hits=["pump pressure"],
            assistant_core_diagnostic_priority={"score": 1}) for n in range(30)]
        check = row("step-check", "Osservare pressione pompa e segnale sensore senza azionare la macchina.", kind="step")
        legend = row("hmi-legend", "Pump pressure indication: green means available; gray means absent.")
        selected, meta = select(covers + [check, legend])
        self.assertEqual({r["citation_id"] for r in selected}, {"step-check", "hmi-legend"})
        self.assertEqual(meta["input_count"], 32)
        self.assertEqual(meta["added_model_calls"], 0)

    def test_common_subsystem_in_focused_corpus_is_not_erased(self):
        for count in (2, 3, 7):
            with self.subTest(count=count):
                selected, _ = select([row(str(i), "Pump pressure observation condition " + str(i)) for i in range(count)])
                self.assertEqual(len(selected), count)

    def test_focused_corpus_does_not_require_optional_router_annotations(self):
        for query in ("pressure loss", "pressure loss after service"):
            with self.subTest(query=query):
                selected, _ = select([row(str(i), "Pressure loss observation " + str(i)) for i in range(3)],
                    request(query), decision(diagnostic_subsystems=(), facet_queries=()))
                self.assertEqual(len(selected), 3)

    def test_rare_single_alarm_code_remains_evidence_without_cosine(self):
        selected, _ = select([row("alarm", "E749 indicates interrupted position confirmation.", semantic_similarity=0),
                             row("intro", "General machine introduction")],
                            request("E749"), decision(diagnostic_subsystems=(), facet_queries=()))
        self.assertEqual([r["citation_id"] for r in selected], ["alarm"])

    def test_split_model_identifier_is_still_one_clue(self):
        selected, _ = select([row("cover", "ZX-500/60T conformity introduction.", semantic_similarity=0.9),
                              row("check", "Pump pressure: observe the indicator before operation.")],
                             request("ZX-500/60T pump pressure alarm"))
        self.assertEqual([r["citation_id"] for r in selected], ["check"])

    def test_router_id_and_facet_tags_cannot_replace_source_body(self):
        cover = row("cover", "General introduction and conformity declaration.",
                    assistant_core_facet_hits=["pump pressure"], v13_score=1000)
        selected, _ = select([cover], dec=decision(relevant_evidence_ids=("cover",)))
        self.assertFalse(selected)

    def test_complete_text_has_priority_over_a_deceptive_snippet(self):
        good = row("good", "placeholder")
        good.pop("chunk_full")
        good.update(text="Pump pressure condition: read the indicator.", snippet="General introduction")
        bad = row("bad", "placeholder")
        bad.pop("chunk_full")
        bad.update(text="General introduction.", snippet="Pump pressure indicator")
        selected, _ = select([bad, good])
        self.assertEqual([r["citation_id"] for r in selected], ["good"])

    def test_same_prefix_different_complete_body_and_units_survive(self):
        prefix = "Pump pressure observation instructions. " * 20
        rows = [row("a", prefix+"Limit 3 Pa.", owner="manual"),
                row("b", prefix+"Limit 3 pA.", owner="manual")]
        selected, _ = select(rows)
        self.assertEqual(len(selected), 2)
        self.assertEqual({r["chunk_full"] for r in selected}, {r["chunk_full"] for r in rows})

    def test_late_foreign_record_and_duplicate_conflict_fail_before_cap(self):
        baseline = [row(str(i), "Pump pressure record " + str(i)) for i in range(30)]
        for extra in (row("foreign", "Unrelated", company_id="foreign"),
                      row("0", "Different full body for one identifier"),
                      row("foreign", "Unrelated", machine_id="foreign")):
            with self.subTest(extra=extra["citation_id"]):
                with self.assertRaises(smart_evidence.SmartEvidenceError):
                    select(baseline + [extra], max_items=1)

    def test_full_aliases_deduplicate_but_page_and_owner_do_not(self):
        a = row("a", "Pump pressure indication.", owner="manual")
        alias = {**a, "citation_id": "alias"}
        other_page = {**a, "citation_id": "page2", "page_from": 2, "page_to": 2}
        other_owner = {**a, "citation_id": "other", "bubble_document_id": "another"}
        selected, _ = select([a, alias, other_page, other_owner])
        self.assertEqual({r["citation_id"] for r in selected}, {"a", "page2", "other"})

    def test_bound_and_input_immutability(self):
        rows = [row(str(i), "Pump pressure condition " + str(i)) for i in range(20)]
        before = deepcopy(rows)
        selected, _ = select(rows, max_items=3)
        self.assertEqual(len(selected), 3)
        self.assertEqual(rows, before)

    def test_owner_body_conflicts_fail_before_smart_selection(self):
        original = row("same", "Pump pressure reading.")
        for changes in ({"company_id": "foreign"}, {"machine_id": "foreign"},
                        {"chunk_full": "Different body"}, {"bubble_document_id": "another"}):
            with self.subTest(changes=changes):
                with self.assertRaises(smart_evidence.SmartEvidenceError):
                    select([original, {**original, **changes}])

    def test_legacy_producer_projection_merge_remains_compatible(self):
        body = "Pump pressure observation and applicable conditions. " * 55
        views = [row("same", body[:n], exact_machine_scope=True) for n in (900, 2000, 2400)]
        # Dense/FTS are already SQL-scoped producer views, with differing lengths
        # and legacy binding projections. We preserve that existing boundary;
        # this test does NOT claim all pre-merge occurrences reach Smart.
        fts = {**views[0]}
        fts.pop("machine_id")
        fts.pop("exact_machine_scope")
        merged = ranking.v13_merge_candidates([views, [fts]])
        selected, _ = select(merged)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["chunk_full"], body[:2400])

    def test_smart_scorer_preserves_complete_occurrences_without_legacy_dedup(self):
        rows = [row("a", "Pump pressure reading."), row("b", "Pump pressure reading.")]
        result = ranking.v13_score_candidates("pressure", rows, runtime=scoring_runtime(), preserve_complete=True)
        self.assertEqual(len(result), 2)
        legacy = ranking.v13_score_candidates("pressure", rows,
            runtime=scoring_runtime(lambda rows, **kw: rows[:1]))
        self.assertEqual(len(legacy), 1)

    def test_initial_smart_mode_preserves_received_bodies_without_changing_other_modes(self):
        rows = [row("a", "Pump pressure observation first."), row("b", "Pump pressure observation second.")]
        for mode, expected in (("smart_neutral", 2), ("neutral", 1), ("ask", 1)):
            calls = []
            result = orchestration.v13_initial_retrieval(q="pump pressure", **SCOPE,
                doc_ids=None, bubble_document_id=None, response_language="en", mode=mode,
                runtime=initial_runtime(rows, calls))
            self.assertEqual(len(result["candidates"]), expected)
            self.assertEqual(calls, [("score", mode == "smart_neutral")])

    def test_title_merge_smart_opt_in_retains_default_behavior(self):
        rows = [row("a", "Pump pressure observation first."), row("b", "Pump pressure observation second.")]
        def score(q, data, **kwargs):
            return ranking.v13_score_candidates(q, data,
                runtime=scoring_runtime(lambda rows, **kw: rows[:1]), **kwargs)
        runtime = ranking.V13MergeSourceTitleCandidatesRuntime(8, 12,
            lambda rows: {}, ranking.v13_merge_candidates, score)
        for preserve, expected in ((False, 1), (True, 2)):
            result = ranking.v13_merge_source_title_candidates("pump pressure", {"candidates": rows[:1]},
                rows[1:], runtime=runtime, preserve_complete=preserve)
            self.assertEqual(len(result["candidates"]), expected)

    def test_prepare_smart_does_not_apply_causal_filter_or_extra_retrieval(self):
        values = {f.name: bomb if f.name.startswith("_") else f.name.lower()
                  for f in fields(orchestration.AssistantCorePrepareEvidenceRuntime)}
        values.update(MODE_SMART_DIAGNOSTIC="smart_diagnostic", EVIDENCE_SUPPORTED="supported",
            EVIDENCE_PARTIAL="partial", EVIDENCE_REFINE="refine", _content_term_set=terms,
            _assistant_core_retrieval_query=lambda req: req.query,
            _v13_candidate_text=lambda r: r["chunk_full"], _v13_evidence_metrics=lambda rows: {})
        runtime = orchestration.AssistantCorePrepareEvidenceRuntime(**values)
        data = {"candidates": [row("check", "Pump pressure: observe the panel indication.", kind="step")]}
        result = orchestration.assistant_core_prepare_evidence(request(), data, decision(), runtime=runtime)
        self.assertTrue(result["supported"])
        self.assertEqual(result["retrieval"]["citations"][0]["citation_id"], "check")
        for route in (decision(degraded=True), decision(evidence_state="unsupported")):
            self.assertFalse(orchestration.assistant_core_prepare_evidence(request(), data, route,
                                                                         runtime=runtime)["supported"])


if __name__ == "__main__":
    unittest.main()
