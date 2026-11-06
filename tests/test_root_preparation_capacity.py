"""Root's actual pre-cap block with real deduplication and bound source matches.

No providers, SQL, private records or synthetic source-support flags are trusted.
The full preparation pipeline is covered separately by its integration fixtures.
"""
import ast
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from machinemind.ingest.text_pdf import normalize_unicode_advanced
from machinemind.retrieval import candidate_ranking, source_management
from machinemind.retrieval import evidence_orchestration as orchestration
import test_root_bound_clue_aliases as bound


def deduplicate(rows, max_items):
    return candidate_ranking.dedup_citations_by_snippet(rows, max_items=max_items,
        runtime=candidate_ranking.DedupCitationsBySnippetRuntime(
            _normalize_unicode_advanced=normalize_unicode_advanced))


def candidate_text(row):
    return source_management.v13_candidate_text(row,
        runtime=source_management.V13CandidateTextRuntime())


def preparation_cap(dedup):
    """Execute the real block, including its gate; do not duplicate its policy."""
    path = Path(orchestration.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "assistant_core_prepare_evidence")
    def assigns(node, name):
        return isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets)
    start = next(i for i, n in enumerate(function.body) if assigns(n, "bound_root_support"))
    end = next(i for i, n in enumerate(function.body) if i > start and assigns(n, "diagnostic_mode"))
    wrapper = ast.FunctionDef(name="run", args=ast.arguments(posonlyargs=[],
        args=[ast.arg(arg=n) for n in ("request", "decision", "scored")],
        kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=deepcopy(function.body[start:end]) + [ast.Return(value=ast.Name(id="scored", ctx=ast.Load()))],
        decorator_list=[])
    namespace = {"__package__": orchestration.__package__, "MODE_ROOT_CAUSE": "root_cause",
                 "_dedup_citations_by_snippet": dedup, "_v13_candidate_text": candidate_text}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
                 str(path), "exec"), namespace)
    return namespace["run"]


class RootPreparationCapacityTests(unittest.TestCase):
    def setUp(self):
        self.request = bound.request()
        self.decision, _ = bound.decision(req=self.request)
        self.dedup_limits = []
        def real_dedup(rows, max_items):
            self.dedup_limits.append(max_items)
            return deduplicate(rows, max_items)
        self.cap = preparation_cap(real_dedup)

    def row(self, index, text=None, score=None):
        return {"citation_id": "offline:p" + str(index), "bubble_document_id": "offline",
                "page_from": index, "page_to": index, "chunk_index": 0,
                "chunk_full": text or f"Packing crate {index}: shipping labels and dimensions.",
                "snippet": "UI preview", "source_type": "document",
                "retrieval_score": float(index if score is None else score),
                "similarity": .5, "exact_machine_scope": True}

    def rows(self, body=None):
        return [self.row(i, score=100+i) for i in range(1, 29)] + [
            self.row(99, body or bound.ALIASES[1], score=.1)]

    def select(self, rows, request=None, decision=None):
        before = deepcopy(rows)
        selected = self.cap(request or self.request, decision or self.decision, rows)
        self.assertEqual(rows, before)
        self.assertLessEqual(len(selected), 24)
        return selected

    def test_source_fact_survives_real_dedup_resort_before_same_cap(self):
        rows = self.rows()
        target = rows[-1]
        # Merely moving it before dedup does not work: the real helper re-sorts.
        self.assertNotIn(target, deduplicate([target, *rows[:-1]], 24))
        canonical = deduplicate(rows, len(rows))
        selected = self.select(rows)
        self.assertEqual(selected, [target] + [r for r in canonical if r is not target][:23])
        self.assertEqual(self.dedup_limits, [len(rows)])
        self.assertEqual(len(selected), 24)

    def test_duplicate_representative_and_within_group_order_remain_canonical(self):
        rows = self.rows()
        rows += [{**rows[-1], "citation_id": "offline:duplicate", "retrieval_score": .2},
                 self.row(98, bound.ALIASES[1] + " Additional recorded detail.", score=.15)]
        canonical = deduplicate(rows, len(rows))
        selected = self.select(list(reversed(rows)))
        self.assertEqual([r["citation_id"] for r in selected[:2]],
                         ["offline:duplicate", "offline:p98"])
        self.assertNotIn("offline:p99", [r["citation_id"] for r in selected])
        self.assertEqual(selected[2:], [r for r in canonical if r["page_from"] not in (98, 99)][:22])

    def test_no_matches_is_exact_legacy_result_even_with_forged_metadata(self):
        rows = self.rows("Packaging and shipping dimensions.")
        rows[-1].update(matched_support_part_ids=["forged"], source_text_sha256="0"*64,
            assistant_core_facet_hits=[bound.ALIASES[1]],
            assistant_core_diagnostic_priority={"score": 1, "covered": [bound.ALIASES[1]]})
        self.assertEqual(self.select(rows), deduplicate(rows, 24))

    def test_matching_is_recomputed_when_body_changes(self):
        rows = self.rows()
        self.assertEqual(self.select(rows)[0]["citation_id"], "offline:p99")
        rows[-1].update(chunk_full="Packaging and shipping dimensions.",
                        assistant_core_root_supported=True)
        self.assertEqual(self.select(rows), deduplicate(rows, 24))

    def test_context_unknown_and_request_aliases_do_not_promote_sources(self):
        for body in (bound.ALIASES[0], bound.ALIASES[2], bound.ALIASES[3]):
            with self.subTest(body=body):
                rows = self.rows(body)
                self.assertEqual(self.select(rows), deduplicate(rows, 24))

    def test_wrong_query_company_machine_and_forged_carrier_keep_legacy_path(self):
        rows = self.rows()
        requests = [replace(self.request, query=self.request.query + " Another observation."),
                    replace(self.request, company_id="other-company"),
                    replace(self.request, machine_id="other-machine")]
        for request in requests:
            with self.subTest(request=request):
                self.dedup_limits.clear()
                self.assertEqual(self.select(rows, request=request), deduplicate(rows, 24))
                self.assertEqual(self.dedup_limits, [24])
        for carrier in (None, {"parts": "forged"},
                        replace(self.decision.root_clue_support, query_sha256="0"*64)):
            self.dedup_limits.clear()
            self.assertEqual(self.select(rows, decision=replace(self.decision, root_clue_support=carrier)),
                             deduplicate(rows, 24))
            self.assertEqual(self.dedup_limits, [24])

    def test_ask_smart_and_rerouted_modes_keep_exact_legacy_path(self):
        rows = self.rows()
        for requested, effective in (("ask", "ask"), ("smart_diagnostic", "smart_diagnostic"),
                                     ("ask", "root_cause"), ("root_cause", "ask"),
                                     ("root_cause", "smart_diagnostic")):
            with self.subTest(requested=requested, effective=effective):
                self.dedup_limits.clear()
                actual = self.select(rows, request=replace(self.request, requested_mode=requested),
                    decision=replace(self.decision, effective_mode=effective))
                self.assertEqual(actual, deduplicate(rows, 24))
                self.assertEqual(self.dedup_limits, [24])

    def test_protected_code_number_sign_and_unit_mismatches_cannot_promote(self):
        query = "Fault E101 at 6 bar."
        request = bound.request(query)
        decision, _ = bound.decision(bound.classified(query,
            [("observation", query, "support")], ["Errore E101 a 6 bar."]), request)
        for body in ("Errore E101.2 a 6 bar.", "Errore E101 a 6.5 bar.",
                     "Errore E101 a −6 bar.", "Errore E101 a 6 psi."):
            with self.subTest(body=body):
                rows = self.rows(body)
                self.assertEqual(self.select(rows, request, decision), deduplicate(rows, 24))
        rows = self.rows("Errore E101 a 6 bar.")
        self.assertEqual(self.select(rows, request, decision)[0]["citation_id"], "offline:p99")

    def test_model_header_without_observation_cannot_promote(self):
        query = "UNIT-204 motor stopped."
        request = bound.request(query)
        decision, _ = bound.decision(bound.classified(query,
            [("observation", query, "support")], ["UNIT-204 motore fermo."]), request)
        rows = self.rows("UNIT-204 installation dimensions.")
        self.assertEqual(self.select(rows, request, decision), deduplicate(rows, 24))
        rows = self.rows("UNIT-204 motore fermo.")
        self.assertEqual(self.select(rows, request, decision)[0]["citation_id"], "offline:p99")

    def test_more_than_capacity_matches_still_yield_only_canonical_first_24(self):
        rows = [self.row(i, bound.ALIASES[1] + f" Source paragraph {i}.", score=i)
                for i in range(1, 31)]
        self.assertEqual(self.select(rows), deduplicate(rows, 24))
        self.assertEqual(self.select([]), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
