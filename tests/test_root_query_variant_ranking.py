"""Offline Root ranking: query variants are hints, never admission or source text."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from assistant_core_v2 import AssistantCoreDecision, AssistantCoreFacetQuery, AssistantCoreRequest
from machinemind.retrieval import evidence_orchestration as orchestration
import test_ask_bounded_acquisition as harness


class RootQueryVariantRankingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        harness.AskBoundedAcquisitionTests.setUpClass()
        cls.m = harness.AskBoundedAcquisitionTests.m

    @classmethod
    def tearDownClass(cls):
        harness.AskBoundedAcquisitionTests.tearDownClass()

    def fixture(self, language="it"):
        alternatives = {
            "it": ("carrello avanzamento materiale fermo", "carrello avanzamento materiale", "carrello avanzamento materiale fermo"),
            "en": ("feed carriage stock stationary", "feed carriage stock", "feed carriage stock stationary"),
            "es": ("carro avance material inmovil", "carro avance material", "carro avance material inmovil"),
        }
        alternative, generic, specific = alternatives[language]
        facet = "Observed feed carriage movement while stock remains stationary"
        request = AssistantCoreRequest(query=facet, requested_mode="root_cause",
            response_language="en", company_id="offline-company", machine_id="offline-machine",
            ai_scope="machine_all", top_k=8)
        decision = AssistantCoreDecision(request_kind="fault_diagnostic", effective_mode="root_cause",
            confidence=.95, requested_mode_fit=True, evidence_state="refine", evidence_policy="machine_sources_required",
            information_task="fault_diagnostic", preferred_source_types=("document",),
            required_answer_types=("diagnostic_causes",), required_facets=(facet,),
            diagnostic_observables=(facet,), diagnostic_subsystems=("material feed",),
            facet_queries=(AssistantCoreFacetQuery(facet=facet, answer_type="diagnostic_causes",
                dense_queries=(facet,), lexical_queries=(alternative,), exact_terms=("UNIT-204",),
                preferred_source_types=("document",)),))
        rows = []
        for index in range(4):
            text = "UNIT-204. " + (specific if index == 3 else generic) + ". Documented mechanism."
            rows.append({"citation_id":f"offline-manual:p{index+1}:c0", "bubble_document_id":"offline-manual",
                "source_type":"document", "page_from":index+1,"page_to":index+1,
                "chunk_full":text,"snippet":text,"exact_machine_scope":True,
                "semantic_similarity":.8,"similarity":.8,"v13_score":1.0-index*.02,
                "causal_strength_score":.25,"subsystem_score":.2,"context_fit_score":.1,
                "role_group":"core","matched_subsystems":["material_feed"],
                "assistant_core_facet_hits":[facet]})
        return request, decision, rows

    def prepare(self, request, decision, rows):
        captured=[]
        runtime=self.m._assistant_core_prepare_evidence_runtime()
        def select(candidates, **kwargs):
            captured.extend(candidates)
            return runtime._assistant_core_root_source_selection(candidates, **kwargs)
        # Fixtures are already scope-admitted and Root-rescored. Exercise the real
        # final ranking, viability, source caps and selector without remote recall.
        local=replace(runtime,
            _assistant_core_root_diagnostic_evidence_assurance=lambda r,p,d:p,
            _v13_current_budget=lambda:None,
            _v13_rescore_root_candidates=lambda q,c:c,
            _assistant_core_root_source_selection=select)
        result=orchestration.assistant_core_prepare_evidence(request,{"candidates":rows},decision,runtime=local)
        return result,captured

    def test_source_language_variants_survive_real_viability_and_source_cap(self):
        for language in ("it","en","es"):
            with self.subTest(language=language):
                request,decision,rows=self.fixture(language)
                result,seen=self.prepare(request,decision,rows)
                selected=result["retrieval"]["candidates"]
                self.assertTrue(result["supported"])
                # Source-only diagnostics may prune broad peers previously
                # corroborated only by their query facet. Keep the source cap,
                # not a requirement to fill it with weaker excerpts.
                self.assertGreaterEqual(len(selected),1)
                self.assertLessEqual(len(selected),3)
                self.assertEqual(selected[0]["page_from"],4)
                self.assertTrue(all(c["assistant_core_root_viable"] for c in selected))
                self.assertEqual({c["citation_id"] for c in seen},{c["citation_id"] for c in rows})

    def test_tags_and_common_identifier_cannot_fabricate_query_support(self):
        request,decision,rows=self.fixture()
        rows[0]["chunk_full"]=rows[0]["snippet"]="UNIT-204. Installation dimensions and packaging."
        rows[0]["assistant_core_facet_hits"]=[decision.required_facets[0],*decision.facet_queries[0].lexical_queries]
        result,seen=self.prepare(request,decision,rows)
        source=next(c for c in seen if c["page_from"]==1)
        self.assertEqual(source["assistant_core_root_query_support"],0.0)
        self.assertNotEqual(result["retrieval"]["candidates"][0]["page_from"],1)

    def test_matching_source_cannot_bypass_existing_hard_exclusion(self):
        request,decision,rows=self.fixture()
        rows[3]["hard_excluded"]=True
        result,seen=self.prepare(request,decision,rows)
        self.assertNotIn(4,[c["page_from"] for c in seen])
        self.assertNotIn(4,[c["page_from"] for c in result["retrieval"]["candidates"]])

    def test_matching_words_without_causal_evidence_cannot_enter_selector(self):
        request,decision,rows=self.fixture()
        for row in rows:
            row.update(causal_strength_score=0,subsystem_score=0,context_fit_score=0,
                semantic_similarity=0,similarity=0,assistant_core_facet_hits=[],matched_subsystems=[])
        result,seen=self.prepare(request,decision,rows)
        self.assertEqual(seen,[])
        self.assertFalse(result["supported"])

    def test_model_code_only_variant_has_no_ranking_bonus(self):
        request,decision,rows=self.fixture()
        decision=replace(decision,facet_queries=(replace(decision.facet_queries[0],
            dense_queries=("UNIT-204",),lexical_queries=("UNIT 204",)),))
        _,seen=self.prepare(request,decision,rows)
        self.assertTrue(all(c["assistant_core_root_query_support"]==0 for c in seen))

    def test_distinctive_fault_code_is_preserved_against_neighbouring_wrong_code(self):
        request,decision,rows=self.fixture()
        decision=replace(decision,facet_queries=(replace(decision.facet_queries[0],
            dense_queries=("UNIT-204 fault E101 motor locked",),lexical_queries=(),
            exact_terms=("UNIT-204","E101")),))
        for row in rows:
            text="UNIT-204 fault "+("E101" if row["page_from"]==4 else "E102")+" motor locked."
            row["chunk_full"]=row["snippet"]=text
        result,seen=self.prepare(request,decision,rows)
        scores={row["page_from"]:row["assistant_core_root_query_support"] for row in seen}
        self.assertGreater(scores[4],scores[1])
        self.assertEqual(result["retrieval"]["candidates"][0]["page_from"],4)

    def test_distinctive_numeric_setpoint_is_not_treated_as_a_scope_identifier(self):
        request,decision,rows=self.fixture()
        decision=replace(decision,facet_queries=(replace(decision.facet_queries[0],
            dense_queries=("pressure 6bar motor stopped",),lexical_queries=(),exact_terms=("6bar",)),))
        for row in rows:
            text="Pressure "+("6bar" if row["page_from"]==4 else "8bar")+" motor stopped."
            row["chunk_full"]=row["snippet"]=text
        result,seen=self.prepare(request,decision,rows)
        scores={row["page_from"]:row["assistant_core_root_query_support"] for row in seen}
        self.assertGreater(scores[4],scores[1])
        self.assertEqual(result["retrieval"]["candidates"][0]["page_from"],4)

    def test_ask_keeps_its_existing_candidates_without_root_annotation(self):
        request,decision,rows=self.fixture()
        request=replace(request,requested_mode="ask")
        decision=replace(decision,effective_mode="ask",request_kind="factual",information_task="explanation",
            required_answer_types=("explanation",),diagnostic_observables=(),diagnostic_subsystems=())
        result,_=self.prepare(request,decision,rows)
        self.assertTrue(result["supported"])
        self.assertTrue(all("assistant_core_root_query_support" not in c for c in result["retrieval"]["candidates"]))


if __name__ == "__main__":
    unittest.main()
