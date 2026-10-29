"""Offline operation anchors: inflection recall without widening named locations."""
from contextlib import ExitStack
from copy import deepcopy
import importlib
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import test_compound_procedure_selection as fixtures


def denied(*a, **k):
    raise AssertionError("OFFLINE_EXTERNAL_IO_DENIED")


class ProcedureInflectionAnchorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = ExitStack()
        cls.guard.enter_context(patch.dict(os.environ, {"MM_USAGE_ENFORCEMENT": "off",
            "AI_INTERNAL_SECRET": "offline-fixture-only", "OPENAI_API_KEY": "offline-unused",
            "MM_INGEST_LEDGER_AUTO_DDL": "0"}))
        for name in ("socket.create_connection", "socket.socket.connect",
                     "requests.sessions.Session.request", "psycopg2.connect"):
            cls.guard.enter_context(patch(name, denied))
        cls.m = importlib.import_module("main")

    @classmethod
    def tearDownClass(cls):
        cls.guard.close()

    def setUp(self):
        self.rows = [
            fixtures.step(1, "Isolare il sistema", "Prima del caricamento isolare le energie."),
            fixtures.step(2, "Pulizia esterna", "Pulire il pannello esterno."),
            fixtures.step(3, "Collegamento alimentatore", "Collegare il connettore alimentatore."),
            fixtures.step(4, "Percorso alimentatore", "Instradare il materiale sull'alimentatore."),
            fixtures.step(5, "Allineamento ingresso", "Allineare il materiale verso l'ingresso."),
            fixtures.step(6, "Caricare il materiale", "Spingere il materiale nel canale interno."),
            fixtures.step(7, "Richiudere la guida", "Richiudere la leva della guida."),
            fixtures.step(8, "Avvio produzione", "Avviare il ciclo automatico."),
            fixtures.step(9, "Calibrazione ottica", "Calibrare il sensore ottico."),
        ]
        self.q = "come si carica il consumabile"
        self.planner = {"information_task": "procedure_segment", "required_answer_types": ["ordered_actions", "safety_conditions"],
            "facet_queries": [fixtures.facet("Instradare il materiale sull'alimentatore", "ordered_actions"),
                              fixtures.facet("Allineare il materiale verso l'ingresso", "ordered_actions")]}

    def select(self, q=None, planner=None):
        result = self.m._v12_select_response_steps(all_steps=self.rows, selected_step_ids=[],
            model_used_citations=[], q=self.q if q is None else q, planner=self.planner if planner is None else planner)
        return [self.m._v12_step_sort_key(s)[0] for s in result]

    def legacy(self, q=None, planner=None):
        with patch.object(self.m, "_v12_operation_inflection_scores", side_effect=lambda steps, q: [0.0] * len(steps)):
            return self.select(q, planner)

    def test_nonempty_preparation_prefix_no_longer_erases_inflected_operation(self):
        self.assertEqual(self.legacy(), [4, 5])
        self.assertEqual(self.select(), [4, 5, 6, 7])
        self.assertNotIn(2, self.select())
        self.assertNotIn(8, self.select())

    def test_safety_prerequisite_body_does_not_anchor_unrelated_earlier_steps(self):
        scores = self.m._v12_operation_inflection_scores(self.rows, self.q)
        self.assertEqual(scores[0], 0)
        self.assertGreater(scores[5], 0.055)
        self.assertTrue(all(score == 0 for i, score in enumerate(scores) if i != 5))

    def test_known_location_qualifier_blocks_similar_internal_operation(self):
        q = "come si carica il materiale sull'alimentatore"
        scores = self.m._v12_operation_inflection_scores(self.rows, q)
        self.assertEqual(scores[5], 0)
        planner = {"information_task": "procedure_segment", "required_answer_types": ["ordered_actions"]}
        self.assertEqual(self.select(q, planner), self.legacy(q, planner))
        self.assertNotIn(6, self.select(q, planner))

    def test_narrow_entry_and_restoration_queries_do_not_expand(self):
        planner = {"information_task": "procedure_segment", "required_answer_types": ["ordered_actions"]}
        for q in ("Allineare il materiale verso l'ingresso", "Richiudere la leva della guida"):
            with self.subTest(q=q):
                self.assertEqual(self.select(q, planner), self.legacy(q, planner))
                self.assertNotIn(8, self.select(q, planner))

    def test_explicit_requested_range_never_uses_implicit_inflection_recall(self):
        q = "from Percorso alimentatore to Allineamento ingresso"
        expected = self.legacy(q)
        with patch.object(self.m, "_v12_operation_inflection_scores", side_effect=AssertionError("implicit recall used")):
            self.assertEqual(self.select(q), expected)
        self.assertNotIn(6, expected)

    def test_existing_comparator_handles_english_inflection(self):
        rows = [fixtures.step(1, "Secure the latch", "Engage the retaining lever."),
                fixtures.step(2, "External cleaning", "Wipe the enclosure.")]
        scores = self.m._v12_operation_inflection_scores(rows, "How is it secured?")
        self.assertGreater(scores[0], 0.055)
        self.assertEqual(scores[1], 0)

    def test_shared_roots_short_words_numbers_and_identifiers_are_not_new_anchors(self):
        for query, title in (("pressa", "pressione"), ("stampo", "stampaggio"),
                             ("ciclo", "cicloturismo"), ("wire", "wireless"),
                             ("motor123", "motor1234"), ("12345", "123456"),
                             ("cambio", "cambiare")):
            with self.subTest(query=query, title=title):
                rows = [fixtures.step(1, title, "Execute the documented task.")]
                self.assertEqual(self.m._v12_operation_inflection_scores(rows, query), [0.0])

    def test_sparse_checklists_and_source_records_are_unchanged(self):
        before = deepcopy(self.rows)
        planner = {"information_task": "procedure_segment", "required_answer_types": ["checklist", "safety_conditions"]}
        expected = self.legacy("Which checks apply?", planner)
        with patch.object(self.m, "_v12_operation_inflection_scores", side_effect=AssertionError("implicit recall used")):
            self.assertEqual(self.select("Which checks apply?", planner), expected)
        self.select()
        self.assertEqual(self.rows, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
