"""Semantic factual checklist admission, independent of lexical list triggers."""
from dataclasses import replace
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unicodedata
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assistant_core_v2 import AssistantCoreDecision, AssistantCoreRequest
from machinemind.retrieval.context_expansion import (
    AssistantCoreEnumerationRequestedRuntime, assistant_core_enumeration_requested,
)


class SemanticChecklistEnumerationTests(unittest.TestCase):
    def setUp(self):
        self.request = AssistantCoreRequest(
            query='Documented intervals and locations.', requested_mode='ask',
            response_language='en', company_id='company-a', machine_id='machine-a',
            ai_scope='machine_all', top_k=6)
        self.decision = AssistantCoreDecision(
            request_kind='factual', effective_mode='ask', confidence=.9,
            requested_mode_fit=True, evidence_state='supported',
            evidence_policy='machine_sources_required',
            information_task='document_explanation',
            required_answer_types=('checklist', 'explanation'))
        self.runtime = AssistantCoreEnumerationRequestedRuntime(
            lambda text: unicodedata.normalize('NFKC', text), re)

    def check(self, request=None, decision=None):
        return assistant_core_enumeration_requested(
            request or self.request, decision or self.decision, runtime=self.runtime)

    def test_supported_factual_checklist_does_not_require_list_keywords(self):
        cases = (
            # Actual failure shape, with generic question and no tenant/media IDs.
            ('it', 'Quali punti di lubrificazione e frequenze di manutenzione sono documentati per questa macchina? Se mancano dati dichiaralo.'),
            ('en', 'Give documented service frequencies and lubrication locations.'),
            ('de', 'Nenne dokumentierte Schmierstellen und Wartungsintervalle.'),
        )
        for language, query in cases:
            with self.subTest(language=language):
                request = replace(self.request, query=query, response_language=language)
                self.assertTrue(self.check(request))
                self.assertEqual(request.query, query)

    def test_only_supported_factual_information_tasks_opt_in(self):
        for task in ('document_explanation', 'comparison', 'numeric_specification'):
            for state in ('supported', 'partial'):
                with self.subTest(task=task, state=state):
                    self.assertTrue(self.check(decision=replace(
                        self.decision, information_task=task, evidence_state=state)))
        self.assertFalse(self.check(decision=replace(
            self.decision, required_answer_types=('explanation',))))
        for state in ('refine', 'unsupported', 'clarify'):
            with self.subTest(state=state):
                self.assertFalse(self.check(decision=replace(self.decision, evidence_state=state)))

    def test_other_modes_kinds_and_tasks_retain_legacy_false(self):
        for mode in ('root_cause', 'smart_diagnostic'):
            with self.subTest(requested_mode=mode):
                self.assertFalse(self.check(request=replace(self.request, requested_mode=mode)))
            with self.subTest(effective_mode=mode):
                self.assertFalse(self.check(decision=replace(self.decision, effective_mode=mode)))
        for kind in ('procedure', 'comparison', 'source_retrieval', 'fault_diagnostic',
                     'guided_diagnostic', 'general_technical', 'ambiguous',
                     'out_of_scope', 'unsafe_request'):
            with self.subTest(kind=kind):
                self.assertFalse(self.check(decision=replace(self.decision, request_kind=kind)))
        for task in ('procedure_full', 'procedure_segment', 'fault_diagnostic',
                     'sequence_or_synchronization', 'interface_navigation',
                     'source_retrieval', 'general_technical', 'out_of_scope', 'other'):
            with self.subTest(task=task):
                self.assertFalse(self.check(decision=replace(self.decision, information_task=task)))
        self.assertFalse(self.check(decision=replace(self.decision,
            information_task='procedure_segment', procedure_scope='conditions_only')))

    def test_existing_lexical_positive_is_not_suppressed_for_other_consumers(self):
        for mode in ('ask', 'root_cause', 'smart_diagnostic'):
            with self.subTest(mode=mode):
                self.assertTrue(self.check(
                    request=replace(self.request, requested_mode=mode, query='Which control types are documented?'),
                    decision=replace(self.decision, effective_mode=mode,
                                     information_task='other', required_answer_types=())))
        # Existing facet hint and abbreviated legacy objects remain compatible.
        self.assertTrue(self.check(
            request=SimpleNamespace(query='Which are available?'),
            decision=SimpleNamespace(required_facets=('control modes',))))
        self.assertFalse(self.check(
            request=SimpleNamespace(query='Documented values.'),
            decision=SimpleNamespace(required_facets=())))

    def test_constrained_checklist_keeps_original_scope_and_contract(self):
        query = 'Give only two documented examples; exclude electrical work.'
        request = replace(self.request, query=query)
        decision = replace(self.decision, required_facets=(
            'Two examples only, excluding electrical work',))
        before = (request, decision)
        self.assertTrue(self.check(request, decision))
        self.assertEqual((request, decision), before)
        self.assertEqual(request.query, query)
        # The signal enables an inventory answering this constrained question;
        # it does not expand the required facets or invent a universal item set.


if __name__ == '__main__':
    unittest.main()
