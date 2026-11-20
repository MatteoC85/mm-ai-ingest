"""Request-local immutable admission reuse preserves all current gates."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from machinemind.retrieval import ask_composition as comp
from machinemind.evidence.contracts import EvidenceContractError
import test_ask_prepared_scope_admission as prepared

class RetentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prepared.AskPreparedScopeAdmissionTests.setUpClass()
    @classmethod
    def tearDownClass(cls):
        prepared.AskPreparedScopeAdmissionTests.tearDownClass()
    def setUp(self):
        self.fixture = prepared.AskPreparedScopeAdmissionTests()
        args = self.fixture.harness()
        self.request, _, self.owner, self.data, _, self.session, _, _ = args
        self.handles = self.owner._selection[0].records
        self.allowance = self.session._nodes[0].bundle.assembly.manifest.allowed_sources
    def tearDown(self):
        self.fixture.finish(self.owner, self.session)
    def admit(self, name, *, rows=None, handles=None, allowance=None):
        return self.session.admission(request=self.request,
            retrieval={name: deepcopy(self.data['candidates']) if rows is None else rows},
            selections=(comp.AskSelection(name, self.handles if handles is None else handles),),
            current_allowed_sources=self.allowance if allowance is None else allowance)
    def test_alternating_selections_reuse_exact_immutable_envelope(self):
        with patch.object(comp, 'build_ask_evidence_input', wraps=comp.build_ask_evidence_input) as build:
            first = self.admit('candidates').evidence_input
            second = self.admit('citations').evidence_input
            self.assertIs(self.admit('candidates').evidence_input, first)
            self.assertIs(self.admit('citations').evidence_input, second)
            self.assertEqual(build.call_count, 2)
        self.assertLessEqual(self.session._bytes + sum(r[3].bundle.size_bytes for r in self.session._admission_cache),
                             self.session._limits.max_bytes)
    def test_cached_selection_checks_current_grants_handle_and_mutable_rows(self):
        self.admit('candidates'); self.admit('citations')
        for field, value in [('chunk_full', 'injected'), ('similarity', True), ('page_from', 99)]:
            rows = deepcopy(self.data['candidates']); rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(EvidenceContractError):
                self.admit('candidates', rows=rows)
        with self.assertRaises(EvidenceContractError): self.admit('candidates', allowance=frozenset())
        with self.assertRaises(EvidenceContractError):
            self.admit('candidates', handles=(comp.RecordHandle(object(), 0),))
        self.admit('candidates')
    def test_optional_retention_evicts_before_capacity_and_close(self):
        self.admit('candidates'); self.admit('citations')
        size = self.session._admission_cache[-1][3].bundle.size_bytes
        self.session._limits = replace(self.session._limits, max_bytes=self.session._bytes + size)
        self.admit('candidates')
        self.assertEqual(len(self.session._admission_cache), 1)
        with self.assertRaises(EvidenceContractError): self.session._check_capacity(0, size + 1)
        self.session.close(); self.assertEqual(self.session._admission_cache, [])
        with self.assertRaises(EvidenceContractError): self.admit('candidates')
    def test_envelopes_alone_never_exceed_remaining_retention_budget(self):
        rows = deepcopy(self.data['candidates'])
        largest = self.admit('candidates', rows=rows*3, handles=self.handles*3)
        self.session._limits = replace(self.session._limits,
            max_bytes=self.session._bytes + largest.evidence_input.bundle.size_bytes)
        retained = self.session._bytes
        for name in ('candidates', 'citations', 'candidates'):
            for count in (1, 2, 3):
                self.admit(name, rows=deepcopy(rows*count), handles=self.handles*count)
                self.assertEqual(self.session._bytes, retained)
                self.assertLessEqual(len(self.session._admission_cache), 4)
                self.assertLessEqual(retained + sum(row[3].bundle.size_bytes
                    for row in self.session._admission_cache), self.session._limits.max_bytes)


if __name__ == "__main__": unittest.main()
