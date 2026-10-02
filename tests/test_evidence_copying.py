"""Exact detached-copy semantics at the ASK intake/binding boundaries; offline."""
import copy
from pathlib import Path
import struct
import sys
import unittest
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from machinemind.evidence.copying import deepcopy
from machinemind.evidence.ask_input import _same_value
from machinemind.ask import request_binding, core_intake
from machinemind.retrieval import receipt_producers
from machinemind.evidence.ask_input import (AskRequestKey, AskCollectionInput,
    AskEvidenceAdmission, build_ask_evidence_input, apply_ask_evidence_input)
from machinemind.evidence.contracts import EvidenceContractError
from test_lineage_copy_reuse import fixture


class EvidenceCopyTests(unittest.TestCase):
    def test_all_three_boundaries_use_same_detached_copier(self):
        for module in (request_binding, core_intake, receipt_producers):
            self.assertIs(module.deepcopy, deepcopy)

    def test_nested_aliases_are_preserved_but_input_is_detached(self):
        vector = [0.0, -0.0, 0.25, 0.5]
        shared = {'vector': vector, 'safety': ['isolate', {'hold': True}]}
        original = {'candidates': [shared, shared], 'citations': [shared], 'vector': vector}
        result = deepcopy(original)
        self.assertTrue(_same_value(result, copy.deepcopy(original)))
        self.assertIs(result['candidates'][0], result['citations'][0])
        self.assertIs(result['candidates'][0], result['candidates'][1])
        self.assertIs(result['vector'], result['citations'][0]['vector'])
        result['vector'][1] = 9.0
        result['citations'][0]['safety'][1]['hold'] = False
        self.assertEqual(struct.pack('d', vector[1]), struct.pack('d', -0.0))
        self.assertTrue(shared['safety'][1]['hold'])

    def test_cycles_and_tuple_backreference_match_stdlib(self):
        rows = []
        root = (rows,)
        rows.extend([root, rows])
        result = deepcopy(root)
        self.assertIs(result[0][0], result)
        self.assertIs(result[0][1], result[0])
        self.assertIsNot(result[0], rows)
        mapping = {}
        mapping['self'] = mapping
        result = deepcopy(mapping)
        self.assertIs(result['self'], result)

    def test_explicit_memo_overrides_root_container_and_float_components(self):
        scalar = float('0.123456789')
        row = [scalar, scalar]
        marker = object()
        self.assertIs(deepcopy(row, {id(row): marker}), marker)
        result = deepcopy(row, {id(scalar): marker})
        self.assertEqual(result, copy.deepcopy(row, {id(scalar): marker}))
        self.assertTrue(all(x is marker for x in result))
        inner = {'x': 1}
        self.assertEqual(deepcopy([inner, inner], {id(inner): marker}), [marker, marker])

    def test_unknown_types_and_float_subclasses_keep_hooks(self):
        class Scalar(float):
            def __deepcopy__(self, memo):
                return 'custom-float'
        class Object:
            def __deepcopy__(self, memo):
                result = {'owner': 'hook'}
                memo[id(self)] = result
                return result
        obj = Object()
        result = deepcopy([obj, obj, Scalar(0.5)])
        self.assertIs(result[0], result[1])
        self.assertEqual(result[2], 'custom-float')

    def test_dict_hooks_keep_value_before_key_order(self):
        seen = []
        class Hook:
            def __init__(self, label): self.label = label
            def __deepcopy__(self, memo):
                seen.append(self.label)
                return self
        original = {Hook('key'): Hook('value')}
        deepcopy(original)
        actual = seen[:]
        seen.clear()
        copy.deepcopy(original)
        self.assertEqual(actual, seen)
        self.assertEqual(actual, ['value', 'key'])

    def test_nonstandard_memo_delegates_to_stdlib(self):
        class Memo(dict): pass
        value = [1.5, {'x': ['data']}]
        self.assertEqual(deepcopy(value, Memo()), copy.deepcopy(value, Memo()))

    def test_copied_boundary_still_rejects_revocation_tenant_and_deep_mutation(self):
        trace, rows = fixture(2, vector_size=4)
        inputs = tuple(trace.roots.values())
        key = AskRequestKey('Operation?', 'offline-company', 'offline-machine',
                            'machine_all', 'en', 6, False, (), None)
        envelope = build_ask_evidence_input(request_key=key,
            collections=(AskCollectionInput('candidates', inputs, 'list'),),
            allowed_sources=inputs[0].context.allowed_sources,
            adapter_limits=trace.limits.adapter, assembly_limits=trace.limits.assembly,
            legacy_limits=trace.limits.legacy)
        admission = AskEvidenceAdmission(envelope, inputs[0].context.allowed_sources,
                                         trace.limits.adapter)
        packet = request_binding.deepcopy({'candidates': rows})
        out = apply_ask_evidence_input(packet, request_key=key, admission=admission)
        self.assertTrue(_same_value(out, packet))
        with self.assertRaises(EvidenceContractError):
            apply_ask_evidence_input(packet, request_key=key,
                admission=replace(admission, current_allowed_sources=frozenset()))
        with self.assertRaises(EvidenceContractError):
            apply_ask_evidence_input(packet, request_key=replace(key, company_id='another-company'),
                                     admission=admission)
        packet['candidates'][0]['embedding'][1] = -0.0
        with self.assertRaises(EvidenceContractError):
            apply_ask_evidence_input(packet, request_key=key, admission=admission)


if __name__ == '__main__':
    unittest.main()
