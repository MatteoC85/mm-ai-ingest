"""Actual emitted ownership constraints, with independent canonical gates intact."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from machinemind.ask import factual_context_review as review
from context_review_wire_fixture import encode_context_reply, original_context_sources
import test_duplicate_table_context as fx


class ContextOwnerBoundWireTests(unittest.TestCase):
    def fixture(self, bodies=None):
        first, frame, sources, _ = fx.candidate(bodies)
        messages = review.build_messages(frame, question='Which intervals?', response_language='en', sources_block=sources)
        canonical = fx.verdict(frame)
        canonical['reason'] = 'accepted'
        wire = encode_context_reply(canonical, messages, original_sources=sources, json_schema=review.schema(frame))
        return first, frame, sources, messages, canonical, wire

    def test_emitted_schema_requires_every_owner_and_excludes_cell_only_or_other_owner(self):
        first, frame, sources, messages, canonical, wire = self.fixture()
        full_schema = review.schema(frame)['schema']
        schema = full_schema['properties']['table_contexts']
        refs = review._atomic_references(frame)
        targets = review._context_targets(frame)
        self.assertEqual(schema['type'], 'object')
        self.assertFalse(schema['additionalProperties'])
        self.assertEqual(schema['required'], [target for target, _, _, _ in targets])
        for target, _, owner, _ in targets:
            slot = schema['properties'][target]['anyOf'][0]
            self.assertEqual(set(slot['required']), set(slot['properties']))
            choices = slot['properties']['owning_source_quote']['enum']
            self.assertTrue(choices)
            prose = review._documentary_prose(dict(frame.source_bodies)[owner])
            self.assertTrue(all(refs[ref][0] == owner and refs[ref][1] in prose for ref in choices))
            self.assertFalse(any(refs[ref][1] == 'Oil temperature' for ref in choices))
            supplement = slot['properties']['supplementary_source_quotes']
            self.assertEqual(supplement['maxItems'], 1)
            self.assertEqual(supplement['items'], {'$ref': '#/$defs/source_quote_reference'})
            self.assertEqual(set(full_schema['$defs']['source_quote_reference']['enum']), set(refs))
        out = review.apply_wire_review(first, wire, frame)
        self.assertEqual(out, review.apply_review(first, canonical, frame))
        self.assertEqual(out['expected_list_items'], first['expected_list_items'])
        self.assertEqual(out['table_record_coverage'], first['table_record_coverage'])
        emitted = messages[1]['content'].split('SOURCES:\n', 1)[1].split('\n\nCOMPLETE_CANDIDATE:', 1)[0]
        self.assertEqual(original_context_sources(emitted, sources), sources)
        self.assertEqual(out['independent_context_review']['answer_sha256'], hashlib.sha256(out['answer'].encode()).hexdigest())

    def test_both_original_table_owner_failures_stay_rejected_even_if_provider_ignores_schema(self):
        first, frame, _, _, _, good = self.fixture()
        refs = review._atomic_references(frame)
        target, _, owner, _ = review._context_targets(frame)[1]
        invalid = [next(ref for ref, (cid, literal) in refs.items() if cid != owner),
                   next(ref for ref, (cid, literal) in refs.items() if cid == owner and literal == 'Oil temperature')]
        choices = review.schema(frame)['schema']['properties']['table_contexts']['properties'][target]['anyOf'][0]['properties']['owning_source_quote']['enum']
        for selected in invalid:
            self.assertNotIn(selected, choices)
            bad = deepcopy(good)
            bad['table_contexts'][target]['owning_source_quote'] = selected
            # Supplying the correct quote as supplementary cannot repair an
            # invalid explicit primary selection.
            bad['table_contexts'][target]['supplementary_source_quotes'] = [good['table_contexts'][target]['owning_source_quote']]
            result = review.apply_wire_review(first, bad, frame)
            self.assertEqual(result['independent_context_review']['gate'], 'table_owner')
            self.assertEqual(result['answer'], '')

    def test_bound_user_dedup_preserves_schema_source_candidate_and_exact_owner_choices(self):
        for label, qualifier in [('lathe-v', 'Only for the documented auxiliary spindle.'),
                                 ('pompe-z', 'Applicable uniquement au groupe auxiliaire documenté.')]:
            bodies={label+':p1':fx.SCOPE+'\n'+fx.CELLS,
                    label+':p2':qualifier+'\n'+fx.CELLS}
            first,frame,sources,_=fx.candidate(bodies)
            before=deepcopy(review.schema(frame))
            messages=review.build_messages(frame,question='Which interval?',response_language='en',sources_block=sources)
            self.assertEqual(review.schema(frame),before)
            user=messages[1]['content']
            targets=json.loads(user.split('TABLE_CONTEXT_TARGETS:\n',1)[1].split('\n\n',1)[0])
            candidate=json.loads(user.split('COMPLETE_CANDIDATE:\n',1)[1].split('\n\n',1)[0])
            self.assertEqual([x['text']for x in candidate['inventory']],list(frame.items))
            emitted=user.split('SOURCES:\n',1)[1].split('\n\nCOMPLETE_CANDIDATE:',1)[0]
            self.assertEqual(original_context_sources(emitted,sources),sources)
            refs=review._atomic_references(frame)
            slots=before['schema']['properties']['table_contexts']['properties']
            self.assertEqual({x['target']for x in targets},set(slots))
            for target in targets:
                self.assertNotIn('owner_prose_refs',target)
                self.assertNotIn('owning_source_refs',target)
                owner=review._context_targets(frame)[[x[0]for x in review._context_targets(frame)].index(target['target'])][2]
                choices=slots[target['target']]['anyOf'][0]['properties']['owning_source_quote']['enum']
                expected=review._owner_prose_refs(frame,owner,refs)
                self.assertEqual(choices,expected)
                self.assertTrue(all(refs[ref][0]==owner for ref in choices))

    def test_missing_extra_null_and_legacy_slots_cannot_approve_and_no_sources_needs_no_invention(self):
        first, frame, _, _, _, good = self.fixture()
        first_target = next(iter(good['table_contexts']))
        mutations = [lambda r: r['table_contexts'].pop(first_target),
            lambda r: r['table_contexts'].update(foreign=deepcopy(r['table_contexts'][first_target])),
            lambda r: r['table_contexts'].update({first_target: None}),
            lambda r: r.update(table_contexts=[]),
            lambda r: r['table_contexts'][first_target].update(supplementary_source_quotes=['0:1', '0:2'])]
        for mutate in mutations:
            bad = deepcopy(good); mutate(bad)
            self.assertEqual(review.apply_wire_review(first, bad, frame)['outcome'], 'no_sources')
        rejected = deepcopy(good)
        rejected.update(outcome='no_sources', table_contexts={target: None for target in good['table_contexts']})
        self.assertEqual(review.apply_wire_review(first, rejected, frame)['independent_context_review']['gate'], 'model_rejected')

    def test_semantics_documentary_limits_and_frame_binding_are_not_schema_approvals(self):
        first, frame, _, _, _, good = self.fixture()
        for check in review.CHECKS:
            bad = deepcopy(good); bad['checks'][check] = False
            self.assertEqual(review.apply_wire_review(first, bad, frame)['independent_context_review']['gate'], 'checks_false')
        bad = deepcopy(good)
        bad['table_contexts'][next(iter(bad['table_contexts']))]['documentary_limits_checked'] = False
        self.assertEqual(review.apply_wire_review(first, bad, frame)['independent_context_review']['gate'], 'documentary_limits_unchecked')
        changed = deepcopy(first); changed['answer'] += ' Changed'
        self.assertEqual(review.apply_wire_review(changed, good, frame)['independent_context_review']['gate'], 'frame_mismatch')

    def test_distinct_documents_and_translated_prose_keep_separate_contexts(self):
        for prefix, sentence in [('pump-z', 'Use these intervals only for the auxiliary pump.'),
                                 ('press-x', 'Ces intervalles concernent seulement la presse auxiliaire.')]:
            bodies = {prefix + ':p7': fx.SCOPE + '\n' + fx.CELLS,
                      prefix + ':p12': sentence + '\n' + fx.CELLS}
            first, frame, sources, messages, canonical, _ = self.fixture(bodies)
            canonical['table_contexts'][1]['context'] = sentence
            canonical['table_contexts'][1]['source_quotes'] = [{'citation_id': prefix + ':p12', 'quote': sentence}]
            wire = encode_context_reply(canonical, messages, original_sources=sources, json_schema=review.schema(frame))
            out = review.apply_wire_review(first, wire, frame)
            self.assertEqual(out['outcome'], 'rewrite')
            self.assertIn(sentence, out['answer'])
            self.assertEqual(len(wire['table_contexts']), 2)
            self.assertEqual(out['answer'].count(first['expected_list_items'][0]), 1)

    def test_table_only_owner_uses_its_literal_units_without_inventing_prose(self):
        first, frame, sources, _, = fx.candidate({'table-only:p8': fx.CELLS})
        refs = review._atomic_references(frame)
        choices = review._bound_context_references(frame, refs)['item:0']
        self.assertEqual(set(choices), {ref for ref, (cid, _) in refs.items() if cid == 'table-only:p8'})
        self.assertIsNone(review._owner_prose_refs(frame, 'table-only:p8', refs))

    def test_schema_capacity_fallback_preserves_prior_complete_wire_and_all_semantic_gates(self):
        first, frame, sources, _, canonical, _ = self.fixture()
        # Repeated context owners force repeated enums beyond the strict limit;
        # source text and the complete atomic reference universe remain intact.
        oversized = replace(frame, context_owners=tuple(
            ('item:0:context:' + str(i), 0, frame.table_owners[0][1], (frame.table_owners[0][1],))
            for i in range(80)))
        refs = review._atomic_references(oversized)
        self.assertIsNotNone(refs)
        self.assertIsNone(review._bound_context_references(oversized, refs))
        self.assertEqual(review.schema(oversized)['schema']['properties']['table_contexts']['type'], 'array')
        messages = review.build_messages(oversized, question='q', response_language='en', sources_block=sources)
        targets=json.loads(messages[1]['content'].split('TABLE_CONTEXT_TARGETS:\n',1)[1].split('\n\n',1)[0])
        self.assertEqual(len(targets),80)
        for target in targets:
            self.assertTrue(target['owner_prose_refs'])
            self.assertNotIn('owning_source_refs',target)
        emitted = messages[1]['content'].split('SOURCES:\n', 1)[1].split('\n\nCOMPLETE_CANDIDATE:', 1)[0]
        self.assertEqual(original_context_sources(emitted, sources), sources)
        short = encode_context_reply(canonical, messages, original_sources=sources, json_schema=review.schema(oversized))
        self.assertEqual(review.apply_wire_review(first, short, oversized)['independent_context_review']['gate'], 'table_count')

    def test_472_real_literal_pairs_and_three_contexts_use_one_complete_supplementary_enum(self):
        bodies = {f'doc-{i}:p1': fx.SCOPE + '\n' + fx.CELLS + '\n' + '\n'.join(
            f'Document {i} qualification sentence {j}.' for j in range(145)) for i in range(3)}
        first, frame, sources, _ = fx.candidate(bodies)
        refs = review._atomic_references(frame)
        self.assertIsNotNone(refs)
        # Adjust only the number of genuine distinct source lines, never create
        # aliases absent from the emitted packet or copy the production parser.
        self.assertLess(len(refs), 472)
        bodies['doc-0:p1'] += '\n' + '\n'.join(f'Additional literal qualification {j}.' for j in range(472-len(refs)))
        first, frame, sources, _ = fx.candidate(bodies)
        refs = review._atomic_references(frame)
        self.assertEqual(len(refs), 472)
        self.assertEqual(len(review._context_targets(frame)), 3)
        bound = review._bound_context_references(frame, refs)
        self.assertIsNotNone(bound)
        schema = review.schema(frame)['schema']
        self.assertEqual(schema['properties']['table_contexts']['type'], 'object')
        self.assertEqual(set(schema['$defs']['source_quote_reference']['enum']), set(refs))
        enums = []
        def visit(node):
            if isinstance(node, dict):
                if 'enum' in node: enums.append(node['enum'])
                for child in node.values(): visit(child)
            elif isinstance(node, list):
                for child in node: visit(child)
        visit(schema)
        self.assertEqual(sum(set(values)==set(refs) for values in enums), 1)
        self.assertLessEqual(sum(map(len, enums)), 1000)
        messages = review.build_messages(frame, question='Which intervals?', response_language='en', sources_block=sources)
        canonical = fx.verdict(frame); canonical['reason'] = 'accepted'
        wire = encode_context_reply(canonical, messages, original_sources=sources, json_schema=review.schema(frame))
        self.assertIsInstance(wire['table_contexts'], dict)
        self.assertEqual(review.apply_wire_review(first, wire, frame), review.apply_review(first, canonical, frame))
        emitted = messages[1]['content'].split('SOURCES:\n', 1)[1].split('\n\nCOMPLETE_CANDIDATE:', 1)[0]
        self.assertEqual(original_context_sources(emitted, sources), sources)


if __name__ == '__main__': unittest.main(verbosity=2)
