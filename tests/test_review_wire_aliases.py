"""Lossless short wire identities, decoded before canonical review validation."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from machinemind.ask import procedure_review as review
import test_source_ordered_boundary as boundary_fixture


def compact_reply(canonical, aliases, *, layout=None):
    """Test-only inverse encoder; production only decodes provider wire data."""
    result=deepcopy(canonical)
    sources={cid:name for name,cid in aliases['sources'].items()}
    blocks={bid:name for name,bid in aliases['blocks'].items()}
    result['citation_ids']=[sources[cid]for cid in result.get('citation_ids',[])]
    layout=layout or review.block_layout(boundary_fixture.SourceOrderedBoundaryTests.DRAFT)
    originals={b['block_id']:review._block_parts(b['text'])[1]for b in layout['blocks']}
    for edit in result.get('edits',[]):
        bid=edit['block_id'];body=edit.pop('text')
        edit.update(block_id=blocks[bid],old_text=originals[bid],new_text=body)
    audit=result['operation_boundary']
    audit['checked_blocks']=[blocks[bid]for bid in audit['checked_blocks']]
    for i,edge in enumerate(audit['precedence']):
        audit['precedence'][i]={'from':blocks[edge['before_block_id']],'action':edge['before_answer_quote'],
            'to':blocks[edge['after_block_id']],'next':edge['after_answer_quote'],
            'source':sources[edge['source_citation_id']],'proof':edge['source_quote']}
    return result


class ReviewWireAliasTests(unittest.TestCase):
    def decode(self, wire, aliases):
        return review.decode_review_wire_aliases(wire,aliases,
            layout=review.block_layout(boundary_fixture.SourceOrderedBoundaryTests.DRAFT))

    def fixture(self):
        f=boundary_fixture.SourceOrderedBoundaryTests()
        canonical=f.reply()
        source='1780000000000x123456789012345678:p50-50:manualsupport:v13:49'
        for edge in canonical['operation_boundary']['precedence']:edge['source_citation_id']=source
        canonical['operation_boundary'].pop('changes')
        canonical['operation_boundary']['change_edges']=[{'edge_index':1,'resolution':'restored'}]
        canonical['citation_ids']=['step:four',source]
        aliases=review.review_wire_aliases(review.block_layout(f.DRAFT),canonical['citation_ids'])
        return f,source,canonical,aliases

    def test_all_identity_fields_roundtrip_exactly_and_keep_validation_result(self):
        f,source,canonical,aliases=self.fixture()
        wire=compact_reply(canonical,aliases)
        original=deepcopy(wire)
        decoded=self.decode(wire,aliases)
        self.assertEqual(decoded,canonical)
        self.assertEqual(wire,original)
        kwargs={'layout':review.block_layout(f.DRAFT),'source_texts':{source:f.SOURCE},'source_ordered':True}
        self.assertEqual(review.verify_operation_boundary(decoded,**kwargs),
                         review.verify_operation_boundary(canonical,**kwargs))
        self.assertTrue(review.verify_operation_boundary(decoded,**kwargs)['operation_boundary_validation']['complete'])
        self.assertLess(len(json.dumps(wire)),len(json.dumps(canonical)))

    def test_unknown_source_or_block_cannot_become_authority(self):
        _,source,canonical,aliases=self.fixture()
        mutations=(lambda r:r.update(citation_ids=['foreign']),
            lambda r:r['edits'][0].update(block_id='b999'),
            lambda r:r['operation_boundary'].update(checked_blocks=['b999']),
            lambda r:r['operation_boundary']['precedence'][0].update(source=source),
            lambda r:r['operation_boundary']['precedence'][0].update(to='foreign'))
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                wire=compact_reply(canonical,aliases);mutate(wire)
                with self.assertRaisesRegex(ValueError,'alias_unknown'):
                    self.decode(wire,aliases)

    def test_mixed_or_missing_edge_fields_are_rejected(self):
        _,_,canonical,aliases=self.fixture()
        for mutate in (lambda e:e.update(source_citation_id='s1'),lambda e:e.pop('proof'),
                       lambda e:e.update(source=['s1'])):
            wire=compact_reply(canonical,aliases);mutate(wire['operation_boundary']['precedence'][0])
            with self.assertRaises(ValueError):self.decode(wire,aliases)

    def test_quotes_are_never_rewritten_even_if_they_contain_alias_like_text(self):
        _,_,canonical,aliases=self.fixture()
        canonical['operation_boundary']['precedence'][0]['source_quote']='Do not replace s1 or b2 in literal source text.'
        canonical['operation_boundary']['precedence'][0]['before_answer_quote']='The label reads s2.'
        decoded=self.decode(compact_reply(canonical,aliases),aliases)
        self.assertEqual(decoded,canonical)

    def test_schema_uses_only_admitted_aliases_and_keeps_all_proof_fields(self):
        f,_,canonical,aliases=self.fixture()
        legacy={'name':'test','schema':{'properties':{'citation_ids':{'type':'array','items':{'type':'string'}}},'required':[]}}
        schema=review.review_schema(legacy,review.block_layout(f.DRAFT),[],[],source_ordered=True,
            operation_boundary=True,operation_source_ids=canonical['citation_ids'],wire_aliases=aliases)['schema']
        self.assertEqual(schema['properties']['citation_ids']['items']['enum'],['s1','s2'])
        edges=schema['properties']['operation_boundary']['properties']['precedence']['items']
        self.assertEqual(set(edges['required']),{'from','action','to','next','source','proof'})
        self.assertEqual(edges['properties']['source']['enum'],['s1','s2'])
        self.assertEqual(edges['properties']['from']['enum'],['b1','b2'])
        self.assertEqual(edges['properties']['proof']['maxLength'],400)
        changed=deepcopy(aliases);changed['sources']['s2']='foreign'
        with self.assertRaisesRegex(ValueError,'catalog_drift'):
            review.review_schema(legacy,review.block_layout(f.DRAFT),[],[],source_ordered=True,
                operation_boundary=True,operation_source_ids=canonical['citation_ids'],wire_aliases=changed)

    def test_duplicate_views_share_source_identity_but_not_new_block_authority(self):
        layout=review.block_layout('3. First.\n8. Second.')
        aliases=review.review_wire_aliases(layout,['known','known','other'])
        self.assertEqual(aliases,{'sources':{'s1':'known','s2':'other'},
                                  'blocks':{'b1':'step_1','b2':'step_2'}})
        self.assertNotEqual(aliases['blocks']['b1'],'step_3')


if __name__=='__main__':unittest.main(verbosity=2)
