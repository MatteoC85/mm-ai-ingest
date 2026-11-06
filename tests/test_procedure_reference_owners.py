"""Owner-bound review schema and strict reference decoding, without provider IO."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from machinemind.ask import procedure_review as r

def accepts(value,schema,root):
    """Check the generated schema subset independently; no provider claim."""
    if '$ref' in schema:
        ref=root
        for part in schema['$ref'].removeprefix('#/').split('/'):ref=ref[part]
        return accepts(value,ref,root)
    if 'anyOf' in schema:return any(accepts(value,choice,root) for choice in schema['anyOf'])
    types={'object':dict,'array':list,'string':str,'integer':int,'boolean':bool}
    if 'type' in schema and type(value)is not types[schema['type']]:return False
    if 'enum' in schema and value not in schema['enum']:return False
    if type(value)is int and not schema.get('minimum',value)<=value<=schema.get('maximum',value):return False
    if type(value)is str and not schema.get('minLength',0)<=len(value)<=schema.get('maxLength',len(value)):return False
    if type(value)is list:
        return (schema.get('minItems',0)<=len(value)<=schema.get('maxItems',len(value))
                and all(accepts(item,schema['items'],root)for item in value))
    if type(value)is dict:
        props=schema.get('properties',{})
        return (set(schema.get('required',[])).issubset(value)
                and (schema.get('additionalProperties')is not False or set(value).issubset(props))
                and all(accepts(item,props[key],root)for key,item in value.items()if key in props))
    return True

def span(first=0,last=None):return {'first':first,'last':first if last is None else last}
def endpoint(owner,first=0,last=None,kind='original'):return {'block_id':owner,'kind':kind,'range':span(first,last)}

class OwnerReferenceTests(unittest.TestCase):
    def setUp(self):
        self.answer='1. Release the guide.\n2. Isolate supply. Feed strip. Secure the guide.'
        self.sources={'short':'Keep supply isolated.', 'long':'Release the guide. Feed strip. Secure the guide.'}
        self.layout=r.block_layout(self.answer)
        self.aliases=r.review_wire_aliases(self.layout,list(self.sources))
        text='\n\n'.join('['+key+'] source_type=document\n'+body for key,body in self.sources.items())
        self.catalog=r.reference_catalog(self.layout,self.aliases,text,self.sources)
        legacy={'name':'offline','schema':{'type':'object','additionalProperties':False,
            'properties':{'citation_ids':{'type':'array','items':{'type':'string'}},'answer':{'type':'string'}},
            'required':['citation_ids','answer']}}
        self.schema=r.review_schema(legacy,self.layout,[],[],operation_boundary=True,source_ordered=True,
            operation_source_ids=list(self.sources),wire_aliases=self.aliases,reference_catalogue=self.catalog)['schema']
        self.edge=self.schema['properties']['operation_boundary']['properties']['precedence']['items']
        self.wire={'outcome':'pass','reply_mode':'retain','answer':'','edits':[],
            'citation_ids':['s1','s2'],'covered_facets':[],'missing_facets':[],
            'covered_answer_types':[],'missing_answer_types':[],'missing_list_items':[],
            'operation_boundary':{'complete':True,'checked_blocks':['b1','b2'],
                'precedence':[{'action':endpoint('b1'),'next':endpoint('b2',2),
                               'proof':{'source':'s2','range':span(0,2)}}],
                'change_edges':[{'edge_index':0,'resolution':'restored'}]}}

    def decoded(self,value):
        return r.decode_reference_reply(value,self.aliases,layout=self.layout,catalog=self.catalog)

    def failure(self,value,reason,stage):
        with self.assertRaises(ValueError)as caught:self.decoded(value)
        diagnostic=r.protocol_failure_diagnostic(caught.exception)
        self.assertEqual(diagnostic['reason'],reason);self.assertEqual(diagnostic['stage'],stage)
        self.assertTrue(set(diagnostic)=={'version','stage','reason','unit_count','first','last'})
        return diagnostic

    def test_valid_nested_wire_retains_exact_canonical_answer_and_proof(self):
        before=deepcopy(self.wire);decoded,proof=self.decoded(self.wire)
        resolved=r.resolve_reply(decoded,answer=self.answer,layout=self.layout,required_facets=[],required_types=[],source_ordered=True)
        out=r.verify_operation_boundary(resolved,layout=self.layout,source_texts=self.sources,source_ordered=True,reference_proof=proof)
        self.assertEqual(self.wire,before);self.assertEqual(out['answer'],self.answer)
        self.assertTrue(out['operation_boundary_validation']['complete'])
        self.assertEqual(out['operation_boundary_validation']['version'],'ask-operation-boundary-v1')
        self.assertEqual(out['operation_boundary_validation']['reference_protocol'],'ask-procedure-references-v2')

    def test_every_original_owner_has_its_own_bounds_not_global_maximum(self):
        for owner,item in self.catalog['blocks'].items():
            for first in range(-1,len(item['units'])+2):
                value=endpoint(owner,first)
                self.assertEqual(accepts(value,self.edge['properties']['action'],self.schema),0<=first<len(item['units']))
        self.assertFalse(accepts(endpoint('b1',1),self.edge['properties']['next'],self.schema))
        self.assertTrue(accepts(endpoint('b2',1),self.edge['properties']['next'],self.schema))

    def test_every_source_owner_has_local_bounds_and_cannot_borrow_other_units(self):
        for owner,item in self.catalog['sources'].items():
            for first in range(-1,len(item['units'])+2):
                value={'source':owner,'range':span(first)}
                self.assertEqual(accepts(value,self.edge['properties']['proof'],self.schema),0<=first<len(item['units']))
        bad=deepcopy(self.wire);bad['operation_boundary']['precedence'][0]['proof']={'source':'s1','range':span(1)}
        self.assertFalse(accepts(bad['operation_boundary']['precedence'][0],self.edge,self.schema))
        self.failure(bad,'out_of_bounds','proof')

    def test_edits_cannot_borrow_longer_block_bounds(self):
        item=self.schema['properties']['edits']['items']
        value={'block_id':'b1','original':span(1),'replacement_parts':['Replacement.']}
        self.assertFalse(accepts(value,item,self.schema))
        value['block_id']='b2';self.assertTrue(accepts(value,item,self.schema))

    def test_unknown_owner_boolean_index_and_old_detached_owners_fail(self):
        for value in (endpoint('b999'),endpoint('b1',True),endpoint('b1',kind='guessed')):
            self.assertFalse(accepts(value,self.edge['properties']['action'],self.schema))
        bad=deepcopy(self.wire);bad['operation_boundary']['precedence'][0]['from']='b2'
        self.assertFalse(accepts(bad['operation_boundary']['precedence'][0],self.edge,self.schema))
        with self.assertRaisesRegex(ValueError,'procedure_review_wire_edge_invalid'):self.decoded(bad)

    def test_reversed_ranges_still_reject_never_swap_or_claim_schema_eliminates_them(self):
        bad=deepcopy(self.wire);bad['operation_boundary']['precedence'][0]['next']=endpoint('b2',1,0)
        self.assertTrue(accepts(bad['operation_boundary']['precedence'][0],self.edge,self.schema))
        self.failure(bad,'reversed','next')

    def test_replacement_without_edit_remains_rejected_not_inferred(self):
        bad=deepcopy(self.wire);bad['operation_boundary']['precedence'][0]['action']=endpoint('b1',kind='replacement')
        self.assertTrue(accepts(bad['operation_boundary']['precedence'][0],self.edge,self.schema))
        self.failure(bad,'replacement_missing','action')

    def test_replacement_length_uses_same_declared_edit_at_decode(self):
        bad=deepcopy(self.wire);bad.update(outcome='rewrite',reply_mode='edit')
        bad['edits']=[{'block_id':'b1','original':span(),'replacement_parts':['Release the marked guide.']}]
        bad['operation_boundary']['precedence'][0]['action']=endpoint('b1',1,kind='replacement')
        self.failure(bad,'out_of_bounds','action')

    def test_diagnostic_never_leaks_raw_strings_aliases_or_unbounded_numbers(self):
        bad=deepcopy(self.wire);bad['operation_boundary']['precedence'][0]['action']=endpoint('b1',10**100)
        diagnostic=self.failure(bad,'out_of_bounds','action')
        self.assertEqual(diagnostic['first'],1000000)
        self.assertNotIn('b1',json.dumps(diagnostic));self.assertNotIn(self.answer,json.dumps(diagnostic))
        arbitrary=ValueError('SECRET SOURCE TEXT')
        arbitrary.diagnostic={'source':self.answer}
        self.assertEqual(r.protocol_failure_diagnostic(arbitrary),{})
        try:r._reference_range(span(10),((0,1,'x'),),stage='proof')
        except ValueError as exc:
            exc.diagnostic['raw_source']='SECRET';self.assertNotIn('SECRET',json.dumps(r.protocol_failure_diagnostic(exc)))

    def test_whitespace_only_span_has_distinct_failure_not_false_evidence(self):
        with self.assertRaises(ValueError)as caught:r._reference_range(span(),((0,3,'   '),),stage='proof')
        self.assertEqual(r.protocol_failure_diagnostic(caught.exception)['reason'],'empty_span')

    def test_empty_numbered_body_is_rejected_during_catalog_preflight(self):
        for answer in ('1. \n2. Keep supply isolated.','   \n1. Keep supply isolated.'):
            layout=r.block_layout(answer);aliases=r.review_wire_aliases(layout,['short'])
            with self.subTest(answer=answer),self.assertRaisesRegex(ValueError,'procedure_reference_catalog_invalid'):
                r.reference_catalog(layout,aliases,'[short] source_type=document\n'+self.sources['short'],{'short':self.sources['short']})

    def test_definitions_are_shared_alias_only_and_within_provider_limits(self):
        schema=self.schema;serialized=json.dumps(schema)
        self.assertNotIn(self.sources['short'],serialized);self.assertNotIn(self.sources['long'],serialized)
        self.assertEqual(self.edge['properties']['action'],self.edge['properties']['next'])
        forbidden={'allOf','not','dependentRequired','dependentSchemas','if','then','else'}
        properties=enums=0
        def walk(node):
            nonlocal properties,enums
            if isinstance(node,dict):
                self.assertFalse(forbidden.intersection(node))
                if node.get('type')=='object':
                    self.assertIs(node.get('additionalProperties'),False)
                    self.assertEqual(set(node.get('required',[])),set(node.get('properties',{})))
                    properties+=len(node.get('properties',{}))
                enums+=len(node.get('enum',[]))
                for child in node.values():walk(child)
            elif isinstance(node,list):
                for child in node:walk(child)
        walk(schema)
        self.assertLess(properties,5000);self.assertLess(enums,1000)
        self.assertNotIn('anyOf',schema)
        # Resolve definitions to measure actual nested containers, not textual refs.
        def nesting(node,seen=frozenset()):
            if '$ref' in node:
                ref=node['$ref'];self.assertNotIn(ref,seen)
                return nesting(schema['$defs'][ref.rsplit('/',1)[1]],seen|{ref})
            if 'anyOf' in node:return max(nesting(x,seen)for x in node['anyOf'])
            if node.get('type')=='object':return 1+max([nesting(x,seen)for x in node.get('properties',{}).values()]+[0])
            if node.get('type')=='array':return 1+nesting(node['items'],seen)
            return 0
        self.assertLessEqual(nesting(schema),10)

if __name__=='__main__':unittest.main(verbosity=2)
