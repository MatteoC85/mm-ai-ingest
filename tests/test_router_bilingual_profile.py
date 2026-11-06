"""Router-owned translated terms must reach the bounded lexical profile."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import ast
import re
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from machinemind.retrieval import query_fallbacks as policy


def dedup(values,limit):
    return list(dict.fromkeys(v for v in values if isinstance(v,str) and v))[:limit]


class RouterBilingualProfileTests(unittest.TestCase):
    def setUp(self):
        self.profile={'search_terms_it':['clamp'],'search_terms_en':['clamp'],
                      'search_phrases':['clamp stays open'],'important_codes_or_numbers':[]}
        self.runtime=policy.V13BuildProfileFromPlanRuntime(
            _ask_evidence_fallback_profile=lambda *a:self.profile,
            _dedup_text_values=dedup,
            _ask_evidence_tokenize=lambda text:re.findall(r'[a-z]+',text.lower()))

    def test_translated_terms_survive_when_the_complete_phrase_is_absent(self):
        plan={'dense_queries':['clamp remains open','pinza rimane aperta'],
              'lexical_queries':['carrello avanzamento chiusura pinza'],
              'required_facets':['diagnostic distinction'],'exact_terms':['E42']}
        original=deepcopy((self.profile,plan))
        out=policy.v13_build_profile_from_plan('clamp stays open','en',plan,runtime=self.runtime)
        page='La chiusura della pinza permette il trascinamento del materiale.'
        self.assertNotIn('pinza rimane aperta',page)
        for key in ('search_terms_it','search_terms_en'):
            self.assertIn('pinza',out[key])
            self.assertIn('chiusura',out[key])
            self.assertIn('E42',out[key])
            self.assertIn('diagnostic distinction',out[key])
            self.assertLessEqual(len(out[key]),32)
        self.assertIn('pinza rimane aperta',out['search_phrases'])
        self.assertEqual((self.profile,plan),original)
        self.assertIsNot(out,self.profile)

    def test_existing_terms_keep_priority_and_all_caps_stay_bounded(self):
        self.profile['search_terms_en']=[f'original{i}'for i in range(32)]
        plan={'dense_queries':[f'query {i} words'for i in range(40)],
              'lexical_queries':['nuovo termine'],'exact_terms':['E42']}
        out=policy.v13_build_profile_from_plan('q','en',plan,runtime=self.runtime)
        self.assertEqual(out['search_terms_en'],self.profile['search_terms_en'])
        self.assertLessEqual(len(out['search_terms_it']),32)
        self.assertLessEqual(len(out['search_phrases']),24)

    def test_unobserved_context_is_not_promoted_to_a_search_term(self):
        out=policy.v13_build_profile_from_plan('clamp','en',{
            'dense_queries':['clamp'], 'missing_information':['unobservedrupture'],
            'diagnostic_exclusions':['foreignexclusion']},runtime=self.runtime)
        self.assertNotIn('unobservedrupture',str(out))
        self.assertNotIn('foreignexclusion',str(out))

    def test_absent_plan_and_legacy_runtime_remain_compatible(self):
        baseline=deepcopy(self.profile)
        old=policy.V13BuildProfileFromPlanRuntime(lambda *a:self.profile,dedup)
        out=policy.v13_build_profile_from_plan('clamp','en',None,runtime=old)
        self.assertEqual(out['search_terms_en'],baseline['search_terms_en'])
        self.assertEqual(self.profile,baseline)

    def test_real_main_wires_the_existing_tokenizer(self):
        path=Path(__file__).resolve().parents[1]/'main.py'
        node=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
                  if isinstance(n,ast.FunctionDef)and n.name=='_v13_build_profile_from_plan')
        namespace={'Optional':__import__('typing').Optional,
            '_retrieval_query_fallbacks':policy,
            '_ask_evidence_fallback_profile':self.runtime._ask_evidence_fallback_profile,
            '_dedup_text_values':dedup,'_ask_evidence_tokenize':self.runtime._ask_evidence_tokenize}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
        out=namespace[node.name]('clamp','en',{'lexical_queries':['chiusura pinza']})
        self.assertIn('chiusura',out['search_terms_it'])


if __name__=='__main__':unittest.main(verbosity=2)
