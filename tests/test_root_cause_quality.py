"""Root Cause evidence regression tests, with no provider or database calls.

These fixtures prove retention/binding of evidence, not semantic LLM quality.
The same production selector and immutable review validator are exercised.
"""
from copy import deepcopy
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from machinemind.retrieval import diagnostic_sources as sources
from machinemind.retrieval import review_packet, review_decisions, review_references, source_management


RUNTIME=sources.DiagnosticSourceRuntime(
    source_type=lambda candidate:candidate.get('source_type','document'),
    stable_key=lambda candidate:candidate.get('citation_id',''),
    family_key=lambda candidate:candidate.get('family','same-section'))


def candidate(cid,text,*,page=7,source='manual-new',kind='document',family='same-section'):
    return {'citation_id':cid,'company_id':'company-new','machine_id':'machine-new',
            'bubble_document_id':source,'source_type':kind,'page_from':page,'page_to':page,
            'family':family,'chunk_full':text,'snippet':text[:220],
            'assistant_core_root_viable':True,'role_group':'core',
            'semantic_similarity':0.65,'causal_strength_score':0.20,
            'subsystem_score':0.20,'context_fit_score':0.15,
            'assistant_core_facet_coverage':0.5,
            'assistant_core_diagnostic_priority':{'score':0.5,'base_score':0.5,
                'groups':{'observables':0.5,'subsystems':0.5}}}


def select(items,limit=8):
    return sources.select_root_cause_candidates(items,limit=limit,runtime=RUNTIME)


def semantic_dedup(items,limit=24):
    return source_management.dedup_root_cause_candidates_semantic(items,limit,
        runtime=source_management.DedupRootCauseCandidatesSemanticRuntime(
            _normalize_unicode_advanced=lambda text:text,re=re))


class RescoringDedupTests(unittest.TestCase):
    def test_both_rescoring_passes_preserve_distinct_continuations(self):
        prefix='Observed delivery is below the commanded rate. Consult the applicable mechanism. '*5
        rows=[candidate('joint',prefix+'A leaking suction joint can admit air.'),
              candidate('filter',prefix+'A blocked filter can restrict flow.')]
        first_pass=semantic_dedup(rows)
        selected=select(first_pass)
        final_pass=semantic_dedup(list(selected.candidates))
        self.assertEqual([c['citation_id'] for c in final_pass],['joint','filter'])

    def test_heading_and_unit_case_remain_part_of_the_evidence(self):
        variants=[
            ('SECTION: Suction joint\nCompare the observed pressure.','SECTION: Discharge filter\nCompare the observed pressure.'),
            ('Check whether the control value is 12 MPa.','Check whether the control value is 12 mPa.'),
        ]
        for left,right in variants:
            with self.subTest(left=left):
                self.assertEqual(len(semantic_dedup([candidate('a',left),candidate('b',right)])),2)

    def test_standalone_numbers_beyond_preview_are_preserved(self):
        prefix='Compare the documented threshold before interpreting the observed reading. '*5
        self.assertEqual(len(semantic_dedup([candidate('a',prefix+'\n4\nbar'),candidate('b',prefix+'\n7\nbar')])),2)

    def test_scope_and_source_identity_are_never_merged(self):
        row=candidate('a','Use this pressure observation only for the selected source and machine.')
        for field,value in (('company_id','other-company'),('machine_id','other-machine'),
                            ('source_type','ps'),('bubble_document_id','another-source')):
            with self.subTest(field=field):
                other=deepcopy(row);other.update(citation_id='b');other[field]=value
                self.assertEqual(len(semantic_dedup([row,other])),2)

    def test_true_duplicates_keep_the_existing_best_score_and_limit(self):
        row=candidate('a','A blocked outlet increases pressure. Compare the inlet and outlet measurements.')
        better=deepcopy(row);better.update(citation_id='b',causal_strength_score=0.3)
        self.assertEqual([c['citation_id'] for c in semantic_dedup([row,better])],['b'])
        other=candidate('c','A loose coupling interrupts rotation. Check relative shaft movement.')
        self.assertEqual(len(semantic_dedup([row,other],limit=1)),1)


class EvidenceRetentionTests(unittest.TestCase):
    def test_distinct_mechanisms_on_one_page_survive_in_both_languages(self):
        examples=[
            ('Se la presa aspira aria, la pompa perde portata. Controllare la tenuta della presa.',
             'Un filtro ostruito riduce la portata della pompa. Misurare la pressione prima e dopo il filtro.'),
            ('Air drawn through the suction joint reduces pump delivery. Check the suction joint seal.',
             'A clogged filter reduces pump delivery. Compare pressure readings before and after the filter.'),
        ]
        for left,right in examples:
            with self.subTest(language='EN' if left.startswith('Air') else 'IT'):
                result=select([candidate('suction',left),candidate('filter',right)])
                self.assertEqual([c['citation_id'] for c in result.candidates],['suction','filter'])
                self.assertEqual(result.summary['unique_count'],2)

    def test_unpaginated_structured_parts_are_not_treated_as_one_excerpt(self):
        for kind in ('procedure','step','ps'):
            with self.subTest(kind=kind):
                result=select([
                    candidate('part-a','Check the upstream supply before testing the control input.',page=0,source=kind+':new',kind=kind),
                    candidate('part-b','After the upstream check, compare the input with the physical switch position.',page=0,source=kind+':new',kind=kind)])
                self.assertEqual(len(result.candidates),2)

    def test_identical_preview_does_not_hide_a_different_complete_mechanism(self):
        prefix='Inspect the documented operating state and preserve the recorded observations. '*5
        rows=[candidate('first',prefix+'A loose coupling permits relative shaft movement.'),
              candidate('second',prefix+'An interrupted drive signal prevents shaft rotation.')]
        self.assertEqual(rows[0]['snippet'],rows[1]['snippet'])
        self.assertEqual(len(select(rows).candidates),2)

    def test_rejected_excerpt_cannot_hide_valid_same_page_evidence(self):
        weak=candidate('weak','Keep the equipment clean and observe general safety requirements.')
        weak.update(role_group='collateral',generic_downranked=True,
                    semantic_similarity=0.05,causal_strength_score=0,subsystem_score=0,
                    context_fit_score=0,assistant_core_facet_coverage=0,
                    assistant_core_diagnostic_priority={})
        strong=candidate('strong','A blocked discharge causes pressure to rise; compare the upstream and downstream readings.')
        self.assertEqual([c['citation_id'] for c in select([weak,strong]).candidates],['strong'])

    def test_same_identity_and_same_complete_excerpt_still_deduplicate(self):
        text='A blocked discharge causes pressure to rise; compare the upstream and downstream readings.'
        original=candidate('one',text)
        alias=candidate('two','  '+text.replace(' ','  ')+'\n')
        result=select([original,deepcopy(original),alias])
        self.assertEqual([c['citation_id'] for c in result.candidates],['one'])

    def test_different_page_applicability_is_preserved_even_for_identical_excerpt(self):
        text='Compare the control indication with the measured input before replacing a component.'
        rows=[candidate('first',text,page=3),candidate('second',text,page=9)]
        self.assertEqual(len(select(rows).candidates),2)

    def test_case_sensitive_units_are_not_collapsed(self):
        rows=[candidate('pressure','Compare the specified Pa value at the instrument.'),
              candidate('current','Compare the specified pA value at the instrument.')]
        self.assertEqual(len(select(rows).candidates),2)

    def test_caps_limits_hard_exclusions_and_weak_abstention_remain(self):
        rows=[candidate('c'+str(i),'Distinct source mechanism and check number '+str(i),family='family-'+str(i)) for i in range(7)]
        self.assertEqual(len(select(rows).candidates),3)  # Existing per-source cap.
        self.assertEqual(len(select(rows,limit=1).candidates),1)
        for row in rows:row['family']='same-family'
        self.assertEqual(len(select(rows).candidates),2)
        for row in rows:row['hard_excluded']=True
        self.assertEqual(select(rows).candidates,())
        weak=candidate('weak','General overview only.')
        weak.update(role_group='collateral',generic_downranked=True,
                    semantic_similarity=0.1,causal_strength_score=0,subsystem_score=0,
                    context_fit_score=0,assistant_core_facet_coverage=0,
                    assistant_core_diagnostic_priority={})
        self.assertEqual(select([weak]).candidates,())

    def test_selection_does_not_rewrite_inputs_or_provenance(self):
        rows=[candidate('one','A displaced drive coupling can interrupt motion. Check coupling alignment.'),
              candidate('two','Loss of drive enable inhibits motion. Compare enable state with the command.')]
        before=deepcopy(rows)
        result=select(rows)
        self.assertEqual(rows,before)
        for row,chosen in zip(rows,result.candidates):
            for key,value in row.items():self.assertEqual(chosen[key],value)


class ReviewBindingTests(unittest.TestCase):
    def evidence(self):
        rows=list(select([
            candidate('joint','The pump suction joint draws air if its seal leaks. Check the suction joint seal.'),
            candidate('filter','The pump filter restricts delivery when clogged. Compare pressures across the filter.')]).candidates)
        records=[{'citation_id':c['citation_id'],'text':c['chunk_full'],'source_type':'document',
                  'page_from':7,'page_to':7,'context_pages':[]} for c in rows]
        scope={'company_id':'company-new','machine_id':'machine-new','ai_scope':'machine_all'}
        return rows,records,scope

    def test_two_mechanisms_can_reach_immutable_review_without_borrowed_citations(self):
        rows,records,scope=self.evidence()
        self.assertEqual(len(rows),2)
        packet=review_packet.build_review_packet(scope=scope,candidates=rows,records=records,
                                                 company_general_sentinel='GENERAL')
        bound=packet['validator_records']
        drafts=[{'cause':'Possible suction joint leak','why':'A leak can admit air; this is not yet verified.',
                 'checks':['Check the suction joint seal.']},
                {'cause':'Possible filter restriction','why':'A restriction can reduce delivery; this is not yet verified.',
                 'checks':['Compare pressures across the filter.']}]
        manifest=review_decisions.manifest(drafts,bound,max_causes=3)
        observed='Pump delivery decreased under the same operating conditions.'
        prepared=review_references.prepare(packet=packet['model_packet'],proposal_manifest=manifest,
            records=bound,original_query=observed,observed_query=observed)
        frozen=prepared['frozen']
        decisions={'decisions':[{'proposal_index':i,'verdict':'accept','reason':'supported',
            'blocking_checks':[],'note':'','proofs':[{
            'source_index':i,'supports_cause':True,'observation_units':frozen['observed_ids'],
            'source_units':frozen['source_sets'][i],'target_units':frozen['target_sets'][i],
            'applicability':'same_target','support_type':'bounded_inference','check_indices':[0]}]}
            for i,record in enumerate(bound)]}
        result=review_references.validate(parsed=decisions,frozen=frozen,records=bound,observed_query=observed)
        self.assertEqual(result['citation_ids'],['joint','filter'])
        self.assertEqual([c['checks'] for c in result['causes']],[d['checks'] for d in drafts])
        self.assertTrue(result['summary']['all_checks_covered'])
        self.assertFalse(result['summary']['semantic_truth_verified_by_code'])
        bad=deepcopy(decisions);bad['decisions'][1]['proofs'][0]['check_indices']=[]
        with self.assertRaisesRegex(review_references.ReferenceError,'not_all_checks_supported'):
            review_references.validate(parsed=bad,frozen=frozen,records=bound,observed_query=observed)
        bad=deepcopy(decisions);bad['decisions'][1]['proofs'][0]['source_units']=frozen['source_sets'][0]
        with self.assertRaisesRegex(review_references.ReferenceError,'reference_outside_authorized_set'):
            review_references.validate(parsed=bad,frozen=frozen,records=bound,observed_query=observed)
        with self.assertRaisesRegex(review_references.ReferenceError,'observed_request_changed'):
            review_references.validate(parsed=decisions,frozen=frozen,records=bound,observed_query='A different symptom')

    def test_multi_tenant_machine_and_document_scope_guards_still_reject(self):
        for field,value,expected in (
                ('company_id','another-company','candidate_company_scope_mismatch'),
                ('machine_id','another-machine','candidate_machine_scope_mismatch'),
                ('bubble_document_id','unselected-document','candidate_document_scope_mismatch')):
            with self.subTest(field=field):
                rows,records,scope=self.evidence()
                rows[0][field]=value
                if field=='bubble_document_id':scope.update(ai_scope='document_ids',document_ids=['manual-new'])
                with self.assertRaisesRegex(review_packet.ReviewPacketError,expected):
                    review_packet.build_review_packet(scope=scope,candidates=rows,records=records,
                                                     company_general_sentinel='GENERAL')


if __name__=='__main__':unittest.main(verbosity=2)
