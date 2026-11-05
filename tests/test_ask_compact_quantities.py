"""ASK quantity representations, independent of company, document or query.

Pure parser/final-contract checks: no network, models, database or authority
changes. Numeric compatibility is not semantic proof of the measured fluid.
"""
from pathlib import Path
import json,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from machinemind.ask import unit_grounding as u,final_contract

def row(number,values,role='RECORD'):
 return f'- [PDF_TABLE_{role} rows={number} columns={len(values)}] '+ ' | '.join(
  f'c{index} r{number}='+json.dumps(value,ensure_ascii=False)for index,value in enumerate(values,1))

class CompactQuantityTests(unittest.TestCase):
 def supported(self,answer,source):
  kept,removed=u._assistant_core_filter_unsupported_claim_sentences(answer,source)
  self.assertEqual(removed,[],(answer,source,removed));self.assertEqual(kept,answer)
 def rejected(self,answer,source):
  kept,removed=u._assistant_core_filter_unsupported_claim_sentences(answer,source)
  self.assertEqual(kept,'',(answer,source,kept));self.assertTrue(removed)
 def test_leading_decimal_forms_are_exact_bidirectional_quantities(self):
  forms=('.75','0.75','0,75',',75','00.750','+.75')
  for left in forms:
   for right in forms:
    for unit in ('l/min','kW','bar'):
     with self.subTest(left=left,right=right,unit=unit):self.supported(f'Value: {left}{unit}.',f'Value: {right} {unit}.')
  for wrong in ('75','7.5','0.075','-.75','0.7501'):
   self.rejected(f'Flow: {wrong} l/min.','Water flow: .75l/min.')
 def test_compact_and_spaced_units_preserve_dimensions_and_case(self):
  for number,unit in (('100','psi'),('30','CFM'),('1','kW'),('.75','l/min'),('0.85','m3/min')):
   for left,right in ((number+unit,number+' '+unit),(number+' '+unit,number+unit)):
    self.supported('Value: '+left+'.','Value: '+right+'.')
   self.supported(f'Value: **{number} {unit}**.',f'Value: {number}{unit}.')
  self.supported('Flow: 0.85 m³/min.','Flow: 0.85m3/min.')
  self.supported('Power: 1 KW.','Power: 1kW.')
 def test_wrong_units_values_and_unrequested_conversions_still_fail(self):
  for answer,source in (('Air supply: .75 CFM.','Water cooling: .75l/min.'),
   ('Water flow: .75 l/min.','Air supply: .75CFM.'),('Pressure: 100 bar.','Pressure: 100psi.'),
   ('Power: 1 W.','Power: 1kW.'),('Power: 1000 W.','Power: 1kW.'),
   ('Flow: 30 l/min.','Flow: 30CFM.'),('Pressure: 7 bar.','Pressure: 100psi.'),
   ('Pressure: 101 psi.','Pressure: 100psi.'),('Flow: .76 l/min.','Flow: .75l/min.')):
   with self.subTest(answer=answer,source=source):self.rejected(answer,source)
 def test_whole_suffix_required_and_composite_identifiers_are_exact(self):
  for source in ('Power: 1kWh.','Power: 1kW_model.','Power: 1kW-model.','Power: 1kW/s.'):
   self.rejected('Power: 1 kW.',source)
  self.rejected('Flow: .75 l/min.','Flow: .75l/min/s.')
  for code in ('QTR-100psi','LC170','AME-1000SN','PWR_1kW','QTR-.75'):
   self.supported('Designation: '+code+'.','Designation: '+code+'.')
   claims=u._assistant_core_claims('Designation: '+code+'.')
   self.assertTrue(any(c['kind']=='code'and c['raw'].rstrip('.')==code for c in claims),claims)
  self.rejected('Designation: QTR-101psi.','Designation: QTR-100psi.')
  self.rejected('Pressure: 100 psi.','Designation: QTR-100psi.')
 def test_compound_dimensions_and_dot_codes_cannot_supply_a_quantity_prefix(self):
  for suffix in ('/s','/ s',' /s',' / s','.model','*bar','**bar'):
   for number,unit in (('1','kW'),('100','psi'),('.75','l/min')):
    with self.subTest(suffix=suffix,unit=unit):
     self.rejected(f'Value: {number} {unit}.',f'Source: {number}{unit}{suffix}.')
  for code in ('1kW.model','100psi.model','1kW/second'):
   self.supported('Designation: '+code+'.','Designation: '+code+'.')
   self.rejected('Designation: '+code.replace('1','2',1)+'.','Designation: '+code+'.')
  self.supported('Power: **1 kW**.','Power: 1kW.')
  self.supported('Power: 1 kW.','Power: **1kW**. Next sentence.')
 def test_same_column_inheritance_and_conflicts_remain_strict(self):
  source=row(1,['Water flow (l/min)','Power (kW)'],'HEADER')+'\n'+row(2,['.75','1'])
  self.supported('Flow: 0,75 l/min.',source);self.supported('Power: 1kW.',source)
  self.rejected('Flow: 1 l/min.',source);self.rejected('Power: .75 kW.',source)
  for value in ('.75CFM','0.75 CFM'):
   conflict=row(1,['Water flow (l/min)'],'HEADER')+'\n'+row(2,[value])
   self.rejected('Flow: .75 l/min.',conflict);self.rejected('Flow: .75 CFM.',conflict)
  self.rejected('Flow: .75 l/min.',row(2,['.75','l/min']))
 def test_serialized_literals_remain_unchanged(self):
  source=row(2,['Water cooling','Flow rate minimum: ".75l/min".'])+'\n'+row(3,['Applicability','Power > 1kW'])
  before=source.encode();self.supported('Water cooling: 0,75 l/min.',source)
  self.supported('Applicable above 1 kW.',source);self.assertEqual(source.encode(),before)
 def test_leading_decimal_ranges_keep_both_endpoint_units(self):
  self.supported('Flow: .25 l/min.','Flow: .25 to .75l/min.')
  self.supported('Flow: 0.75 l/min.','Flow: .25 to .75l/min.')
  self.rejected('Flow: .25 CFM.','Flow: .25 to .75l/min.')
 def test_valid_review_binding_survives_quantity_normalization(self):
  source=row(2,['Water cooling','Minimum .75l/min.'])+'\nPower > 1kW.\nAir: 30CFM at 100psi.'
  answer='Air: 30 CFM at 100 psi.\nWater cooling: 0,75 l/min above 1 kW.'
  projected,removed=u._assistant_core_filter_unsupported_claim_sentences(answer,source)
  self.assertEqual(removed,[])
  semantic={'outcome':'rewrite','answer':answer,'missing_facets':[],'missing_answer_types':[],'missing_list_items':[]}
  contract=final_contract.final_contract(deterministic={'passed':True},semantic=semantic,
   reviewed_answer=answer,answer=projected,semantic_complete=True,semantic_partial=False,
   grounding_proof={'basis':'existing_grounding_projection','removed_claims':0,
    'input_answer_sha256':final_contract.answer_digest(answer),'output_answer_sha256':final_contract.answer_digest(projected)})
  self.assertTrue(contract['passed']);self.assertTrue(contract['semantic_answer_bound'])
 def test_semantic_rejection_of_wrong_fluid_still_blocks_publication(self):
  # Equal numeric dimensions do not establish which subsystem/medium they
  # describe; the independent semantic review retains that responsibility.
  answer='Air supply: .75 l/min.';source='Water cooling: .75l/min.'
  projected,_=u._assistant_core_filter_unsupported_claim_sentences(answer,source)
  semantic={'outcome':'no_sources','answer':'','missing_facets':['correct fluid'],
   'missing_answer_types':[],'missing_list_items':[]}
  contract=final_contract.final_contract(deterministic={'passed':True},semantic=semantic,
   reviewed_answer=answer,answer=projected,semantic_complete=False,semantic_partial=False)
  self.assertFalse(contract['passed']);self.assertFalse(contract['complete'])

if __name__=='__main__':unittest.main(verbosity=2)
