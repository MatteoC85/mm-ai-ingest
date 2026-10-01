import assert from 'node:assert/strict';
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const dir=path.dirname(fileURLToPath(import.meta.url));
const file=process.argv[2] || path.join(dir,'worker.js');
const source=readFileSync(file);
const sha256=createHash('sha256').update(source).digest('hex');
assert.equal(sha256,'aaf5c8b78ac89a8032da3062044a6f7e89693212dfcd81559080faf8942926e6',
 'Candidate source SHA256 must match the reviewed language-only Worker');
const TOKEN='local-synthetic-transport-0000000000000000';
const APP='local-synthetic-authority-1111111111111111';
let fixture={},calls=[];
globalThis.fetch=async(input,init={})=>{
 const url=new URL(String(input));
 assert.ok(['cloud.example.test','bubble.example.test'].includes(url.hostname));
 const body=JSON.parse(init.body||'{}');calls.push({path:url.pathname,body,hasSignal:!!init.signal});
 if(url.pathname==='/quota')return Response.json({ok:true});
 if(url.pathname==='/v1/ai/ask/authorize')return Response.json({
  ok:true,status:'authorized',result_code:'REQUEST_AUTHORIZED',
  authority_version:'application-authority-p6b4l-v1',
  authority_nonce:new Headers(init.headers).get('X-MM-Authority-Nonce'),
  company_id:body.company_id,machine_id:body.machine_id,ai_scope:body.ai_scope,
  document_ids:body.document_ids||[],bubble_document_id:body.bubble_document_id||null,
  canonical_evidence_active:false});
 if(url.pathname.startsWith('/v1/ai/'))return Response.json(fixture);
 throw Error('Unexpected fixture path; no original fetch is available');
};
const worker=(await import('data:text/javascript;base64,'+source.toString('base64'))).default;
const liveSource=readFileSync(path.join(dir,'fixtures','worker.live-88ec46a7.js'));
assert.equal(createHash('sha256').update(liveSource).digest('hex'),'7ae430e0830003d3fe227acfb7035443fff261dfdbe14db6359113f656cb000a');
const liveWorker=(await import('data:text/javascript;base64,'+liveSource.toString('base64'))).default;
const env={AI_INTERNAL_SECRET:TOKEN,MM_APP_AUTHORITY_SECRET:APP,
 MM_ASK_REQUEST_AUTHORITY:'required',MM_AUTHORITY_PREFLIGHT_TIMEOUT_MS:'60000',
 CLOUD_RUN_BASE_URL:'https://cloud.example.test',BUBBLE_QUOTA_CHECK_URL:'https://bubble.example.test/quota',
 MM_ASK_CACHE_ENABLED:'0',MM_RC_CACHE_ENABLED:'0'};
const base=()=>({auth_token:TOKEN,company_id:'company-A',machine_id:'machine-A',
 ai_scope:'machine_all',query:'Controlled synthetic fault',language:'en',
 options:{top_k:8,max_causes:3}});
const rc=()=>({ok:true,status:'answered',language:'en',effective_mode:'root_cause',
 possible_causes:[{rank:1,cause:'Restricted filter',why:'Differential pressure increases',
 checks:['Read the differential pressure gauge','Inspect the isolated filter']}],
 recommended_next_checks:['Compare the documented operating range'],
 citations:[{citation_id:'S1',bubble_document_id:'doc-A',page_from:2,page_to:2,snippet:'Source fixture'}],
 rg_links:[{citation_id:'S1',url:'https://app.example.test/doc-A'}],
 meta:{cached:false,cache_status:'miss',cacheable:true,v13_estimated_cost_usd:0.02}});
async function request(route,body=base(),target=worker){
 const r=await target.fetch(new Request('https://worker.example.test'+route,{method:'POST',
  headers:{'Content-Type':'application/json','X-MM-App-Authority':APP},body:JSON.stringify(body)}),env);
 assert.equal(r.status,200);return r.json();
}
const tests=[];
async function test(name,fn){fixture={};calls=[];try{await fn();tests.push({name,status:'PASS'});}
catch(e){tests.push({name,status:'FAIL',error:e.message.slice(0,300)});}}
await test('root_cause_English_display_labels',async()=>{
 fixture=rc();const r=await request('/v1/ai/root-cause');
 assert.match(r.possible_causes_text,/Why: Differential pressure/);
 assert.match(r.possible_causes_text,/Recommended checks:/);
 assert.doesNotMatch(r.possible_causes_text,/Perch\u00e9|Controlli consigliati/);
});
await test('root_cause_Italian_display_labels_preserved',async()=>{
 fixture={...rc(),language:'it'};const r=await request('/v1/ai/root-cause',{...base(),language:'it'});
 assert.match(r.possible_causes_text,/Perch\u00e9:/);assert.match(r.possible_causes_text,/Controlli consigliati:/);
});
await test('root_cause_meta_language_controls_display_when_root_language_absent',async()=>{
 fixture=rc();delete fixture.language;fixture.meta.language='en';
 const r=await request('/v1/ai/root-cause',{...base(),language:'it'});assert.match(r.possible_causes_text,/Why:/);
});
await test('root_cause_requested_language_fallback',async()=>{
 fixture=rc();delete fixture.language;const r=await request('/v1/ai/root-cause');
 assert.match(r.possible_causes_text,/Why:/);
});
await test('root_cause_cache_hit_metadata_and_cost_unchanged_from_live',async()=>{
 fixture=rc();fixture.meta={...fixture.meta,cached:true,cache_status:'semantic_hit',cache_layer:'cloud_run_semantic',v13_estimated_cost_usd:0};
 const r=await request('/v1/ai/root-cause');
 const original=await request('/v1/ai/root-cause',base(),liveWorker);
 assert.deepEqual(r.meta,original.meta);assert.equal(r.meta.cached,false);
 assert.equal(r.meta.cache_status,'disabled');assert.equal(r.meta.v13_estimated_cost_usd,0);
 assert.equal(r.meta.cache_layer,'cloud_run_semantic');
});
await test('root_cause_cache_miss_metadata_unchanged_from_live',async()=>{
 fixture=rc();const r=await request('/v1/ai/root-cause');
 const original=await request('/v1/ai/root-cause',base(),liveWorker);
 assert.deepEqual(r.meta,original.meta);assert.equal(r.meta.cache_status,'disabled');
 assert.equal(r.meta.worker_cache_status,undefined);
});
await test('root_cause_complete_causes_checks_citations_and_links_preserved',async()=>{
 fixture=rc();for(let i=2;i<=8;i++)fixture.possible_causes.push({...fixture.possible_causes[0],rank:i,cause:'Cause '+i});
 const r=await request('/v1/ai/root-cause');
 assert.deepEqual(r.possible_causes,fixture.possible_causes);assert.deepEqual(r.citations,fixture.citations);
 assert.deepEqual(r.rg_links,fixture.rg_links);assert.match(r.possible_causes_text,/8\. Cause 8/);
 assert.deepEqual(JSON.parse(r.raw_payload_json),fixture);
});
await test('root_cause_routed_ASK_answer_preserved',async()=>{
 fixture={...rc(),effective_mode:'ask',possible_causes:[],answer:'Complete six-step source procedure'};
 const r=await request('/v1/ai/root-cause');assert.equal(r.possible_causes_text,fixture.answer);
});
await test('ASK_output_and_request_contract_preserved',async()=>{
 fixture={ok:true,status:'answered',answer:'Source-grounded reply',language:'en',
 answer_html:'<p>Source-grounded reply</p>',citations:rc().citations,rg_links:rc().rg_links,
 meta:{cached:true,cache_status:'semantic_hit',v13_estimated_cost_usd:0}};
 const r=await request('/v1/ai/ask');
 assert.equal(r.answer,fixture.answer);assert.equal(r.answer_html,fixture.answer_html);
 assert.deepEqual(r.citations,fixture.citations);assert.equal(r.meta.cached,true);
 assert.equal(calls.at(-1).body.top_k,8);assert.equal(calls.at(-1).body.ai_scope,'machine_all');
});
for(const action of ['start','answer','finalize']){
 await test('smart_'+action+'_request_response_fields_preserved',async()=>{
  fixture={ok:true,status:action==='finalize'?'completed':'active',final_ready:action==='finalize',
   language:'en',session_state_json:'{"schema":1}',question:{question_id:'q1',question_text:'Is pressure low?'},
   hypotheses:[{id:'h1',label:'Restricted filter',checks:['Inspect pressure']}],
   citations:rc().citations,rg_links:rc().rg_links,
   final_result:action==='finalize'?{summary:'Source-grounded result',recommended_checks:['Check filter']}:null,
   meta:{model:'controlled-no-provider',budget_cost_usd:0}};
  const state={schema:1,history:[{question_id:'q0',answer:'yes'}]};
  const body={...base(),session_id:'session-A',symptom_text:'Low pressure',
    context:{context_type:'machine',context_id:'machine-A'},
    question_id:'q1',state_json:state,
    answer:{api_value:'low',value:'low',label:'Low',free_text:'measured low'},
    options:{top_k:8,max_hypotheses:4,max_questions:6}};
  const r=await request('/v1/ai/smart-diagnostic/'+action,body);
  for(const field of ['status','final_ready','session_state_json','question','hypotheses','citations','rg_links','final_result'])
   assert.deepEqual(r[field],fixture[field]);
  assert.deepEqual(JSON.parse(r.raw_payload_json),fixture);
  if(action==='start'){assert.deepEqual(calls.at(-1).body.options,body.options);assert.deepEqual(calls.at(-1).body.context,body.context);}
  else assert.deepEqual(calls.at(-1).body.state_json,state);
  if(action==='answer')assert.deepEqual(calls.at(-1).body.answer,body.answer);
 });
}
const report={suite:'offline Worker quality adapter differential',worker_file:path.basename(file),
 worker_sha256:sha256,node_version:process.version,external_network_calls:0,provider_calls:0,
 passed:tests.filter(x=>x.status==='PASS').length,failed:tests.filter(x=>x.status==='FAIL').length,tests};
const reportDir=path.join(dir,'reports');
mkdirSync(reportDir,{recursive:true});
const target=path.join(reportDir,'worker-quality-report.json');
writeFileSync(target,JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));process.exitCode=report.failed?1:0;
