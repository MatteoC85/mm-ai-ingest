import assert from 'node:assert/strict';
import {readFileSync, writeFileSync, mkdirSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import path from 'node:path';

const dir = path.dirname(fileURLToPath(import.meta.url));
const workerPath = process.argv[2] || path.join(dir, 'worker.js');
const expectedHash = 'aaf5c8b78ac89a8032da3062044a6f7e89693212dfcd81559080faf8942926e6';
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const source = readFileSync(workerPath);
assert.equal(sha256(source), expectedHash, 'Candidate source SHA256 must match the reviewed language-only Worker');

// This harness uses local Request/Response objects only. global fetch is
// replaced before importing the Worker and every outbound request is handled
// by an allowlisted, in-memory fixture. No production credentials are loaded.
const TRANSPORT = 'synthetic-transport-fixture-0000000000000000000000';
const APP_AUTH = 'synthetic-app-authority-fixture-111111111111111111';
const USAGE_AUTH = 'synthetic-usage-authority-fixture-2222222222222222';
const usageVersion = 'p6-interactive-ledger-v1';
const usage = () => ({
  request_id:'fixture-request-000000000001', actor_id:'actor-A', company_id:'company-A',
  enabled:true, daily_limit:100, monthly_limit:1000, opening_day:0, opening_month:0,
  day_key:'2026-10-01', month_key:'2026-10'
});
const payload = () => ({
  auth_token:TRANSPORT, company_id:'company-A', machine_id:'machine-A',
  ai_scope:'machine_all', query:'Synthetic controlled question', language:'it',
  top_k:5
});
const json = (body, status=200) => new Response(JSON.stringify(body), {
  status, headers:{'Content-Type':'application/json'}
});
let fixture = {};
let requests = [];
let kvReads = 0;
let kvWrites = 0;
globalThis.fetch = async (input, init={}) => {
  const url = new URL(String(input));
  assert.ok(['cloud.example.test','bubble.example.test'].includes(url.hostname),
    'Unexpected outbound destination; no real fetch is available');
  const headers = new Headers(init.headers);
  let body = null;
  try { body = JSON.parse(init.body || '{}'); } catch { throw Error('Fixture request must be JSON'); }
  requests.push({path:url.pathname,origin:url.origin,method:init.method,
    hasSignal:!!init.signal,redirect:init.redirect,body,
    hasUsageHeader:headers.get('X-MM-Usage-Authority')===USAGE_AUTH,
    usageContext:headers.get('X-MM-Usage-Context')});
  if (url.pathname === '/v1/ai/ask/authorize') {
    if (fixture.authorityThrows) throw new DOMException('Controlled deadline', 'TimeoutError');
    if (fixture.authorityDenies) return json({ok:false},403);
    return json({
      ok:true,status:'authorized',result_code:'REQUEST_AUTHORIZED',
      authority_version:'application-authority-p6b4l-v1',
      authority_nonce:headers.get('X-MM-Authority-Nonce'),
      company_id:body.company_id,machine_id:body.machine_id,
      ai_scope:body.ai_scope || 'machine_all', document_ids:body.document_ids || [],
      bubble_document_id:body.bubble_document_id || null, canonical_evidence_active:false
    });
  }
  if (url.pathname === '/quota') return json(fixture.quota || {ok:true});
  if (url.pathname === '/v1/ai/usage/snapshot') return json({
    ok:true,usage_version:usageVersion,daily_used:4,monthly_used:10,
    daily_limit:100,monthly_limit:1000
  });
  if (url.pathname === '/v1/ai/ask' || url.pathname === '/v1/ai/usage/execute/ask') {
    if (fixture.askThrows) throw new DOMException('Controlled deadline', 'TimeoutError');
    if (fixture.httpStatus) return json({ok:false},fixture.httpStatus);
    if (fixture.invalidJson) return new Response('<html>bad upstream</html>',{status:200});
    if (fixture.upstreamError) return json({
      ok:false,status:'error',result_code:'PROVIDER_LIMIT',error:{code:'PROVIDER_LIMIT',message:'Controlled denial'},
      meta:{cacheable:true,semantic_cacheable:true}
    });
    return json({
      ok:true,status:'answered',answer:fixture.emptyAnswer?'':'Synthetic grounded answer',
      citations:[],rg_links:[],
      meta:fixture.receiptMissing?{}:{
        usage_version:usageVersion,usage_recorded:true,
        usage_request_id:'fixture-request-000000000001'
      }
    });
  }
  throw Error('Unrecognized outbound fixture route: '+url.pathname);
};
const worker = (await import('data:text/javascript;base64,'+source.toString('base64'))).default;
const baselineEnv = () => ({
  AI_INTERNAL_SECRET:TRANSPORT,MM_APP_AUTHORITY_SECRET:APP_AUTH,
  MM_ASK_REQUEST_AUTHORITY:'required',MM_AUTHORITY_PREFLIGHT_TIMEOUT_MS:'60000',
  CLOUD_RUN_BASE_URL:'https://cloud.example.test',
  BUBBLE_QUOTA_CHECK_URL:'https://bubble.example.test/quota',
  MM_ASK_CACHE_ENABLED:'1',MM_ASK_CACHE_TTL_SECONDS:'300',
  MM_RC_CACHE:{
    async get(){kvReads++; throw Error('ASK must bypass KV with authority required');},
    async put(){kvWrites++; throw Error('ASK must bypass KV with authority required');}
  }
});
const requiredEnv = () => ({
  ...baselineEnv(),MM_USAGE_ENFORCEMENT:'required',MM_USAGE_AUTHORITY_SECRET:USAGE_AUTH
});
async function call({env=baselineEnv(), body=payload(), route='/v1/ai/ask',
  appHeader=true,usageHeader=false,method='POST',extraHeaders={}}={}) {
  const headers={'Content-Type':'application/json',...extraHeaders};
  if(appHeader)headers['X-MM-App-Authority']=APP_AUTH;
  if(usageHeader)headers['X-MM-Usage-Authority']=USAGE_AUTH;
  const response = await worker.fetch(new Request('https://worker.example.test'+route,{
    method,headers,...(method==='POST'?{body:JSON.stringify(body)}:{})
  }),env);
  return {status:response.status,data:await response.json()};
}
const errorCode = r => r.data.error?.code || r.data.error_code || r.data.result_code;
const tests=[];
async function test(name, fn) {
  fixture={};requests=[];kvReads=0;kvWrites=0;
  try {
    await fn();
    tests.push({name,status:'PASS',outbound_fixture_requests:requests.length,
      ask_fixture_requests:requests.filter(x=>['/v1/ai/ask','/v1/ai/usage/execute/ask'].includes(x.path)).length,
      kv_reads:kvReads,kv_writes:kvWrites});
  } catch(e) {
    tests.push({name,status:'FAIL',error:String(e.message).slice(0,500)});
  }
}
await test('maintenance_denies_before_any_upstream_even_with_usage_off',async()=>{
  const r=await call({env:{...baselineEnv(),MM_AI_MAINTENANCE:'1'}});
  assert.equal(r.status,503);assert.equal(errorCode(r),'AI_MAINTENANCE');assert.equal(requests.length,0);
});
await test('usage_required_without_secret_fails_closed',async()=>{
  const r=await call({env:{...baselineEnv(),MM_USAGE_ENFORCEMENT:'required'}});
  assert.equal(r.status,503);assert.equal(errorCode(r),'USAGE_CONFIGURATION_INVALID');assert.equal(requests.length,0);
});
await test('usage_secret_must_differ_from_existing_transport_secret',async()=>{
  const r=await call({env:{...requiredEnv(),MM_USAGE_AUTHORITY_SECRET:TRANSPORT}});
  assert.equal(errorCode(r),'USAGE_CONFIGURATION_INVALID');assert.equal(requests.length,0);
});
await test('usage_required_without_usage_header_fails_closed',async()=>{
  const r=await call({env:requiredEnv(),body:{...payload(),usage:usage()}});
  assert.equal(r.status,401);assert.equal(errorCode(r),'USAGE_AUTH_REQUIRED');assert.equal(requests.length,0);
});
await test('missing_usage_context_rejected_before_upstream',async()=>{
  const r=await call({env:requiredEnv(),usageHeader:true});
  assert.equal(r.status,400);assert.equal(errorCode(r),'USAGE_CONTEXT_INVALID');assert.equal(requests.length,0);
});
await test('usage_context_with_extra_field_rejected',async()=>{
  const r=await call({env:requiredEnv(),usageHeader:true,body:{...payload(),usage:{...usage(),untrusted:true}}});
  assert.equal(r.status,400);assert.equal(errorCode(r),'USAGE_CONTEXT_INVALID');assert.equal(requests.length,0);
});
await test('usage_context_with_negative_limit_rejected',async()=>{
  const r=await call({env:requiredEnv(),usageHeader:true,body:{...payload(),usage:{...usage(),daily_limit:-1}}});
  assert.equal(errorCode(r),'USAGE_CONTEXT_INVALID');assert.equal(requests.length,0);
});
await test('usage_company_mismatch_rejected',async()=>{
  const r=await call({env:requiredEnv(),usageHeader:true,body:{...payload(),usage:{...usage(),company_id:'company-B'}}});
  assert.equal(r.status,403);assert.equal(errorCode(r),'USAGE_COMPANY_MISMATCH');assert.equal(requests.length,0);
});
await test('usage_plan_disabled_rejected',async()=>{
  const r=await call({env:requiredEnv(),usageHeader:true,body:{...payload(),usage:{...usage(),enabled:false}}});
  assert.equal(r.status,403);assert.equal(errorCode(r),'PLAN_AI_DISABLED');assert.equal(requests.length,0);
});
await test('usage_snapshot_returns_totals_without_authority_quota_or_ask',async()=>{
  const r=await call({env:requiredEnv(),usageHeader:true,route:'/v1/ai/usage/snapshot',
    body:{...payload(),usage:usage()}});
  assert.equal(r.data.ok,true);assert.equal(r.data.daily_used,4);assert.equal(requests.length,1);
  assert.equal(requests[0].path,'/v1/ai/usage/snapshot');assert.equal(requests[0].hasUsageHeader,true);
});
await test('snapshot_with_live_missing_usage_bindings_is_configuration_error',async()=>{
  const r=await call({route:'/v1/ai/usage/snapshot'});
  assert.equal(r.status,503);assert.equal(errorCode(r),'USAGE_CONFIGURATION_INVALID');assert.equal(requests.length,0);
});
await test('authority_required_bypasses_read_and_write_cache',async()=>{
  const r=await call();
  assert.equal(r.data.ok,true);assert.equal(r.data.status,'answered');
  assert.equal(r.data.meta.worker_cache_status,'bypass');assert.equal(kvReads,0);assert.equal(kvWrites,0);
  assert.equal(r.data.meta.worker_timing.cloud_run_ask_requests,1);
  assert.deepEqual(requests.map(x=>x.path),['/v1/ai/ask/authorize','/quota','/v1/ai/ask']);
});
await test('live_usage_off_ask_upstream_has_no_deadline_signal',async()=>{
  const r=await call();assert.equal(r.data.ok,true);
  assert.equal(requests.find(x=>x.path==='/v1/ai/ask').hasSignal,false);
  assert.equal(requests.find(x=>x.path==='/v1/ai/ask/authorize').hasSignal,true);
});
await test('required_usage_routes_execute_only_and_skips_bubble_quota',async()=>{
  const r=await call({env:requiredEnv(),usageHeader:true,body:{...payload(),usage:usage()}});
  assert.equal(r.data.ok,true);assert.equal(r.data.meta.usage_recorded,true);
  assert.deepEqual(requests.map(x=>x.path),['/v1/ai/ask/authorize','/v1/ai/usage/execute/ask']);
  const upstream=requests.at(-1);
  assert.equal(upstream.hasUsageHeader,true);assert.equal(upstream.hasSignal,true);assert.equal(upstream.redirect,'manual');
  assert.deepEqual(JSON.parse(Buffer.from(upstream.usageContext,'base64').toString()),usage());
});
await test('usage_receipt_missing_cannot_publish_answer',async()=>{
  fixture.receiptMissing=true;
  const r=await call({env:requiredEnv(),usageHeader:true,body:{...payload(),usage:usage()}});
  assert.equal(r.data.ok,false);assert.equal(errorCode(r),'USAGE_RECEIPT_MISSING');
  assert.equal(r.data.meta.cacheable,false);
});
await test('upstream_timeout_exception_maps_to_terminal_ASK_TIMEOUT',async()=>{
  fixture.askThrows=true;const r=await call();
  assert.equal(r.status,200);assert.equal(r.data.ok,false);assert.equal(errorCode(r),'ASK_TIMEOUT');
  assert.equal(r.data.meta.cacheable,false);assert.equal(r.data.meta.semantic_cacheable,false);
});
await test('upstream_524_maps_to_terminal_ASK_TIMEOUT',async()=>{
  fixture.httpStatus=524;const r=await call();
  assert.equal(r.data.ok,false);assert.equal(errorCode(r),'ASK_TIMEOUT');assert.equal(r.data.meta.cacheable,false);
});
await test('empty_answered_response_is_rejected',async()=>{
  fixture.emptyAnswer=true;const r=await call();
  assert.equal(r.data.ok,false);assert.equal(errorCode(r),'ASK_RESPONSE_INVALID');assert.equal(r.data.meta.cacheable,false);
});
await test('invalid_upstream_json_is_terminal_error',async()=>{
  fixture.invalidJson=true;const r=await call();
  assert.equal(r.data.ok,false);assert.equal(errorCode(r),'ASK_FAILED');assert.equal(r.data.meta.cacheable,false);
});
await test('structured_upstream_denial_cannot_become_answered_or_cacheable',async()=>{
  fixture.upstreamError=true;const r=await call();
  assert.equal(r.data.ok,false);assert.equal(errorCode(r),'PROVIDER_LIMIT');
  assert.equal(r.data.answer,'');assert.equal(r.data.meta.cacheable,false);assert.equal(r.data.meta.semantic_cacheable,false);
});
await test('quota_denial_stops_before_ask',async()=>{
  fixture.quota={ok:false,error_code:'AI_QUOTA_EXCEEDED'};const r=await call();
  assert.equal(r.data.ok,false);assert.equal(errorCode(r),'AI_QUOTA_EXCEEDED');
  assert.deepEqual(requests.map(x=>x.path),['/v1/ai/ask/authorize','/quota']);
});
await test('authority_timeout_stops_before_quota_and_ask',async()=>{
  fixture.authorityThrows=true;const r=await call();
  assert.equal(r.status,503);assert.equal(errorCode(r),'AUTHORITY_PROVIDER_UNAVAILABLE');
  assert.deepEqual(requests.map(x=>x.path),['/v1/ai/ask/authorize']);
});
const report={
  suite:'new controlled Worker verification, separate from historical 67-test suite',
  worker_file:path.basename(workerPath),worker_sha256:sha256(source),worker_bytes:source.length,
  node_version:process.version,external_network_calls:0,paid_ai_calls:0,
  production_credentials_loaded:false,
  executed_at_utc:new Date().toISOString(),
  passed:tests.filter(x=>x.status==='PASS').length,
  failed:tests.filter(x=>x.status==='FAIL').length,
  scope:'Actual downloaded Worker exported fetch; local Request/Response and mocked fetch/KV/authority/quota/backend only. This does not certify Cloudflare, Bubble or Cloud Run live behavior.',
  tests
};
const reportDir=path.join(dir,'reports');
mkdirSync(reportDir,{recursive:true});
writeFileSync(path.join(reportDir,'worker-controlled-report.json'),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
process.exitCode=report.failed?1:0;
