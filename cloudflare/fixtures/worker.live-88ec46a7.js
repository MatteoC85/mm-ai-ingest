// Phase 6: private application usage context; accounting stays in PostgreSQL.
// The execute-only path cannot accidentally run on an older Cloud Run revision.
const MM_USAGE_VERSION = 'p6-interactive-ledger-v1';
const MM_USAGE_PATHS = new Set(['/v1/ai/ask','/v1/ai/root-cause','/v1/ai/draft_ps',
 '/v1/ai/smart-diagnostic/start','/v1/ai/smart-diagnostic/answer','/v1/ai/smart-diagnostic/finalize']);
function mmUsageBoundary(request, env, parsed) {
 const path = new URL(request.url).pathname;
 const applicable = MM_USAGE_PATHS.has(path) || path === '/v1/ai/usage/snapshot';
 const mode = env.MM_USAGE_ENFORCEMENT ?? 'off';
 if (!applicable) return null;
 const fail = (code, status=503) => ({error:mmB4lError(code,status)});
 // A cutover/emergency switch is effective even before enforcement is enabled.
 if (env.MM_AI_MAINTENANCE === '1') return fail('AI_MAINTENANCE',503);
 if (mode === 'off' && path !== '/v1/ai/usage/snapshot') return null;
 if (mode !== 'required') return fail('USAGE_CONFIGURATION_INVALID');
 const secret = env.MM_USAGE_AUTHORITY_SECRET;
 if (typeof secret !== 'string' || !/^[\x21-\x7e]{32,256}$/.test(secret) ||
     secret === env.AI_INTERNAL_SECRET || secret === env.MM_APP_AUTHORITY_SECRET)
   return fail('USAGE_CONFIGURATION_INVALID');
 if (!mmB4lEqual(request.headers.get('X-MM-Usage-Authority'),secret))
   return fail('USAGE_AUTH_REQUIRED',401);
 if (new URL(request.url).searchParams.get('debug')==='1' || request.headers.get('x-debug')==='1' || parsed?.debug===true)
   return fail('USAGE_DEBUG_DISABLED',400);
 const u = parsed?.usage;
 const names=['request_id','actor_id','company_id','enabled','daily_limit','monthly_limit','opening_day','opening_month','day_key','month_key'];
 if (!u || Array.isArray(u) || typeof u!=='object' || Object.keys(u).length!==names.length || names.some(n=>!(n in u)))
   return fail('USAGE_CONTEXT_INVALID',400);
 const validId=x=>typeof x==='string' && /^[A-Za-z0-9_.:-]{1,128}$/.test(x);
 if (!validId(u.request_id) || u.request_id.length<16 || !validId(u.actor_id) || !validId(u.company_id) ||
     typeof u.enabled!=='boolean' || ['daily_limit','monthly_limit','opening_day','opening_month'].some(k=>!Number.isSafeInteger(u[k]) || u[k]<0 || u[k]>1e9) ||
     typeof u.day_key!=='string' || !/^\d{4}-\d{2}-\d{2}$/.test(u.day_key) ||
     typeof u.month_key!=='string' || !/^\d{4}-\d{2}$/.test(u.month_key))
   return fail('USAGE_CONTEXT_INVALID',400);
 if (u.company_id!==parsed?.company_id) return fail('USAGE_COMPANY_MISMATCH',403);
 if (!u.enabled) return fail('PLAN_AI_DISABLED',403);
 let origin;
 try {const url=new URL(getCloudRunBase(env));
  if(url.protocol!=='https:' || url.username || url.password || url.pathname!=='/' || url.search || url.hash)throw Error();
  origin=url.origin;
 }catch{return fail('USAGE_CONFIGURATION_INVALID');}
 // All accepted values are ASCII; a fresh copy cannot follow later mutation.
 const context=btoa(JSON.stringify(Object.fromEntries(names.map(n=>[n,u[n]]))));
 return {secret,context,origin};
}
function mmUsageFetch(boundary) {
 const realFetch=globalThis.fetch.bind(globalThis);
 return async (input, init={}) => {
  if(!boundary)return realFetch(input,init);
  const url=new URL(String(input));
  const metered=MM_USAGE_PATHS.has(url.pathname);
  if(url.origin!==boundary.origin || (!metered && url.pathname!=='/v1/ai/usage/snapshot'))
    return realFetch(input,init);
  if(metered)url.pathname=url.pathname.replace('/v1/ai/','/v1/ai/usage/execute/');
  const headers=new Headers(init.headers);
  headers.set('X-MM-Usage-Authority',boundary.secret);
  headers.set('X-MM-Usage-Context',boundary.context);
  const result=await realFetch(url.toString(),{...init,headers,redirect:'manual',cache:'no-store',
    signal:AbortSignal.timeout(115000)});
  // The legacy adapters understand the structured error envelope. Preserve a
  // ledger denial (401/409/429/503), rather than mislabeling it as an AI timeout.
  if(!result.ok){
    let data;
    try{data=await mmB4lReadJson(result,65536);}catch{return jsonResponse({ok:false,status:'error',error:{code:'USAGE_UPSTREAM_UNAVAILABLE',message:'USAGE_UPSTREAM_UNAVAILABLE'}},200);}
    if(data?.meta?.usage_version===MM_USAGE_VERSION && data.ok===false)return jsonResponse(data,200);
    return jsonResponse({ok:false,status:'error',error:{code:'USAGE_UPSTREAM_UNAVAILABLE',message:'USAGE_UPSTREAM_UNAVAILABLE'}},200);
  }
  let data;
  try {data=await mmB4lReadJson(result,4194304);} catch {
    return jsonResponse({ok:false,status:'error',error:{code:'USAGE_UPSTREAM_INVALID',message:'USAGE_UPSTREAM_INVALID'}},200);
  }
  if(!metered && data?.ok===true && data?.usage_version===MM_USAGE_VERSION &&
     ['daily_used','monthly_used','daily_limit','monthly_limit'].every(k=>Number.isSafeInteger(data[k]) && data[k]>=0))
    return jsonResponse(data,200);
  if(data?.meta?.usage_version!==MM_USAGE_VERSION || data?.meta?.usage_recorded!==true){
    if(data?.ok===false && data?.meta?.usage_version===MM_USAGE_VERSION)return jsonResponse(data,200);
    return jsonResponse({ok:false,status:'error',error:{code:'USAGE_RECEIPT_MISSING',message:'USAGE_RECEIPT_MISSING'}},200);
  }
  return jsonResponse(data,200);
 };
}

// BEGIN B4L REQUEST AUTHORITY — independent server-only application boundary.
// Trusts only the server-side application boundary; no end-user identity header is accepted.
const MM_B4L_AUTHORITY_VERSION = "application-authority-p6b4l-v1";
function mmB4lError(code, status = 503) {
  return jsonResponse({ok:false,status:"error",result_code:code,
    error:{code,message:code},answer:"",citations:[],rg_links:[],
    meta:{cacheable:false,semantic_cacheable:false}},status);
}
function mmB4lEqual(a, b) {
  if (typeof a !== "string" || typeof b !== "string" || a.length !== b.length) return false;
  let mismatch = 0;
  for (let i=0;i<a.length;i++) mismatch |= a.charCodeAt(i)^b.charCodeAt(i);
  return mismatch === 0; // No early character exit; Python verifies with compare_digest again.
}
function mmB4lAskPayload(parsed) {
  const aiScope=getAiScope(parsed);
  const rawTop=parsed?.options?.top_k ?? parsed?.top_k;
  return {company_id:String(parsed?.company_id || parsed?.company?.id || "").trim(),
    machine_id:String(parsed?.machine_id || parsed?.scope?.machine_id || "").trim(),
    ...(aiScope ? {ai_scope:aiScope} : {}), query:String(parsed?.query || "").trim(),
    language:String(parsed?.language || "").trim().toLowerCase() || null,
    top_k:Number.isFinite(Number(rawTop)) ? Number(rawTop) : 5,
    bubble_document_id:String(parsed?.bubble_document_id || parsed?.bubble_document || "").trim() || null,
    document_ids:shouldForwardDocumentIds(aiScope) ? normalizeDocumentIds(parsed?.document_ids ?? parsed?.scope?.document_ids ?? null) : null,
    knowledge_version:getExternalKnowledgeVersion(parsed) || null,debug:false};
}
async function mmB4lReadJson(response, maximum) {
  const reader=response.body?.getReader();
  if (!reader) throw new Error("AUTHORITY_RESPONSE_INVALID");
  let size=0;const chunks=[];
  try {
    while(true){const item=await reader.read();if(item.done)break;
      size+=item.value.byteLength;
      if(size>maximum)throw new Error("AUTHORITY_RESPONSE_TOO_LARGE");
      chunks.push(item.value);}
  } finally { await reader.cancel().catch(()=>{}); }
  const bytes=new Uint8Array(size);let offset=0;
  for(const chunk of chunks){bytes.set(chunk,offset);offset+=chunk.byteLength;}
  return JSON.parse(new TextDecoder("utf-8",{fatal:true}).decode(bytes));
}
async function mmB4lPreflight(request,env,parsed,expectedToken,debugOn) {
  const mode=env.MM_ASK_REQUEST_AUTHORITY ?? "off";
  if(mode==="off")return null;
  if(mode!=="required")return mmB4lError("AUTHORITY_CONFIGURATION_INVALID");
  const key=env.MM_APP_AUTHORITY_SECRET;
  if(typeof key!=="string" || !/^[\x21-\x7e]{32,256}$/.test(key) || key===expectedToken)
    return mmB4lError("AUTHORITY_CONFIGURATION_INVALID");
  if(!mmB4lEqual(request.headers.get("X-MM-App-Authority"),key))
    return mmB4lError("AUTH_REQUIRED",401);
  if(debugOn)return mmB4lError("AUTHORITY_DEBUG_DISABLED",400);
  if(!parsed || typeof parsed!=="object" || Array.isArray(parsed))
    return mmB4lError("AUTHORITY_REQUEST_INVALID",400);
  const timeout=Number(env.MM_AUTHORITY_PREFLIGHT_TIMEOUT_MS);
  if(!Number.isSafeInteger(timeout) || timeout<1 || timeout>120000)
    return mmB4lError("AUTHORITY_CONFIGURATION_INVALID");
  let base;
  try {base=new URL(getCloudRunBase(env));
    if(base.protocol!=="https:" || base.username || base.password || base.search || base.hash || base.pathname!=="/")
      throw new Error("configured Cloud Run origin required");
  } catch {return mmB4lError("AUTHORITY_CONFIGURATION_INVALID");}
  const payload=mmB4lAskPayload(parsed),nonce=crypto.randomUUID().replaceAll("-","");
  const headers={"Content-Type":"application/json","X-AI-Internal-Secret":expectedToken,
    "X-MM-App-Authority":key};
  try {
    const response=await fetch(`${base.origin}/v1/ai/ask/authorize`,{method:"POST",
      headers:{...headers,"X-MM-Authority-Nonce":nonce},body:JSON.stringify(payload),
      redirect:"manual",cache:"no-store",signal:AbortSignal.timeout(timeout)});
    if(!response.ok)return mmB4lError(response.status===401 ? "AUTH_REQUIRED" :
      response.status===403 ? "SCOPE_DENIED" : "AUTHORITY_PROVIDER_UNAVAILABLE",
      response.status===401 || response.status===403 ? response.status : 503);
    const result=await mmB4lReadJson(response,16384);
    if(result?.ok!==true || result.status!=="authorized" || result.result_code!=="REQUEST_AUTHORIZED" ||
       result.authority_version!==MM_B4L_AUTHORITY_VERSION || result.authority_nonce!==nonce ||
       result.company_id!==payload.company_id || result.canonical_evidence_active!==false ||
       !mmB4lScopeEchoMatches(payload,result))
      return mmB4lError("AUTHORITY_RESPONSE_INVALID");
    // Reuse the exact payload and authenticated headers, not another derivation
    // after preflight. This is NOT permission to reuse a cached ASK answer.
    return {headers,payload,origin:base.origin,required:true};
  } catch {return mmB4lError("AUTHORITY_PROVIDER_UNAVAILABLE");}
}
// END B4L REQUEST AUTHORITY

// Phase 6 closure: ASK-only transport/response instrumentation. No provider or cache calls.
function mmAskClock() { return performance.now(); }
function mmAskTrace() { return {started: mmAskClock(), spans: {}, active: null, cloudRunRequests: 0}; }
function mmAskSpanStart(trace, name) {
  if (!trace) return;
  mmAskSpanEnd(trace);
  trace.active = {name, started: mmAskClock()};
}
function mmAskSpanEnd(trace) {
  if (!trace?.active) return;
  const {name, started} = trace.active;
  trace.spans[name] = (trace.spans[name] || 0) + Math.max(0, (mmAskClock() - started) / 1000);
  trace.active = null;
}
function mmAskResponse(payload, status, trace) {
  mmAskSpanEnd(trace);
  if (!trace) return jsonResponse(payload, status);
  const elapsed = Math.max(0, (mmAskClock() - trace.started) / 1000);
  const rounded = (value) => Math.round(value * 1000000) / 1000000;
  const measured = Object.values(trace.spans).reduce((sum, value) => sum + value, 0);
  const timing = {
    version: "phase6-worker-timing-v1",
    elapsed_seconds: rounded(elapsed),
    preflight_seconds: rounded(trace.spans.preflight || 0),
    quota_seconds: rounded(trace.spans.quota || 0),
    cloud_run_response_seconds: rounded(trace.spans.cloud_run || 0),
    response_adapter_seconds: rounded(trace.spans.adapter || 0),
    other_seconds: rounded(Math.max(0, elapsed - measured)),
    cloud_run_ask_requests: trace.cloudRunRequests,
    scope: "worker_entry_through_response_adapter"
  };
  return jsonResponse({...payload, meta: {...(payload?.meta || {}),
    ...(payload?.ok !== true ? {cacheable:false, semantic_cacheable:false} : {}),
    worker_timing: timing}}, status);
}
function mmAskError(code, trace, extra = {}) {
  return mmAskResponse({ok:false, status:"error", result_code:code,
    error_code:code, error_message:code, answer:"", citations:[], rg_links:[],
    error:{code, message:code}, ...extra,
    meta:{...(extra.meta || {}), cacheable:false, semantic_cacheable:false}}, 200, trace);
}
async function mmAskQuotaCheck({env, companyId, endpoint, authToken, bubbleDocumentId, isInteractiveAiCall, trace}) {
  mmAskSpanStart(trace, "quota");
  const failure = (code) => ({ok:false, error_code:code, error_message:code});
  try {
    const timeout = Number(env.MM_ASK_QUOTA_TIMEOUT_MS ?? 5000);
    const rawUrl = String(env.BUBBLE_QUOTA_CHECK_URL || "").trim();
    let quotaUrl;
    try { quotaUrl = new URL(rawUrl); } catch { return failure("ASK_QUOTA_CONFIGURATION_INVALID"); }
    if (quotaUrl.protocol !== "https:" || quotaUrl.username || quotaUrl.password || quotaUrl.hash ||
        !Number.isSafeInteger(timeout) || timeout < 1 || timeout > 10000)
      return failure("ASK_QUOTA_CONFIGURATION_INVALID");
    const response = await fetch(quotaUrl.toString(), {method:"POST",
      headers:{"Content-Type":"application/json"}, redirect:"manual", cache:"no-store",
      body:JSON.stringify({p_internal_secret:authToken, p_company_id:companyId, p_endpoint:endpoint,
        p_bubble_document_id:bubbleDocumentId || null, p_is_interactive_ai_call:Boolean(isInteractiveAiCall)}),
      signal:AbortSignal.timeout(timeout)});
    if (!response.ok) return failure("ASK_QUOTA_UNAVAILABLE");
    let value;
    try { value = await mmB4lReadJson(response, 65536); }
    catch (error) {
      return failure(isLikelyTimeoutException(error) ? "ASK_QUOTA_TIMEOUT" : "ASK_QUOTA_RESPONSE_INVALID");
    }
    if (value && typeof value === "object" && !Array.isArray(value) &&
        value.response && typeof value.response === "object" && !Array.isArray(value.response))
      value = value.response;
    if (!value || typeof value !== "object" || Array.isArray(value))
      return failure("ASK_QUOTA_RESPONSE_INVALID");
    if (![true, false, "true", "false", "yes", "no"].includes(value.ok))
      return failure("ASK_QUOTA_RESPONSE_INVALID");
    return value;
  } catch (error) {
    return failure(isLikelyTimeoutException(error) ? "ASK_QUOTA_TIMEOUT" : "ASK_QUOTA_UNAVAILABLE");
  } finally { mmAskSpanEnd(trace); }
}
// Validate the echo against the existing machinemind/core/scope.py contract.
// This grants no source access and does not cache authorization.
function mmB4lScopeEchoMatches(payload, result) {
  let scope = payload.ai_scope || "machine_all";
  let machine = payload.machine_id || "__MM_COMPANY_GENERAL__";
  let docs = payload.document_ids || [];
  let single = payload.bubble_document_id || null;
  if (scope === "company_general") { machine = "__MM_COMPANY_GENERAL__"; docs = []; single = null; }
  else if (scope === "document_ids" || (!payload.ai_scope && (docs.length || single))) scope = "document_ids";
  else if (scope === "machine_all") { docs = []; single = null; }
  else return false;
  return result.machine_id === machine && result.ai_scope === scope &&
    result.bubble_document_id === single && Array.isArray(result.document_ids) &&
    result.document_ids.length === docs.length &&
    docs.every((id, index) => result.document_ids[index] === id);
}
// End Phase 6 ASK-only helpers.


function jsonResponse(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

// V13: immutable cache schema marker. Changing this invalidates all older
// V12/V11 exact-cache entries even when MM_*_CACHE_VERSION env vars are set.
const MM_WORKER_V13_CACHE_SCHEMA = "machinemind-canonical-no-buffer-v9-1-20260804-7";

function getExternalKnowledgeVersion(parsed) {
  return String(
    parsed?.knowledge_version ||
      parsed?.index_revision ||
      parsed?.scope?.knowledge_version ||
      parsed?.scope?.index_revision ||
      ""
  ).trim();
}

function safeSlice(s, n) {
  if (typeof s !== "string") return "";
  return s.length > n ? s.slice(0, n) : s;
}

function isLikelyTimeoutException(error) {
  const text = `${String(error?.name || "")} ${String(error?.message || error || "")}`
    .toLowerCase();
  return (
    text.includes("timeout") ||
    text.includes("timed out") ||
    text.includes("deadline") ||
    text.includes("aborterror") ||
    text.includes("aborted") ||
    text.includes("http 524") ||
    text.includes("http 504")
  );
}

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  let binary = "";

  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode(...chunk);
  }

  return btoa(binary);
}

function getFormString(form, key) {
  const v = form.get(key);
  if (v === null || v === undefined) return "";
  return String(v).trim();
}

function normalizeBooleanLike(value) {
  if (value === true || value === 1) return true;
  if (value === false || value === 0 || value === null || value === undefined) return false;

  const s = String(value).normalize("NFKC").trim().toLowerCase();
  return ["true", "yes", "1", "y", "si", "sì"].includes(s);
}

// Ingest-credit metering is intentionally separate from the Ask/Root Cause cache schema.
// Changing this marker must not invalidate the existing exact-response caches.
const MM_INGEST_CREDITS_VERSION = "ingest-credits-v1-post-threshold-block";
const PLAN_INGEST_CREDITS_LIMIT_EXCEEDED = "PLAN_INGEST_CREDITS_LIMIT_EXCEEDED";

function finiteNonNegativeNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) && n >= 0 ? n : fallback;
}

function isIngestCreditsForwardEnabled(env) {
  // Default OFF so this Worker can be deployed before Cloud Run accepts the new fields.
  return normalizeBooleanLike(env?.MM_INGEST_CREDITS_FORWARD_TO_CLOUD_RUN ?? false);
}

function firstNonEmptyString(...values) {
  for (const value of values) {
    const s = String(value ?? "").trim();
    if (s) return s;
  }
  return "";
}

function getIngestCreditQuota(qc) {
  const limit = finiteNonNegativeNumber(
    qc?.ingest_credits_limit_month ??
      qc?.ai_ingest_credits_limit_month ??
      qc?.ingest_credit_limit_month ??
      0
  );

  const used = finiteNonNegativeNumber(
    qc?.ingest_credits_used_month ??
      qc?.ai_ingest_credits_used_month ??
      qc?.ingest_credit_used_month ??
      qc?.ingest_credits_used ??
      0
  );

  const enforced = normalizeBooleanLike(
    qc?.ingest_credits_enforced ??
      qc?.ai_ingest_credits_enforced ??
      qc?.ingest_credit_enforced ??
      false
  );

  const monthKey = firstNonEmptyString(
    qc?.ingest_month_key,
    qc?.month_key,
    qc?.billing_month_key
  );

  const requestAlreadyAdmitted = normalizeBooleanLike(
    qc?.ingest_request_already_admitted ??
      qc?.ingest_document_already_admitted ??
      false
  );

  return {
    limit,
    used,
    enforced,
    monthKey,
    requestAlreadyAdmitted,
    remaining: limit > 0 ? Math.max(0, limit - used) : 0,
    reached:
      enforced &&
      limit > 0 &&
      used >= limit &&
      !requestAlreadyAdmitted,
  };
}

function getIngestRequestKey({ parsed = null, form = null, bubbleDocumentId = "" } = {}) {
  return firstNonEmptyString(
    form ? getFormString(form, "request_key") : "",
    form ? getFormString(form, "trace_id") : "",
    parsed?.ingest_request_key,
    parsed?.request_key,
    parsed?.trace_id,
    parsed?.source?.request_key,
    parsed?.source?.trace_id,
    bubbleDocumentId
  );
}

function isIngestCreditLimitCode(value) {
  return String(value || "").trim() === PLAN_INGEST_CREDITS_LIMIT_EXCEEDED;
}

function getCloudRunLimitReason(crJson) {
  return firstNonEmptyString(
    crJson?.reason,
    crJson?.error_code,
    crJson?.error?.code
  );
}

function isAnyDocumentIngestLimitCode(value) {
  const code = String(value || "").trim();
  return (
    code === "PLAN_EMBED_CHARS_LIMIT_EXCEEDED" ||
    code === "PLAN_INDEX_STORAGE_LIMIT_EXCEEDED" ||
    code === PLAN_INGEST_CREDITS_LIMIT_EXCEEDED
  );
}

function buildIngestCreditLimitPayload({ quota, requestKey }) {
  const q = quota || getIngestCreditQuota({});

  return {
    ok: false,
    status: "limit_exceeded",
    reason: PLAN_INGEST_CREDITS_LIMIT_EXCEEDED,
    error_code: PLAN_INGEST_CREDITS_LIMIT_EXCEEDED,
    ai_quota_exceeded: true,

    request_key: requestKey || "",
    ingest_request_key: requestKey || "",
    ingest_month_key: q.monthKey || "",
    ingest_credits_actual: 0,
    ingest_credits_used_before: q.used,
    ingest_credits_used_month: q.used,
    ingest_credits_limit_month: q.limit,
    ingest_credits_remaining: q.remaining,
    ingest_credits_enforced: q.enforced,
    ingest_request_already_admitted: q.requestAlreadyAdmitted,
    ingest_limit_reached: true,
    ingest_metering_status: "not_started_quota_blocked",
    ingest_metering_version: MM_INGEST_CREDITS_VERSION,

    // Keep the legacy flat fields too, so existing Bubble expressions can read them.
    limit: q.limit,
    used: q.used,
    remaining: q.remaining,

    error: {
      code: PLAN_INGEST_CREDITS_LIMIT_EXCEEDED,
      message:
        "Limite mensile di elaborazione AI documenti già raggiunto. Il documento resta salvato ma non viene indicizzato.",
      detail: {
        ingest_month_key: q.monthKey || "",
        ingest_credits_used_month: q.used,
        ingest_credits_limit_month: q.limit,
      },
    },
  };
}

function getCloudRunIngestMeteringFields({ crJson, quota, requestKey }) {
  const source = crJson && typeof crJson === "object" ? crJson : {};
  const q = quota || getIngestCreditQuota({});

  const actualRaw =
    source?.ingest_credits_actual ??
    source?.metering?.ingest_credits_actual ??
    source?.usage?.ingest_credits_actual;

  const hasActual =
    actualRaw !== null &&
    actualRaw !== undefined &&
    String(actualRaw).trim() !== "" &&
    Number.isFinite(Number(actualRaw));

  const actual = hasActual ? finiteNonNegativeNumber(actualRaw, 0) : 0;
  const usedAfterHint = q.used + actual;
  const cloudRunLimitReason = getCloudRunLimitReason(source);
  const quotaExceeded =
    isIngestCreditLimitCode(cloudRunLimitReason) ||
    normalizeBooleanLike(source?.ai_quota_exceeded);

  const usageObject =
    source?.ingest_usage ??
    source?.metering?.ingest_usage ??
    source?.usage?.ingest ??
    null;

  let usageJson = "";
  if (usageObject && typeof usageObject === "object") {
    try {
      usageJson = JSON.stringify(usageObject);
    } catch (_) {
      usageJson = "";
    }
  }

  return {
    request_key: firstNonEmptyString(source?.request_key, source?.ingest_request_key, requestKey),
    ingest_request_key: firstNonEmptyString(
      source?.ingest_request_key,
      source?.request_key,
      requestKey
    ),
    ingest_usage_event_id: firstNonEmptyString(
      source?.ingest_usage_event_id,
      source?.metering?.usage_event_id,
      source?.usage_event_id
    ),
    ingest_month_key: firstNonEmptyString(source?.ingest_month_key, q.monthKey),
    ingest_credits_actual: actual,
    ingest_credits_used_before: q.used,
    ingest_credits_used_after_hint: usedAfterHint,
    ingest_credits_limit_month: q.limit,
    ingest_credits_remaining_after_hint:
      q.limit > 0 ? Math.max(0, q.limit - usedAfterHint) : 0,
    ingest_credits_enforced: q.enforced,
    ingest_request_already_admitted: q.requestAlreadyAdmitted,
    ingest_limit_reached_after:
      q.enforced && q.limit > 0 && usedAfterHint >= q.limit,
    ingest_pricing_version: firstNonEmptyString(
      source?.ingest_pricing_version,
      source?.metering?.pricing_version,
      source?.pricing_version
    ),
    ingest_metering_status: firstNonEmptyString(
      source?.ingest_metering_status,
      source?.metering?.status,
      quotaExceeded
        ? "not_started_quota_blocked"
        : hasActual
          ? "measured"
          : "missing_from_cloud_run"
    ),
    ingest_metering_version: firstNonEmptyString(
      source?.ingest_metering_version,
      source?.metering?.version,
      MM_INGEST_CREDITS_VERSION
    ),
    ingest_usage_json: usageJson,
    ai_quota_exceeded: quotaExceeded,
  };
}

function normalizeDocumentCategoryCode(value) {
  return String(value ?? "")
    .normalize("NFKC")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "_");
}

function isAllowedIngestFilename(filename) {
  const lower = String(filename || "").toLowerCase();
  return lower.endsWith(".pdf") || lower.endsWith(".xlsx");
}

function getBearerToken(request) {
  const h = request.headers.get("authorization") || request.headers.get("Authorization") || "";
  const m = h.match(/^Bearer\s+(.+)$/i);
  return m ? m[1].trim() : "";
}

// ✅ Cloud Run base URL (da env; fallback sicuro)
function getCloudRunBase(env) {
  const v = String(env.CLOUD_RUN_BASE_URL || "").trim().replace(/\/+$/, "");
  if (!v) {
    throw new Error("CLOUD_RUN_BASE_URL missing");
  }
  return v;
}

function getAiScope(parsed) {
  const raw = String(parsed?.ai_scope || parsed?.scope?.ai_scope || "").trim().toLowerCase();

  if (!raw) return "";

  if (["machine", "machine_all", "machine_all_plus_company"].includes(raw)) {
    return "machine_all";
  }

  if (["company", "company_general", "company_only", "general"].includes(raw)) {
    return "company_general";
  }

  if (["document_ids", "documents", "document"].includes(raw)) {
    return "document_ids";
  }

  // Lascia a Cloud Run la validazione finale degli scope non supportati.
  return raw;
}

function normalizeDocumentIds(docIds) {
  if (typeof docIds === "string") {
    docIds = docIds
      .split(",")
      .map((x) => String(x || "").trim())
      .filter(Boolean);
  }

  if (Array.isArray(docIds)) {
    docIds = docIds.map((x) => String(x || "").trim()).filter(Boolean);
    if (docIds.length === 0) return null;
    return docIds;
  }

  return null;
}

function shouldForwardDocumentIds(aiScope) {
  // Compatibilità vecchia: se ai_scope non è passato, mantieni document_ids.
  // Nuovo comportamento: machine_all e company_general non devono restringere ai soli document_ids.
  return !aiScope || aiScope === "document_ids";
}

function signedUrlExpiresAtMs(urlString) {
  try {
    const exp = new URL(String(urlString || "")).searchParams.get("Expires");
    const n = Number(exp || 0);
    if (!Number.isFinite(n) || n <= 0) return 0;
    return n * 1000;
  } catch (_) {
    return 0;
  }
}

function isExpiredSignedUrl(urlString) {
  const expMs = signedUrlExpiresAtMs(urlString);
  if (!expMs) return false;
  // Treat URLs expiring in the next minute as unusable.
  return expMs <= Date.now() + 60000;
}

function normalizeUrlCandidateValue(value) {
  if (value === null || value === undefined) return "";

  const s = String(value).trim();
  const low = s.toLowerCase();

  // Bubble can pass empty optional params as literal strings.
  if (!s || low === "null" || low === "undefined" || low === "[object object]") {
    return "";
  }

  return s;
}

function firstUsableUrlCandidate(candidates) {
  const inspected = [];

  for (const c of candidates || []) {
    const source = String(c?.source || "");
    const value = normalizeUrlCandidateValue(c?.value);

    if (!value) {
      inspected.push({ source, status: "empty" });
      continue;
    }

    let parsedUrl = null;
    try {
      parsedUrl = new URL(value.startsWith("//") ? "https:" + value : value);
    } catch (_) {
      inspected.push({ source, status: "invalid_url", value_head: value.slice(0, 120) });
      continue;
    }

    if (parsedUrl.protocol !== "http:" && parsedUrl.protocol !== "https:") {
      inspected.push({ source, status: "invalid_protocol", protocol: parsedUrl.protocol });
      continue;
    }

    if (isExpiredSignedUrl(parsedUrl.toString())) {
      inspected.push({
        source,
        status: "expired",
        expires_at_ms: signedUrlExpiresAtMs(parsedUrl.toString()),
      });
      continue;
    }

    inspected.push({ source, status: "selected" });
    return { url: parsedUrl.toString(), source, inspected };
  }

  return { url: "", source: "", inspected };
}

function hasRootCauseCacheKv(env) {
  return !!env.MM_RC_CACHE;
}

function isRootCauseCacheEnabled(env) {
  if (!hasRootCauseCacheKv(env)) return false;
  return String(env.MM_RC_CACHE_ENABLED || "1").trim() !== "0";
}

function getRootCauseCacheTtlSeconds(env) {
  const raw = Number(env.MM_RC_CACHE_TTL_SECONDS || 900);
  if (!Number.isFinite(raw)) return 900;
  return Math.max(60, Math.min(86400, Math.floor(raw)));
}

function normalizeRootCauseCacheText(value) {
  return String(value || "")
    .normalize("NFKC")
    .toLowerCase()
    .replace(/\s+/g, " ")
    .trim();
}

function normalizeRootCauseCacheDocIds(docIds) {
  if (typeof docIds === "string") {
    docIds = docIds
      .split(",")
      .map((x) => String(x || "").trim())
      .filter(Boolean);
  }

  if (!Array.isArray(docIds)) return [];

  return docIds
    .map((x) => String(x || "").trim())
    .filter(Boolean)
    .sort();
}

function stableStringify(value) {
  if (value === null || value === undefined) return "null";

  if (Array.isArray(value)) {
    return "[" + value.map((x) => stableStringify(x)).join(",") + "]";
  }

  if (typeof value === "object") {
    return (
      "{" +
      Object.keys(value)
        .sort()
        .map((k) => JSON.stringify(k) + ":" + stableStringify(value[k]))
        .join(",") +
      "}"
    );
  }

  return JSON.stringify(value);
}

async function sha256Hex(value) {
  const data = new TextEncoder().encode(String(value || ""));
  const digest = await crypto.subtle.digest("SHA-256", data);
  return [...new Uint8Array(digest)]
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

async function getRootCauseKnowledgeVersion(env, companyId) {
  if (!hasRootCauseCacheKv(env)) return "no_kv";

  const key = `rcv:${String(companyId || "").trim() || "unknown"}`;
  try {
    return (await env.MM_RC_CACHE.get(key)) || "0";
  } catch (e) {
    return "0";
  }
}

async function bumpRootCauseKnowledgeVersion(env, companyId) {
  if (!hasRootCauseCacheKv(env)) return;

  const cid = String(companyId || "").trim();
  if (!cid) return;

  const key = `rcv:${cid}`;
  const value = String(Date.now());

  try {
    await env.MM_RC_CACHE.put(key, value, { expirationTtl: 60 * 60 * 24 * 30 });
  } catch (e) {
    // Cache invalidation must never break ingest/delete workflows.
  }
}

function shouldBypassRootCauseCache({ url, parsed }) {
  if (url.searchParams.get("cache") === "0") return true;
  if (url.searchParams.get("no_cache") === "1") return true;

  const opt = parsed?.options || {};
  if (opt.no_cache === true || opt.no_cache === "true" || opt.no_cache === "yes") return true;
  if (parsed?.no_cache === true || parsed?.no_cache === "true" || parsed?.no_cache === "yes") return true;

  return false;
}

async function buildRootCauseCacheKey({
  env,
  parsed,
  companyId,
  machineId,
  aiScope,
  query,
  language,
  top_k,
  max_causes,
  docIds,
  bubbleDocumentId,
}) {
  const workerCacheVersion = String(env.MM_RC_CACHE_VERSION || "rc_v13").trim() || "rc_v13";
  const companyKnowledgeVersion = await getRootCauseKnowledgeVersion(env, companyId);

  const externalKnowledgeVersion = getExternalKnowledgeVersion(parsed);

  const keyPayload = {
    worker_cache_schema: MM_WORKER_V13_CACHE_SCHEMA,
    worker_cache_version: workerCacheVersion,
    company_knowledge_version: companyKnowledgeVersion,
    external_knowledge_version: externalKnowledgeVersion,
    company_id: String(companyId || "").trim(),
    machine_id: String(machineId || "").trim(),
    ai_scope: String(aiScope || "").trim().toLowerCase(),
    language: String(language || "it").trim().toLowerCase(),
    query: normalizeRootCauseCacheText(query),
    top_k: Number(top_k || 8),
    max_causes: Number(max_causes || 3),
    bubble_document_id: String(bubbleDocumentId || "").trim(),
    document_ids: normalizeRootCauseCacheDocIds(docIds),
  };

  return `rc:${String(companyId || "unknown").trim()}:${await sha256Hex(stableStringify(keyPayload))}`;
}

async function getCachedRootCauseResponse(env, cacheKey) {
  if (!isRootCauseCacheEnabled(env) || !cacheKey) return null;

  try {
    const cached = await env.MM_RC_CACHE.get(cacheKey, "json");
    if (!cached || typeof cached !== "object") return null;
    if (cached.ok !== true) return null;

    return cached;
  } catch (e) {
    return null;
  }
}

async function putCachedRootCauseResponse(env, cacheKey, payload, ttlSeconds) {
  if (!isRootCauseCacheEnabled(env) || !cacheKey) return;
  if (!payload || typeof payload !== "object") return;
  if (payload.ok !== true) return;
  if (String(payload.status || "").trim().toLowerCase() !== "answered") return;
  if (payload?.meta?.cacheable === false) return;
  if (payload?.meta?.technical_failure) return;

  try {
    await env.MM_RC_CACHE.put(cacheKey, JSON.stringify(payload), {
      expirationTtl: ttlSeconds,
    });
  } catch (e) {
    // Cache write must never break user responses.
  }
}

function withRootCauseCacheMeta(payload, metaPatch) {
  const out = { ...(payload || {}) };
  out.meta = {
    ...(out.meta || {}),
    ...(metaPatch || {}),
  };
  return out;
}


function hasAskCacheKv(env) {
  // Reuse the same KV namespace already bound for Root Cause.
  return !!env.MM_RC_CACHE;
}

function isAskCacheEnabled(env) {
  if (!hasAskCacheKv(env)) return false;
  return String(env.MM_ASK_CACHE_ENABLED || "1").trim() !== "0";
}

function getAskCacheTtlSeconds(env) {
  const raw = Number(env.MM_ASK_CACHE_TTL_SECONDS || 300);
  if (!Number.isFinite(raw)) return 300;
  return Math.max(60, Math.min(86400, Math.floor(raw)));
}

function shouldBypassAskCache({ url, parsed }) {
  if (url.searchParams.get("cache") === "0") return true;
  if (url.searchParams.get("no_cache") === "1") return true;

  const opt = parsed?.options || {};
  if (opt.no_cache === true || opt.no_cache === "true" || opt.no_cache === "yes") return true;
  if (parsed?.no_cache === true || parsed?.no_cache === "true" || parsed?.no_cache === "yes") return true;

  return false;
}

async function buildAskCacheKey({
  env,
  parsed,
  companyId,
  machineId,
  aiScope,
  query,
  language,
  top_k,
  docIds,
  bubbleDocumentId,
}) {
  const workerCacheVersion = String(env.MM_ASK_CACHE_VERSION || "ask_v13").trim() || "ask_v13";
  const companyKnowledgeVersion = await getRootCauseKnowledgeVersion(env, companyId);

  const externalKnowledgeVersion = getExternalKnowledgeVersion(parsed);

  const keyPayload = {
    worker_cache_schema: MM_WORKER_V13_CACHE_SCHEMA,
    worker_cache_version: workerCacheVersion,
    company_knowledge_version: companyKnowledgeVersion,
    external_knowledge_version: externalKnowledgeVersion,
    company_id: String(companyId || "").trim(),
    machine_id: String(machineId || "").trim(),
    ai_scope: String(aiScope || "").trim().toLowerCase(),
    language: String(language || "it").trim().toLowerCase(),
    query: normalizeRootCauseCacheText(query),
    top_k: Number(top_k || 5),
    bubble_document_id: String(bubbleDocumentId || "").trim(),
    document_ids: normalizeRootCauseCacheDocIds(docIds),
  };

  return `ask:${String(companyId || "unknown").trim()}:${await sha256Hex(stableStringify(keyPayload))}`;
}

async function getCachedAskResponse(env, cacheKey) {
  if (!isAskCacheEnabled(env) || !cacheKey) return null;

  try {
    const cached = await env.MM_RC_CACHE.get(cacheKey, "json");
    if (!cached || typeof cached !== "object") return null;
    if (cached.ok !== true) return null;

    return cached;
  } catch (e) {
    return null;
  }
}

async function putCachedAskResponse(env, cacheKey, payload, ttlSeconds) {
  if (!isAskCacheEnabled(env) || !cacheKey) return;
  if (!payload || typeof payload !== "object") return;
  if (payload.ok !== true) return;
  if (String(payload.status || "").trim().toLowerCase() !== "answered") return;
  if (payload?.meta?.cacheable === false) return;
  if (payload?.meta?.technical_failure) return;

  try {
    await env.MM_RC_CACHE.put(cacheKey, JSON.stringify(payload), {
      expirationTtl: ttlSeconds,
    });
  } catch (e) {
    // Cache write must never break user responses.
  }
}

function withAskCacheMeta(payload, metaPatch) {
  const out = { ...(payload || {}) };
  out.meta = {
    ...(out.meta || {}),
    ...(metaPatch || {}),
  };
  return out;
}

async function readAndParseBody(request) {
  const contentType = request.headers.get("content-type") || "";
  const raw = await request.text();

  let parsed = null;
  let parseError = null;

  if (raw && raw.trim().length > 0) {
    try {
      parsed = JSON.parse(raw);
      if (typeof parsed === "string") {
        parsed = JSON.parse(parsed);
      }
    } catch (e) {
      parseError = String(e);
      parsed = null;
    }
  }

  return { contentType, raw, parsed, parseError };
}

async function bubbleQuotaCheck({
  env,
  companyId,
  endpoint,
  authToken,
  bubbleDocumentId,
  isInteractiveAiCall = false,
}) {
  const url = (env.BUBBLE_QUOTA_CHECK_URL || "").trim();
  if (!url) {
    // fail-open durante setup
    return { ok: true, status: "ok", limit: 0, used: 0, remaining: 999999 };
  }

  const payload = {
    p_internal_secret: authToken,
    p_company_id: companyId,
    p_endpoint: endpoint,
    p_bubble_document_id: bubbleDocumentId || null,
    p_is_interactive_ai_call: Boolean(isInteractiveAiCall),
  };

  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  let txt = "";
  try {
    txt = await r.text();
  } catch (e) {
    return { ok: true, status: "ok", limit: 0, used: 0, remaining: 999999 };
  }

  let j = null;
  try {
    j = JSON.parse(txt);
  } catch (e) {
    j = null;
  }

  if (j && typeof j === "object" && j.response && typeof j.response === "object") j = j.response;

  if (!j || typeof j !== "object") {
    return { ok: true, status: "ok", limit: 0, used: 0, remaining: 999999 };
  }

  return j;
}

async function handleIngestDocumentFile(request, env, expectedToken) {
  let form;

  try {
    form = await request.formData();
  } catch (e) {
    return jsonResponse(
      { ok: false, status: "error", error: { code: "BAD_REQUEST", message: "Invalid multipart/form-data" } },
      200
    );
  }

  const authToken = getFormString(form, "auth_token");

  if (!expectedToken || authToken !== expectedToken) {
    return jsonResponse(
      { ok: false, status: "error", error: { code: "UNAUTHORIZED", message: "Invalid auth_token" } },
      200
    );
  }

  const companyId = getFormString(form, "company_id");
  const bubbleDocumentId = getFormString(form, "bubble_document_id");
  const aiScope = getFormString(form, "ai_scope") || "company_general";
  const machineId = getFormString(form, "machine_id");
  const documentCategoryCode = normalizeDocumentCategoryCode(
    getFormString(form, "document_category_code") || getFormString(form, "category")
  );
  const documentIsTechnical = normalizeBooleanLike(
    getFormString(form, "document_is_technical") || getFormString(form, "is_tecnico")
  );

  if (!companyId || !bubbleDocumentId || (aiScope !== "company_general" && !machineId)) {
    return jsonResponse(
      {
        ok: false,
        status: "error",
        error: { code: "BAD_REQUEST", message: "company_id, machine_id, bubble_document_id are required" },
        detail: {
          company_id: companyId ? "ok" : "missing",
          machine_id: machineId || aiScope === "company_general" ? "ok" : "missing",
          bubble_document_id: bubbleDocumentId ? "ok" : "missing",
          ai_scope: aiScope || "not_set",
        },
      },
      200
    );
  }

  const file = form.get("file");

  if (!file || typeof file.arrayBuffer !== "function") {
    return jsonResponse(
      { ok: false, status: "error", error: { code: "BAD_REQUEST", message: "Missing file" } },
      200
    );
  }

  const filename = getFormString(form, "filename") || file.name || "document";

  if (!isAllowedIngestFilename(filename)) {
    return jsonResponse(
      {
        ok: false,
        status: "error",
        error: { code: "UNSUPPORTED_FILE_TYPE", message: "Only PDF and XLSX are supported for document ingest" },
      },
      200
    );
  }

  const endpoint = "ingest_document";

  const qc = await bubbleQuotaCheck({
    env,
    companyId,
    endpoint,
    authToken: expectedToken,
    bubbleDocumentId,
    isInteractiveAiCall: false,
  });

  const qcOk = qc?.ok === true || qc?.ok === "true" || qc?.ok === "yes";
  const qcErrCode = String(qc?.error_code || qc?.error?.code || "").trim();
  const ingestCreditQuota = getIngestCreditQuota(qc);
  const ingestRequestKey = getIngestRequestKey({ form, bubbleDocumentId });

  // Backward-compatible safety: Bubble may return either ok=true + quota fields
  // or ok=false + the dedicated reason. Both forms become the same limit response.
  if (!qcOk && isIngestCreditLimitCode(qcErrCode)) {
    return jsonResponse(
      buildIngestCreditLimitPayload({ quota: ingestCreditQuota, requestKey: ingestRequestKey }),
      200
    );
  }

  if (!qcOk) {
    return jsonResponse(
      {
        ok: false,
        status: "error",
        error_code: qcErrCode || "AI_QUOTA_EXCEEDED",
        error: {
          code: qcErrCode || "AI_QUOTA_EXCEEDED",
          message: qc?.error_message || qc?.error?.message || "Daily AI quota exceeded",
        },
        limit: qc?.limit ?? 0,
        used: qc?.used ?? 0,
        remaining: qc?.remaining ?? 0,
      },
      200
    );
  }

  // Post-threshold policy: if the monthly total was already at/over the limit
  // before this new document starts, do not read or forward the file to Cloud Run.
  if (ingestCreditQuota.reached) {
    return jsonResponse(
      buildIngestCreditLimitPayload({ quota: ingestCreditQuota, requestKey: ingestRequestKey }),
      200
    );
  }

  const planEmbedCharsLimitTotal = Number(qc?.embed_chars_limit_total ?? 0) || 0;
  const planIndexStorageLimitBytes = Number(qc?.index_storage_limit_bytes ?? 0) || 0;
  const usedEmbedCharsTotal = Number(qc?.embed_chars_used_total ?? 0) || 0;
  const usedIndexStorageTotal = Number(qc?.index_storage_used_total ?? 0) || 0;
  const docPrevEmbedChars = Number(qc?.doc_prev_embed_chars ?? 0) || 0;
  const docPrevIndexStorageBytes = Number(qc?.doc_prev_index_storage_bytes ?? 0) || 0;

  const contentType =
    getFormString(form, "content_type") ||
    file.type ||
    (filename.toLowerCase().endsWith(".xlsx")
      ? "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
      : "application/pdf");

  let fileBase64 = "";

  try {
    const buffer = await file.arrayBuffer();
    fileBase64 = arrayBufferToBase64(buffer);
  } catch (e) {
    return jsonResponse(
      { ok: false, status: "error", error: { code: "FILE_READ_FAILED", message: String(e && e.message ? e.message : e) } },
      200
    );
  }

  const fileUrl = getFormString(form, "file_url");
  const docId = bubbleDocumentId;

  let crStatus = 0;
  let crText = "";
  let crJson = null;

  try {
    const r = await fetch(`${getCloudRunBase(env)}/v1/ai/ingest/document`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-AI-Internal-Secret": expectedToken,
      },
      body: JSON.stringify({
        file_url: fileUrl,
        file_base64: fileBase64,
        filename,
        content_type: contentType,

        company_id: companyId,
        machine_id: machineId,
        bubble_document_id: bubbleDocumentId,
        ...(aiScope ? { ai_scope: aiScope } : {}),
        document_category_code: documentCategoryCode || null,
        document_is_technical: documentIsTechnical,

        // Cost-based ingest metering context. Keep forwarding OFF until Cloud Run
        // accepts these optional fields, to avoid breaking a strict request schema.
        ...(isIngestCreditsForwardEnabled(env)
          ? {
              ingest_request_key: ingestRequestKey,
              ingest_month_key: ingestCreditQuota.monthKey || null,
              ingest_credits_limit_month: ingestCreditQuota.limit,
              ingest_credits_used_before: ingestCreditQuota.used,
              ingest_credits_enforced: ingestCreditQuota.enforced,
              ingest_request_already_admitted: ingestCreditQuota.requestAlreadyAdmitted,
              ingest_metering_version: MM_INGEST_CREDITS_VERSION,
            }
          : {}),

        plan_embed_chars_limit_total: planEmbedCharsLimitTotal,
        plan_index_storage_limit_bytes: planIndexStorageLimitBytes,
        embed_chars_used_total: usedEmbedCharsTotal,
        index_storage_used_total: usedIndexStorageTotal,
        doc_prev_embed_chars: docPrevEmbedChars,
        doc_prev_index_storage_bytes: docPrevIndexStorageBytes,
      }),
    });

    crStatus = r.status;
    crText = await r.text();

    try {
      crJson = JSON.parse(crText);
    } catch {
      crJson = null;
    }

    if (!r.ok || !crJson || typeof crJson !== "object") {
      return jsonResponse(
        {
          ok: false,
          status: "error",
          error: {
            code: "INGEST_FAILED",
            message: `Cloud Run ingest failed (HTTP ${crStatus})`,
            detail: crJson || crText || null,
          },
          ...getCloudRunIngestMeteringFields({
            crJson,
            quota: ingestCreditQuota,
            requestKey: ingestRequestKey,
          }),
        },
        200
      );
    }

    if (crJson.ok !== true) {
      const reason = getCloudRunLimitReason(crJson);
      const isLimit = isAnyDocumentIngestLimitCode(reason);

      if (isLimit) {
        return jsonResponse(
          {
            ok: false,
            status: "limit_exceeded",
            reason,
            error: {
              code: reason,
              message: crJson?.error?.message || "Limit exceeded",
              detail: crJson,
            },
            text_chars: Number(crJson.text_chars ?? 0),
            pages_detected: Number(crJson.pages_detected ?? 0),
            est_storage_bytes: Number(crJson.est_storage_bytes ?? 0),
            ...getCloudRunIngestMeteringFields({
              crJson,
              quota: ingestCreditQuota,
              requestKey: ingestRequestKey,
            }),
          },
          200
        );
      }

      return jsonResponse(
        {
          ok: false,
          status: "error",
          error: {
            code: crJson?.error?.code || crJson?.error_code || "NOT_INDEXABLE",
            message: crJson?.error?.message || crJson?.error_message || "Documento non indicizzabile",
            detail: crJson,
          },
          text_chars: Number(crJson.text_chars ?? 0),
          pages_detected: Number(crJson.pages_detected ?? 0),
          est_storage_bytes: Number(crJson.est_storage_bytes ?? 0),
          ...getCloudRunIngestMeteringFields({
            crJson,
            quota: ingestCreditQuota,
            requestKey: ingestRequestKey,
          }),
        },
        200
      );
    }

    await bumpRootCauseKnowledgeVersion(env, companyId);

    return jsonResponse(
      {
        ok: true,
        status: "indexed",
        source_id: `mm:doc:${docId}`,
        text_chars: Number(crJson.text_chars ?? 0),
        pages_detected: Number(crJson.pages_detected ?? 0),
        est_storage_bytes: Number(crJson.est_storage_bytes ?? 0),
        electrical_candidate: crJson.electrical_candidate === true,
        electrical_pipeline_enabled: crJson.electrical_pipeline_enabled === true,
        electrical_document_id: crJson.electrical_document_id ?? null,
        electrical_index_status: String(crJson.electrical_index_status || "not_applicable"),
        electrical_latest_version_no: Number(crJson.electrical_latest_version_no ?? 0),
        electrical_registry_error: crJson.electrical_registry_error || null,
        ...getCloudRunIngestMeteringFields({
          crJson,
          quota: ingestCreditQuota,
          requestKey: ingestRequestKey,
        }),
        root_cause_cache_invalidated: true,
        ask_cache_invalidated: true,
        ai_cache_invalidated: true,
      },
      200
    );
  } catch (e) {
    return jsonResponse(
      {
        ok: false,
        status: "error",
        error: { code: "INGEST_FAILED", message: "Cloud Run request failed", detail: String(e) },
        ...getCloudRunIngestMeteringFields({
          crJson,
          quota: ingestCreditQuota,
          requestKey: ingestRequestKey,
        }),
      },
      200
    );
  }
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const askTrace = url.pathname === "/v1/ai/ask" ? mmAskTrace() : null;

    if (
      url.pathname !== "/v1/ai/ingest/document" &&
      url.pathname !== "/v1/ai/ingest/document-file" &&
      url.pathname !== "/v1/ai/ingest/source" &&
      url.pathname !== "/v1/ai/ask" &&
      url.pathname !== "/v1/ai/usage/snapshot" &&
      url.pathname !== "/v1/ai/draft_ps" &&
      url.pathname !== "/v1/ai/root-cause" &&
      url.pathname !== "/v1/ai/smart-diagnostic/start" &&
      url.pathname !== "/v1/ai/smart-diagnostic/answer" &&
      url.pathname !== "/v1/ai/smart-diagnostic/finalize" &&
      url.pathname !== "/v1/ai/delete/document" &&
      url.pathname !== "/v1/ai/delete/company-index" &&
      url.pathname !== "/v1/ai/delete/company_index"
    ) {
      return jsonResponse(
        { ok: false, status: "error", error: { code: "NOT_FOUND", message: "Endpoint not found" } },
        404
      );
    }

    if (request.method !== "POST") {
      return jsonResponse(
        { ok: false, status: "error", error: { code: "METHOD_NOT_ALLOWED", message: "Use POST" } },
        405
      );
    }

    const expectedToken = env.AI_INTERNAL_SECRET || "";
    if (!expectedToken) {
      return jsonResponse(
        {
          ok: false,
          status: "error",
          error: { code: "CONFIG_MISSING", message: "AI_INTERNAL_SECRET not set in Worker env" },
        },
        200
      );
    }

    const debugOn =
      url.searchParams.get("debug") === "1" || (request.headers.get("x-debug") || "") === "1";

    if (url.pathname === "/v1/ai/ingest/document-file") {
      return await handleIngestDocumentFile(request, env, expectedToken);
    }

    const { contentType, raw, parsed, parseError } = await readAndParseBody(request);

    const tokenFromBody =
      parsed && typeof parsed === "object" && parsed !== null ? parsed.auth_token || "" : "";
    const tokenFromHeader = getBearerToken(request);
    const receivedToken = (tokenFromBody || tokenFromHeader || "").trim();

    if (receivedToken !== expectedToken) {
      return jsonResponse(
        {
          ok: false,
          status: "error",
          error: { code: "UNAUTHORIZED", message: "Invalid auth token" },
        },
        200
      );
    }

    const usageBoundary = mmUsageBoundary(request, env, parsed);
    if (usageBoundary?.error) return usageBoundary.error;
    const usageFetch = mmUsageFetch(usageBoundary);
    if (url.pathname === "/v1/ai/usage/snapshot") {
      return usageFetch(`${getCloudRunBase(env)}/v1/ai/usage/snapshot`, {
        method:"POST", headers:{"Content-Type":"application/json","X-AI-Internal-Secret":expectedToken},
        body:JSON.stringify({company_id:parsed.company_id})});
    }

    // B4l precedes debug, quota and ALL Worker response-cache access.
    let mmRequestAuthority = null;
    if (url.pathname === "/v1/ai/ask") {
      mmAskSpanStart(askTrace, "preflight");
      mmRequestAuthority = await mmB4lPreflight(request, env, parsed, expectedToken, debugOn);
      mmAskSpanEnd(askTrace);
      if (mmRequestAuthority instanceof Response)
        return mmAskResponse(await mmRequestAuthority.json(), mmRequestAuthority.status, askTrace);
    }

    // Legacy debug is unchanged only while request authority is OFF.
    if (debugOn) {
      return jsonResponse(
        {
          ok: true,
          status: "debug",
          method: request.method,
          content_type: contentType,
          raw_body_len: raw.length,
          raw_body_head: safeSlice(raw, 600),
          parse_error: parseError,
          parsed_type: parsed === null ? "null" : typeof parsed,
          parsed_keys:
            parsed && typeof parsed === "object" && parsed !== null ? Object.keys(parsed) : null,
          received_auth_from: tokenFromBody
            ? "body.auth_token"
            : tokenFromHeader
              ? "header.authorization"
              : "none",
          auth_match: true,
          path: url.pathname,
          cloud_run_base: getCloudRunBase(env), // ✅ utile per debug
        },
        200
      );
    }

    if (!parsed || typeof parsed !== "object") {
      return jsonResponse(
        {
          ok: false,
          status: "error",
          error: { code: "BAD_REQUEST", message: "Invalid or missing JSON body" },
        },
        200
      );
    }

    // ROUTE: /v1/ai/delete/company-index  (REAL -> Cloud Run)
    // Nota: sta prima del QUOTA CHECK perché una cancellazione indice non deve
    // essere bloccata da limiti giornalieri AI o limiti piano.
    const isDeleteCompanyIndexPath =
      url.pathname === "/v1/ai/delete/company-index" ||
      url.pathname === "/v1/ai/delete/company_index";

    if (isDeleteCompanyIndexPath) {
      const companyIdForCompanyIndex = String(parsed?.company_id || parsed?.company?.id || "").trim();

      if (!companyIdForCompanyIndex) {
        return jsonResponse(
          {
            ok: false,
            status: "error",
            error: { code: "BAD_REQUEST", message: "company_id is required" },
          },
          200
        );
      }

      let crStatus = 0;
      let crText = "";
      let crJson = null;

      try {
        const r = await usageFetch(`${getCloudRunBase(env)}/v1/ai/delete/company-index`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-AI-Internal-Secret": expectedToken,
          },
          body: JSON.stringify({
            company_id: companyIdForCompanyIndex,
          }),
        });

        crStatus = r.status;
        crText = await r.text();

        try {
          crJson = JSON.parse(crText);
        } catch {
          crJson = null;
        }

        if (!r.ok || !crJson || typeof crJson !== "object") {
          return jsonResponse(
            {
              ok: false,
              status: "error",
              error: {
                code: "DELETE_COMPANY_INDEX_FAILED",
                message: `Cloud Run company index delete failed (HTTP ${crStatus})`,
                detail: crJson || crText || null,
              },
            },
            200
          );
        }

        if (crJson.ok !== true) {
          return jsonResponse(
            {
              ok: false,
              status: "error",
              error: {
                code: crJson?.error?.code || crJson?.error_code || "DELETE_COMPANY_INDEX_FAILED",
                message:
                  crJson?.error?.message ||
                  crJson?.error_message ||
                  "Company index delete failed",
                detail: crJson,
              },
            },
            200
          );
        }

        await bumpRootCauseKnowledgeVersion(env, companyIdForCompanyIndex);

        return jsonResponse(
          {
            ok: true,
            status: crJson.status || "deleted",
            deleted: crJson.deleted || {},
            meta: {
              company_id: companyIdForCompanyIndex,
              root_cause_cache_invalidated: true,
              ask_cache_invalidated: true,
              ai_cache_invalidated: true,
            },
          },
          200
        );
      } catch (e) {
        return jsonResponse(
          {
            ok: false,
            status: "error",
            error: {
              code: "DELETE_COMPANY_INDEX_FAILED",
              message: "Cloud Run request failed",
              detail: String(e),
            },
          },
          200
        );
      }
    }

    // QUOTA CHECK    
    const companyId = String(parsed?.company_id || parsed?.company?.id || "").trim();
    if (!companyId) {
      return jsonResponse(
        { ok: false, status: "error", error: { code: "BAD_REQUEST", message: "company_id is required" } },
        200
      );
    }

    const endpoint =
      url.pathname === "/v1/ai/ask"
        ? "ask"
        : url.pathname === "/v1/ai/root-cause"
          ? "root_cause"
          : url.pathname === "/v1/ai/draft_ps"
            ? "draft_ps"
            : url.pathname === "/v1/ai/smart-diagnostic/start" ||
              url.pathname === "/v1/ai/smart-diagnostic/answer" ||
              url.pathname === "/v1/ai/smart-diagnostic/finalize"
              ? "smart_diagnostic"
              : url.pathname === "/v1/ai/delete/document"
                ? "delete_document"
                : url.pathname === "/v1/ai/ingest/source"
                  ? "ingest_document"
                  : "ingest_document";

    const isInteractiveAiCall =
      endpoint === "ask" ||
      endpoint === "root_cause" ||
      endpoint === "draft_ps" ||
      endpoint === "smart_diagnostic";

    const bubbleDocumentId =
      String(parsed?.bubble_document_id || parsed?.bubble_document || "").trim() || null;

    const quotaArgs = {env, companyId, endpoint, authToken:expectedToken, bubbleDocumentId, isInteractiveAiCall};
    const qc = usageBoundary && isInteractiveAiCall
      ? {ok:true} // NOT an admission: /usage/execute rejects before AI if quota is unavailable.
      : endpoint === "ask"
        ? await mmAskQuotaCheck({...quotaArgs, trace:askTrace})
        : await bubbleQuotaCheck(quotaArgs);

    const qcOk = qc?.ok === true || qc?.ok === "true" || qc?.ok === "yes";
    const qcErrCode = String(qc?.error_code || qc?.error?.code || "").trim();
    if (endpoint === "ask" && !qcOk) {
      return mmAskError(qcErrCode || "AI_QUOTA_EXCEEDED", askTrace, {
        limit:qc?.limit ?? 0, used:qc?.used ?? 0, remaining:qc?.remaining ?? 0
      });
    }
    const ingestCreditQuota = getIngestCreditQuota(qc);
    const ingestRequestKey = getIngestRequestKey({ parsed, bubbleDocumentId });
    const isDocumentIngestPath = url.pathname === "/v1/ai/ingest/document";

    // The dedicated monthly ingest-credit limit applies only to uploaded Documents,
    // not to /ingest/source (procedures, steps and other structured sources).
    if (!qcOk && isDocumentIngestPath && isIngestCreditLimitCode(qcErrCode)) {
      return jsonResponse(
        buildIngestCreditLimitPayload({ quota: ingestCreditQuota, requestKey: ingestRequestKey }),
        200
      );
    }

    if (!qcOk) {
      return jsonResponse(
        {
          ok: false,
          status: "error",
          error_code: qcErrCode || "AI_QUOTA_EXCEEDED",
          error: {
            code: qcErrCode || "AI_QUOTA_EXCEEDED",
            message: qc?.error_message || qc?.error?.message || "Daily AI quota exceeded",
          },
          limit: qc?.limit ?? 0,
          used: qc?.used ?? 0,
          remaining: qc?.remaining ?? 0,
        },
        200
      );
    }

    // Post-threshold policy: a document already in progress may finish and push
    // the total over the limit; only the next new document is blocked.
    if (isDocumentIngestPath && ingestCreditQuota.reached) {
      return jsonResponse(
        buildIngestCreditLimitPayload({ quota: ingestCreditQuota, requestKey: ingestRequestKey }),
        200
      );
    }

    // ROUTE: /v1/ai/smart-diagnostic/*  (REAL -> Cloud Run)
    const isSmartDiagnosticPath =
      url.pathname === "/v1/ai/smart-diagnostic/start" ||
      url.pathname === "/v1/ai/smart-diagnostic/answer" ||
      url.pathname === "/v1/ai/smart-diagnostic/finalize";

    if (isSmartDiagnosticPath) {
      const machineId = String(parsed?.machine_id || parsed?.scope?.machine_id || "").trim();
      const sessionId = String(parsed?.session_id || parsed?.smart_diagnostic_session_id || "").trim();
      const language = String(parsed?.language || "it").trim().toLowerCase() || "it";

      if (!machineId) {
        return jsonResponse(
          { ok: false, status: "error", error: { code: "BAD_REQUEST", message: "machine_id is required" } },
          200
        );
      }

      if (!sessionId) {
        return jsonResponse(
          { ok: false, status: "error", error: { code: "BAD_REQUEST", message: "session_id is required" } },
          200
        );
      }

      let bodyForCloudRun = null;

      if (url.pathname === "/v1/ai/smart-diagnostic/start") {
        const symptomText = String(parsed?.symptom_text || parsed?.query || parsed?.symptom || "").trim();
        if (!symptomText) {
          return jsonResponse(
            { ok: false, status: "error", error: { code: "BAD_REQUEST", message: "symptom_text is required" } },
            200
          );
        }

        bodyForCloudRun = {
          company_id: companyId,
          machine_id: machineId,
          session_id: sessionId,
          symptom_text: symptomText,
          language,
          context: parsed?.context || null,
          options: parsed?.options || {
            max_questions: 6,
            max_hypotheses: 4,
            top_k: 8,
          },
          debug: false,
        };
      }

      if (url.pathname === "/v1/ai/smart-diagnostic/answer") {
        const questionId = String(parsed?.question_id || parsed?.current_question_id || "").trim();
        const stateJson = parsed?.state_json ?? parsed?.session_state_json ?? "";
        const answer = parsed?.answer || {
          value: String(parsed?.answer_value || "").trim(),
          api_value: String(parsed?.answer_api_value || parsed?.selected_option_id || parsed?.answer_value || "").trim(),
          label: String(parsed?.answer_label || "").trim(),
          free_text: String(parsed?.answer_free_text || "").trim(),
        };

        if (!questionId) {
          return jsonResponse(
            { ok: false, status: "error", error: { code: "BAD_REQUEST", message: "question_id is required" } },
            200
          );
        }

        if (!stateJson) {
          return jsonResponse(
            { ok: false, status: "error", error: { code: "BAD_REQUEST", message: "state_json/session_state_json is required" } },
            200
          );
        }

        bodyForCloudRun = {
          company_id: companyId,
          machine_id: machineId,
          session_id: sessionId,
          question_id: questionId,
          answer,
          state_json: stateJson,
          language,
          debug: false,
        };
      }

      if (url.pathname === "/v1/ai/smart-diagnostic/finalize") {
        const stateJson = parsed?.state_json ?? parsed?.session_state_json ?? "";
        if (!stateJson) {
          return jsonResponse(
            { ok: false, status: "error", error: { code: "BAD_REQUEST", message: "state_json/session_state_json is required" } },
            200
          );
        }

        bodyForCloudRun = {
          company_id: companyId,
          machine_id: machineId,
          session_id: sessionId,
          state_json: stateJson,
          language,
          debug: false,
        };
      }

      let crStatus = 0;
      let crText = "";
      let crJson = null;

      try {
        const r = await usageFetch(`${getCloudRunBase(env)}${url.pathname}`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-AI-Internal-Secret": expectedToken,
          },
          body: JSON.stringify(bodyForCloudRun),
        });

        crStatus = r.status;
        crText = await r.text();

        try {
          crJson = JSON.parse(crText);
        } catch {
          crJson = null;
        }

        if (!r.ok || !crJson || typeof crJson !== "object") {
          const errorCode = "SMART_DIAGNOSTIC_FAILED";
          const errorMessage =
            `Cloud Run Smart Diagnostic failed (HTTP ${crStatus})`;

          return jsonResponse(
            {
              ok: false,
              status: "error",

              // Campi flat leggibili direttamente da Bubble
              error_code: errorCode,
              error_message: errorMessage,

              // Manteniamo anche l'oggetto nested esistente
              error: {
                code: errorCode,
                message: errorMessage,
                detail: crJson || crText || null,
              },

              raw_payload_json: JSON.stringify(crJson || {}),
            },
            200
          );
        }

        if (crJson.ok !== true) {
          const errorCode = String(
            crJson?.error_code ||
            crJson?.error?.code ||
            "SMART_DIAGNOSTIC_FAILED"
          ).trim();

          const errorMessage = String(
            crJson?.error_message ||
            crJson?.error?.message ||
            "Smart Diagnostic failed"
          ).trim();

          return jsonResponse(
            {
              ok: false,
              status: "error",

              // Campi flat leggibili direttamente da Bubble
              error_code: errorCode,
              error_message: errorMessage,

              // Manteniamo anche l'oggetto nested esistente
              error: {
                code: errorCode,
                message: errorMessage,
                detail: crJson,
              },

              raw_payload_json: JSON.stringify(crJson),
            },
            200
          );
        }

        return jsonResponse(
          {
            ...crJson,

            // Valori non vuoti: Bubble li acquisisce durante Reinitialize call
            error_code: "NONE",
            error_message: "NONE",

            raw_payload_json: JSON.stringify(crJson),

            meta: {
              ...(crJson.meta || {}),
              endpoint: endpoint,
              language: crJson.language || language,
            },
          },
          200
        );
      } catch (e) {
        const errorCode = "SMART_DIAGNOSTIC_FAILED";
        const errorMessage = "Cloud Run request failed";

        return jsonResponse(
          {
            ok: false,
            status: "error",

            // Campi flat leggibili direttamente da Bubble
            error_code: errorCode,
            error_message: errorMessage,

            // Manteniamo anche l'oggetto nested esistente
            error: {
              code: errorCode,
              message: errorMessage,
              detail: String(e),
            },

            raw_payload_json: JSON.stringify({
              error: String(e),
            }),
          },
          200
        );
      }
    }

    // ROUTE: /v1/ai/draft_ps  (REAL -> Cloud Run)
    if (url.pathname === "/v1/ai/draft_ps") {
      const query = String(parsed?.query || parsed?.prompt || parsed?.context?.user_prompt || "").trim();
      if (!query) {
        return jsonResponse(
          { ok: false, status: "error", error: { code: "BAD_REQUEST", message: "query is required" } },
          200
        );
      }

      const aiScope = getAiScope(parsed);
      const machineId = String(parsed?.machine_id || parsed?.scope?.machine_id || "").trim();
      if (aiScope !== "company_general" && !machineId) {
        return jsonResponse(
          { ok: false, status: "error", error: { code: "BAD_REQUEST", message: "machine_id is required" } },
          200
        );
      }

      const language = String(parsed?.language || "it").trim().toLowerCase() || "it";

      const top_k_raw = parsed?.options?.top_k;
      const top_k = Number.isFinite(Number(top_k_raw)) ? Number(top_k_raw) : 8;

      const max_causes_raw = parsed?.options?.max_causes;
      const max_causes = Number.isFinite(Number(max_causes_raw)) ? Number(max_causes_raw) : 3;

      let crStatus = 0;
      let crText = "";
      let crJson = null;

      try {
        const rawDocIds = parsed?.document_ids ?? parsed?.scope?.document_ids ?? null;
        const docIds = shouldForwardDocumentIds(aiScope) ? normalizeDocumentIds(rawDocIds) : null;

        const r = await usageFetch(`${getCloudRunBase(env)}/v1/ai/draft_ps`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-AI-Internal-Secret": expectedToken,
          },
          body: JSON.stringify({
            company_id: companyId,
            machine_id: machineId,
            ...(aiScope ? { ai_scope: aiScope } : {}),
            query,
            language,
            document_ids: docIds,
            options: {
              top_k,
              max_causes,
            },
            debug: false,
          }),
        });

        crStatus = r.status;
        crText = await r.text();

        try {
          crJson = JSON.parse(crText);
        } catch {
          crJson = null;
        }

        if (!r.ok || !crJson || typeof crJson !== "object") {
          return jsonResponse(
            {
              ok: false,
              status: "error",
              error: {
                code: "DRAFT_PS_FAILED",
                message: `Cloud Run draft_ps failed (HTTP ${crStatus})`,
                detail: crJson || crText || null,
              },
            },
            200
          );
        }

        if (crJson.ok !== true) {
          return jsonResponse(
            {
              ok: false,
              status: "error",
              error: {
                code: crJson?.error?.code || crJson?.error_code || "DRAFT_PS_FAILED",
                message: crJson?.error?.message || crJson?.error_message || "Draft P&S failed",
                detail: crJson,
              },
            },
            200
          );
        }

        const possibleCauses = Array.isArray(crJson.possible_causes) ? crJson.possible_causes : [];
        const citations = Array.isArray(crJson.citations) ? crJson.citations : [];
        const rgLinks = Array.isArray(crJson.rg_links) ? crJson.rg_links : [];

        const solutionText = possibleCauses
          .map((c, idx) => {
            const rank = Number(c?.rank ?? idx + 1);
            const cause = String(c?.cause || "").trim();
            const why = String(c?.why || "").trim();
            const checks = Array.isArray(c?.checks) ? c.checks : [];

            const checksText = checks
              .map((x) => `- ${String(x || "").trim()}`)
              .filter(Boolean)
              .join("\n");

            return [
              `[Causa #${rank}] ${cause}`,
              "",
              "Perché:",
              why,
              "",
              "Controlli:",
              checksText || "-",
            ].join("\n");
          })
          .filter(Boolean)
          .join("\n\n");

        const cleanSnippet = (value, maxLen = 260) => {
          const s = String(value || "").replace(/\s+/g, " ").trim();
          return s.length > maxLen ? `${s.slice(0, maxLen - 1).trim()}…` : s;
        };

        const fallbackCitationsText = citations
          .map((c) => {
            const label = cleanSnippet(
              c?.display_label ||
              c?.display_title ||
              c?.citation_id ||
              "Fonte",
              160
            );
            const snippet = cleanSnippet(c?.snippet_clean || c?.snippet || "", 260);
            return snippet ? `- ${label} — ${snippet}` : `- ${label}`;
          })
          .filter(Boolean)
          .join("\n");

        const citationsText = String(
          crJson.citations_text_clean ||
          crJson.notes_clean ||
          crJson.citations_text ||
          fallbackCitationsText ||
          ""
        ).trim();

        const linksText = String(crJson.links_text_clean || crJson.links_text || "").trim();
        const notesClean = String(crJson.notes_clean || citationsText || "").trim();

        const rawPayloadJson = JSON.stringify(crJson);

        return jsonResponse(
          {
            ok: true,
            status: crJson.status || "drafted",
            title: String(crJson.title || ""),
            problem_summary: String(crJson.problem_summary || ""),
            solution_text: solutionText,
            citations_text: citationsText,
            links_text: linksText,
            citations_text_clean: citationsText,
            links_text_clean: linksText,
            notes_clean: notesClean,
            raw_payload_json: rawPayloadJson,
            possible_causes: possibleCauses,
            citations: citations,
            rg_links: rgLinks,
            meta: {
              top_k: crJson?.meta?.top_k ?? top_k,
              max_causes: crJson?.meta?.max_causes ?? max_causes,
              similarity_max: crJson?.meta?.similarity_max ?? null,
              chat_model: crJson?.meta?.chat_model ?? null,
              language: crJson?.meta?.language ?? language,
              reason: crJson?.meta?.reason ?? null,
            },
          },
          200
        );
      } catch (e) {
        return jsonResponse(
          {
            ok: false,
            status: "error",
            error: { code: "DRAFT_PS_FAILED", message: "Cloud Run request failed", detail: String(e) },
          },
          200
        );
      }
    }

    // ROUTE: /v1/ai/root-cause  (REAL -> Cloud Run)
    if (url.pathname === "/v1/ai/root-cause") {
      const query = String(parsed?.query || parsed?.prompt || parsed?.context?.user_prompt || "").trim();
      if (!query) {
        return jsonResponse(
          { ok: false, status: "error", error: { code: "BAD_REQUEST", message: "query is required" } },
          200
        );
      }

      const aiScope = getAiScope(parsed);
      const machineId = String(parsed?.machine_id || parsed?.scope?.machine_id || "").trim();
      if (aiScope !== "company_general" && !machineId) {
        return jsonResponse(
          { ok: false, status: "error", error: { code: "BAD_REQUEST", message: "machine_id is required" } },
          200
        );
      }

      const top_k_raw = parsed?.options?.top_k ?? parsed?.top_k;
      const top_k = Number.isFinite(Number(top_k_raw)) ? Number(top_k_raw) : 8;

      const max_causes_raw = parsed?.options?.max_causes ?? parsed?.max_causes;
      const max_causes = Number.isFinite(Number(max_causes_raw)) ? Number(max_causes_raw) : 3;
      const language = String(parsed?.language || "").trim().toLowerCase();
      const cacheLanguage = language || "auto";

      let crStatus = 0;
      let crText = "";
      let crJson = null;

      try {
        const rawDocIds = parsed?.document_ids ?? parsed?.scope?.document_ids ?? null;
        const docIds = shouldForwardDocumentIds(aiScope) ? normalizeDocumentIds(rawDocIds) : null;

        let rcCacheKey = "";
        const rcCacheTtl = getRootCauseCacheTtlSeconds(env);
        const rcCacheBypass = !!usageBoundary || shouldBypassRootCauseCache({ url, parsed });

        if (isRootCauseCacheEnabled(env) && !rcCacheBypass) {
          rcCacheKey = await buildRootCauseCacheKey({
            env,
            parsed,
            companyId,
            machineId,
            aiScope,
            query,
            language: cacheLanguage,
            top_k,
            max_causes,
            docIds,
            bubbleDocumentId,
          });

          const cachedPayload = await getCachedRootCauseResponse(env, rcCacheKey);
          if (cachedPayload) {
            return jsonResponse(
              withRootCauseCacheMeta(cachedPayload, {
                cached: true,
                cache_status: "hit",
                cache_layer: "worker_exact",
                cache_hit_at: new Date().toISOString(),
                cache_ttl_seconds: rcCacheTtl,
                v13_origin_route: cachedPayload?.meta?.v13_origin_route || cachedPayload?.meta?.v13_route || null,
                v13_origin_llm_calls: cachedPayload?.meta?.v13_origin_llm_calls ?? cachedPayload?.meta?.v13_llm_calls ?? null,
                v13_origin_estimated_cost_usd: cachedPayload?.meta?.v13_origin_estimated_cost_usd ?? cachedPayload?.meta?.v13_estimated_cost_usd ?? null,
                v13_origin_elapsed_seconds: cachedPayload?.meta?.v13_origin_elapsed_seconds ?? cachedPayload?.meta?.v13_elapsed_seconds ?? null,
                v13_origin_chat_model: cachedPayload?.meta?.v13_origin_chat_model || cachedPayload?.meta?.chat_model || null,
                v13_route: "worker_exact_cache",
                v13_llm_calls: 0,
                v13_estimated_cost_usd: 0,
                v13_elapsed_seconds: 0,
                chat_model: "worker_exact_cache",
              }),
              200
            );
          }
        }

        const r = await usageFetch(`${getCloudRunBase(env)}/v1/ai/root-cause`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-AI-Internal-Secret": expectedToken,
          },
          body: JSON.stringify({
            company_id: companyId,
            machine_id: machineId,
            ...(aiScope ? { ai_scope: aiScope } : {}),
            query,
            language: language || null,
            top_k,
            max_causes,
            bubble_document_id: bubbleDocumentId,
            document_ids: docIds,
            knowledge_version: getExternalKnowledgeVersion(parsed) || null,
            debug: false,
          }),
        });

        crStatus = r.status;
        crText = await r.text();

        try {
          crJson = JSON.parse(crText);
        } catch {
          crJson = null;
        }

        if (!r.ok || !crJson || typeof crJson !== "object") {
          const isTimeout = [408, 502, 503, 504, 524].includes(Number(crStatus));
          const errorCode = isTimeout ? "ROOT_CAUSE_TIMEOUT" : "ROOT_CAUSE_FAILED";
          const errorMessage = isTimeout
            ? "Tempo massimo di analisi superato. Riprova."
            : `Cloud Run root-cause failed (HTTP ${crStatus})`;
          return jsonResponse(
            {
              ok: false,
              status: "error",
              error_code: errorCode,
              error_message: errorMessage,
              error: {
                code: errorCode,
                message: errorMessage,
                detail: crJson || crText || null,
              },
            },
            200
          );
        }

        if (crJson.ok !== true) {
          const errorCode = String(
            crJson?.error?.code || crJson?.error_code || "ROOT_CAUSE_FAILED"
          ).trim();
          const errorMessage = String(
            crJson?.error?.message || crJson?.error_message || "Root cause failed"
          ).trim();
          return jsonResponse(
            {
              ok: false,
              status: "error",
              error_code: errorCode,
              error_message: errorMessage,
              error: {
                code: errorCode,
                message: errorMessage,
                detail: crJson,
              },
            },
            200
          );
        }

        const possibleCauses = Array.isArray(crJson.possible_causes) ? crJson.possible_causes : [];
        const recommendedNextChecks = Array.isArray(crJson.recommended_next_checks)
          ? crJson.recommended_next_checks
          : [];
        const citations = Array.isArray(crJson.citations) ? crJson.citations : [];
        const rgLinks = Array.isArray(crJson.rg_links) ? crJson.rg_links : [];

        let possibleCausesText = possibleCauses
          .map((c, idx) => {
            const rank = Number(c?.rank ?? idx + 1);
            const cause = String(c?.cause || "").trim();
            const why = String(c?.why || "").trim();
            const checks = Array.isArray(c?.checks) ? c.checks : [];

            const checksText = checks
              .map((x) => `- ${String(x || "").trim()}`)
              .filter(Boolean)
              .join("\n");

            return [
              `${rank}. ${cause}`,
              `Perché: ${why || "-"}`,
              `Controlli consigliati:`,
              checksText || "-",
            ].join("\n");
          })
          .filter(Boolean)
          .join("\n\n");

        const effectiveMode = String(crJson.effective_mode || "root_cause").trim().toLowerCase();
        const routedAnswer = String(crJson.answer || "").trim();
        if (!possibleCausesText && effectiveMode === "ask" && routedAnswer) {
          // Backward compatibility for the current Bubble Root Cause text box.
          possibleCausesText = routedAnswer;
        }

        const recommendedNextChecksText = recommendedNextChecks
          .map((x) => `- ${String(x || "").trim()}`)
          .filter(Boolean)
          .join("\n");

        const citationsText = citations
          .map((c) =>
            [
              `- [${String(c?.citation_id || "").trim()}] doc=${String(c?.bubble_document_id || "").trim()} p.${String(c?.page_from ?? "")}-${String(c?.page_to ?? "")}`,
              `  ${String(c?.snippet || "").trim()}`,
            ].join("\n")
          )
          .filter(Boolean)
          .join("\n\n");

        const linksText = rgLinks
          .map((l) => `- [${String(l?.citation_id || "").trim()}] ${String(l?.url || "").trim()}`)
          .filter(Boolean)
          .join("\n");

        const rawPayloadJson = JSON.stringify(crJson);

        const responsePayload = {
          ok: true,
          status: crJson.status || "answered",
          symptom: String(crJson.symptom || query),
          problem_summary: String(crJson.problem_summary || ""),
          answer: routedAnswer,
          answer_html: String(crJson.answer_html || ""),
          answer_format: String(crJson.answer_format || (crJson.answer_html ? "html" : "text")),
          answer_render_version: String(crJson.answer_render_version || ""),
          requested_mode: String(crJson.requested_mode || "root_cause"),
          effective_mode: effectiveMode,
          routed: crJson.routed === true,
          result_code: String(crJson.result_code || ""),
          request_kind: String(crJson.request_kind || ""),
          evidence_state: String(crJson.evidence_state || ""),
          language: crJson.language || crJson?.meta?.language || language || "it",
          possible_causes_text: possibleCausesText,
          recommended_next_checks_text: recommendedNextChecksText,
          citations_text: citationsText,
          links_text: linksText,
          raw_payload_json: rawPayloadJson,
          possible_causes: possibleCauses,
          recommended_next_checks: recommendedNextChecks,
          citations: citations,
          rg_links: rgLinks,
          meta: {
            ...(crJson?.meta || {}),
            top_k: crJson?.top_k ?? crJson?.meta?.top_k ?? top_k,
            max_causes,
            similarity_max: crJson?.similarity_max ?? crJson?.meta?.similarity_max ?? null,
            chat_model: crJson?.chat_model ?? crJson?.meta?.chat_model ?? null,
            language: crJson.language || crJson?.meta?.language || language || "it",
            cached: false,
            cache_status: rcCacheBypass ? "bypass" : rcCacheKey ? "miss_stored" : "disabled",
            cache_ttl_seconds: rcCacheTtl,
            worker_cache_schema: MM_WORKER_V13_CACHE_SCHEMA,
          },
        };

        if (rcCacheKey && !rcCacheBypass) {
          await putCachedRootCauseResponse(
            env,
            rcCacheKey,
            withRootCauseCacheMeta(responsePayload, {
              cached: false,
              cache_status: "stored",
              cache_stored_at: new Date().toISOString(),
              cache_ttl_seconds: rcCacheTtl,
            }),
            rcCacheTtl
          );
        }

        return jsonResponse(responsePayload, 200);
      } catch (e) {
        const isTimeout = isLikelyTimeoutException(e);
        const errorCode = isTimeout ? "ROOT_CAUSE_TIMEOUT" : "ROOT_CAUSE_FAILED";
        const errorMessage = isTimeout
          ? "Tempo massimo di analisi superato. Riprova."
          : "Cloud Run request failed";
        return jsonResponse(
          {
            ok: false,
            status: "error",
            error_code: errorCode,
            error_message: errorMessage,
            error: { code: errorCode, message: errorMessage, detail: String(e) },
          },
          200
        );
      }
    }

    // ROUTE: /v1/ai/ask  (REAL -> Cloud Run)
    if (url.pathname === "/v1/ai/ask") {
      const askJsonResponse = (payload, status = 200) => mmAskResponse(payload, status, askTrace);
      const query = String(parsed?.query || "").trim();
      if (!query) {
        return askJsonResponse(
          { ok: false, status: "error", error: { code: "BAD_REQUEST", message: "query is required" } },
          200
        );
      }

      // machine_id obbligatorio solo per scope macchina.
      // Per ai_scope=company_general la pagina Machines può chiamare senza macchina.
      const aiScope = getAiScope(parsed);
      const machineId = String(parsed?.machine_id || parsed?.scope?.machine_id || "").trim();
      if (aiScope !== "company_general" && !machineId) {
        return askJsonResponse(
          { ok: false, status: "error", error: { code: "BAD_REQUEST", message: "machine_id is required" } },
          200
        );
      }

      // top_k: options.top_k oppure default 5
      const top_k_raw = parsed?.options?.top_k ?? parsed?.top_k;
      const top_k = Number.isFinite(Number(top_k_raw)) ? Number(top_k_raw) : 5;
      const language = String(parsed?.language || "").trim().toLowerCase();
      const cacheLanguage = language || "auto";

      let crStatus = 0;
      let crText = "";
      let crJson = null;

      try {
        // document_ids: accetta array o stringa comma-separated da Bubble.
        // Con ai_scope=machine_all/company_general non inoltriamo document_ids,
        // perché altrimenti Cloud Run restringerebbe lo scope ai soli documenti passati.
        const rawDocIds = parsed?.document_ids ?? parsed?.scope?.document_ids ?? null;
        const docIds = shouldForwardDocumentIds(aiScope) ? normalizeDocumentIds(rawDocIds) : null;

        let askCacheKey = "";
        const askCacheTtl = getAskCacheTtlSeconds(env);
        const askCacheBypass = !!mmRequestAuthority || shouldBypassAskCache({ url, parsed });

        if (isAskCacheEnabled(env) && !askCacheBypass) {
          askCacheKey = await buildAskCacheKey({
            env,
            parsed,
            companyId,
            machineId,
            aiScope,
            query,
            language: cacheLanguage,
            top_k,
            docIds,
            bubbleDocumentId,
          });

          const cachedPayload = await getCachedAskResponse(env, askCacheKey);
          if (cachedPayload) {
            return askJsonResponse(
              withAskCacheMeta(cachedPayload, {
                cached: true,
                cache_status: "hit",
                cache_layer: "worker_exact",
                cache_hit_at: new Date().toISOString(),
                cache_ttl_seconds: askCacheTtl,
                v13_origin_route: cachedPayload?.meta?.v13_origin_route || cachedPayload?.meta?.v13_route || null,
                v13_origin_llm_calls: cachedPayload?.meta?.v13_origin_llm_calls ?? cachedPayload?.meta?.v13_llm_calls ?? null,
                v13_origin_estimated_cost_usd: cachedPayload?.meta?.v13_origin_estimated_cost_usd ?? cachedPayload?.meta?.v13_estimated_cost_usd ?? null,
                v13_origin_elapsed_seconds: cachedPayload?.meta?.v13_origin_elapsed_seconds ?? cachedPayload?.meta?.v13_elapsed_seconds ?? null,
                v13_origin_chat_model: cachedPayload?.meta?.v13_origin_chat_model || cachedPayload?.meta?.chat_model || null,
                v13_route: "worker_exact_cache",
                v13_llm_calls: 0,
                v13_estimated_cost_usd: 0,
                v13_elapsed_seconds: 0,
                chat_model: "worker_exact_cache",
              }),
              200
            );
          }
        }

        mmAskSpanStart(askTrace, "cloud_run");
        askTrace.cloudRunRequests += 1;
        const r = await usageFetch(`${mmRequestAuthority?.origin || getCloudRunBase(env)}/v1/ai/ask`, {
          method: "POST",
          ...(mmRequestAuthority ? {redirect: "manual", cache: "no-store"} : {}),
          headers: mmRequestAuthority?.headers || {
            "Content-Type": "application/json",
            "X-AI-Internal-Secret": expectedToken,
          },
          body: JSON.stringify(mmRequestAuthority?.payload || {
            company_id: companyId,
            machine_id: machineId,
            ...(aiScope ? { ai_scope: aiScope } : {}),
            query,
            language: language || null,
            top_k,
            bubble_document_id: bubbleDocumentId,
            document_ids: docIds,
            knowledge_version: getExternalKnowledgeVersion(parsed) || null,
            debug: false
          }),
        });


        crStatus = r.status;
        crText = await r.text();
        mmAskSpanStart(askTrace, "adapter");

        try {
          crJson = JSON.parse(crText);
        } catch {
          crJson = null;
        }

        if (!r.ok || !crJson || typeof crJson !== "object") {
          const isTimeout = [408, 504, 524].includes(Number(crStatus));
          const errorCode = isTimeout ? "ASK_TIMEOUT" : "ASK_FAILED";
          const errorMessage = isTimeout
            ? "Tempo massimo di risposta superato. Riprova."
            : `Cloud Run ask failed (HTTP ${crStatus})`;
          return askJsonResponse(
            {
              ok: false,
              status: "error",
              result_code: errorCode,
              error_code: errorCode,
              error_message: errorMessage,
              error: {
                code: errorCode,
                message: errorMessage,
                detail: crJson || crText || null,
              },
            },
            200
          );
        }

        if (crJson.ok !== true) {
          const errorCode = String(
            crJson?.error?.code || crJson?.error_code || crJson?.result_code || "ASK_FAILED"
          ).trim();
          const errorMessage = String(
            crJson?.error?.message || crJson?.error_message || "Ask failed"
          ).trim();
          return askJsonResponse(
            {
              ok: false,
              status: typeof crJson.status === "string" && crJson.status ? crJson.status : "error",
              result_code: String(crJson.result_code || errorCode),
              error_code: errorCode,
              error_message: errorMessage,
              language: crJson.language || crJson?.meta?.language || language || "it",
              answer: "", citations: [], rg_links: [],
              meta: {...(crJson.meta || {}), cacheable:false, semantic_cacheable:false},
              error: {code:errorCode, message:errorMessage},
            },
            200
          );
        }

        if (typeof crJson.status !== "string" || !/^[a-z][a-z0-9_]{0,63}$/.test(crJson.status) ||
            (crJson.status === "answered" &&
             !((typeof crJson.answer === "string" && crJson.answer.trim()) ||
               (typeof crJson.answer_html === "string" && crJson.answer_html.trim())))) {
          return mmAskError("ASK_RESPONSE_INVALID", askTrace);
        }

        const responsePayload = {
          ok: true,
          status: crJson.status,
          answer: crJson.answer || "",
          answer_html: String(crJson.answer_html || ""),
          answer_format: String(crJson.answer_format || (crJson.answer_html ? "html" : "text")),
          answer_render_version: String(crJson.answer_render_version || ""),
          requested_mode: String(crJson.requested_mode || "ask"),
          effective_mode: String(crJson.effective_mode || "ask"),
          routed: crJson.routed === true,
          result_code: String(crJson.result_code || ""),
          request_kind: String(crJson.request_kind || ""),
          evidence_state: String(crJson.evidence_state || ""),
          language: crJson.language || crJson?.meta?.language || language || "it",
          citations: Array.isArray(crJson.citations) ? crJson.citations : [],
          rg_links: Array.isArray(crJson.rg_links) ? crJson.rg_links : [],
          meta: {
            ...(crJson?.meta || {}),
            top_k: crJson.top_k ?? crJson?.meta?.top_k ?? top_k,
            similarity_max: crJson.similarity_max ?? crJson?.meta?.similarity_max ?? null,
            chat_model: crJson.chat_model ?? crJson?.meta?.chat_model ?? null,
            language: crJson.language || crJson?.meta?.language || language || "it",
            cached: crJson?.meta?.cached === true,
            cache_status: crJson?.meta?.cache_status || "not_reported",
            worker_cache_status: askCacheBypass ? "bypass" : askCacheKey ? "miss" : "disabled",
            worker_cache_ttl_seconds: askCacheTtl,
            worker_cache_schema: MM_WORKER_V13_CACHE_SCHEMA,
          },
        };

        if (askCacheKey && !askCacheBypass) {
          await putCachedAskResponse(
            env,
            askCacheKey,
            withAskCacheMeta(responsePayload, {
              cached: false,
              cache_status: "stored",
              cache_stored_at: new Date().toISOString(),
              cache_ttl_seconds: askCacheTtl,
            }),
            askCacheTtl
          );
        }

        return askJsonResponse(responsePayload, 200);
      } catch (e) {
        const isTimeout = isLikelyTimeoutException(e);
        const errorCode = isTimeout ? "ASK_TIMEOUT" : "ASK_FAILED";
        const errorMessage = isTimeout
          ? "Tempo massimo di risposta superato. Riprova."
          : "Cloud Run request failed";
        return askJsonResponse(
          {
            ok: false,
            status: "error",
            result_code: errorCode,
            error_code: errorCode,
            error_message: errorMessage,
            error: { code: errorCode, message: errorMessage, detail: String(e) },
          },
          200
        );
      }
    }

    // ROUTE: /v1/ai/delete/document (REAL -> Cloud Run)
    if (url.pathname === "/v1/ai/delete/document") {
      const companyId = String(parsed?.company_id || parsed?.company?.id || "").trim();
      const bubbleDocumentId = String(parsed?.bubble_document_id || parsed?.bubble_document_id || "").trim();

      if (!companyId || !bubbleDocumentId) {
        return jsonResponse(
          {
            ok: false,
            status: "error",
            error: { code: "BAD_REQUEST", message: "company_id and bubble_document_id are required" },
          },
          200
        );
      }

      let crStatus = 0;
      let crText = "";
      let crJson = null;

      try {
        const r = await usageFetch(`${getCloudRunBase(env)}/v1/ai/delete/document`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-AI-Internal-Secret": expectedToken,
          },
          body: JSON.stringify({
            company_id: companyId,
            bubble_document_id: bubbleDocumentId,
          }),
        });

        crStatus = r.status;
        crText = await r.text();

        try {
          crJson = JSON.parse(crText);
        } catch {
          crJson = null;
        }

        if (!r.ok || !crJson || typeof crJson !== "object") {
          return jsonResponse(
            {
              ok: false,
              status: "error",
              error: {
                code: "DELETE_FAILED",
                message: `Cloud Run delete failed (HTTP ${crStatus})`,
                detail: crJson || crText || null,
              },
            },
            200
          );
        }

        if (crJson.ok !== true) {
          return jsonResponse(
            {
              ok: false,
              status: "error",
              error: {
                code: crJson?.error?.code || crJson?.error_code || "DELETE_FAILED",
                message: crJson?.error?.message || crJson?.error_message || "Delete failed",
                detail: crJson,
              },
            },
            200
          );
        }

        await bumpRootCauseKnowledgeVersion(env, companyId);

        return jsonResponse(
          {
            ok: true,
            status: crJson.status || "deleted",
            deleted: crJson.deleted || {},
            meta: {
              company_id: companyId,
              bubble_document_id: bubbleDocumentId,
              root_cause_cache_invalidated: true,
              ask_cache_invalidated: true,
              ai_cache_invalidated: true,
            },
          },
          200
        );
      } catch (e) {
        return jsonResponse(
          {
            ok: false,
            status: "error",
            error: { code: "DELETE_FAILED", message: "Cloud Run request failed", detail: String(e) },
          },
          200
        );
      }
    }

    // ROUTE: /v1/ai/ingest/source  (REAL -> Cloud Run)
    if (url.pathname === "/v1/ai/ingest/source") {
      const companyId2 = String(parsed?.company_id || parsed?.company?.id || "").trim();
      const machineId2 = String(parsed?.machine_id || parsed?.machine?.id || parsed?.scope?.machine_id || "").trim();
      const sourceType = String(parsed?.source_type || "").trim();
      const sourceId = String(parsed?.source_id || parsed?.bubble_source_id || parsed?.unique_id || "").trim();
      let sourceUrl = String(parsed?.source_url || parsed?.url || "").trim();

      const title = parsed?.title ?? null;
      const description = parsed?.description ?? null;
      const shortDescription = parsed?.short_description ?? null;
      const procedureType = parsed?.procedure_type ?? parsed?.type ?? null;
      const stepNumberRaw = parsed?.step_number;
      const parentProcedureId = String(
        parsed?.parent_procedure_id ??
        parsed?.procedure_id ??
        parsed?.parent?.procedure_id ??
        ""
      ).trim();
      const parentProcedureCode = String(
        parsed?.parent_procedure_code ??
        parsed?.procedure_code ??
        parsed?.parent?.procedure_code ??
        ""
      ).trim();
      const parentProcedureTitle = String(
        parsed?.parent_procedure_title ??
        parsed?.procedure_title ??
        parsed?.parent?.procedure_title ??
        ""
      ).trim();
      const category = parsed?.category ?? null;
      const solution = parsed?.solution ?? null;
      const notes = parsed?.notes ?? null;

      const stepNumber =
        Number.isFinite(Number(stepNumberRaw)) && String(stepNumberRaw).trim() !== ""
          ? Number(stepNumberRaw)
          : null;

      if (!companyId2 || !machineId2 || !sourceType || !sourceId) {
        return jsonResponse(
          {
            ok: false,
            status: "error",
            error: {
              code: "BAD_REQUEST",
              message: "company_id, machine_id, source_type and source_id are required",
            },
            detail: {
              company_id: companyId2 ? "ok" : "missing",
              machine_id: machineId2 ? "ok" : "missing",
              source_type: sourceType ? "ok" : "missing",
              source_id: sourceId ? "ok" : "missing",
            },
          },
          200
        );
      }

      if (sourceUrl.startsWith("//")) sourceUrl = "https:" + sourceUrl;

      let crStatus = 0;
      let crText = "";
      let crJson = null;

      try {
        const planEmbedCharsLimitTotal = Number(qc?.embed_chars_limit_total ?? 0) || 0;
        const planIndexStorageLimitBytes = Number(qc?.index_storage_limit_bytes ?? 0) || 0;
        const usedEmbedCharsTotal = Number(qc?.embed_chars_used_total ?? 0) || 0;
        const usedIndexStorageTotal = Number(qc?.index_storage_used_total ?? 0) || 0;
        const docPrevEmbedChars = Number(qc?.doc_prev_embed_chars ?? 0) || 0;
        const docPrevIndexStorageBytes = Number(qc?.doc_prev_index_storage_bytes ?? 0) || 0;

        if (planEmbedCharsLimitTotal > 0 && usedEmbedCharsTotal >= planEmbedCharsLimitTotal) {
          return jsonResponse(
            {
              ok: false,
              status: "error",
              error: {
                code: "PLAN_EMBED_CHARS_LIMIT_EXCEEDED",
                message: "Limite totale caratteri AI già raggiunto per questa Company.",
              },
            },
            200
          );
        }

        if (planIndexStorageLimitBytes > 0 && usedIndexStorageTotal >= planIndexStorageLimitBytes) {
          return jsonResponse(
            {
              ok: false,
              status: "error",
              error: {
                code: "PLAN_INDEX_STORAGE_LIMIT_EXCEEDED",
                message: "Limite totale storage AI già raggiunto per questa Company.",
              },
            },
            200
          );
        }

        const r = await usageFetch(`${getCloudRunBase(env)}/v1/ai/ingest/source`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-AI-Internal-Secret": expectedToken,
          },
          body: JSON.stringify({
            company_id: companyId2,
            machine_id: machineId2,
            source_type: sourceType,
            source_id: sourceId,
            source_url: sourceUrl || null,

            title,
            description,
            short_description: shortDescription,
            procedure_type: procedureType,
            step_number: stepNumber,
            parent_procedure_id: parentProcedureId || null,
            parent_procedure_code: parentProcedureCode || null,
            parent_procedure_title: parentProcedureTitle || null,
            category,
            solution,
            notes,

            plan_embed_chars_limit_total: planEmbedCharsLimitTotal,
            plan_index_storage_limit_bytes: planIndexStorageLimitBytes,
            embed_chars_used_total: usedEmbedCharsTotal,
            index_storage_used_total: usedIndexStorageTotal,
            doc_prev_embed_chars: docPrevEmbedChars,
            doc_prev_index_storage_bytes: docPrevIndexStorageBytes,
          }),
        });

        crStatus = r.status;
        crText = await r.text();

        try {
          crJson = JSON.parse(crText);
        } catch {
          crJson = null;
        }

        if (!r.ok || !crJson || typeof crJson !== "object") {
          return jsonResponse(
            {
              ok: false,
              status: "error",
              error: {
                code: "INGEST_SOURCE_FAILED",
                message: `Cloud Run structured ingest failed (HTTP ${crStatus})`,
                detail: crJson || crText || null,
              },
            },
            200
          );
        }

        if (crJson.ok !== true) {
          const reason = String(crJson?.reason || "").trim();

          const isLimit =
            reason === "PLAN_EMBED_CHARS_LIMIT_EXCEEDED" ||
            reason === "PLAN_INDEX_STORAGE_LIMIT_EXCEEDED";

          if (isLimit) {
            return jsonResponse(
              {
                ok: false,
                status: "limit_exceeded",
                reason,
                error: {
                  code: reason,
                  message: crJson?.error?.message || "Limit exceeded",
                  detail: crJson,
                },
                text_chars: Number(crJson.text_chars ?? 0),
                pages_detected: Number(crJson.pages_detected ?? 0),
                est_storage_bytes: Number(crJson.est_storage_bytes ?? 0),
              },
              200
            );
          }

          return jsonResponse(
            {
              ok: false,
              status: "error",
              error: {
                code: crJson?.error?.code || crJson?.error_code || "INGEST_SOURCE_FAILED",
                message: crJson?.error?.message || crJson?.error_message || "Structured source not indexable",
                detail: crJson,
              },
              text_chars: Number(crJson.text_chars ?? 0),
              pages_detected: Number(crJson.pages_detected ?? 0),
              est_storage_bytes: Number(crJson.est_storage_bytes ?? 0),
            },
            200
          );
        }

        const returnedSourceKey = String(crJson?.source_key || `${sourceType}:${sourceId}`).trim();

        await bumpRootCauseKnowledgeVersion(env, companyId2);

        return jsonResponse(
          {
            ok: true,
            status: "indexed",
            source_type: String(crJson?.source_type || sourceType).trim(),
            source_key: returnedSourceKey,
            source_id: `mm:${returnedSourceKey}`,
            text_chars: Number(crJson.text_chars ?? 0),
            pages_detected: Number(crJson.pages_detected ?? 1),
            est_storage_bytes: Number(crJson.est_storage_bytes ?? 0),
            chunks_written: Number(crJson.chunks_written ?? 0),
            parent_source_key: String(crJson.parent_source_key || ""),
            structured_relation_written: crJson.structured_relation_written === true,
            root_cause_cache_invalidated: true,
            ask_cache_invalidated: true,
            ai_cache_invalidated: true,
          },
          200
        );
      } catch (e) {
        return jsonResponse(
          {
            ok: false,
            status: "error",
            error: {
              code: "INGEST_SOURCE_FAILED",
              message: "Cloud Run request failed",
              detail: String(e),
            },
          },
          200
        );
      }
    }

    // ROUTE: /v1/ai/ingest/document  (REAL -> Cloud Run)
    const fileUrlPick = firstUsableUrlCandidate([
      { source: "runtime_file_url", value: parsed?.runtime_file_url },
      { source: "source.runtime_file_url", value: parsed?.source?.runtime_file_url },
      { source: "source.file_url", value: parsed?.source?.file_url },
      { source: "file_url", value: parsed?.file_url },
      { source: "file.url", value: parsed?.file?.url },
    ]);

    const fileUrl = fileUrlPick.url;
    const companyId2 = String(parsed?.company_id || parsed?.company?.id || "").trim();
    const aiScope = getAiScope(parsed);
    const machineId2 = String(parsed?.machine_id || parsed?.machine?.id || parsed?.scope?.machine_id || "").trim();
    const bubbleDocumentId2 = String(parsed?.bubble_document_id || parsed?.bubble_document || "").trim();

    const docId = String(
      parsed?.bubble_document_id ||
      parsed?.bubble_docum ||
      parsed?.bubble_document ||
      parsed?.source?.bubble_document_id ||
      "unknown"
    ).trim();

    if (!fileUrl) {
      return jsonResponse(
        {
          ok: false,
          status: "error",
          error: {
            code: "BAD_REQUEST",
            message: "runtime_file_url/file_url is required or all provided signed URLs are expired",
            detail: { inspected_file_url_candidates: fileUrlPick.inspected },
          },
        },
        200
      );
    }

    if (!companyId2 || !bubbleDocumentId2 || (aiScope !== "company_general" && !machineId2)) {
      return jsonResponse(
        {
          ok: false,
          status: "error",
          error: { code: "BAD_REQUEST", message: "company_id, machine_id, bubble_document_id are required" },
          detail: {
            company_id: companyId2 ? "ok" : "missing",
            machine_id: machineId2 || aiScope === "company_general" ? "ok" : "missing",
            bubble_document_id: bubbleDocumentId2 ? "ok" : "missing",
            ai_scope: aiScope || "not_set",
          },
        },
        200
      );
    }

    let normalizedUrl = fileUrl;
    if (normalizedUrl.startsWith("//")) normalizedUrl = "https:" + normalizedUrl;

    let crStatus = 0;
    let crText = "";
    let crJson = null;

    try {
      const planEmbedCharsLimitTotal = Number(qc?.embed_chars_limit_total ?? 0) || 0;
      const planIndexStorageLimitBytes = Number(qc?.index_storage_limit_bytes ?? 0) || 0;
      const usedEmbedCharsTotal = Number(qc?.embed_chars_used_total ?? 0) || 0;
      const usedIndexStorageTotal = Number(qc?.index_storage_used_total ?? 0) || 0;
      const docPrevEmbedChars = Number(qc?.doc_prev_embed_chars ?? 0) || 0;
      const docPrevIndexStorageBytes = Number(qc?.doc_prev_index_storage_bytes ?? 0) || 0;

      // Pre-enforcement: blocca se già pieno
      if (planEmbedCharsLimitTotal > 0 && usedEmbedCharsTotal >= planEmbedCharsLimitTotal) {
        return jsonResponse(
          {
            ok: false,
            status: "error",
            error: {
              code: "PLAN_EMBED_CHARS_LIMIT_EXCEEDED",
              message: "Limite totale caratteri AI già raggiunto per questa Company.",
            },
          },
          200
        );
      }

      if (planIndexStorageLimitBytes > 0 && usedIndexStorageTotal >= planIndexStorageLimitBytes) {
        return jsonResponse(
          {
            ok: false,
            status: "error",
            error: {
              code: "PLAN_INDEX_STORAGE_LIMIT_EXCEEDED",
              message: "Limite totale storage AI già raggiunto per questa Company.",
            },
          },
          200
        );
      }

      let fetchedFileBase64 = "";
      let fetchedContentType = "";
      let fetchedFilename = "";

      try {
        const fr = await usageFetch(normalizedUrl, {
          method: "GET",
          redirect: "follow",
          headers: {
            "User-Agent": "MachineMind-AI-Worker/1.0",
          },
        });

        if (!fr.ok) {
          let bodyHead = "";
          try {
            bodyHead = (await fr.text()).slice(0, 500);
          } catch (_) {}

          return jsonResponse(
            {
              ok: false,
              status: "error",
              error: {
                code: "FILE_FETCH_FAILED",
                message: `Worker could not fetch file_url (HTTP ${fr.status})`,
                detail: {
                  url: normalizedUrl,
                  file_url_source: fileUrlPick.source,
                  inspected_file_url_candidates: fileUrlPick.inspected,
                  content_type: fr.headers.get("content-type") || "",
                  body_head: bodyHead,
                },
              },
            },
            200
          );
        }

        const ab = await fr.arrayBuffer();

        if (!ab || ab.byteLength <= 0) {
          return jsonResponse(
            {
              ok: false,
              status: "error",
              error: {
                code: "FILE_FETCH_EMPTY",
                message: "Worker fetched an empty file",
                detail: {
                  url: normalizedUrl,
                  file_url_source: fileUrlPick.source,
                  inspected_file_url_candidates: fileUrlPick.inspected,
                },
              },
            },
            200
          );
        }

        fetchedFileBase64 = arrayBufferToBase64(ab);
        fetchedContentType = String(fr.headers.get("content-type") || "")
          .split(";", 1)[0]
          .trim()
          .toLowerCase();

        try {
          fetchedFilename = decodeURIComponent(
            new URL(normalizedUrl).pathname.split("/").pop() || ""
          );
        } catch (_) {
          fetchedFilename = "";
        }
      } catch (e) {
        return jsonResponse(
          {
            ok: false,
            status: "error",
            error: {
              code: "FILE_FETCH_FAILED",
              message: "Worker file fetch crashed",
              detail: String(e && e.message ? e.message : e),
              url: normalizedUrl,
              file_url_source: fileUrlPick.source,
              inspected_file_url_candidates: fileUrlPick.inspected,
            },
          },
          200
        );
      }

      const filenameForCloudRun = String(
        parsed?.filename ||
        parsed?.source?.filename ||
        parsed?.file?.filename ||
        fetchedFilename ||
        docId ||
        "document"
      ).trim();

      const contentTypeForCloudRun = String(
        parsed?.content_type ||
        parsed?.source?.content_type ||
        fetchedContentType ||
        (filenameForCloudRun.toLowerCase().endsWith(".xlsx")
          ? "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
          : "application/pdf")
      ).trim();

      const documentCategoryCode = normalizeDocumentCategoryCode(
        parsed?.document_category_code ??
        parsed?.category ??
        parsed?.source?.document_category_code ??
        parsed?.source?.category ??
        ""
      );
      const documentIsTechnical = normalizeBooleanLike(
        parsed?.document_is_technical ??
        parsed?.is_tecnico ??
        parsed?.source?.document_is_technical ??
        parsed?.source?.is_tecnico ??
        false
      );

      const r = await usageFetch(`${getCloudRunBase(env)}/v1/ai/ingest/document`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-AI-Internal-Secret": expectedToken,
        },
        body: JSON.stringify({
          file_url: normalizedUrl,
          file_base64: fetchedFileBase64,
          filename: filenameForCloudRun,
          content_type: contentTypeForCloudRun,
          company_id: companyId2,
          machine_id: machineId2,
          bubble_document_id: bubbleDocumentId2,
          ...(aiScope ? { ai_scope: aiScope } : {}),
          document_category_code: documentCategoryCode || null,
          document_is_technical: documentIsTechnical,

          // Cost-based ingest metering context. Keep forwarding OFF until Cloud Run
          // accepts these optional fields, to avoid breaking a strict request schema.
          ...(isIngestCreditsForwardEnabled(env)
            ? {
                ingest_request_key: ingestRequestKey,
                ingest_month_key: ingestCreditQuota.monthKey || null,
                ingest_credits_limit_month: ingestCreditQuota.limit,
                ingest_credits_used_before: ingestCreditQuota.used,
                ingest_credits_enforced: ingestCreditQuota.enforced,
                ingest_request_already_admitted: ingestCreditQuota.requestAlreadyAdmitted,
                ingest_metering_version: MM_INGEST_CREDITS_VERSION,
              }
            : {}),

          // Existing capacity limits remain unchanged.
          plan_embed_chars_limit_total: planEmbedCharsLimitTotal,
          plan_index_storage_limit_bytes: planIndexStorageLimitBytes,
          embed_chars_used_total: usedEmbedCharsTotal,
          index_storage_used_total: usedIndexStorageTotal,
          doc_prev_embed_chars: docPrevEmbedChars,
          doc_prev_index_storage_bytes: docPrevIndexStorageBytes,
        }),
      });

      crStatus = r.status;
      crText = await r.text();

      try {
        crJson = JSON.parse(crText);
      } catch {
        crJson = null;
      }

      if (!r.ok || !crJson || typeof crJson !== "object") {
        return jsonResponse(
          {
            ok: false,
            status: "error",
            error: {
              code: "INGEST_FAILED",
              message: `Cloud Run ingest failed (HTTP ${crStatus})`,
              detail: crJson || crText || null,
            },
            ...getCloudRunIngestMeteringFields({
              crJson,
              quota: ingestCreditQuota,
              requestKey: ingestRequestKey,
            }),
          },
          200
        );
      }

      if (crJson.ok !== true) {
        const reason = getCloudRunLimitReason(crJson);
        const isLimit = isAnyDocumentIngestLimitCode(reason);

        if (isLimit) {
          return jsonResponse(
            {
              ok: false,
              status: "limit_exceeded",
              reason,
              error: {
                code: reason,
                message: crJson?.error?.message || "Limit exceeded",
                detail: crJson,
              },
              text_chars: Number(crJson.text_chars ?? 0),
              pages_detected: Number(crJson.pages_detected ?? 0),
              est_storage_bytes: Number(crJson.est_storage_bytes ?? 0),
              ...getCloudRunIngestMeteringFields({
                crJson,
                quota: ingestCreditQuota,
                requestKey: ingestRequestKey,
              }),
            },
            200
          );
        }

        return jsonResponse(
          {
            ok: false,
            status: "error",
            error: {
              code: crJson?.error?.code || crJson?.error_code || "NOT_INDEXABLE",
              message: crJson?.error?.message || crJson?.error_message || "Documento non indicizzabile",
              detail: crJson,
            },
            text_chars: Number(crJson.text_chars ?? 0),
            pages_detected: Number(crJson.pages_detected ?? 0),
            est_storage_bytes: Number(crJson.est_storage_bytes ?? 0),
            ...getCloudRunIngestMeteringFields({
              crJson,
              quota: ingestCreditQuota,
              requestKey: ingestRequestKey,
            }),
          },
          200
        );
      }

      await bumpRootCauseKnowledgeVersion(env, companyId2);

      return jsonResponse(
        {
          ok: true,
          status: "indexed",
          source_id: `mm:doc:${docId}`,
          text_chars: Number(crJson.text_chars ?? 0),
          pages_detected: Number(crJson.pages_detected ?? 0),
          est_storage_bytes: Number(crJson.est_storage_bytes ?? 0),
          electrical_candidate: crJson.electrical_candidate === true,
          electrical_pipeline_enabled: crJson.electrical_pipeline_enabled === true,
          electrical_document_id: crJson.electrical_document_id ?? null,
          electrical_index_status: String(crJson.electrical_index_status || "not_applicable"),
          electrical_latest_version_no: Number(crJson.electrical_latest_version_no ?? 0),
          electrical_registry_error: crJson.electrical_registry_error || null,
          ...getCloudRunIngestMeteringFields({
            crJson,
            quota: ingestCreditQuota,
            requestKey: ingestRequestKey,
          }),
          root_cause_cache_invalidated: true,
          ask_cache_invalidated: true,
          ai_cache_invalidated: true,
        },
        200
      );
    } catch (e) {
      return jsonResponse(
        {
          ok: false,
          status: "error",
          error: { code: "INGEST_FAILED", message: "Cloud Run request failed", detail: String(e) },
          ...getCloudRunIngestMeteringFields({
            crJson,
            quota: ingestCreditQuota,
            requestKey: ingestRequestKey,
          }),
        },
        200
      );
    }
  },
};
