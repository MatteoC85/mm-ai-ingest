"""Private, request-owned FREE exact input counting after an unsent ASK refusal.

No source text or credentials are retained. Proofs are capabilities of the actual
protected request budget, not dictionaries or model/client metadata.
"""
from __future__ import annotations
import hashlib,json,threading,time
from copy import deepcopy

ENDPOINT='https://api.openai.com/v1/responses/input_tokens'
CREATE_ENDPOINT='https://api.openai.com/v1/responses'
VERSION='ask-owned-exact-input-count-v1'
from .request_budget import _V13BudgetExceeded

class _RevalidationDenied(_V13BudgetExceeded):
    def __init__(self, original):
        self.original=original
        super().__init__('Counted dispatch revalidation denied')
PURPOSES=frozenset({'ask_final_synthesis','assistant_core_answer_contract_verifier'})
MODELS=frozenset({'gpt-5.6-sol','gpt-5.6-terra','gpt-5.6-luna'})
_INPUT_KEYS=frozenset({'model','input','instructions','reasoning','text','truncation'})
_CREATE_KEYS=_INPUT_KEYS|{'store','service_tier','safety_identifier','max_output_tokens'}

def _wire(value):
    return json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')
def _sha(value):return hashlib.sha256(_wire(value)).hexdigest()
def _projection(payload):
    if (type(payload)is not dict or not set(payload)<=_CREATE_KEYS or payload.get('model')not in MODELS
            or payload.get('store')is not False or payload.get('service_tier')!='default'
            or payload.get('truncation','disabled')!='disabled'):
        raise ValueError('count_projection')
    if 'instructions'in payload and type(payload['instructions'])is not str:raise ValueError('count_projection')
    if 'reasoning'in payload:
        reason=payload['reasoning']
        if (type(reason)is not dict or not set(reason)<={'context','effort','generate_summary','mode','summary'}
                or any(type(v)is not str for v in reason.values())):raise ValueError('count_projection')
    if 'text'in payload:
        text=payload['text']
        if type(text)is not dict or not set(text)<={'format','verbosity'}:raise ValueError('count_projection')
        if 'format'in text and type(text['format'])is not dict:raise ValueError('count_projection')
        if 'verbosity'in text and text['verbosity']not in {'low','medium','high'}:raise ValueError('count_projection')
    inputs=payload.get('input')
    if type(inputs)is not list or not inputs:raise ValueError('count_projection')
    for msg in inputs:
        if type(msg)is not dict or set(msg)!={'role','content'} or msg['role']not in {'system','developer','user','assistant'}:raise ValueError('count_projection')
        value=msg['content']
        if type(value)is str:continue
        if (type(value)is not list or any(type(x)is not dict or set(x)!={'type','text'}
                or x['type']not in {'text','input_text'} or type(x['text'])is not str for x in value)):raise ValueError('count_projection')
    return {k:deepcopy(v)for k,v in payload.items()if k in _INPUT_KEYS}

def _live(budget,owner,completion):
    from machinemind.ask import routing_runtime,request_completion
    from .request_budget import _v13_current_budget
    return (type(owner)is routing_runtime.ProtectedExecution and owner.active
        and routing_runtime._ACTIVE.get()is owner and request_completion.current()is completion
        and completion is not None and completion.budget is budget and _v13_current_budget()is budget
        and budget.mode=='assistant_core_ask')

class _Proof:
    __slots__=('session','purpose','model','planned_sha','projection_sha','tokens','requested','used','finalized','event')
    def __init__(self,session,purpose,planned,projection,tokens,requested,event):
        self.session,self.purpose,self.model=session,purpose,planned['model']
        self.planned_sha,self.projection_sha=_sha(planned),_sha(projection)
        self.tokens,self.requested=tokens,requested
        self.used=self.finalized=False;self.event=event

class _Session:
    def __init__(self,budget,owner,completion):
        self.budget,self.owner,self.completion=budget,owner,completion
        self.events=[];self.roles=set();self.proofs=[];self.lock=threading.RLock()
    def live(self):return _live(self.budget,self.owner,self.completion)
    def count(self,*,purpose,payload,requested,api_key,post_fn,url,old_error):
        if not self.live()or purpose not in PURPOSES or url!=CREATE_ENDPOINT:return None
        try:
            self.budget.ensure_time(7.)
            planned=deepcopy(payload);planned['max_output_tokens']=requested
            projection=_projection(planned);encoded=_wire(projection)
            from .request_budget import _text_input_bound,_V13OutputCapacityInsufficient
            old_bound=_text_input_bound(payload['input'],payload)[0]
            if len(encoded)>2*1024*1024:return None
            remaining_cost_before=round(self.budget.remaining_cost_usd,8)
            with self.lock:
                if purpose in self.roles or len(self.roles)>=2:return None
                self.roles.add(purpose)
                event={'purpose':purpose,'model':planned['model'],'status':'count_attempted',
                    'planned_create_sha256':_sha(planned),'input_projection_sha256':_sha(projection),
                    'provider_calls':0,'cost_usd':0,'count_HTTP_attempts':1,
                    'old_UTF8_input_bound':old_bound,'configured_output_ceiling':requested,'remaining_cost_before_usd':remaining_cost_before,
                    'refusal':'structured_output_capacity'if type(old_error)is _V13OutputCapacityInsufficient else'useful_output_capacity'}
                self.events.append(event)
            started=time.monotonic();response=None
            try:
                remaining=self.budget.remaining()
                each=min(2.5,max(.1,(remaining-2.)/2.))
                response=post_fn(ENDPOINT,data=encoded,headers={'Authorization':'Bearer '+api_key,'Content-Type':'application/json'},timeout=(each,each),allow_redirects=False,stream=True)
                if response.status_code!=200:raise ValueError('count_http')
                body=bytearray()
                for chunk in response.iter_content(chunk_size=4096):
                    if time.monotonic()-started>5.:raise ValueError('count_time')
                    if len(body)+len(chunk)>16384:raise ValueError('count_size')
                    body.extend(chunk)
                value=json.loads(bytes(body))
                if (type(value)is not dict or set(value)!={'object','input_tokens'}or value['object']!='response.input_tokens'
                        or type(value['input_tokens'])is not int or not 0<value['input_tokens']<=1048576):raise ValueError('count_reply')
                if value['input_tokens']>old_bound:raise ValueError('count_no_reduction')
                if not self.live():raise ValueError('count_owner')
                self.budget.ensure_time(2.)
                proof=_Proof(self,purpose,planned,projection,value['input_tokens'],requested,event)
                with self.lock:self.proofs.append(proof)
                with self.lock:event.update(status='count_verified',input_tokens=proof.tokens)
                return proof
            except Exception:
                with self.lock:event['status']='count_unavailable'
                return None
            finally:
                with self.lock:event['elapsed_seconds']=round(max(0.,time.monotonic()-started),6)
                if response is not None:
                    try:response.close()
                    except Exception:pass
        except Exception:return None
    def snapshot(self):
        with self.lock:
            return {'version':VERSION,'count_calls':len(self.events),'maximum_count_calls':2,
                'cost_usd':0,'llm_calls':0,'events':deepcopy(self.events)}if self.events else None

def for_owner(budget):
    from machinemind.ask import routing_runtime,request_completion
    owner,completion=routing_runtime._ACTIVE.get(),request_completion.current()
    if not _live(budget,owner,completion):return None
    with budget._lock:
        existing=getattr(budget,'_ask_input_count_session',None)
        if existing is None:
            existing=_Session(budget,owner,completion);budget._ask_input_count_session=existing
        return existing if type(existing)is _Session and existing.budget is budget and existing.live()else None

def eligible_refusal(budget,error,*,purpose,max_output_tokens,model):
    from .request_budget import _V13BudgetExceeded,_V13OutputCapacityInsufficient,_v13_model_rates
    if purpose not in PURPOSES or model not in MODELS:return False
    full=type(error)is _V13OutputCapacityInsufficient
    useful=(type(error)is _V13BudgetExceeded and str(error)==f'V13 {budget.mode} insufficient reserved cost for useful output')
    if not(full or useful):return False
    _,rate=_v13_model_rates(model)
    required=max_output_tokens if full else min(800,max_output_tokens)
    return type(required)is int and required>0 and required*rate/1000000<=budget.remaining_cost_usd

def consume(proof,*,budget,purpose,model,payload,requested):
    if type(proof)is not _Proof:raise ValueError('input_count_proof')
    session=proof.session;planned=deepcopy(payload);planned['max_output_tokens']=requested
    with session.lock:
        if (session.budget is not budget or not session.live()or proof not in session.proofs or proof.used
                or proof.purpose!=purpose or proof.model!=model or proof.requested!=requested
                or proof.planned_sha!=_sha(planned)or proof.projection_sha!=_sha(_projection(planned))):raise ValueError('input_count_proof')
        proof.used=True
        return proof.tokens

def finalize(proof,*,budget,payload,index):
    if type(proof)is not _Proof:raise ValueError('input_count_proof')
    session=proof.session
    with budget._lock,session.lock:
        cap=payload.get('max_output_tokens');planned=deepcopy(payload);planned['max_output_tokens']=proof.requested
        if (session.budget is not budget or not session.live()or proof not in session.proofs or not proof.used or proof.finalized
                or type(cap)is not int or not 0<cap<=proof.requested or _sha(planned)!=proof.planned_sha
                or _sha(_projection(payload))!=proof.projection_sha):raise ValueError('input_count_final_payload')
        row=budget._row(index)
        if (row['accounting_state']!='pending'or row['dispatched']or row['model']!=proof.model
                or row['purpose']!=proof.purpose or cap!=row['max_output_tokens']):raise ValueError('input_count_final_payload')
        sent=_sha(payload);proof.finalized=True
        proof.event.update(status='applied_to_unsent_reservation',sent_create_sha256=sent,effective_output_tokens=cap)
        row.update(input_bound_basis='official_free_input_tokens',input_projection_sha256=proof.projection_sha,input_count_planned_create_sha256=proof.planned_sha,input_count_sent_create_sha256=sent)

def snapshot(budget):
    value=getattr(budget,'_ask_input_count_session',None)
    return value.snapshot()if type(value)is _Session else None
