"""ASGI admission/settlement around existing interactive routes.

The semantic handlers, budgets and source fences remain the originals. Streaming
whitespace heartbeats are forwarded; JSON answers are released only after durable
settlement. No full answer is stored/replayed by this accounting layer.
"""
from __future__ import annotations
import asyncio
from datetime import datetime, timezone
import hmac
import os
from .contracts import Context, PATHS, SHADOW_PATHS, UsageError, VERSION, canonical, strict_json, settlement, Settlement
from .ledger import production_ledger
from . import cache_receipt

MAX_REQUEST=1_048_576
MAX_RESPONSE=4_194_304
HEALTH='/v1/ai/usage/health'
SNAPSHOT='/v1/ai/usage/snapshot'


def _secret_match(actual, expected):
    return (isinstance(expected,str) and 32 <= len(expected) <= 256
            and expected.isascii() and all(33<=ord(ch)<=126 for ch in expected)
            and isinstance(actual,str) and actual.isascii() and hmac.compare_digest(actual,expected))


def error_body(code):
    return {'ok':False,'status':'error','result_code':code,'error_code':code,
        'error_message':code,'error':{'code':code,'message':code},
        'answer':'','citations':[],'rg_links':[],
        'meta':{'cacheable':False,'semantic_cacheable':False,'usage_version':VERSION}}


class InteractiveUsageMiddleware:
    def __init__(self, app, *, environ=None, ledger=None, now=None):
        self.app=app
        self.environ=os.environ if environ is None else environ
        self.ledger_override=ledger
        self.now=now or (lambda:datetime.now(timezone.utc))

    async def __call__(self, scope, receive, send):
        if scope['type']!='http': return await self.app(scope,receive,send)
        incoming=scope.get('path','').rstrip('/')
        aliases={p.replace('/v1/ai/','/v1/ai/usage/execute/',1):p for p in PATHS}
        path=aliases.get(incoming,incoming)
        internal_scope={**scope,'path':path,'raw_path':path.encode('ascii')} if incoming in aliases else scope
        applies=path in PATHS or path in SHADOW_PATHS or path in {HEALTH,SNAPSHOT}
        mode=self.environ.get('MM_USAGE_ENFORCEMENT','off')
        if not applies or (mode=='off' and incoming not in aliases and path not in {HEALTH,SNAPSHOT}):
            return await self.app(scope,receive,send)
        headers={}
        duplicated=False
        for key,value in scope.get('headers',[]):
            name=key.decode('latin1').lower()
            if name in headers and name.startswith('x-mm-'):duplicated=True
            headers[name]=value.decode('latin1')
        started=False
        async def emit(body,status=200):
            nonlocal started
            raw=canonical(body)
            if not started:
                await send({'type':'http.response.start','status':status,'headers':[
                    (b'content-type',b'application/json'),(b'cache-control',b'no-store')]})
                started=True
            await send({'type':'http.response.body','body':raw,'more_body':False})
        try:
            if mode!='required':raise UsageError('USAGE_CONFIGURATION_INVALID')
            if not _secret_match(headers.get('x-mm-usage-authority'),self.environ.get('MM_USAGE_AUTHORITY_SECRET')):
                raise UsageError('USAGE_AUTH_REQUIRED',401)
            internal=self.environ.get('AI_INTERNAL_SECRET','')
            if not internal or not internal.isascii() or not headers.get('x-ai-internal-secret','').isascii() or not hmac.compare_digest(headers.get('x-ai-internal-secret',''),internal):
                raise UsageError('USAGE_AUTH_REQUIRED',401)
            if path=='/v1/ai/ask' and not _secret_match(headers.get('x-mm-app-authority'),self.environ.get('MM_APP_AUTHORITY_SECRET')):
                raise UsageError('USAGE_AUTH_REQUIRED',401)
            if duplicated:raise UsageError('USAGE_HEADERS_INVALID',400)
            if path in SHADOW_PATHS:raise UsageError('USAGE_SHADOW_DISABLED',403)
            if scope.get('method')!='POST':raise UsageError('USAGE_METHOD_INVALID',405)
            if self.environ.get('MM_USAGE_AUTHORITY_SECRET') in {self.environ.get('AI_INTERNAL_SECRET'),self.environ.get('MM_APP_AUTHORITY_SECRET')}:
                raise UsageError('USAGE_SECRET_SEPARATION_REQUIRED')
            ledger=self.ledger_override or production_ledger(self.environ)
            if path==HEALTH:
                ready=await asyncio.to_thread(ledger.health)
                return await emit({'ok':True,**ready})
            data=bytearray()
            while True:
                item=await receive()
                if item['type']=='http.disconnect':return
                if item['type']!='http.request':raise UsageError('USAGE_BODY_INVALID',400)
                data.extend(item.get('body',b''))
                if len(data)>MAX_REQUEST:raise UsageError('USAGE_BODY_TOO_LARGE',413)
                if not item.get('more_body',False):break
            try: payload=strict_json(bytes(data))
            except Exception:raise UsageError('USAGE_BODY_INVALID',400) from None
            ctx=Context.parse(headers.get('x-mm-usage-context'),payload,self.now(),
                self.environ.get('MM_USAGE_TIMEZONE','Europe/Rome'))
            if path==SNAPSHOT:
                counts=await asyncio.to_thread(ledger.snapshot,ctx)
                return await emit({'ok':True,'usage_version':VERSION,'daily_used':counts['day'],
                    'monthly_used':counts['month'],'daily_limit':ctx.daily_limit,'monthly_limit':ctx.monthly_limit})
            operation,cap=PATHS[path]
            fingerprint=ctx.fingerprint(path,payload)
            counts=await asyncio.to_thread(ledger.admit,ctx,operation,fingerprint,cap)
        except UsageError as exc:
            return await emit(error_body(exc.code),exc.status)
        except Exception:
            return await emit(error_body('USAGE_UNAVAILABLE'),503)

        receipt=cache_receipt.CacheReceipt(ledger,ctx,fingerprint,cap)
        receipt_token=cache_receipt.activate(receipt)
        response=bytearray(); http_status=500; consumed=False
        async def replay_receive():
            nonlocal consumed
            if not consumed:
                consumed=True
                return {'type':'http.request','body':bytes(data),'more_body':False}
            return await receive()
        async def collect(message):
            nonlocal started,http_status
            if message['type']=='http.response.start':
                http_status=int(message['status'])
            elif message['type']=='http.response.body':
                chunk=message.get('body',b'')
                if not response and chunk and not chunk.strip() and message.get('more_body'):
                    if not started:
                        await send({'type':'http.response.start','status':200,'headers':[
                            (b'content-type',b'application/json'),(b'cache-control',b'no-store')]})
                        started=True
                    await send({'type':'http.response.body','body':chunk,'more_body':True})
                else:
                    response.extend(chunk)
                    if len(response)>MAX_RESPONSE:raise UsageError('USAGE_RESPONSE_TOO_LARGE')
        try:
            await self.app(internal_scope,replay_receive,collect)
            try:
                body=strict_json(bytes(response))
                if type(body) is not dict:raise ValueError()
            except Exception:
                body=error_body('UPSTREAM_RESPONSE_INVALID')
            result=settlement(body,http_status,cap)
            if receipt.checked is not None and result.known < receipt.checked:
                # A hard-timeout envelope can lack the last thread snapshot.
                # Retain known spend; the remaining cap is not counted twice.
                result=Settlement('uncertain',receipt.checked,
                    max(cap-receipt.checked,result.known+result.uncertain-receipt.checked,0),
                    result.result_code)
            await asyncio.to_thread(ledger.complete,ctx,fingerprint,result)
            body=dict(body)
            body['meta']={**(body.get('meta') if type(body.get('meta')) is dict else {}),
                'usage_version':VERSION, 'usage_request_id':ctx.request_id,
                'usage_recorded':True,'usage_state':result.state,
                'usage_daily_used':counts['day']-int(result.refund),
                'usage_monthly_used':counts['month']-int(result.refund),
                'usage_known_usd':str(result.known),'usage_uncertain_usd':str(result.uncertain)}
            if result.state=='uncertain':
                body['meta'].update(cacheable=False,semantic_cacheable=False)
            await emit(body,http_status if http_status<400 else 200)
        except asyncio.CancelledError:
            # The durable started row keeps the entire liability and quota slot.
            # Never cancel/retry a provider merely to turn a timeout into zero.
            raise
        except Exception:
            # Completion may have committed despite a network timeout. Its exact
            # immutable request ID prevents redispatch; never attempt a second ASK.
            await emit(error_body('USAGE_COMPLETION_UNCONFIRMED'),503)
        finally:
            receipt.close()
            cache_receipt.deactivate(receipt_token)


def runtime_health(environ=None):
    env=os.environ if environ is None else environ
    mode=env.get('MM_USAGE_ENFORCEMENT','off')
    if mode!='required':return {'version':VERSION,'mode':mode,'ready':False}
    secret=env.get('MM_USAGE_AUTHORITY_SECRET')
    if not _secret_match(secret,secret) or secret in {env.get('AI_INTERNAL_SECRET'),env.get('MM_APP_AUTHORITY_SECRET')}:
        return {'version':VERSION,'mode':mode,'ready':False,'code':'USAGE_CONFIGURATION_INVALID'}
    try:return {**production_ledger(env).health(),'mode':mode}
    except Exception:return {'version':VERSION,'mode':mode,'ready':False,'code':'USAGE_DATABASE_UNAVAILABLE'}
