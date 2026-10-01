"""Request-local proof that cacheable provider usage was durably checkpointed.

Only the admitted ASGI owner activates it. Context propagation uses the existing
execution helpers; OFF/non-HTTP callers have exactly their previous behaviour.
No permission, response body or mutable caller flag is accepted as a receipt.
"""
from contextvars import ContextVar
from threading import RLock
from .contracts import settlement

_ACTIVE = ContextVar('mm_interactive_usage_receipt_v6', default=None)

class CacheReceipt:
    def __init__(self, ledger, context, fingerprint, cap):
        self.ledger=ledger; self.context=context; self.fingerprint=fingerprint; self.cap=cap
        self.active=True; self.checked=None; self.lock=RLock()
    def close(self):
        # Do not wait for a DB/network call on cancellation. The second active
        # check below rejects a callback whose request has already terminated.
        self.active=False
    def allow(self, company, budget):
        with self.lock:
            if not self.active or company!=self.context.company_id or budget is None:return False
            try:
                m=budget.public_meta()
                result=settlement({'status':'CACHE_CHECKPOINT','meta':{
                    'v13_estimated_cost_usd':m['estimated_cost_usd'],
                    'v13_committed_cost_usd':m['committed_cost_usd'],
                    'v13_uncertain_cost_usd':m['uncertain_cost_usd'],
                    'v13_accounting_complete':m['accounting_complete']}},200,self.cap)
                if result.state!='settled':return False
                if self.checked!=result.known:
                    self.ledger.checkpoint(self.context,self.fingerprint,result.known)
                    self.checked=result.known
                return self.active
            except Exception:
                return False

def activate(receipt):return _ACTIVE.set(receipt)
def deactivate(token):_ACTIVE.reset(token)
def allow_cache(company, budget):
    owner=_ACTIVE.get()
    return True if owner is None else owner.allow(company,budget)
