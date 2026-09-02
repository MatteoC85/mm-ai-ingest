from __future__ import annotations

import copy
import json
import sys
import types
from contextlib import contextmanager
from typing import Any


def install_stubs() -> None:
    psycopg2 = types.ModuleType('psycopg2')
    psycopg2.connect = lambda **kwargs: None
    sys.modules['psycopg2'] = psycopg2

    google = sys.modules.get('google') or types.ModuleType('google')
    if not hasattr(google, '__path__'):
        google.__path__ = []
    cloud = types.ModuleType('google.cloud')
    cloud.__path__ = []
    tasks = types.ModuleType('google.cloud.tasks_v2')
    class CloudTasksClient:
        pass
    tasks.CloudTasksClient = CloudTasksClient
    cloud.tasks_v2 = tasks
    google.cloud = cloud
    sys.modules['google'] = google
    sys.modules['google.cloud'] = cloud
    sys.modules['google.cloud.tasks_v2'] = tasks


install_stubs()
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import main


_MISSING = object()


@contextmanager
def patched(obj: Any, **values: Any):
    old = {name: getattr(obj, name, _MISSING) for name in values}
    for name, value in values.items():
        setattr(obj, name, value)
    try:
        yield
    finally:
        for name, value in old.items():
            if value is _MISSING:
                delattr(obj, name)
            else:
                setattr(obj, name, value)


def serialize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): serialize(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [serialize(v) for v in value]
    if isinstance(value, set):
        return sorted(serialize(v) for v in value)
    if isinstance(value, float):
        return round(value, 12)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return repr(value)


def capture(fn, *args, **kwargs) -> dict:
    try:
        return {'value': serialize(fn(*args, **kwargs)), 'error': None}
    except Exception as exc:
        return {
            'value': None,
            'error': {
                'type': type(exc).__name__,
                'message': str(exc),
                'status_code': getattr(exc, 'status_code', None),
                'detail': getattr(exc, 'detail', None),
            },
        }


class FakeResponse:
    def __init__(self, status_code: int = 200, data: Any = None, text: str | None = None):
        self.status_code = int(status_code)
        self._data = copy.deepcopy(data)
        self.text = text if text is not None else json.dumps(data, ensure_ascii=False, default=str)

    def json(self):
        return copy.deepcopy(self._data)


class PostRecorder:
    def __init__(self, responses: list[Any]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, url, *, headers=None, json=None, timeout=None, **kwargs):
        self.calls.append({
            'url': url,
            'headers': copy.deepcopy(headers),
            'json': copy.deepcopy(json),
            'timeout': timeout,
            'extra': copy.deepcopy(kwargs),
        })
        if not self.responses:
            raise RuntimeError('no fake response queued')
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeMeter:
    def __init__(self):
        self.calls: list[dict] = []

    def record_embedding(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))


class FakeBudget:
    def __init__(self, *, remaining: float = 30.0, cache: dict | None = None):
        self._remaining = float(remaining)
        self.embedding_cache = dict(cache or {})
        self.embedding_cache_hits = 0
        self.ensure_calls: list[float] = []
        self.embedding_calls: list[dict] = []
        self.llm_calls = 0
        self.reserve_calls: list[dict] = []
        self.usage_calls: list[dict] = []
        self.failed_calls: list[dict] = []
        self.retry_calls: list[dict] = []
        self.reserve_exception: BaseException | None = None
        self.reserve_timeout = 17
        self.reserve_output_cap = 1234

    def ensure_time(self, seconds: float):
        self.ensure_calls.append(float(seconds))

    def remaining(self):
        return self._remaining

    def record_embedding(self, **kwargs):
        self.embedding_calls.append(copy.deepcopy(kwargs))

    def reserve_call(self, **kwargs):
        self.reserve_calls.append(copy.deepcopy(kwargs))
        if self.reserve_exception is not None:
            raise self.reserve_exception
        self.llm_calls += 1
        return self.reserve_timeout, self.reserve_output_cap, self.llm_calls

    def record_usage(self, call_index, model, usage):
        self.usage_calls.append({
            'call_index': call_index,
            'model': model,
            'usage': copy.deepcopy(usage),
        })

    def mark_call_failed(self, call_index, error):
        self.failed_calls.append({'call_index': call_index, 'error': str(error)})

    def grant_retry_allowance(self, **kwargs):
        self.retry_calls.append(copy.deepcopy(kwargs))
        return int(kwargs.get('failed_attempts') or 0)

    def snapshot(self):
        return serialize({
            'remaining': self._remaining,
            'embedding_cache': self.embedding_cache,
            'embedding_cache_hits': self.embedding_cache_hits,
            'ensure_calls': self.ensure_calls,
            'embedding_calls': self.embedding_calls,
            'llm_calls': self.llm_calls,
            'reserve_calls': self.reserve_calls,
            'usage_calls': self.usage_calls,
            'failed_calls': self.failed_calls,
            'retry_calls': self.retry_calls,
        })


BASE_SETTINGS = {
    'OPENAI_API_KEY': 'secret-key',
    'OPENAI_EMBED_MODEL': 'embed-model',
    'OPENAI_EMBED_URL': 'https://provider.test/embeddings',
    'OPENAI_CHAT_MODEL': 'chat-default',
    'OPENAI_CHAT_URL': 'https://provider.test/chat',
    'OPENAI_RESPONSES_URL': 'https://provider.test/responses',
    'V13_FAST_MODEL': 'gpt-5.6-terra',
}


def with_settings(**overrides):
    values = dict(BASE_SETTINGS)
    values.update(overrides)
    return patched(main, **values)


out: dict[str, Any] = {
    'signatures': {},
    'normalize': {},
    'chat_text': {},
    'chat_json': {},
    'chat_json_models': {},
    'embed': {},
    'safety_identifier': {},
    'response_text': {},
    'responses_json': {},
    'json_models': {},
}

import inspect
for name in [
    '_openai_embed_texts', '_openai_chat', '_openai_chat_json',
    '_normalize_model_candidates', '_openai_chat_json_models',
    '_v13_safety_identifier', '_v13_response_text', '_v13_responses_json', '_v13_json_models',
]:
    out['signatures'][name] = str(inspect.signature(getattr(main, name)))

# Pure normalizer
for label, value in {
    'none': None,
    'empty': [],
    'dedupe': [' a ', '', 'a', 'b', None, ' b ', 'c'],
}.items():
    out['normalize'][label] = capture(main._normalize_model_candidates, value)

# Basic chat transport
with with_settings(OPENAI_API_KEY=''):
    out['chat_text']['missing_key'] = capture(main._openai_chat, [{'role': 'user', 'content': 'x'}])
rec = PostRecorder([FakeResponse(data={'choices': [{'message': {'content': 'hello'}}]})])
with with_settings(), patched(main.requests, post=rec):
    result = capture(main._openai_chat, [{'role': 'user', 'content': 'x'}])
    out['chat_text']['success_default'] = {'result': result, 'calls': serialize(rec.calls)}
rec = PostRecorder([FakeResponse(data={'choices': [{'message': {'content': 'custom'}}]})])
with with_settings(), patched(main.requests, post=rec):
    result = capture(main._openai_chat, [{'role': 'system', 'content': 's'}], model='m2', temperature=0.7)
    out['chat_text']['success_custom'] = {'result': result, 'calls': serialize(rec.calls)}
rec = PostRecorder([FakeResponse(status_code=429, data={'error': 'rate'}, text='RATE')])
with with_settings(), patched(main.requests, post=rec):
    out['chat_text']['http_error'] = {'result': capture(main._openai_chat, []), 'calls': serialize(rec.calls)}

# JSON chat transport
schema = {'name': 'x', 'strict': True, 'schema': {'type': 'object'}}
rec = PostRecorder([FakeResponse(data={'choices': [{'message': {'content': '{"a":1}'}}]})])
with with_settings(), patched(main.requests, post=rec):
    out['chat_json']['string_content'] = {
        'result': capture(main._openai_chat_json, [{'role': 'user', 'content': 'q'}], model='m-json', json_schema=schema, timeout=9),
        'calls': serialize(rec.calls),
    }
rec = PostRecorder([FakeResponse(data={'choices': [{'message': {'content': [
    {'type': 'text', 'text': '{"a":'}, {'type': 'ignored', 'text': 'bad'}, {'type': 'text', 'text': '2}'},
]}}]})])
with with_settings(), patched(main.requests, post=rec):
    out['chat_json']['list_content'] = {
        'result': capture(main._openai_chat_json, [], timeout=11),
        'calls': serialize(rec.calls),
    }
for label, response in {
    'empty': FakeResponse(data={'choices': [{'message': {'content': ''}}]}),
    'invalid': FakeResponse(data={'choices': [{'message': {'content': '{bad'}}]}),
    'http_error': FakeResponse(status_code=500, data={'error': 'x'}, text='BROKEN'),
}.items():
    rec = PostRecorder([response])
    with with_settings(), patched(main.requests, post=rec):
        out['chat_json'][label] = {'result': capture(main._openai_chat_json, []), 'calls': serialize(rec.calls)}

# Legacy model fallback helper
calls = []
def fake_chat_json(messages, *, model=None, json_schema=None, timeout=60):
    calls.append({'model': model, 'schema': copy.deepcopy(json_schema), 'timeout': timeout, 'messages': copy.deepcopy(messages)})
    if model == 'bad':
        raise RuntimeError('bad model')
    return {'model': model}
with with_settings(), patched(main, _openai_chat_json=fake_chat_json):
    out['chat_json_models']['fallback'] = {
        'result': capture(main._openai_chat_json_models, [{'x': 1}], models=[' bad ', 'bad', '', 'good'], json_schema=schema, timeout=13),
        'calls': serialize(calls),
    }
calls = []
def all_bad(messages, *, model=None, json_schema=None, timeout=60):
    calls.append(model)
    raise RuntimeError(f'fail-{model}')
with with_settings(), patched(main, _openai_chat_json=all_bad):
    out['chat_json_models']['all_fail'] = {
        'result': capture(main._openai_chat_json_models, [], models=['x', 'y']),
        'calls': serialize(calls),
    }
calls = []
def default_ok(messages, *, model=None, json_schema=None, timeout=60):
    calls.append(model)
    return {'ok': True}
with with_settings(), patched(main, _openai_chat_json=default_ok):
    out['chat_json_models']['default_model'] = {
        'result': capture(main._openai_chat_json_models, []),
        'calls': serialize(calls),
    }

# Embeddings
with with_settings(OPENAI_API_KEY=''):
    out['embed']['missing_key'] = capture(main._openai_embed_texts, ['x'])
with with_settings():
    out['embed']['empty'] = capture(main._openai_embed_texts, [])

budget = FakeBudget(remaining=7.8)
meter = FakeMeter()
rec = PostRecorder([FakeResponse(data={
    'usage': {'prompt_tokens': 7},
    'data': [
        {'index': 1, 'embedding': [2.0, 2.5]},
        {'index': 0, 'embedding': [1.0, 1.5]},
    ],
})])
with with_settings(), patched(main, _v13_current_budget=lambda: budget, _current_ingest_meter=lambda: meter), patched(main.requests, post=rec):
    out['embed']['provider_usage_duplicates'] = {
        'result': capture(main._openai_embed_texts, ['abc', 'abc', 'def'], timeout=60),
        'calls': serialize(rec.calls),
        'budget': budget.snapshot(),
        'meter': serialize(meter.calls),
    }

budget = FakeBudget(remaining=30.0, cache={('embed-model', 'cached'): [9.0]})
meter = FakeMeter()
rec = PostRecorder([])
with with_settings(), patched(main, _v13_current_budget=lambda: budget, _current_ingest_meter=lambda: meter), patched(main.requests, post=rec):
    out['embed']['full_cache'] = {
        'result': capture(main._openai_embed_texts, ['cached', 'cached']),
        'calls': serialize(rec.calls),
        'budget': budget.snapshot(),
        'meter': serialize(meter.calls),
    }

budget = FakeBudget(remaining=30.0, cache={('embed-model', 'cached'): [9.0]})
meter = FakeMeter()
rec = PostRecorder([FakeResponse(data={'usage': {}, 'data': [{'index': 0, 'embedding': [3.0]}]})])
with with_settings(), patched(main, _v13_current_budget=lambda: budget, _current_ingest_meter=lambda: meter), patched(main.requests, post=rec):
    out['embed']['partial_cache_fallback_usage'] = {
        'result': capture(main._openai_embed_texts, ['cached', 'new', 'new']),
        'calls': serialize(rec.calls),
        'budget': budget.snapshot(),
        'meter': serialize(meter.calls),
    }

for label, response in {
    'missing_vector': FakeResponse(data={'data': []}),
    'http_error': FakeResponse(status_code=503, data={'error': 'down'}, text='DOWN'),
}.items():
    budget = FakeBudget()
    meter = FakeMeter()
    rec = PostRecorder([response])
    with with_settings(), patched(main, _v13_current_budget=lambda: budget, _current_ingest_meter=lambda: meter), patched(main.requests, post=rec):
        out['embed'][label] = {
            'result': capture(main._openai_embed_texts, ['one']),
            'calls': serialize(rec.calls),
            'budget': budget.snapshot(),
            'meter': serialize(meter.calls),
        }

# Stable per-company safety identifier
for label, value in {
    'empty': '',
    'ascii': 'company-1',
    'unicode': 'azienda-à/测试',
}.items():
    out['safety_identifier'][label] = capture(main._v13_safety_identifier, value)

# Responses text parser
for label, data in {
    'direct': {'output_text': '  direct  '},
    'parts': {'output': [{'type': 'message', 'content': [
        {'type': 'output_text', 'text': ' a '}, {'type': 'output_text', 'text': 'b'},
    ]}]},
    'refusal': {'output': [{'type': 'message', 'content': [{'type': 'refusal', 'refusal': 'denied'}]}]},
    'empty': {'output': []},
}.items():
    out['response_text'][label] = capture(main._v13_response_text, data)

# Responses API structured call
messages = [{'role': 'user', 'content': 'q'}]
resp_schema = {'name': 'schema-name', 'strict': False, 'schema': {'type': 'object', 'properties': {}}}
budget = FakeBudget()
rec = PostRecorder([FakeResponse(data={
    'status': 'completed', 'output_text': '{"answer":"ok"}',
    'usage': {'input_tokens': 10, 'output_tokens': 4},
})])
with with_settings(), patched(main, _v13_current_budget=lambda: budget), patched(main.requests, post=rec):
    out['responses_json']['success'] = {
        'result': capture(main._v13_responses_json, messages, model='gpt-5.6-terra', json_schema=resp_schema, effort='high', reasoning_mode='pro', timeout=25, max_output_tokens=2000, company_id='company-1', purpose='test-purpose'),
        'calls': serialize(rec.calls),
        'budget': budget.snapshot(),
    }

with with_settings(OPENAI_API_KEY=''):
    out['responses_json']['missing_key'] = capture(main._v13_responses_json, messages, model='gpt-5.6-terra', json_schema=resp_schema, effort='medium', timeout=20, max_output_tokens=1000, company_id='c', purpose='p')
with with_settings(), patched(main, _v13_current_budget=lambda: None):
    out['responses_json']['missing_budget'] = capture(main._v13_responses_json, messages, model='gpt-5.6-terra', json_schema=resp_schema, effort='medium', timeout=20, max_output_tokens=1000, company_id='c', purpose='p')

response_cases = {
    'http_error': FakeResponse(status_code=502, data={'error': 'bad'}, text='BAD'),
    'incomplete': FakeResponse(data={'status': 'incomplete', 'incomplete_details': {'reason': 'limit'}, 'usage': {}}),
    'failed': FakeResponse(data={'status': 'failed', 'error': {'message': 'boom'}, 'usage': {}}),
    'parse_error': FakeResponse(data={'status': 'completed', 'output_text': '{bad', 'usage': {}}),
    'non_object': FakeResponse(data={'status': 'completed', 'output_text': '[1,2]', 'usage': {}}),
}
for label, response in response_cases.items():
    budget = FakeBudget()
    rec = PostRecorder([response])
    with with_settings(), patched(main, _v13_current_budget=lambda: budget), patched(main.requests, post=rec):
        out['responses_json'][label] = {
            'result': capture(main._v13_responses_json, messages, model='gpt-5.6-terra', json_schema=resp_schema, effort='medium', timeout=20, max_output_tokens=1000, company_id='c', purpose='p'),
            'calls': serialize(rec.calls),
            'budget': budget.snapshot(),
        }

# Multi-provider V13 fallback orchestration
class BudgetExceeded(main._V13BudgetExceeded):
    pass

budget = FakeBudget()
provider_calls = []
def fake_responses(messages, **kwargs):
    provider_calls.append({'kind': 'responses', **copy.deepcopy(kwargs)})
    budget.llm_calls += 1
    if kwargs['model'] == 'gpt-5.6-bad':
        raise RuntimeError('responses-down')
    return {'via': 'responses'}
def fake_legacy(messages, *, model=None, json_schema=None, timeout=60):
    provider_calls.append({'kind': 'chat', 'model': model, 'timeout': timeout, 'json_schema': copy.deepcopy(json_schema)})
    return {'via': 'chat'}
with with_settings(), patched(main, _v13_current_budget=lambda: budget, _v13_responses_json=fake_responses, _openai_chat_json=fake_legacy):
    out['json_models']['fallback_to_legacy'] = {
        'result': capture(main._v13_json_models, messages, models=['gpt-5.6-bad', 'legacy-model'], json_schema=resp_schema, effort='medium', reasoning_mode='', timeout=20, max_output_tokens=1000, company_id='c', purpose='multi'),
        'provider_calls': serialize(provider_calls),
        'budget': budget.snapshot(),
    }

budget = FakeBudget()
def first_budget_exceeded(messages, **kwargs):
    raise main._V13BudgetExceeded('no budget')
with with_settings(), patched(main, _v13_current_budget=lambda: budget, _v13_responses_json=first_budget_exceeded):
    out['json_models']['first_budget_exceeded'] = {
        'result': capture(main._v13_json_models, messages, models=['gpt-5.6-terra'], json_schema=resp_schema, effort='medium', reasoning_mode='', timeout=20, max_output_tokens=1000, company_id='c', purpose='multi'),
        'budget': budget.snapshot(),
    }

budget = FakeBudget()
provider_calls = []
def always_fail_responses(messages, **kwargs):
    provider_calls.append(kwargs['model'])
    budget.llm_calls += 1
    raise RuntimeError(f"down-{kwargs['model']}")
with with_settings(), patched(main, _v13_current_budget=lambda: budget, _v13_responses_json=always_fail_responses):
    out['json_models']['all_fail'] = {
        'result': capture(main._v13_json_models, messages, models=['gpt-5.6-a', 'gpt-5.6-b'], json_schema=resp_schema, effort='medium', reasoning_mode='', timeout=20, max_output_tokens=1000, company_id='c', purpose='multi'),
        'provider_calls': serialize(provider_calls),
        'budget': budget.snapshot(),
    }

print(json.dumps(serialize(out), ensure_ascii=False, sort_keys=True, separators=(',', ':')))
