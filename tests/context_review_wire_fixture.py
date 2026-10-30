"""Encode canonical test replies using the actual emitted source-alias map."""
from copy import deepcopy
import json


def encode_context_reply(reply, messages):
    block = messages[1]['content'].split('\n\nCITATION_ALIASES:\n', 1)[1].split('\n\nTABLE_CONTEXT_TARGETS:', 1)[0]
    aliases = json.loads(block)
    reverse = {cid: alias for alias, cid in aliases.items()}
    assert len(reverse) == len(aliases)
    out = deepcopy(reply)
    out['citation_ids'] = [reverse.get(cid, cid) for cid in out['citation_ids']]
    for context in out.get('table_contexts', []):
        for quote in context['source_quotes']:
            quote['citation_id'] = reverse.get(quote['citation_id'], quote['citation_id'])
    return out


def assembled_context_candidate(messages):
    from machinemind.ask.factual_context_review import _expand
    view = json.loads(messages[1]['content'].split('\n\nCOMPLETE_CANDIDATE:\n', 1)[1].split('\n\nCITATION_ALIASES:', 1)[0])
    if 'read_only_answer' in view:
        return view['read_only_answer']
    return _expand(view['template'], [item['text'] for item in view['inventory']])
