"""Encode canonical test replies using the actual emitted source-alias map."""
from copy import deepcopy
import json
import re


def original_context_sources(view):
    """Undo only the reference markers; the original source wire stays exact."""
    return re.sub(r'\[Q[0-9]+\]|\[/Q\]', '', view)


def encode_context_reply(reply, messages):
    block = messages[1]['content'].split('\n\nCITATION_ALIASES:\n', 1)[1].split('\n\nTABLE_CONTEXT_TARGETS:', 1)[0]
    aliases = json.loads(block)
    reverse = {cid: alias for alias, cid in aliases.items()}
    assert len(reverse) == len(aliases)
    out = deepcopy(reply)
    source_view = messages[1]['content'].split('\n\nSOURCES:\n', 1)[1].split('\n\nCOMPLETE_CANDIDATE:', 1)[0]
    refs = {}
    for ref, literal in re.findall(r'\[(Q[1-9][0-9]*)\](.*?)\[/Q\]', source_view, re.DOTALL):
        assert ref not in refs or refs[ref] == literal
        refs[ref] = literal
    by_literal = {literal: ref for ref, literal in refs.items()}
    out['citation_ids'] = [reverse.get(cid, cid) for cid in out['citation_ids']]
    for context in out.get('table_contexts', []):
        for quote in context['source_quotes']:
            quote['citation_id'] = reverse.get(quote['citation_id'], quote['citation_id'])
            # An invented quotation becomes an unknown reference, never a
            # silently substituted supported sentence in a negative fixture.
            quote['quote_ref'] = by_literal.get(quote.pop('quote'), 'Q99999')
    return out


def assembled_context_candidate(messages):
    from machinemind.ask.factual_context_review import _expand
    view = json.loads(messages[1]['content'].split('\n\nCOMPLETE_CANDIDATE:\n', 1)[1].split('\n\nCITATION_ALIASES:', 1)[0])
    if 'read_only_answer' in view:
        return view['read_only_answer']
    return _expand(view['template'], [item['text'] for item in view['inventory']])
