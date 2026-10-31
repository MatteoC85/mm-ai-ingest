"""Encode canonical test replies using the actual emitted source-alias map."""
from copy import deepcopy
import json
import re


def encode_table_coverage_reply(reply, json_schema):
    """Encode legacy fixture classifications for the actual keyed wire schema.

    Duplicate rows are deliberately not coalesced: raw duplicate-key tests must
    supply response text to the real transport, preserving that failure mode.
    """
    field = json_schema.get('schema', {}).get('properties', {}).get('table_record_coverage', {})
    rows = reply.get('table_record_coverage')
    if field.get('type') != 'object' or type(rows) is not list:
        return reply
    keyed = {rid: None for rid in field['required']}
    seen = set()
    for row in rows:
        match = re.fullmatch(r'(T[1-9][0-9]*)(.*)', row) if type(row) is str else None
        if match is None:
            raise AssertionError('Malformed fixture needs an explicit keyed response')
        rid, suffix = match.groups()
        if rid in seen:
            raise AssertionError('Duplicate fixture needs raw JSON transport coverage')
        seen.add(rid)
        keyed[rid] = suffix
    return {**deepcopy(reply), 'table_record_coverage': keyed}


def original_context_sources(view, original_sources=None):
    """Undo only the reference markers; the original source wire stays exact."""
    from machinemind.ask import review_source_wire
    from machinemind.ask.source_table_review import _HEADER
    result = re.sub(r'\[Q[0-9]+\]|\[/Q\]', '', view)
    if original_sources is not None:
        canonical = sorted(match[1] for match in _HEADER.finditer(review_source_wire.restore_sources_block(original_sources)))
        aliases = {'S'+str(i):cid for i,cid in enumerate(canonical)}
        def restore(match):
            cid = aliases.get(match[1],match[1])
            return '['+cid+']'+match[0][len(match[1])+2:]
        result = _HEADER.sub(restore,result)
    return result


def encode_context_reply(reply, messages, *, json_schema, source_ids=None, original_sources=None):
    from machinemind.ask import review_source_wire
    from machinemind.ask.source_table_review import _HEADER
    user = messages[1]['content']
    atomic = '\n\nCITATION_ALIASES:\n' not in user
    if atomic:
        if original_sources is not None:
            source_ids = [match[1] for match in _HEADER.finditer(review_source_wire.restore_sources_block(original_sources))]
        assert source_ids is not None, 'Atomic fixtures must use the actual original source IDs, never infer them from a model reply.'
        aliases = {'S'+str(i):cid for i,cid in enumerate(sorted(set(source_ids)))}
    else:
        block = user.split('\n\nCITATION_ALIASES:\n', 1)[1].split('\n\nTABLE_CONTEXT_TARGETS:', 1)[0]
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
    out['additional_citation_ids'] = [reverse.get(cid, cid) for cid in out.pop('citation_ids')]
    for context in out.get('table_contexts', []):
        pairs = []
        for quote in context['source_quotes']:
            quote['citation_id'] = reverse.get(quote['citation_id'], quote['citation_id'])
            # An invented quotation becomes an unknown reference, never a
            # silently substituted supported sentence in a negative fixture.
            quote['quote_ref'] = by_literal.get(quote.pop('quote'), 'Q99999')
            if atomic:
                alias,ref = quote['citation_id'],quote['quote_ref']
                pairs.append(alias[1:]+':'+ref[1:] if re.fullmatch(r'S[0-9]+',alias) else '99999:'+ref[1:])
        if atomic:
            context['source_quotes'] = pairs
    context_schema = json_schema['schema']['properties']['table_contexts']
    if atomic and context_schema.get('type') == 'object':
        # The provider receives these exact enums. Do not depend on duplicate
        # user-message hints, invent an owner, or repair an unsupported quote.
        owners = {target: slot['anyOf'][0]['properties']['owning_source_quote']['enum']
                  for target, slot in context_schema['properties'].items()}
        slots = {}
        for context in out.get('table_contexts', []):
            target = context['target']
            if target in slots:
                # A duplicate cannot be represented by an object. Keep the
                # malformed legacy shape so a negative fixture still fails.
                return out
            pairs = list(context['source_quotes'])
            selected = next((i for i, ref in enumerate(pairs) if ref in owners.get(target, [])), 0)
            owner = pairs.pop(selected) if pairs else ''
            slots[target] = {'context': context['context'],
                'documentary_limits_checked': context['documentary_limits_checked'],
                'owning_source_quote': owner, 'supplementary_source_quotes': pairs}
        out['table_contexts'] = slots
    return out


def assembled_context_candidate(messages):
    from machinemind.ask.factual_context_review import _expand, _emitted_source_bodies, _candidate_table_text
    from machinemind.ask import review_source_wire
    from machinemind.ask.source_table_review import _HEADER
    user = messages[1]['content']
    view = json.loads(user.split('\n\nCOMPLETE_CANDIDATE:\n', 1)[1].split('\n\n', 1)[0])
    if 'read_only_answer' in view:
        return view['read_only_answer']
    sources=original_context_sources(user.split('\n\nSOURCES:\n',1)[1].split('\n\nCOMPLETE_CANDIDATE:',1)[0])
    ids=[m[1]for m in _HEADER.finditer(review_source_wire.restore_sources_block(sources))]
    bodies=_emitted_source_bodies(sources,ids)
    language=user.split('\n\nRESPONSE_LANGUAGE: ',1)[1].split('\n\nSOURCES:',1)[0]
    items=[_candidate_table_text(bodies[item['source_table']],item['caption'],language)
           if 'source_table' in item else item['text'] for item in view['inventory']]
    return _expand(view['template'], items)
