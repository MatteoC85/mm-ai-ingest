"""Bounded independent review of an already assembled factual ASK candidate.

This pure representation module supplies no source authority or model approval.
The caller must reuse its admitted source packet and provider wrapper, then run
the ordinary coverage, grounding, language, safety and final binding gates.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import re

from . import enumeration_review
from . import review_source_wire
from . import table_units
from .source_table_review import PREFIX, TableCatalog, _HEADER, _RECORD_ID, _escape, _render as _render_table_rows

MARKER = enumeration_review.INVENTORY_MARKER
MAX_TEXT = 30000
_QUOTE_MARKER = re.compile(r'\[\s*/?\s*Q\s*[0-9]*\s*\]', re.IGNORECASE)
_QUOTE_REF = re.compile(r'Q[1-9][0-9]{0,4}')
_SOURCE_MARKER = re.compile(
    r'(\[PDF_TABLE_[^\]\r\n]*\]|\[TABLE_RECORD T[1-9][0-9]*\]|'
    r'\[\[SOURCE_TABLE:[^\]\r\n]*\]\]|\[\[FACTUAL_INVENTORY\]\])')
CHECKS = ('source_fidelity', 'scope_applicability', 'completeness',
          'safety', 'language', 'provenance')
INSTRUCTIONS = (
    'Independently review the complete candidate against QUESTION and SOURCES. '
    'No prior model verdict is supplied. Check source fidelity, requested scope '
    'and applicability, completeness across every requested category, safety '
    'prerequisites for each physical intervention retained, response language and '
    'citation provenance. Review the fully assembled result after any edits, not '
    'only the changed sentences. Immutable source quotations cannot be edited or '
    'paraphrased; they are documentary context, not additional requested operations. '
    'Correct only a non-immutable inventory target or template-answer: complete supported conditions in the '
    'same item or remove an unnecessary physical intervention. For informational requests, '
    'remove unrequested access, setup, state-change and restart instructions from editable '
    'prose while preserving requested facts, applicability and qualifications. If an intervention '
    'is explicitly requested, retain all its source-supported prerequisites and operating '
    'conditions; generic safety advice cannot replace a specific condition. Do not introduce '
    'an unrequested operation, source, value or assembly equivalence. Do not add, '
    'delete or reorder inventory slots. A template-answer edit must retain '
    '[[FACTUAL_INVENTORY]] exactly once on its own line; the server inserts all '
    'inventory entries, changing only accepted non-table item edits. Return pass '
    'only for the unchanged complete safe candidate, with no edits. Return rewrite '
    'only if the assembled revised candidate meets all six checks. Otherwise return '
    'no_sources. Preserve all current citation IDs and cite additional support only '
    'from SOURCES. A read_only_answer candidate permits no edits: '
    'return pass or no_sources, never rewrite. Return only the strict JSON object.'
    ' TABLE_CONTEXTS: audit every TABLE_CONTEXT_TARGET exactly once, identifying '
    'the assembly/applicability only as far as SOURCES establish it and retaining '
    'all applicable qualifications outside the cells (including adjacent prose). '
    'Give one or two short sentences of reading context and select one or two '
    'supporting quote_ref units from SOURCES, including the owning source. '
    'The server places this context and its quotations immediately before that '
    'unchanged table group. This adds no requested intervention. Do not copy a '
    'whole source or unrelated operations. A context label saying documentary '
    'or not instructions never waives the qualifications. Check any assertion '
    'that information or a chapter is unavailable against all supplied excerpts: '
    'an incomplete excerpt does not establish absence from the complete document. '
    'Correct an overbroad absence claim through the ordinary editable item or '
    'template. Set documentary_limits_checked true only after this check. '
    'With table contexts, return rewrite even if edits is empty, because the '
    'qualified contexts are additions; all six checks concern that final assembled '
    'answer. Without table targets return table_contexts=[] and preserve the '
    'ordinary pass/rewrite rules. If any context remains unsupported, no_sources.'
)

_ATOMIC_INSTRUCTIONS = (INSTRUCTIONS
    .replace('Preserve all current citation IDs and cite additional support only from SOURCES.',
             'Current citations remain preserved. Cite support for edits through additional_citation_ids only from SOURCES.')
    .replace('supporting quote_ref units from SOURCES', 'supporting source_quotes references from the schema')
    .replace('This adds no requested intervention. ', '')
    .replace('because the qualified contexts are additions; all six checks concern that final assembled answer.',
             'because the qualified contexts are additions.')
    .replace('Without table targets return table_contexts=[] and preserve the ordinary pass/rewrite rules.',
             'Without table targets return table_contexts=[].'))
_ATOMIC_REPRESENTATION = (
    '\nCOMPLETE_CANDIDATE is lossless: expand its template inventory marker into the ordered items, each prefixed "- ". '
    'A text item is verbatim. An immutable source_table item is its exact caption followed by ALL HEADER/RECORD rows '
    'from that source occurrence, in source order (including notes and row restarts). Each row is "  - " plus every '
    'nonempty cell as [explicit label+": "]+"«"+literal value+"»", joined by "; ", with Markdown escaping. '
    'Remove only [Qn] and [/Q] markers before literal table expansion. '
    'No other candidate prose is hidden; adjacent source qualifications still apply. '
    'Source header Sx denotes its unchanged admitted source. additional_citation_ids contains only extra Sx support. '
    'Select source_quotes only from the enum: token "a:b" explicitly cites source Sa and literal unit Qb '
    'marked [Qb]...[/Q] in SOURCES. Do not copy quotation text or invent pairs. Existing @n references expand normally. '
    'The server preserves current citations and adds the sources explicitly selected by these references and additional IDs. '
    'References belong only in source_quotes, never as markers in public context or edits.')


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _has_template(body):
    return (type(body) is str and body.count(enumeration_review.INVENTORY_PREFIX) == 1
            and body.splitlines().count(MARKER) == 1)


def _expand(template, items):
    if not _has_template(template):
        raise ValueError('factual_context_review_invalid')
    lines = template.splitlines(keepends=True)
    index = next(i for i,line in enumerate(lines) if line.rstrip('\r\n') == MARKER)
    ending = '\r\n' if lines[index].endswith('\r\n') else '\n' if lines[index].endswith('\n') else ''
    lines[index] = '\n'.join('- ' + item for item in items) + ending
    return ''.join(lines)


@dataclass(frozen=True)
class ReviewFrame:
    first_json: str
    answer: str
    template: str
    items: tuple[str, ...]
    immutable_indices: tuple[int, ...]
    admitted_citation_ids: tuple[str, ...]
    current_citation_ids: tuple[str, ...]
    editable_targets: tuple[str, ...]
    table_owners: tuple[tuple[int, str], ...]
    source_bodies: tuple[tuple[str, str], ...]
    source_packet_sha256: str
    quote_view: str
    quote_units: tuple[tuple[str, str, tuple[str, ...]], ...]


def _emitted_source_bodies(sources_block, allowed):
    """Decode only the already emitted packet; this acquires no source authority."""
    if type(sources_block) is not str:
        raise ValueError('factual_context_review_invalid')
    restored = review_source_wire.restore_sources_block(sources_block)
    headers = list(_HEADER.finditer(restored))
    if (not headers or len(headers) > 64 or len(restored) > MAX_TEXT + 2048
            or restored[:headers[0].start()].strip()):
        raise ValueError('factual_context_review_invalid')
    rows = {}
    for index, header in enumerate(headers):
        cid = header[1]
        if cid not in allowed or cid in rows:
            raise ValueError('factual_context_review_invalid')
        end = headers[index + 1].start() if index + 1 < len(headers) else len(restored)
        body = '\n'.join(line for line in restored[header.end():end].strip().splitlines()
                         if not _RECORD_ID.fullmatch(line))
        rows[cid] = body
    return rows


def _source_quote_view(sources_block, source_bodies):
    """Annotate the existing wire without changing its dictionary or source text.

    References name contiguous literal slices, not claims or semantic support.
    Equal text may share a reference; source membership remains separate.
    """
    if _QUOTE_MARKER.search(sources_block):
        raise ValueError('factual_context_review_invalid')
    units, owners, out = {}, {}, []

    def mark(part):
        if any(prefix in part for prefix in (PREFIX, enumeration_review.INVENTORY_PREFIX,
                                             '[TABLE_RECORD ', '[PDF_TABLE_')):
            return part
        result = []
        while part:
            boundary = min(480, len(part))
            if boundary < len(part):
                space = max(part.rfind(' ', 0, boundary + 1), part.rfind('\t', 0, boundary + 1))
                if space > 0:
                    boundary = space
            piece, part = part[:boundary], part[boundary:]
            literal = piece.strip()
            membership = tuple(cid for cid, body in source_bodies.items() if literal and literal in body)
            if not membership:
                result.append(piece)
                continue
            ref = units.setdefault(literal, 'Q' + str(len(units) + 1))
            owners[ref] = membership
            left = len(piece) - len(piece.lstrip())
            result.append(piece[:left] + '[' + ref + ']' + literal + '[/Q]' + piece[left + len(literal):])
        return ''.join(result)

    compressed = sources_block.startswith(review_source_wire._START)
    records = not compressed
    for line in review_source_wire._lines(sources_block):
        text = line.rstrip('\r\n')
        ending = line[len(text):]
        if compressed and text == review_source_wire._RECORDS.rstrip('\n'):
            records = True
            out.append(line)
            continue
        dictionary = re.fullmatch(r'(@[1-9][0-9]*=)(.*)', text) if compressed and not records else None
        prefix = ''
        if dictionary:
            prefix, text = dictionary.groups()
        elif (not records or _HEADER.fullmatch(text) or _RECORD_ID.fullmatch(text)
                or (compressed and re.fullmatch(r'@[1-9][0-9]*', text))):
            out.append(line)
            continue
        out.append(prefix)
        if '[PDF_TABLE_' in text:
            # Keep all geometry visible but quote only the existing literal
            # JSON value spans. No decoding, header inheritance or row joining.
            match = table_units._LINE.fullmatch(text)
            cells = None
            if match:
                first, last, columns = int(match[2]), int(match[3] or match[2]), int(match[4])
                if 1 <= first <= last <= 100000 and 1 <= columns <= 256:
                    try:
                        cells = table_units._cells(match[5], match.start(5), first, last, columns)
                    except (ValueError, TypeError):
                        pass
            pos = 0
            for cell in cells or ():
                start, end = cell['span']
                out.append(text[pos:start + 1])
                out.append(mark(text[start + 1:end - 1]))
                pos = end - 1
            out.append(text[pos:] + ending)
            continue
        for part in _SOURCE_MARKER.split(text):
            if not part:
                continue
            if _SOURCE_MARKER.fullmatch(part):
                out.append(part)
                continue
            out.append(mark(part))
        out.append(ending)
    view = ''.join(out)
    if _QUOTE_MARKER.sub('', view) != sources_block:
        raise ValueError('factual_context_review_invalid')
    return view, tuple((ref, literal, owners[ref]) for literal, ref in units.items())


def prepare_review(wire_reply, expanded_reply, *, catalog: TableCatalog,
                   admitted_citation_ids, sources_block='') -> ReviewFrame:
    """Bind exact first-stage data; never locate editable blocks by fuzzy text."""
    invalid = ValueError('factual_context_review_invalid')
    if type(wire_reply) is not dict or type(expanded_reply) is not dict or type(catalog) is not TableCatalog:
        raise invalid
    if (wire_reply.get('outcome') not in {'pass', 'rewrite'}
            or expanded_reply.get('outcome') != wire_reply.get('outcome')
            or any(expanded_reply.get(key) for key in
                   ('missing_facets', 'missing_answer_types', 'missing_list_items'))):
        raise invalid
    raw_items, items = wire_reply.get('expected_list_items'), expanded_reply.get('expected_list_items')
    answer, template = expanded_reply.get('answer'), wire_reply.get('answer')
    if (type(raw_items) is not list or type(items) is not list
            or not 0 < len(items) == len(raw_items) <= 40
            or any(type(x) is not str or not x.strip() for x in raw_items + items)
            or type(answer) is not str or not answer.strip() or len(answer) > MAX_TEXT
            or type(template) is not str or len(template) > MAX_TEXT
            or sum(map(len, items)) > MAX_TEXT):
        raise invalid
    entries = {entry.token: entry for entry in catalog.entries}
    immutable = []
    for i, (raw, item) in enumerate(zip(raw_items, items)):
        if PREFIX in raw:
            entry = entries.get(raw)
            if entry is None or item != entry.text:
                raise invalid
            immutable.append(i)
        elif raw != item:
            raise invalid
    if immutable and any(_QUOTE_MARKER.search(value) for value in (answer, template, *items)):
        raise invalid
    if type(admitted_citation_ids) not in (tuple, list, set, frozenset):
        raise invalid
    if any(type(cid) is not str or not cid for cid in admitted_citation_ids):
        raise invalid
    allowed = tuple(sorted(set(admitted_citation_ids)))
    citations = expanded_reply.get('citation_ids')
    if (type(citations) is not list or not citations
            or any(type(cid) is not str or cid not in allowed for cid in citations)):
        raise invalid
    for i in immutable:
        if entries[raw_items[i]].citation_id not in citations:
            raise invalid
    source_bodies = _emitted_source_bodies(sources_block, allowed) if immutable else {}
    owners = []
    for i in immutable:
        entry = entries[raw_items[i]]
        if (entry.citation_id not in source_bodies
                or _digest(source_bodies[entry.citation_id]) != entry.source_sha256):
            raise invalid
        owners.append((i, entry.citation_id))
    template_mode = _has_template(template)
    if template_mode:
        if _expand(template, items) != answer:
            raise invalid
        targets = tuple('item:' + str(i) for i in range(len(items)) if i not in immutable) + ('template-answer',)
    else:
        # Legacy full prose can still be independently accepted unchanged. Its
        # item positions are not inferred from normalized substring matching.
        if (enumeration_review.INVENTORY_PREFIX in template or immutable
                or template != answer):
            raise invalid
        targets = ()
    quote_view, quote_units = _source_quote_view(sources_block, source_bodies) if immutable else (sources_block, ())
    return ReviewFrame(_json(expanded_reply), answer, template, tuple(items), tuple(immutable),
                       allowed, tuple(dict.fromkeys(citations)), targets,
                       tuple(owners), tuple(source_bodies.items()),
                       _digest(sources_block) if sources_block else '', quote_view, quote_units)


def build_messages(frame: ReviewFrame, *, question: str, response_language: str, sources_block: str):
    if type(frame) is not ReviewFrame or any(type(x) is not str for x in (question, response_language, sources_block)):
        raise ValueError('factual_context_review_invalid')
    if frame.source_packet_sha256 and _digest(sources_block) != frame.source_packet_sha256:
        raise ValueError('factual_context_review_invalid')
    if (frame.table_owners and _QUOTE_MARKER.sub('', frame.quote_view) != sources_block
            or not frame.table_owners and frame.quote_view != sources_block):
        raise ValueError('factual_context_review_invalid')
    aliases = _citation_aliases(frame)
    reverse = {cid: alias for alias, cid in aliases.items()}
    atomic = _atomic_references(frame)
    if _has_template(frame.template):
        view = {'template': frame.template, 'inventory': [
            {'target': 'item:' + str(i), 'immutable': i in frame.immutable_indices, 'text': text}
            for i, text in enumerate(frame.items)]}
        if _expand(view['template'], [item['text'] for item in view['inventory']]) != frame.answer:
            raise ValueError('factual_context_review_invalid')
        if atomic is not None:
            view = _compact_candidate_tables(view, frame, response_language)
    else:
        # No-marker prose stays a complete, read-only candidate.
        view = {'read_only_answer': frame.answer}
    source_view = _aliased_source_headers(frame.quote_view, aliases) if atomic is not None else frame.quote_view
    user = ('QUESTION:\n' + question + '\n\nRESPONSE_LANGUAGE: ' + response_language
        + '\n\nSOURCES:\n' + source_view + '\n\nCOMPLETE_CANDIDATE:\n' + _json(view)
        + ('\n\nCITATION_ALIASES:\n' + _json(aliases) if atomic is None else '')
        + '\n\nTABLE_CONTEXT_TARGETS:\n' + json.dumps([
            {'target': 'item:' + str(index), 'citation_id': reverse[cid]}
            for index, cid in frame.table_owners], ensure_ascii=False, separators=(',', ':'))
        + '\n\nCURRENT_CITATION_IDS:\n' + _json([reverse[cid] for cid in frame.current_citation_ids]))
    representation = (
        '\nCOMPLETE_CANDIDATE is lossless: replace the template\'s single full-line '
        '[[FACTUAL_INVENTORY]] by the ordered inventory texts, each prefixed by "- ". '
        'Judge that entire assembled answer against QUESTION. Edit only non-immutable item targets '
        'or template-answer, retaining its inventory marker. read_only_answer is immutable prose. '
        'For citation_ids and source_quotes.citation_id use only CITATION_ALIASES keys; '
        'they denote unchanged original source IDs. All quotation, applicability, limitation '
        'and safety obligations remain unchanged. '
        'SOURCES [Qn]...[/Q] marks an exact quote_ref unit; existing @n dictionary references '
        'expand with these markers intact. Select Qn present in the cited source, never copy '
        'quote text. The server inserts the literal slice; unmarked text remains source context. '
        'Use Q references only in quote_ref, never as markers in context or edits.')
    system = _ATOMIC_INSTRUCTIONS + _ATOMIC_REPRESENTATION if atomic is not None else INSTRUCTIONS + representation
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]


def _citation_aliases(frame):
    return {'S' + str(index): cid for index, cid in enumerate(frame.admitted_citation_ids)}


def _atomic_references(frame):
    """Enumerate complete source/unit pairs only within structured-output limits.

    Oversized or absent catalogs use the complete preceding wire, never a
    selected subset of sources, units, or semantic obligations.
    """
    count = sum(len(owners) for _, _, owners in frame.quote_units)
    other_enums = 3 + len(frame.editable_targets) + len(frame.table_owners) + len(frame.admitted_citation_ids)
    if not frame.table_owners or not count or count + other_enums > 1000:
        return None
    reverse = {cid: alias[1:] for alias, cid in _citation_aliases(frame).items()}
    refs = {}
    for ref, literal, owners in frame.quote_units:
        for cid in owners:
            if cid not in reverse or not _QUOTE_REF.fullmatch(ref):
                return None
            refs[reverse[cid] + ':' + ref[1:]] = (cid, literal)
    if len(refs) != count or (count > 250 and sum(map(len, refs)) > 15000):
        return None
    enum_count, string_chars, valid = 0, 0, True
    def inspect(value):
        nonlocal enum_count, string_chars, valid
        if type(value) is str:
            string_chars += len(value)
        elif type(value) is dict:
            choices = value.get('enum')
            if type(choices) is list:
                enum_count += len(choices)
                if len(choices) > 250 and sum(len(x) for x in choices if type(x) is str) > 15000:
                    valid = False
            for key, item in value.items():
                string_chars += len(key)
                inspect(item)
        elif type(value) is list:
            for item in value:
                inspect(item)
    inspect(_wire_schema(frame, refs))
    if not valid or enum_count > 1000 or string_chars > 120000:
        return None
    return refs


def _aliased_source_headers(view, aliases):
    reverse = {cid: alias for alias, cid in aliases.items()}
    def rename(match, mapping):
        if match[1] not in mapping:
            raise ValueError('factual_context_review_invalid')
        return '[' + mapping[match[1]] + ']' + match[0][len(match[1]) + 2:]
    result = _HEADER.sub(lambda match: rename(match, reverse), view)
    if _HEADER.sub(lambda match: rename(match, aliases), result) != view:
        raise ValueError('factual_context_review_invalid')
    return result


def _candidate_table_text(body, caption, language):
    records = []
    for line in body.splitlines():
        match = table_units._LINE.fullmatch(line)
        if not match:
            if '[PDF_TABLE_' in line:
                raise ValueError('factual_context_review_invalid')
            continue
        first, last, columns = int(match[2]), int(match[3] or match[2]), int(match[4])
        if not 1 <= first <= last <= 100000 or not 1 <= columns <= 256:
            raise ValueError('factual_context_review_invalid')
        cells = table_units._cells(match[5], 0, first, last, columns)
        if not cells:
            raise ValueError('factual_context_review_invalid')
        records.append(cells)
    if not records:
        raise ValueError('factual_context_review_invalid')
    rendered = _render_table_rows(records, None, None, language)
    return caption + '\n' + rendered.split('\n', 1)[1]


def _compact_candidate_tables(view, frame, language):
    result = deepcopy(view)
    aliases = {cid: alias for alias, cid in _citation_aliases(frame).items()}
    bodies, owners = dict(frame.source_bodies), dict(frame.table_owners)
    rebuilt = []
    for index, item in enumerate(result['inventory']):
        text = item['text']
        if index in owners:
            cid = owners[index]
            caption = text.split('\n', 1)[0]
            try:
                restored = _candidate_table_text(bodies[cid], caption, language)
            except (ValueError, TypeError, KeyError):
                restored = None
            if restored == text:
                item.update(source_table=aliases[cid], caption=caption)
                del item['text']
        # Keep full text on any mismatch, including candidate-only prose.
        rebuilt.append(restored if 'source_table' in item else text)
    if tuple(rebuilt) != frame.items or _expand(result['template'], rebuilt) != frame.answer:
        raise ValueError('factual_context_review_invalid')
    return result


def _wire_schema(frame: ReviewFrame, refs=None):
    target = {'type': 'string'}
    if frame.editable_targets:
        target['enum'] = list(frame.editable_targets)
    context_target = {'type': 'string'}
    if frame.table_owners:
        context_target['enum'] = ['item:' + str(index) for index, _ in frame.table_owners]
    result = {'name': 'factual_context_review', 'strict': True, 'schema': {
        'type': 'object', 'additionalProperties': False,
        'properties': {
            'outcome': {'type': 'string', 'enum': ['pass', 'rewrite', 'no_sources']},
            'edits': {'type': 'array', 'maxItems': len(frame.editable_targets), 'items': {
                'type': 'object', 'additionalProperties': False,
                'properties': {'target': target, 'text': {'type': 'string', 'minLength': 1, 'maxLength': 4000}},
                'required': ['target', 'text']}},
            'citation_ids': {'type': 'array', 'maxItems': len(frame.admitted_citation_ids),
                             'items': {'type': 'string', 'enum': list(_citation_aliases(frame))}},
            'table_contexts': {'type': 'array', 'maxItems': len(frame.table_owners), 'items': {
                    'type': 'object', 'additionalProperties': False,
                    'properties': {'target': context_target,
                        'context': {'type': 'string', 'minLength': 1, 'maxLength': 480},
                        'documentary_limits_checked': {'type': 'boolean'},
                        'source_quotes': {'type': 'array', 'minItems': 1, 'maxItems': 2,
                            'items': {'type': 'object', 'additionalProperties': False,
                                'properties': {
                                    'citation_id': {'type': 'string', 'enum': list(_citation_aliases(frame))},
                                    'quote_ref': {'type': 'string', 'pattern': '^Q[1-9][0-9]{0,4}$'}},
                                'required': ['citation_id', 'quote_ref']}}},
                    'required': ['target', 'context', 'documentary_limits_checked', 'source_quotes']}},
            'checks': {'type': 'object', 'additionalProperties': False,
                       'properties': {key: {'type': 'boolean'} for key in CHECKS}, 'required': list(CHECKS)},
            'reason': {'type': 'string', 'maxLength': 500}},
        'required': ['outcome', 'edits', 'citation_ids', 'table_contexts', 'checks', 'reason']}}
    if refs is not None:
        properties = result['schema']['properties']
        properties['additional_citation_ids'] = properties.pop('citation_ids')
        result['schema']['required'] = ['additional_citation_ids' if key == 'citation_ids' else key
                                        for key in result['schema']['required']]
        properties['table_contexts']['items']['properties']['source_quotes']['items'] = {
            'type': 'string', 'enum': list(refs)}
    return result


def schema(frame: ReviewFrame):
    return _wire_schema(frame, _atomic_references(frame))


def _checked_table_contexts(value, frame, citations):
    owners = {'item:' + str(index): cid for index, cid in frame.table_owners}
    if type(value) is not list or len(value) != len(owners):
        raise ValueError('table_count')
    sources, result = dict(frame.source_bodies), {}
    for audit in value:
        if (type(audit) is not dict or set(audit) != {'target', 'context', 'documentary_limits_checked', 'source_quotes'}
                or type(audit.get('target')) is not str
                or type(audit.get('context')) is not str or not audit['context'].strip()
                or len(audit['context']) > 480 or type(audit.get('source_quotes')) is not list
                or not 1 <= len(audit['source_quotes']) <= 2):
            raise ValueError('table_shape')
        if audit['target'] not in owners or audit['target'] in result:
            raise ValueError('table_target')
        if audit.get('documentary_limits_checked') is not True:
            raise ValueError('documentary_limits_unchecked')
        context, quoted_ids, literal = audit['context'], set(), []
        for quote in audit['source_quotes']:
            if (type(quote) is not dict or set(quote) != {'citation_id', 'quote'}
                    or type(quote.get('citation_id')) is not str or type(quote.get('quote')) is not str
                    or not quote['quote'].strip() or len(quote['quote']) > 480):
                raise ValueError('quote_shape')
            if quote['citation_id'] not in citations or quote['citation_id'] not in sources:
                raise ValueError('quote_source')
            if quote['quote'] not in sources[quote['citation_id']]:
                raise ValueError('quote_not_literal')
            quoted_ids.add(quote['citation_id'])
            literal.append('\n'.join('> ' + _escape(line) for line in quote['quote'].splitlines()))
        if owners[audit['target']] not in quoted_ids:
            raise ValueError('table_owner')
        if any(marker in text for marker in (PREFIX, enumeration_review.INVENTORY_PREFIX,
                '[TABLE_RECORD ', '[PDF_TABLE_') for text in [context] + [q['quote'] for q in audit['source_quotes']]):
            raise ValueError('table_marker')
        result[audit['target']] = context + '\n' + '\n'.join(literal) + '\n'
    return result


def _rejected(first_reply, *, reason='factual_context_review_invalid', outcome='invalid', gate='reply_shape'):
    return {**deepcopy(first_reply), 'outcome': 'no_sources', 'answer': '', 'reason': reason,
            '_terminal_review_failure': True,
            'independent_context_review': {'outcome': outcome, 'edit_count': 0, 'gate': gate}}


def apply_wire_review(first_reply, review_reply, frame: ReviewFrame):
    """Decode frame-owned aliases/slices, then apply unchanged canonical gates."""
    if type(frame) is not ReviewFrame or type(review_reply) is not dict:
        return _rejected(first_reply)
    atomic = _atomic_references(frame)
    citation_key = 'additional_citation_ids' if atomic is not None else 'citation_ids'
    if atomic is not None and 'citation_ids' in review_reply:
        return _rejected(first_reply)
    citations, contexts = review_reply.get(citation_key), review_reply.get('table_contexts')
    if type(citations) is not list or type(contexts) is not list:
        return _rejected(first_reply)
    aliases = _citation_aliases(frame)
    if any(type(cid) is not str or cid not in aliases for cid in citations):
        return _rejected(first_reply, gate='citation_alias')
    decoded = deepcopy(review_reply)
    explicit_citations = [aliases[cid] for cid in citations]
    if atomic is not None:
        del decoded['additional_citation_ids']
    decoded['citation_ids'] = explicit_citations
    quoted_sources = []
    quote_units = {ref: (literal, owners) for ref, literal, owners in frame.quote_units}
    if frame.table_owners and decoded.get('outcome') in {'pass', 'rewrite'}:
        for edit in decoded.get('edits', []) if type(decoded.get('edits')) is list else ():
            if type(edit) is dict and type(edit.get('text')) is str and _QUOTE_MARKER.search(edit['text']):
                return _rejected(first_reply, gate='edit_target')
    for audit in decoded['table_contexts']:
        if type(audit) is not dict or type(audit.get('source_quotes')) is not list:
            return _rejected(first_reply, gate='table_shape')
        if (frame.table_owners and decoded.get('outcome') in {'pass', 'rewrite'}
                and type(audit.get('context')) is str and _QUOTE_MARKER.search(audit['context'])):
            return _rejected(first_reply, gate='table_marker')
        canonical_quotes = []
        for quote in audit['source_quotes']:
            if atomic is not None:
                if type(quote) is not str:
                    return _rejected(first_reply, gate='quote_shape')
                unit = atomic.get(quote)
                if unit is None:
                    return _rejected(first_reply, gate='quote_source')
                cid, literal = unit
                quoted_sources.append(cid)
                canonical_quotes.append({'citation_id': cid, 'quote': literal})
                continue
            if (type(quote) is not dict or set(quote) != {'citation_id', 'quote_ref'}
                    or type(quote.get('quote_ref')) is not str or not _QUOTE_REF.fullmatch(quote['quote_ref'])):
                return _rejected(first_reply, gate='quote_shape')
            alias = quote.get('citation_id')
            if type(alias) is not str or alias not in aliases:
                return _rejected(first_reply, gate='citation_alias')
            cid = aliases[alias]
            unit = quote_units.get(quote['quote_ref'])
            if unit is None or cid not in unit[1]:
                return _rejected(first_reply, gate='quote_source')
            quote.clear()
            quote.update(citation_id=cid, quote=unit[0])
            canonical_quotes.append(quote)
        audit['source_quotes'] = canonical_quotes
    if atomic is not None:
        # Every addition is explicitly selected in a bound reference or the
        # extra-support field. No unseen source or inferred citation is added.
        decoded['citation_ids'] = list(dict.fromkeys(
            (*frame.current_citation_ids, *quoted_sources, *explicit_citations)))
    return apply_review(first_reply, decoded, frame)


def apply_review(first_reply, review_reply, frame: ReviewFrame):
    """Apply a complete independent verdict without reclassifying missing facts."""
    def reject(reason='factual_context_review_invalid', outcome='invalid', gate='reply_shape'):
        return _rejected(first_reply, reason=reason, outcome=outcome, gate=gate)
    if type(frame) is not ReviewFrame or type(first_reply) is not dict or _json(first_reply) != frame.first_json:
        return reject(gate='frame_mismatch')
    if (type(review_reply) is not dict or set(review_reply) != {'outcome','edits','citation_ids','table_contexts','checks','reason'}
            or review_reply.get('outcome') not in {'pass','rewrite','no_sources'}
            or type(review_reply.get('reason')) is not str or len(review_reply['reason']) > 500):
        return reject()
    verdict = review_reply['outcome']
    checks, edits, citations = review_reply.get('checks'), review_reply.get('edits'), review_reply.get('citation_ids')
    if (type(checks) is not dict or set(checks) != set(CHECKS)
            or any(type(value) is not bool for value in checks.values())
            or type(edits) is not list or len(edits) > len(frame.editable_targets)
            or type(citations) is not list or len(citations) > len(frame.admitted_citation_ids)
            or any(type(cid) is not str or cid not in frame.admitted_citation_ids for cid in citations)
            or len(set(citations)) != len(citations)):
        return reject()
    if verdict == 'no_sources':
        return reject('factual_context_review_rejected', 'no_sources', 'model_rejected')
    if not all(checks.values()):
        return reject('factual_context_review_incomplete', verdict, 'checks_false')
    if not set(frame.current_citation_ids).issubset(citations):
        return reject(gate='citation_set')
    try:
        contexts = _checked_table_contexts(review_reply['table_contexts'], frame, citations)
    except ValueError as exc:
        # _checked_table_contexts raises only fixed server-owned codes.
        if str(exc) == 'documentary_limits_unchecked':
            return reject('factual_context_review_incomplete', verdict, str(exc))
        return reject(gate=str(exc))
    if verdict == 'pass':
        if edits or contexts or set(citations) != set(frame.current_citation_ids):
            return reject(gate='pass_changed')
        result = deepcopy(first_reply)
    else:
        if (not edits and not contexts) or not frame.editable_targets:
            return reject(gate='edit_target')
        items, template, seen = list(frame.items), frame.template, set()
        for edit in edits:
            if (type(edit) is not dict or set(edit) != {'target','text'}
                    or type(edit.get('target')) is not str or edit['target'] not in frame.editable_targets
                    or edit['target'] in seen or type(edit.get('text')) is not str
                    or not edit['text'].strip() or len(edit['text']) > 4000 or PREFIX in edit['text']):
                return reject(gate='edit_target')
            target, text = edit['target'], edit['text']
            seen.add(target)
            if target == 'template-answer':
                if not _has_template(text):
                    return reject(gate='template')
                template = text
            else:
                if enumeration_review.INVENTORY_PREFIX in text:
                    return reject(gate='template')
                items[int(target.split(':')[1])] = text
        rendered_items = [contexts.get('item:' + str(index), '') + item
                          for index, item in enumerate(items)]
        answer = _expand(template, rendered_items)
        if len(answer) > MAX_TEXT or sum(map(len, items)) > MAX_TEXT:
            return reject(gate='assembled_size')
        result = deepcopy(first_reply)
        result.update(answer=answer, expected_list_items=items, covered_list_items=[], citation_ids=citations)
    # The caller's draft was initially empty: an independent unchanged pass
    # does not turn the first-stage generation rewrite into a pass on emptiness.
    result['reason'] = 'factual_context_review_completed'
    result['independent_context_review'] = {'outcome': verdict, 'edit_count': len(edits), 'gate': 'accepted',
                                          'answer_sha256': _digest(result['answer'])}
    return result
