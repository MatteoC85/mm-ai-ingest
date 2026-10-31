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
from .source_table_review import PREFIX, TableCatalog, _HEADER, _RECORD_ID, _escape, selection_template, _render as _render_table_rows

MARKER = enumeration_review.INVENTORY_MARKER
MAX_TEXT = 30000
_QUOTE_MARKER = re.compile(r'\[\s*/?\s*Q\s*[0-9]*\s*\]', re.IGNORECASE)
_QUOTE_REF = re.compile(r'Q[1-9][0-9]{0,4}')
_SOURCE_MARKER = re.compile(
    r'(\[PDF_TABLE_[^\]\r\n]*\]|\[TABLE_RECORD T[1-9][0-9]*\]|'
    r'\[\[SOURCE_TABLE:[^\]\r\n]*\]\]|\[\[FACTUAL_INVENTORY\]\])')
CHECKS = ('source_fidelity', 'scope_applicability', 'completeness',
          'safety', 'language', 'provenance')
DECISION_REASONS = ('accepted', 'unsupported_fact', 'missing_available_information',
    'unavailable_requested_details', 'missing_prerequisite', 'incorrect_scope',
    'incorrect_language', 'citation_problem', 'inconsistent_sources',
    'uneditable_candidate', 'other')
INSTRUCTIONS = (
    'Independently judge the complete candidate against QUESTION and all SOURCES, '
    'without a prior verdict. All six checks apply to the final assembled answer: '
    'source fidelity, scope/applicability, useful coverage of every requested category, '
    'safety, RESPONSE_LANGUAGE and provenance. '
    'FIRST review every editable item and template sentence, including unchanged prose. '
    'Compare them with the visible inventory and all applicable source passages/tables. '
    'Reconcile different precision explicitly: vague wording in one passage cannot establish '
    'that another applicable source lacks a value, frequency or detail. Categorical values '
    'also specify facts without numerals. Correct every repeated false absence or limitation '
    'in both item and template, attributing differing statements to their actual scope. '
    'Do not transfer values between assemblies or silently resolve conflicting applicability. '
    'An incomplete excerpt cannot establish absence from the whole document. '
    'Preserve useful supported facts and truthful source-scoped limits; do not demand every '
    'manual detail or infer unseen chapters. An incomplete selection must not claim '
    'exhaustiveness; explicit requests for it require coverage or a truthful limitation. '
    'Retain each published fact/value qualification. Check actual intervention guidance '
    'against its source and adjacent conditions: keep required protection, isolation, '
    'permitted mode, access and restoration in the same item. Generic advice or attribution '
    'to a manual cannot replace these conditions. Remove unrequested execution commands '
    'from informational prose, not requested facts to avoid qualifying them. '
    'EDIT only listed non-immutable targets or template-answer; never add, delete or reorder '
    'inventory slots. Preserve [[FACTUAL_INVENTORY]] exactly once on its own line in a '
    'template edit. Immutable quotations remain verbatim documentary context, not new '
    'requested operations. Do not invent operations, sources, values or assembly equivalence. '
    'read_only_answer permits no edits or rewrite. '
    'THEN audit every TABLE_CONTEXT_TARGET exactly once. Give one or two short context '
    'sentences and one or two supporting quotation references, including the owner. '
    'Preserve all applicable adjacent/outside-cell qualifications and documentary limits. '
    'Distinct prose around matching cells needs separate audit: equal cells do not prove '
    'equivalent assemblies or duplicate conditions. Owner documentary prose requires its '
    'quotation, not just a heading, cell or generated SECTION label. Where owner_prose_refs '
    'is listed, choose at least one; null means no owner prose, so use another owner quote. '
    'A reference proves identity, not entailment or complete qualifications. Set '
    'documentary_limits_checked true only after checking these obligations. Contexts and '
    'literal quotations are inserted before their unchanged item; do not copy whole sources '
    'or unrelated operations. A documentary label never waives qualifications. '
    'GROUNDING_CHECKS are bounded diagnostics, not evidence or permission to delete facts. '
    'Resolve flagged values/identifiers from SOURCES in the same target; preserve supported '
    'facts, conditions and exact technical spelling, including unusual identifiers. Missing '
    'diagnostics waive no check; final grounding and answer binding still apply. '
    'The server preserves current citations and adds valid quote owners. Use '
    'additional_citation_ids only for extra admitted support. Return strict schema JSON. '
    'pass requires the unchanged complete safe answer, no edits/contexts/new support. '
    'rewrite requires the entire assembled revision to meet all six checks; with table '
    'contexts use rewrite even without edits. With no table targets return table_contexts=[]. '
    'If any material claim/context remains unsupported or any check fails, return no_sources. '
    'reason is only a diagnostic enum code, never source text or reasoning; it cannot replace checks.'
) + enumeration_review.INFORMATIONAL_SCOPE_INSTRUCTIONS

_ATOMIC_INSTRUCTIONS = INSTRUCTIONS
_ATOMIC_REPRESENTATION = (
    '\nCOMPLETE_CANDIDATE expands its template marker into ordered inventory texts, each prefixed "- ". '
    'All texts, including immutable tables, are verbatim; source geometry/spans and adjacent qualifications still apply. '
    'Sx names an admitted source; additional_citation_ids uses Sx. Quote token "a:b" selects source Sa and its '
    'literal Qb unit marked [Qb]...[/Q] in SOURCES. Use only schema references; never invent pairs or copy quote text. '
    'Existing @n dictionary references expand normally. Quote references belong only in quotation fields, '
    'never as markers in public context or edits.')


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
    context_owners: tuple[tuple[str, int, str, tuple[str, ...]], ...] = ()
    grounding_findings: tuple[tuple[str, int, tuple[tuple[str, str, str], ...]], ...] = ()
    selection_scope_note: str = ''


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


def _table_context_signature(body):
    """Exact cell serialization and its separate prose, never assembly identity."""
    tables, prose = [], []
    for line in body.splitlines():
        match = table_units._LINE.fullmatch(line)
        if match is None:
            if 'PDF_TABLE_' in line:
                return None
            prose.append(line)
            continue
        first, last, columns = int(match[2]), int(match[3] or match[2]), int(match[4])
        if not 1 <= first <= last <= 100000 or not 1 <= columns <= 256:
            return None
        try:
            cells = table_units._cells(match[5], 0, first, last, columns)
        except (ValueError, TypeError):
            return None
        if not cells:
            return None
        tables.append(line)
    return (tuple(tables), '\n'.join(prose)) if tables else None


def _context_bindings(owners, catalog, sources):
    occurrences = []
    for entry in catalog.entries:
        body = sources.get(entry.citation_id)
        if body is None or _digest(body) != entry.source_sha256:
            raise ValueError('factual_context_review_invalid')
        signature = _table_context_signature(body)
        if signature is not None:
            occurrences.append((entry.citation_id, signature))
    result = []
    for index, owner in owners:
        primary = _table_context_signature(sources[owner])
        groups = {}
        if primary is not None:
            for cid, (cells, prose) in occurrences:
                if cells == primary[0]:
                    groups.setdefault((prose, sources[cid]), []).append(cid)
        same = tuple(dict.fromkeys(groups.pop((primary[1], sources[owner]), [owner]))) if primary is not None else (owner,)
        result.append(('item:' + str(index), index, owner, same))
        distinct = [cids for (prose, _), cids in groups.items() if prose.strip()]
        for ordinal, cids in enumerate(distinct, 1):
            ids = tuple(dict.fromkeys(cids))
            result.append(('item:' + str(index) + ':context:' + str(ordinal), index, ids[0], ids))
    return tuple(result)


def _context_targets(frame):
    return frame.context_owners or tuple(('item:' + str(i), i, cid, (cid,)) for i, cid in frame.table_owners)


def _documentary_prose(body):
    """Exclude only the initial chunker-generated SECTION label from proof prose.

    Keep the original body, context identity and later document headings intact.
    This is structural provenance separation, not semantic relevance filtering.
    """
    signature = _table_context_signature(body)
    if signature is None:
        return ''
    prose = signature[1]
    lines = body.splitlines()
    if lines and lines[0].startswith('SECTION: '):
        prose = prose.partition('\n')[2]
    return prose


def _owner_prose_refs(frame, owner, atomic):
    prose = _documentary_prose(dict(frame.source_bodies)[owner])
    if not prose.strip():
        return None
    if atomic is not None:
        return [ref for ref, (cid, literal) in atomic.items() if cid == owner and literal in prose]
    return [ref for ref, literal, owners in frame.quote_units if owner in owners and literal in prose]


def _grounding_findings(items, template, targets, source_text, redact_text):
    if not source_text:
        return ()
    if type(source_text) is not str or len(source_text) > 262144 or not callable(redact_text):
        raise ValueError('factual_context_review_invalid')
    # Use the same scoped ASK matcher as final validation, not a second parser.
    from .unit_grounding import _assistant_core_claims, _assistant_core_claim_supported
    source_claims = _assistant_core_claims(source_text)
    result, remaining_details = [], 12
    for target in targets:
        text = template if target == 'template-answer' else items[int(target.split(':')[1])]
        claims = _assistant_core_claims(redact_text(text))
        unsupported = [claim for claim in claims
                       if not _assistant_core_claim_supported(claim, source_text, source_claims)]
        if not unsupported:
            continue
        details = []
        for claim in unsupported:
            raw = claim.get('raw')
            if (len(details) < 3 and remaining_details and type(raw) is str
                    and 0 < len(raw) <= 96 and claim.get('kind') in {'number', 'code'}):
                details.append((claim['kind'], raw, str(claim.get('unit') or '')))
                remaining_details -= 1
        result.append((target, len(unsupported), tuple(details)))
    return tuple(result)


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
                    admitted_citation_ids, sources_block='', grounding_source_text='', redact_text=None,
                    selected_records_only: bool = False) -> ReviewFrame:
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
    scope_note = ''
    if selected_records_only is True:
        template = selection_template(wire_reply)
        scope_note = wire_reply['selection_scope_note']
        if expanded_reply.get('selection_scope_note') != scope_note:
            raise invalid
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
    bindings = _context_bindings(owners, catalog, source_bodies) if immutable else ()
    findings = _grounding_findings(items, template, targets, grounding_source_text, redact_text)
    return ReviewFrame(_json(expanded_reply), answer, template, tuple(items), tuple(immutable),
                       allowed, tuple(dict.fromkeys(citations)), targets,
                       tuple(owners), tuple(source_bodies.items()),
                       _digest(sources_block) if sources_block else '', quote_view, quote_units, bindings, findings, scope_note)


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
    bound = _bound_context_references(frame, atomic)
    if _has_template(frame.template):
        view = {'template': frame.template, 'inventory': [
            {'target': 'item:' + str(i), 'immutable': i in frame.immutable_indices, 'text': text}
            for i, text in enumerate(frame.items)]}
        if _expand(view['template'], [item['text'] for item in view['inventory']]) != frame.answer:
            raise ValueError('factual_context_review_invalid')
    else:
        # No-marker prose stays a complete, read-only candidate.
        view = {'read_only_answer': frame.answer}
    source_view = _aliased_source_headers(frame.quote_view, aliases) if atomic is not None else frame.quote_view
    user = ('QUESTION:\n' + question + '\n\nRESPONSE_LANGUAGE: ' + response_language
        + '\n\nSOURCES:\n' + source_view + '\n\nCOMPLETE_CANDIDATE:\n' + _json(view)
        + ('\n\nCITATION_ALIASES:\n' + _json(aliases) if atomic is None else '')
        + '\n\nTABLE_CONTEXT_TARGETS:\n' + json.dumps([
            {'target': target, 'inventory_item': index, 'citation_id': reverse[cid],
             'same_context_sources': [reverse[source] for source in same],
             **({'owner_prose_refs': _owner_prose_refs(frame, cid, atomic)} if bound is None else {})}
            for target, index, cid, same in _context_targets(frame)], ensure_ascii=False, separators=(',', ':'))
        + '\n\nCURRENT_CITATION_IDS:\n' + _json([reverse[cid] for cid in frame.current_citation_ids]))
    if frame.grounding_findings:
        user += '\n\nGROUNDING_CHECKS:\n' + _json([
            {'target': target, 'unsupported_count': count,
             'claims': [{'kind': kind, 'raw': raw, 'unit': unit} for kind, raw, unit in details],
             'omitted_details': count - len(details)}
            for target, count, details in frame.grounding_findings])
    representation = (
        '\nCOMPLETE_CANDIDATE is lossless: replace the template\'s single full-line '
        '[[FACTUAL_INVENTORY]] by the ordered inventory texts, each prefixed by "- ". '
        'Judge that entire assembled answer against QUESTION. Edit only non-immutable item targets '
        'or template-answer, retaining its inventory marker. read_only_answer is immutable prose. '
        'For additional_citation_ids and source_quotes.citation_id use only CITATION_ALIASES keys; '
        'they denote unchanged original source IDs. All quotation, applicability, limitation '
        'and safety obligations remain unchanged. '
        'SOURCES [Qn]...[/Q] marks an exact quote_ref unit; existing @n dictionary references '
        'expand with these markers intact. Select Qn present in the cited source, never copy '
        'quote text. The server inserts the literal slice; unmarked text remains source context. '
        'Use Q references only in quote_ref, never as markers in context or edits.')
    system = _ATOMIC_INSTRUCTIONS + _ATOMIC_REPRESENTATION if atomic is not None else INSTRUCTIONS + representation
    if bound is not None:
        system += (
            ' BOUND_CONTEXTS: fill every required table_contexts slot; null is allowed only for no_sources. '
            'owning_source_quote must use its schema enum (owner prose when present, otherwise owner text). '
            'supplementary_source_quotes permits one extra reference. These replace source_quotes; '
            'identity does not prove semantic support or waive any qualification/check.')
    if frame.selection_scope_note:
        system += (
            ' USEFUL_SELECTION: the server computed only exact selected-table mappings. '
            'Unselected records remain unassessed; no exclusion or semantic coverage is inferred. '
            'Check every requested category against the complete supplied sources, all selected '
            'facts and every material qualification. The candidate begins with its immutable '
            'selection_scope_note. Check that it is in RESPONSE_LANGUAGE and truthfully describes '
            'a supported non-exhaustive selection, without calling unselected information '
            'irrelevant or verified. Reject any claim of exhaustiveness or an omitted requested '
            'category; a scope note does not waive useful category coverage. Template edits must '
            'preserve that exact opening note. If the note itself is misleading, reject the candidate.')
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]


def _citation_aliases(frame):
    return {'S' + str(index): cid for index, cid in enumerate(frame.admitted_citation_ids)}


def _atomic_references(frame):
    """Enumerate complete source/unit pairs only within structured-output limits.

    Oversized or absent catalogs use the complete preceding wire, never a
    selected subset of sources, units, or semantic obligations.
    """
    count = sum(len(owners) for _, _, owners in frame.quote_units)
    other_enums = (3 + len(frame.editable_targets) + len(_context_targets(frame))
                   + len(frame.admitted_citation_ids) + len(DECISION_REASONS))
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


def _wire_schema(frame: ReviewFrame, refs=None, bound=None):
    target = {'type': 'string'}
    if frame.editable_targets:
        target['enum'] = list(frame.editable_targets)
    context_target = {'type': 'string'}
    if frame.table_owners:
        context_target['enum'] = [target for target, _, _, _ in _context_targets(frame)]
    result = {'name': 'factual_context_review', 'strict': True, 'schema': {
        'type': 'object', 'additionalProperties': False,
        'properties': {
            'outcome': {'type': 'string', 'enum': ['pass', 'rewrite', 'no_sources']},
            'edits': {'type': 'array', 'maxItems': len(frame.editable_targets), 'items': {
                'type': 'object', 'additionalProperties': False,
                'properties': {'target': target, 'text': {'type': 'string', 'minLength': 1, 'maxLength': 4000}},
                'required': ['target', 'text']}},
            'additional_citation_ids': {'type': 'array', 'maxItems': len(frame.admitted_citation_ids),
                             'items': {'type': 'string', 'enum': list(_citation_aliases(frame))}},
            'table_contexts': {'type': 'array', 'maxItems': len(_context_targets(frame)), 'items': {
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
            'reason': {'type': 'string', 'maxLength': 500, 'enum': list(DECISION_REASONS)}},
        'required': ['outcome', 'edits', 'additional_citation_ids', 'table_contexts', 'checks', 'reason']}}
    if refs is not None:
        properties = result['schema']['properties']
        properties['table_contexts']['items']['properties']['source_quotes']['items'] = {
            'type': 'string', 'enum': list(refs)}
        if bound is not None:
            # Reuse the complete supplementary universe once. Repeating this
            # enum per target can exceed the provider limit on ordinary packets.
            result['schema']['$defs'] = {'source_quote_reference': {'type': 'string', 'enum': list(refs)}}
            slots = {}
            for target, choices in bound.items():
                slots[target] = {'anyOf': [{'type': 'object', 'additionalProperties': False,
                    'properties': {
                        'context': {'type': 'string', 'minLength': 1, 'maxLength': 480},
                        'documentary_limits_checked': {'type': 'boolean'},
                        'owning_source_quote': {'type': 'string', 'enum': choices},
                        'supplementary_source_quotes': {'type': 'array', 'maxItems': 1,
                            'items': {'$ref': '#/$defs/source_quote_reference'}}},
                    'required': ['context', 'documentary_limits_checked',
                                 'owning_source_quote', 'supplementary_source_quotes']}, {'type': 'null'}]}
            properties['table_contexts'] = {'type': 'object', 'additionalProperties': False,
                'properties': slots, 'required': list(slots)}
    return result


def _bound_context_references(frame, refs):
    """Bind every context to its owner without selecting or approving a quote.

    Fall back to the complete prior wire if any required owner has no available
    literal unit or the new schema exceeds conservative structured-output limits.
    """
    if refs is None:
        return None
    bound = {}
    for target, _, owner, _ in _context_targets(frame):
        choices = _owner_prose_refs(frame, owner, refs)
        if choices is None:
            choices = [ref for ref, (cid, _) in refs.items() if cid == owner]
        if not choices or target in bound:
            return None
        bound[target] = choices
    enum_count, string_chars, properties, valid = 0, 0, 0, True
    def inspect(value, depth=0):
        nonlocal enum_count, string_chars, properties, valid
        if type(value) is str:
            string_chars += len(value)
        elif type(value) is dict:
            choices = value.get('enum')
            if type(choices) is list:
                enum_count += len(choices)
                if len(choices) > 250 and sum(map(len, choices)) > 15000:
                    valid = False
            if value.get('type') == 'object':
                properties += len(value.get('properties', {}))
                if depth > 8:
                    valid = False
            for key, item in value.items():
                string_chars += len(key)
                inspect(item, depth + (key in {'properties', 'items'}))
        elif type(value) is list:
            for item in value:
                inspect(item, depth)
    inspect(_wire_schema(frame, refs, bound))
    return bound if valid and enum_count <= 1000 and string_chars <= 120000 and properties <= 100 else None


def schema(frame: ReviewFrame):
    refs = _atomic_references(frame)
    return _wire_schema(frame, refs, _bound_context_references(frame, refs))


def _checked_table_contexts(value, frame, citations):
    bindings = {target: (index, cid) for target, index, cid, _ in _context_targets(frame)}
    owners = {target: cid for target, (_, cid) in bindings.items()}
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
        context, quoted_ids, literal, prose_ids = audit['context'], set(), [], set()
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
            prose = _documentary_prose(sources[quote['citation_id']])
            if quote['quote'] in prose:
                prose_ids.add(quote['citation_id'])
            literal.append('\n'.join('> ' + _escape(line) for line in quote['quote'].splitlines()))
        if owners[audit['target']] not in quoted_ids:
            raise ValueError('table_owner')
        if (_documentary_prose(sources[owners[audit['target']]]).strip()
                and owners[audit['target']] not in prose_ids):
            raise ValueError('table_owner')
        if any(marker in text for marker in (PREFIX, enumeration_review.INVENTORY_PREFIX,
                '[TABLE_RECORD ', '[PDF_TABLE_') for text in [context] + [q['quote'] for q in audit['source_quotes']]):
            raise ValueError('table_marker')
        result[audit['target']] = context + '\n' + '\n'.join(literal) + '\n'
    assembled = {}
    for target, (index, _) in bindings.items():
        item = 'item:' + str(index)
        assembled[item] = assembled.get(item, '') + result[target]
    return assembled


def _rejected(first_reply, *, reason='factual_context_review_invalid', outcome='invalid', gate='reply_shape'):
    return {**deepcopy(first_reply), 'outcome': 'no_sources', 'answer': '', 'reason': reason,
            '_terminal_review_failure': True,
            'independent_context_review': {'outcome': outcome, 'edit_count': 0, 'gate': gate}}


def apply_wire_review(first_reply, review_reply, frame: ReviewFrame):
    """Decode frame-owned aliases/slices, then apply unchanged canonical gates."""
    if type(frame) is not ReviewFrame or type(review_reply) is not dict:
        return _rejected(first_reply)
    atomic = _atomic_references(frame)
    bound = _bound_context_references(frame, atomic)
    if 'citation_ids' in review_reply:
        return _rejected(first_reply)
    citations, contexts = review_reply.get('additional_citation_ids'), review_reply.get('table_contexts')
    if type(citations) is not list:
        return _rejected(first_reply)
    if bound is not None:
        if type(contexts) is not dict or len(contexts) != len(bound):
            return _rejected(first_reply, gate='table_count')
        if set(contexts) != set(bound):
            return _rejected(first_reply, gate='table_target')
        canonical_contexts = []
        for target, choices in bound.items():
            audit = contexts[target]
            if audit is None and review_reply.get('outcome') == 'no_sources':
                continue
            if (type(audit) is not dict or set(audit) != {'context', 'documentary_limits_checked',
                    'owning_source_quote', 'supplementary_source_quotes'}
                    or type(audit.get('supplementary_source_quotes')) is not list
                    or len(audit['supplementary_source_quotes']) > 1):
                return _rejected(first_reply, gate='table_shape')
            selected = audit['owning_source_quote']
            if type(selected) is not str:
                return _rejected(first_reply, gate='quote_shape')
            if selected not in atomic:
                return _rejected(first_reply, gate='quote_source')
            if selected not in choices:
                return _rejected(first_reply, gate='table_owner')
            canonical_contexts.append({'target': target, 'context': audit['context'],
                'documentary_limits_checked': audit['documentary_limits_checked'],
                'source_quotes': [audit['owning_source_quote'], *audit['supplementary_source_quotes']]})
        contexts = canonical_contexts
    elif type(contexts) is not list:
        return _rejected(first_reply)
    aliases = _citation_aliases(frame)
    if any(type(cid) is not str or cid not in aliases for cid in citations):
        return _rejected(first_reply, gate='citation_alias')
    decoded = deepcopy(review_reply)
    decoded['table_contexts'] = deepcopy(contexts)
    explicit_citations = [aliases[cid] for cid in citations]
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
            quoted_sources.append(cid)
            canonical_quotes.append(quote)
        audit['source_quotes'] = canonical_quotes
    # Citation bookkeeping is identical in every wire representation. Every
    # addition was explicitly selected in a validated quotation or extra-support
    # field; preserving the existing frame adds no source authority or approval.
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
                if frame.selection_scope_note and not text.startswith(frame.selection_scope_note + '\n\n'):
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
