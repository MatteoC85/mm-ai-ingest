"""Literal table inventory codec over complete, already emitted source views.

This module grants no authority and makes no relevance/entailment judgement.
The caller owns source admission and must run its existing final validation on
the expanded answer. Original quotations are not asserted to be translations.
"""
from dataclasses import dataclass
from copy import deepcopy
import hashlib
import re

from . import table_units

PREFIX = '[[SOURCE_TABLE:'
_TOKEN = re.compile(r'\[\[SOURCE_TABLE:G([1-9][0-9]*)\]\]')
_HEADER = re.compile(r'^\[([^\]\r\n]+)\] source_type=[^\r\n]+; p=[^\r\n]+$', re.MULTILINE)
_RECORD_ID = re.compile(r'\[TABLE_RECORD (T[1-9][0-9]*)\]')
_COVERAGE = re.compile(r'(T[1-9][0-9]*)=([0-9]+(?:,[0-9]+)*)')
_INVENTORY = '[[FACTUAL_INVENTORY]]'
MAX_TEXT = 30000

INSTRUCTIONS = (
    ' SOURCE_TABLE_QUOTATIONS: whenever any TABLE_RECORD in a listed group supports '
    'the answer, you MUST select its [[SOURCE_TABLE:Gn]] as one whole expected_list_items '
    'string. It expands to the exact preview shown below, including '
    'ALL records and notes of that emitted source occurrence. Do not paraphrase its '
    'table cells into other inventory items or prose: a material column does not create '
    'an action. Select an occurrence when it supports the complete QUESTION, considering '
    'the union of requested categories. In a narrow question, the remaining quoted rows '
    'are source context, not additional requested instructions. Selecting the token explicitly '
    'cites its owner and binds every record in that occurrence to this inventory index. '
    'You may omit these selected records from table_record_coverage: the server supplies '
    'their exact bindings. Any supplied mapping must agree; exclusions or other indices '
    'are contradictions. Normal inventory items '
    'remain necessary for supported non-tabular facts and applicable conditions, in '
    'RESPONSE_LANGUAGE. Preserve necessary safety prerequisites from non-tabular sources. '
    'Source quotations retain their original language and do not constitute a translation; '
    'provide the needed explanation in RESPONSE_LANGUAGE. Use references only for rewrite '
    'with the existing FACTUAL_INVENTORY answer template. Only expected_list_items selects '
    'references; mentions in reason or record-exclusion explanations are inert metadata. '
    'No reference may occur in other fields. All existing semantic, safety, source and completeness checks still apply.'
    ' Avoid repeated quotations from overlapping occurrences: when literal equivalence '
    'is established, select one complete occurrence, cite its owner and mark the other '
    'occurrence records as duplicates. Never merge different assemblies or source identities.'
)

RECORD_INSTRUCTIONS = (
    ' TABLE_RECORD markers identify unchanged literal PDF record occurrences within '
    'their original source context, not new evidence or assembly equivalence. '
    'Account for every unselected ID exactly once in table_record_coverage using compact strings. '
    'Only records bound by a selected whole SOURCE_TABLE token may be omitted. '
    'T1=0,2 means zero-based expected_list_items indices covering all its requested '
    'actions, intervals and conditions; T2!reason excludes it with a brief reason for the complete QUESTION, '
    'not only its first category. IDs and coverage declarations do not prove entailment: '
    'check the full cells, detail rows and notes, including continuation records. '
    'Keep these IDs out of the visible answer.'
)


def quotation_instructions(*, keyed_coverage: bool = False) -> str:
    if keyed_coverage is not True:
        return INSTRUCTIONS
    return INSTRUCTIONS.replace(
        'You may omit these selected records from table_record_coverage: the server supplies their exact bindings.',
        'Explicitly map every selected record in table_record_coverage to its inventory index using "=i"; '
        'do not omit it or use null. The server verifies these exact bindings.')


def record_instructions(*, derive_duplicate_records: bool = False, keyed_coverage: bool = False) -> str:
    text = RECORD_INSTRUCTIONS
    if derive_duplicate_records is True:
        text = text.replace(
        'Only records bound by a selected whole SOURCE_TABLE token may be omitted.',
        'Records bound by selected whole SOURCE_TABLE tokens, and complete byte-identical '
        'table occurrences, may be omitted. The server accounts for missing duplicate '
        'records; every distinct owner context still requires independent review. '
        'Identical cells never establish assembly equivalence. Explicit classifications '
        'are never replaced; different or partial tables must still be classified.')
    if keyed_coverage is True:
        return (
            ' TABLE_RECORD markers identify unchanged literal PDF record occurrences within '
            'their original source context, not new evidence or assembly equivalence. '
            'table_record_coverage is an object with every emitted Tn key exactly once and a '
            'non-null suffix string for each key. Map every record in each selected whole '
            'SOURCE_TABLE to that token\'s zero-based expected_list_items index, e.g. T1:"=0". '
            'For every unselected record use T2:"!brief truthful reason", including exact duplicate '
            'occurrences or facts not included in an explicitly limited useful answer. Do not '
            'falsely call relevant facts unrelated. Null, missing keys and conflicting classifications '
            'are invalid. Never map another occurrence to a selected token from a different source. '
            'All distinct owner contexts still require independent review; identical cells do not '
            'establish assembly equivalence. IDs and declarations do not prove entailment: check '
            'full cells, detail rows, notes and continuation records. Keep IDs out of the visible answer.')
    return text


def reference_schema(schema: dict, catalog: 'TableCatalog', *, derive_duplicate_records: bool = False,
                     keyed_coverage: bool = False) -> dict:
    """Only an emitted catalog permits omission of mechanically bound records."""
    if not catalog.entries:
        return schema
    out = deepcopy(schema)
    coverage = out['schema']['properties']['table_record_coverage']
    coverage['minItems'] = 0
    coverage['description'] = (
        'Classify every unselected TABLE_RECORD once using T1=0,2 or T2!outside-request reason. '
        'Records in selected whole SOURCE_TABLE tokens may be omitted; their owner and exact '
        'inventory index are server-bound. Explicit conflicting mappings/exclusions are invalid.')
    if derive_duplicate_records is True:
        coverage['description'] += (
            ' Missing records in complete byte-identical occurrences of selected tables '
            'are also server-accounted as duplicates, with every distinct source context '
            'independently reviewed. No partial-table or assembly equivalence is inferred.')
    if keyed_coverage is True:
        identities = tuple(rid for entry in catalog.entries for rid in entry.record_ids)
        out['schema']['properties']['table_record_coverage'] = {
            'type': 'object', 'additionalProperties': False,
            'properties': {rid: {'type': 'string', 'minLength': 2, 'maxLength': 512 - len(rid),
                'pattern': r'^(?:=[0-9]+(?:,[0-9]+)*|![^\r\n]+)$'} for rid in identities},
            'required': list(identities),
            'description': 'Every emitted record ID is a required key. Use "=0,2" for zero-based inventory indices, '
                '"!reason" for an explicit truthful exclusion. Every selected record must map to its token index; '
                'every unselected record needs a reason, including duplicates and explicitly limited coverage. '
                'Null and omissions are invalid. Different/partial tables require their own classification. '
                'Explicit conflicts remain invalid; all distinct owner contexts require independent review.'}
        out['reject_duplicate_keys'] = True
    return out


@dataclass(frozen=True)
class TableEntry:
    token: str
    text: str
    citation_id: str
    record_ids: tuple[str, ...]
    source_sha256: str
    table_lines: tuple[str, ...] = ()


@dataclass(frozen=True)
class TableCatalog:
    entries: tuple[TableEntry, ...] = ()
    preview: str = ''


def _escape(value: str) -> str:
    # Existing UI renderer still performs HTML escaping; do not emit raw HTML.
    return re.sub(r'([\\`*_{}\[\]()<>#!|])', r'\\\1', value)


def _render(records, page_from, page_to, language):
    if language == 'it':
        heading = 'Citazione originale della fonte'
        note = ('Dati documentali; eventuali altre voci sono contesto, non istruzioni aggiuntive. '
                'Il materiale indicato non prescrive un intervento.')
        page_label = 'pagina'
    else:
        heading = 'Original source quotation'
        note = ('Other rows are source context, not additional instructions. A material entry '
                'does not prescribe an intervention. The quotation is not a translation.')
        page_label = 'page'
    if type(page_from) is int and type(page_to) is int and 0 < page_from <= page_to:
        heading += ' (' + page_label + ' ' + str(page_from)
        heading += ('–' + str(page_to)) if page_to != page_from else ''
        heading += ')'
    lines = [heading + '. ' + note]
    for cells in records:
        values = []
        for cell in cells:
            value = cell['value']
            if not value:
                continue
            label = cell['label']
            values.append((_escape(label) + ': ' if label else '') + '«' + _escape(value) + '»')
        lines.append('  - ' + '; '.join(values))
    return '\n'.join(lines)


def compile_catalog(sources_block: str, source_views: dict, response_language: str) -> TableCatalog:
    """Build only complete occurrences present in the actual annotated prompt.

    source_views[cid] contains body (the exact formatter body) and optional page
    metadata from the admitted candidate. A truncated or ambiguous body disables
    this optional codec; it never acquires unseen text from source_views.
    """
    empty = TableCatalog()
    if (type(sources_block) is not str or type(source_views) is not dict
            or len(sources_block) > MAX_TEXT + 2048 or PREFIX in sources_block
            or response_language not in {'it', 'en'}):
        return empty
    headers = list(_HEADER.finditer(sources_block))
    if not headers or sources_block[:headers[0].start()].strip():
        return empty
    entries = []
    seen_sources, seen_records = set(), set()
    for index, header in enumerate(headers):
        cid = header[1]
        view = source_views.get(cid)
        if cid in seen_sources or type(view) is not dict or type(view.get('body')) is not str:
            return empty
        seen_sources.add(cid)
        end = headers[index + 1].start() if index + 1 < len(headers) else len(sources_block)
        lines = sources_block[header.end():end].strip().splitlines()
        deannotated = '\n'.join(line for line in lines if not _RECORD_ID.fullmatch(line))
        expected = view['body'].strip()
        if (len(expected) > MAX_TEXT or PREFIX in expected or '[TABLE_RECORD ' in expected
                or deannotated != expected):
            return empty
        records, record_ids, table_lines, pending = [], [], [], None
        for line in lines:
            identity = _RECORD_ID.fullmatch(line)
            if identity:
                if pending is not None or identity[1] in seen_records:
                    return empty
                pending = identity[1]
                seen_records.add(pending)
                continue
            match = table_units._LINE.fullmatch(line)
            if not match:
                if pending is not None or 'PDF_TABLE_' in line:
                    return empty
                continue
            role, first, last, columns = match[1], int(match[2]), int(match[3] or match[2]), int(match[4])
            if not 1 <= first <= last <= 100000 or not 1 <= columns <= 256:
                return empty
            try:
                cells = table_units._cells(match[5], 0, first, last, columns)
            except (ValueError, TypeError):
                return empty
            if not cells or any('\n' in c['value'] or '\r' in c['value'] or '\n' in (c['label'] or '') or '\r' in (c['label'] or '') for c in cells):
                return empty
            table_lines.append(line)
            if role == 'HEADER':
                if pending is not None:
                    return empty
                # No label inheritance across row restarts or source occurrences.
                # Keep the literal header itself: it can contain units/conditions
                # that do not occur again in individual record labels.
                records.append(cells)
                continue
            if pending is None:
                return empty
            record_ids.append(pending)
            records.append(cells)
            pending = None
        if pending is not None or len(seen_records) > 64:
            return empty
        if record_ids:
            token = '[[SOURCE_TABLE:G' + str(len(entries) + 1) + ']]'
            text = _render(records, view.get('page_from'), view.get('page_to'), response_language)
            entries.append(TableEntry(token, text, cid, tuple(record_ids), hashlib.sha256(expected.encode()).hexdigest(), tuple(table_lines)))
    if sum(len(e.text) for e in entries) > MAX_TEXT:
        return empty
    preview = '\n\n'.join(e.token + ' owner=' + e.citation_id + '\n' + e.text for e in entries)
    return TableCatalog(tuple(entries), preview)


def _strings(value):
    if type(value) is str:
        yield value
    elif type(value) is dict:
        for item in value.values():
            yield from _strings(item)
    elif type(value) in (list, tuple):
        for item in value:
            yield from _strings(item)


def _control_strings(result):
    """Explanations neither select quotations nor supply visible answer text."""
    for key, value in result.items():
        if key == 'reason' and type(value) is str:
            continue
        if key == 'table_record_coverage' and type(value) is list:
            for row in value:
                match = (re.fullmatch(r'T[1-9][0-9]*!([^\r\n]+)', row)
                         if type(row) is str and len(row) <= 512 else None)
                if match and match[1].strip():
                    continue
                yield from _strings(row)
        else:
            yield from _strings(value)


def resolve_reply(result: dict, catalog: TableCatalog, *, admitted_citation_ids=None,
                  diagnostics: dict | None = None, derive_duplicate_records: bool = False,
                  keyed_coverage: bool = False) -> dict:
    """Expand a whole inventory item without changing its ordered index.

    A reference binds to the owning citation and every record in that occurrence.
    Existing table bookkeeping and semantic validation run after this decoder.
    """
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics['source_table_gate'] = 'unused'
    def reject(gate):
        if diagnostics is not None:
            diagnostics['source_table_gate'] = gate
        return {**result, 'outcome': 'no_sources', 'answer': '',
                'reason': 'source_table_reference_invalid', '_terminal_review_failure': True}
    if keyed_coverage is True and catalog.entries and result.get('outcome') in {'pass', 'rewrite'}:
        keyed = result.get('table_record_coverage')
        identities = tuple(rid for entry in catalog.entries for rid in entry.record_ids)
        if type(keyed) is not dict or set(keyed) != set(identities):
            return reject('coverage_shape')
        rows = []
        for rid in identities:
            suffix = keyed[rid]
            if (type(suffix) is not str or len(rid + suffix) > 512
                    or re.fullmatch(r'(?:=[0-9]+(?:,[0-9]+)*|![^\r\n]+)', suffix) is None):
                return reject('coverage_shape')
            rows.append(rid + suffix)
        result = {**result, 'table_record_coverage': rows}
    has_reference = any(PREFIX in value for value in _control_strings(result))
    coverage = result.get('table_record_coverage')
    covered_records = {
        match[1] for row in (coverage if type(coverage) is list else ())
        if type(row) is str and (match := _COVERAGE.fullmatch(row)) is not None}
    required = {e.token for e in catalog.entries if covered_records.intersection(e.record_ids)}
    if not has_reference and (not required or result.get('outcome') not in {'pass', 'rewrite'}):
        return result
    items = result.get('expected_list_items')
    body = result.get('answer')
    if result.get('outcome') != 'rewrite':
        return reject('rewrite_required')
    if type(items) is not list or not 0 < len(items) <= 40:
        return reject('inventory_shape')
    if (type(body) is not str or body.splitlines().count(_INVENTORY) != 1
            or body.count('[[FACTUAL_INVENTORY') != 1):
        return reject('template_invalid')
    if any(PREFIX in value for value in _control_strings(
            {key: value for key, value in result.items() if key != 'expected_list_items'})):
        return reject('marker_outside_inventory')
    entries = {e.token: e for e in catalog.entries}
    citations = result.get('citation_ids')
    if type(citations) is not list or any(type(cid) is not str or not cid for cid in citations):
        return reject('citation_shape')
    if admitted_citation_ids is not None:
        if (type(admitted_citation_ids) not in (set, frozenset, tuple, list)
                or any(type(cid) is not str or not cid for cid in admitted_citation_ids)):
            return reject('source_scope_invalid')
        if any(cid not in admitted_citation_ids for cid in citations):
            return reject('citation_not_admitted')
    if type(coverage) is not list:
        return reject('coverage_shape')
    bound_citations, bound_coverage = list(citations), list(coverage)
    parsed_coverage = {}
    for row in coverage:
        if type(row) is not str:
            return reject('coverage_shape')
        match = _COVERAGE.fullmatch(row)
        rid = row.split('=', 1)[0].split('!', 1)[0]
        if rid in parsed_coverage:
            return reject('coverage_duplicate')
        parsed_coverage[rid] = tuple(int(n) for n in match[2].split(',')) if match else None
    expanded, selected = [], set()
    for index, item in enumerate(items):
        if type(item) is not str:
            return reject('item_shape')
        if PREFIX not in item:
            expanded.append(item)
            continue
        entry = entries.get(item)
        if not _TOKEN.fullmatch(item) or entry is None:
            return reject('reference_unknown')
        if item in selected:
            return reject('reference_duplicate')
        if admitted_citation_ids is not None and entry.citation_id not in admitted_citation_ids:
            return reject('source_scope_invalid')
        for rid in entry.record_ids:
            if rid in parsed_coverage:
                if parsed_coverage[rid] != (index,):
                    return reject('selected_record_conflict')
            else:
                bound_coverage.append(rid + '=' + str(index))
        if entry.citation_id not in bound_citations:
            bound_citations.append(entry.citation_id)
        selected.add(item)
        expanded.append(entry.text)
    if derive_duplicate_records is True:
        # Only the request-owned source-first path enables this, after arranging
        # mandatory independent review of every exact-table owner context.
        # No model field can enable it. Explicit classifications stay untouched.
        if admitted_citation_ids is None:
            return reject('source_scope_invalid')
        signatures = {entries[token].table_lines for token in selected if entries[token].table_lines}
        for entry in catalog.entries:
            if entry.token in selected or not entry.table_lines or entry.table_lines not in signatures:
                continue
            if entry.citation_id not in admitted_citation_ids:
                return reject('source_scope_invalid')
            for rid in entry.record_ids:
                if rid not in parsed_coverage:
                    bound_coverage.append(rid + '!Duplicate literal table occurrence; its source context requires independent review.')
    if not required.issubset(selected):
        return reject('required_reference_missing')
    if sum(map(len, expanded)) > MAX_TEXT:
        return reject('expanded_size')
    out = deepcopy(result)
    out['expected_list_items'] = expanded
    out['citation_ids'] = bound_citations
    out['table_record_coverage'] = bound_coverage
    if diagnostics is not None:
        diagnostics.update(source_table_gate='accepted', source_table_selected_count=len(selected),
            source_table_bound_record_count=len(bound_coverage)-len(coverage),
            source_table_bound_citation_count=len(bound_citations)-len(citations))
    return out
