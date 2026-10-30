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
    'are source context, not additional requested instructions. Cite the owning citation '
    'ID. Every TABLE_RECORD in a selected group MUST map to exactly that one inventory '
    'index in table_record_coverage; it cannot be excluded or mapped to a paraphrase. Normal inventory items '
    'remain necessary for supported non-tabular facts and applicable conditions, in '
    'RESPONSE_LANGUAGE. Preserve necessary safety prerequisites from non-tabular sources. '
    'Source quotations retain their original language and do not constitute a translation; '
    'provide the needed explanation in RESPONSE_LANGUAGE. Use references only for rewrite '
    'with the existing FACTUAL_INVENTORY answer template. No reference may occur in other '
    'fields. All existing semantic, safety, source and completeness checks still apply.'
    ' Avoid repeated quotations from overlapping occurrences: when literal equivalence '
    'is established, select one complete occurrence, cite its owner and mark the other '
    'occurrence records as duplicates. Never merge different assemblies or source identities.'
)


@dataclass(frozen=True)
class TableEntry:
    token: str
    text: str
    citation_id: str
    record_ids: tuple[str, ...]
    source_sha256: str


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
        records, record_ids, pending = [], [], None
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
            entries.append(TableEntry(token, text, cid, tuple(record_ids), hashlib.sha256(expected.encode()).hexdigest()))
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


def resolve_reply(result: dict, catalog: TableCatalog) -> dict:
    """Expand a whole inventory item without changing its ordered index.

    A reference binds to the owning citation and every record in that occurrence.
    Existing table bookkeeping and semantic validation run after this decoder.
    """
    def reject():
        return {**result, 'outcome': 'no_sources', 'answer': '',
                'reason': 'source_table_reference_invalid', '_terminal_review_failure': True}
    has_reference = any(PREFIX in value for value in _strings(result))
    coverage = result.get('table_record_coverage')
    covered_records = {
        match[1] for row in (coverage if type(coverage) is list else ())
        if type(row) is str and (match := _COVERAGE.fullmatch(row)) is not None}
    required = {e.token for e in catalog.entries if covered_records.intersection(e.record_ids)}
    if not has_reference and (not required or result.get('outcome') not in {'pass', 'rewrite'}):
        return result
    items = result.get('expected_list_items')
    body = result.get('answer')
    if (result.get('outcome') != 'rewrite' or type(items) is not list or not 0 < len(items) <= 40
            or type(body) is not str or body.splitlines().count(_INVENTORY) != 1
            or body.count('[[FACTUAL_INVENTORY') != 1
            or any(PREFIX in value for key, value in result.items() if key != 'expected_list_items' for value in _strings(value))):
        return reject()
    entries = {e.token: e for e in catalog.entries}
    citations = result.get('citation_ids')
    if type(citations) is not list or type(coverage) is not list:
        return reject()
    parsed_coverage = {}
    for row in coverage:
        if type(row) is not str:
            return reject()
        match = _COVERAGE.fullmatch(row)
        rid = row.split('=', 1)[0].split('!', 1)[0]
        if rid in parsed_coverage:
            return reject()
        parsed_coverage[rid] = tuple(int(n) for n in match[2].split(',')) if match else None
    expanded, selected = [], set()
    for index, item in enumerate(items):
        if type(item) is not str:
            return reject()
        if PREFIX not in item:
            expanded.append(item)
            continue
        entry = entries.get(item)
        if (not _TOKEN.fullmatch(item) or entry is None or item in selected
                or entry.citation_id not in citations
                or any(parsed_coverage.get(rid) != (index,) for rid in entry.record_ids)):
            return reject()
        selected.add(item)
        expanded.append(entry.text)
    if not required.issubset(selected) or sum(map(len, expanded)) > MAX_TEXT:
        return reject()
    out = deepcopy(result)
    out['expected_list_items'] = expanded
    return out
