"""Optional Smart context from exact, scoped Procedure/Step relations.

The caller supplies already admitted RAW retrieval rows, never client/state
metadata. Existing typed readers independently bind relation and page SQL columns.
Two reads maximum, one family, no embeddings, model, fuzzy title or fake header.
Context means the actual parent page and direct predecessors, not a claim that
the whole procedure or all prerequisites have been retrieved.
"""
from . import smart_evidence
from .chunk_evidence import ChunkReadScope, ChunkEvidenceLimits, _source_from_columns
from .relation_evidence import RelationEvidenceRead
from ..evidence.adapter_types import AdapterLimits
from ..evidence.assembly import AssemblyLimits
from ..evidence.legacy_compatibility import LegacyLimits
from ..evidence.manifest import ManifestLimits

MAX_READ_ROWS = 64
PROJECTION_CHARS = smart_evidence.MAX_CHARS + 1
_BYTES = 16 * 1024 * 1024
LIMITS = ChunkEvidenceLimits(AdapterLimits(PROJECTION_CHARS, 64, 4096),
    AssemblyLimits(ManifestLimits(MAX_READ_ROWS, _BYTES), MAX_READ_ROWS, _BYTES),
    LegacyLimits(24, 20000, _BYTES))


class ContextUnavailable(ValueError):
    pass


def _checked_read(read, scope, kind):
    if (not isinstance(read, RelationEvidenceRead) or read.pages.scope != scope
            or read.pages.kind != kind or len(read.relations) > MAX_READ_ROWS
            or read.pages.query_count != 1):
        raise ContextUnavailable('context_read_binding_invalid')
    for relation in read.relations:
        if type(relation.ordinal) is not int or relation.ordinal <= 0:
            raise ContextUnavailable('context_relation_ordinal_invalid')
        if relation.page_observation_index is None:
            raise ContextUnavailable('context_parent_page_missing')
    for page in read.pages.observations:
        # LEFT(text, N) is complete only below its sentinel. No clipping or
        # placeholder is admitted, including for rows later omitted by the cap.
        if not page.text_projection.strip() or len(page.text_projection) >= page.projection_chars:
            raise ContextUnavailable('context_page_incomplete')
    return read


def _row(page, *, parent_key=None, ordinal=None):
    key, number = page.storage_document_id, page.page_number
    return {'citation_id': f'{key}:p{number}-{number}:smart-context',
        'bubble_document_id': key, 'source_type': page.source.source_type.value,
        'source_id': page.source.source_id, 'company_id': page.stored_company_id,
        'machine_id': page.stored_machine_id or '',
        'exact_machine_scope': page.source.scope.machine_id is not None,
        'page_from': number, 'page_to': number,
        'chunk_full': page.text_projection, 'snippet': page.text_projection,
        'parent_source_key': parent_key, 'step_number': ordinal,
        'structured_context_origin': 'canonical_relation_read'}


def acquire(selected_raw, *, scope, parent_reader, step_reader, ensure_time):
    """Return <=3 new complete records and safe omission metadata, atomically.

    reader callbacks are the existing typed readers with their runtime supplied
    by the composition root. ensure_time checks the SAME request/operation budget.
    """
    meta = {'policy': 'smart-canonical-context-v1', 'read_calls': 0,
            'added_ids': [], 'omitted_ids': [], 'reason': 'no_scoped_step_anchor'}
    bound = ChunkReadScope(**scope)
    anchors, keys = [], []
    for row in selected_raw:
        key = str(row.get('bubble_document_id') or '')
        if not key.startswith('step:') or key in keys:
            continue
        # These rows come only from the server's scoped raw producer. A lost
        # machine binding cannot be reconstructed from a title or source ID.
        if (row.get('company_id') not in (None, '', bound.company_id)
                or row.get('machine_id') not in (None, '', bound.machine_id)):
            raise ContextUnavailable('context_anchor_scope_invalid')
        if row.get('machine_id') != bound.machine_id and row.get('exact_machine_scope') is not True:
            continue
        keys.append(key)
        anchors.append(_source_from_columns(bound.company_id, bound.machine_id, key))
    if not anchors:
        return [], meta
    if len(anchors) > smart_evidence.MAX_SELECTED_SOURCES:
        raise ContextUnavailable('context_anchor_limit')
    allowed = frozenset(anchors)
    try:
        ensure_time()
        meta['read_calls'] += 1
        parents = _checked_read(parent_reader(scope=bound, children=tuple(anchors),
            current_allowed_sources=allowed, text_chars=PROJECTION_CHARS, limits=LIMITS),
            bound, 'parent_procedures')
        by_child = {}
        for relation in parents.relations:
            identity = (relation.parent_source_key, relation.ordinal)
            previous = by_child.setdefault(relation.child_source_key, identity)
            if previous != identity:
                raise ContextUnavailable('context_parent_ambiguous')
        anchor_key = next((key for key in keys if key in by_child), None)
        if anchor_key is None:
            meta['reason'] = 'context_relation_missing'
            return [], meta
        parent_key = by_child[anchor_key][0]
        parent_pages = { (p.storage_document_id, p.page_number, p.text_projection): p
            for p in parents.pages.observations if p.storage_document_id == parent_key }
        if len(parent_pages) != 1:
            raise ContextUnavailable('context_parent_page_ambiguous')
        parent_page = next(iter(parent_pages.values()))
        # Only the source actually observed by the typed reader authorizes the
        # second read. Company-general parents keep their original association.
        ensure_time()
        meta['read_calls'] += 1
        children = _checked_read(step_reader(scope=bound, parent=parent_page.source,
            current_allowed_sources=allowed | parents.read_sources,
            text_chars=PROJECTION_CHARS, limits=LIMITS), bound, 'related_steps')
        by_ordinal, child_ordinals = {}, {}
        for relation in children.relations:
            if relation.parent_source_key != parent_key:
                raise ContextUnavailable('context_family_mismatch')
            page = children.pages.observations[relation.page_observation_index]
            identity = (relation.child_source_key, page.page_number, page.text_projection)
            previous = by_ordinal.setdefault(relation.ordinal, (identity, page))
            if previous[0] != identity:
                raise ContextUnavailable('context_predecessor_ambiguous')
            old = child_ordinals.setdefault(relation.child_source_key, relation.ordinal)
            if old != relation.ordinal:
                raise ContextUnavailable('context_child_ordinal_ambiguous')
        # Bind the second snapshot back to each selected anchor in this family.
        ordinals = []
        for key in keys:
            pair = by_child.get(key)
            if pair is None or pair[0] != parent_key:
                continue
            if child_ordinals.get(key) != pair[1]:
                raise ContextUnavailable('context_anchor_relation_changed')
            if pair[1] > 1 and pair[1] - 1 not in ordinals:
                ordinals.append(pair[1] - 1)
        candidates = [_row(parent_page)]
        for ordinal in ordinals:
            if ordinal not in by_ordinal:
                raise ContextUnavailable('context_predecessor_missing')
            candidates.append(_row(by_ordinal[ordinal][1], parent_key=parent_key, ordinal=ordinal))
        existing = {str(row.get('bubble_document_id') or '') for row in selected_raw}
        candidates = [row for row in candidates if row['bubble_document_id'] not in existing]
        selected = candidates[:smart_evidence.MAX_NEW_SOURCES]
        meta.update(reason='context_admitted' if selected else 'context_already_present',
            parent_source_key=parent_key, anchor_ids=keys,
            added_ids=[r['citation_id'] for r in selected],
            omitted_ids=[r['citation_id'] for r in candidates[len(selected):]])
        return selected, meta
    except ContextUnavailable as exc:
        meta['reason'] = str(exc)
    except Exception:
        # No exception text: SQL/connection diagnostics may contain credentials.
        meta['reason'] = 'context_read_unavailable'
    return [], meta
