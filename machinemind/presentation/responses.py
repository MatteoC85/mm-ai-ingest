"""Response finalization and safe HTML/UI rendering.

Extracted from the production composition root. The module receives all
product-specific callbacks and runtime limits explicitly, so it never imports
``main`` and cannot alter retrieval, ranking, evidence admission, source
priorities, prompts, models or tenant scope.
"""
from __future__ import annotations
import html
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, Collection

@dataclass(frozen=True)
class ResponsePresentationRuntime:
    normalize_unicode: Callable[..., str]
    procedure_ui_clean: Callable[..., str]
    procedure_ui_complete_excerpt: Callable[..., str]
    procedure_ui_fields: Callable[..., dict]
    procedure_ui_grounded_by_citation: Callable[..., bool]
    procedure_ui_is_final_verification: Callable[..., bool]
    procedure_ui_is_safety_setup: Callable[..., bool]
    procedure_ui_merge_sources: Callable[..., list]
    procedure_ui_note_is_novel: Callable[..., bool]
    procedure_ui_sections: Callable[..., list]
    safe_int: Callable[..., int]
    evidence_role: Callable[..., str]
    looks_like_target_language: Callable[..., bool]
    manual_note_from_grounded_points: Callable[..., str]
    manual_operation_and_safety_notes_from_support_citations: Callable[..., tuple]
    strip_inline_citation_markers_for_display: Callable[..., str]
    unique_non_empty_strings: Callable[..., list]
    source_type_from_document_id: Callable[..., str]
    structured_source_types: Collection[str]
    assistant_ui_max_html_chars: int
    assistant_ui_render_version: str
    assistant_ask_ui_render_version: str
    ask_ui_max_citations: int
    ask_ui_max_links: int
    ask_ui_structured_max_citations: int
    ask_ui_structured_max_links: int

def build_structured_procedure_ui_model(*, structured_citations: list[dict], manual_support_citations: list[dict], grounded_points: list[dict], response_language: str, q: str='', _runtime: ResponsePresentationRuntime) -> dict:
    """Build one safe, deterministic presentation model from grounded sources.

    The model never contains raw HTML. It is rendered twice: plain text for
    backward compatibility and escaped HTML for Bubble's HTML element.
    """
    is_en = str(response_language or 'it').strip().lower().startswith('en')
    coherent = _runtime.procedure_ui_merge_sources(structured_citations)
    procedures = [c for c in coherent if _runtime.evidence_role(c) == 'procedure']
    step_sources = [c for c in coherent if _runtime.evidence_role(c) == 'step']
    if not procedures and (not step_sources):
        return {}
    grounded_by_citation = _runtime.procedure_ui_grounded_by_citation(grounded_points)
    title = 'Operating procedure' if is_en else 'Procedura operativa'
    recipients = ''
    safety_level = ''
    purpose = ''
    if procedures:
        fields = _runtime.procedure_ui_fields(procedures[0])
        source_title = _runtime.procedure_ui_clean(fields.get('title') or '')
        if source_title and (not is_en or _runtime.looks_like_target_language(source_title, response_language)):
            title = source_title
        description = fields.get('short_description') or fields.get('description') or ''
        sections = _runtime.procedure_ui_sections(description)
        purpose = _runtime.procedure_ui_complete_excerpt(sections.get('purpose') or sections.get('body') or description, max_chars=700)
        recipients = _runtime.procedure_ui_complete_excerpt(sections.get('recipients') or '', max_chars=320)
        safety_level = _runtime.procedure_ui_complete_excerpt(sections.get('safety_level') or '', max_chars=140)
    records: list[dict] = []
    for fallback_no, citation in enumerate(step_sources, start=1):
        fields = _runtime.procedure_ui_fields(citation)
        source_no = _runtime.safe_int(fields.get('step_number'), fallback_no)
        step_title = _runtime.procedure_ui_clean(fields.get('title') or '') or f'Step {source_no}'
        sections = _runtime.procedure_ui_sections(fields.get('description') or '')
        instruction = _runtime.procedure_ui_complete_excerpt(sections.get('instruction') or sections.get('body') or fields.get('description') or '', max_chars=1500)
        safety = _runtime.procedure_ui_complete_excerpt(sections.get('safety') or '', max_chars=800)
        cid = str(citation.get('citation_id') or '').strip()
        grounded = max(grounded_by_citation.get(cid) or [''], key=len)
        if grounded and instruction and (not _runtime.looks_like_target_language(instruction, response_language)):
            instruction = grounded
        elif grounded and (not instruction):
            instruction = grounded
        if is_en and step_title and (not _runtime.looks_like_target_language(step_title, response_language)):
            step_title = f'Step {source_no}'
        records.append({'source_number': source_no, 'title': step_title, 'instruction': instruction, 'safety': safety})
    records.sort(key=lambda item: (int(item.get('source_number') or 9999), str(item.get('title') or '')))
    before: list[dict] = []
    final_checks: list[dict] = []
    operational = list(records)
    if operational:
        first_text = ' '.join((str(operational[0].get(k) or '') for k in ('title', 'instruction')))
        if _runtime.procedure_ui_is_safety_setup(first_text):
            before.append(operational.pop(0))
    if operational:
        last_title = str(operational[-1].get('title') or '')
        if _runtime.procedure_ui_is_final_verification(last_title):
            final_checks.insert(0, operational.pop())
    for display_no, record in enumerate(operational, start=1):
        record['display_number'] = display_no
    manual_notes: list[str] = []
    if manual_support_citations:
        operation_note, safety_note = _runtime.manual_operation_and_safety_notes_from_support_citations(manual_support_citations, q=q, structured_citations=coherent, language=response_language)
        if not safety_note:
            safety_note = _runtime.manual_note_from_grounded_points(grounded_points, language=response_language)
        existing = ' '.join([title, purpose, recipients, safety_level] + [str(r.get('instruction') or '') + ' ' + str(r.get('safety') or '') for r in records])
        for candidate in (operation_note if not records else '', safety_note):
            note = _runtime.procedure_ui_complete_excerpt(candidate, max_chars=700)
            if note and _runtime.procedure_ui_note_is_novel(note, existing + ' ' + ' '.join(manual_notes)):
                manual_notes.append(note)
    return {'kind': 'procedure', 'language': 'en' if is_en else 'it', 'title': title, 'summary': purpose, 'personnel': recipients, 'safety_level': safety_level, 'before': before, 'steps': operational, 'final_checks': final_checks, 'manual_notes': manual_notes[:2]}

def procedure_ui_model_to_text(model: dict, *, response_language: str, _runtime: ResponsePresentationRuntime) -> str:
    if not isinstance(model, dict) or model.get('kind') != 'procedure':
        return ''
    is_en = str(response_language or 'it').strip().lower().startswith('en')
    parts: list[str] = []
    title = _runtime.procedure_ui_clean(model.get('title') or '')
    if title:
        parts.append(title)
    summary = _runtime.procedure_ui_complete_excerpt(model.get('summary') or '', max_chars=700)
    if summary:
        parts.append(summary)
    before_lines: list[str] = []
    for item in model.get('before') or []:
        item_title = _runtime.procedure_ui_clean(item.get('title') or '')
        instruction = _runtime.procedure_ui_complete_excerpt(item.get('instruction') or '', max_chars=1500)
        safety = _runtime.procedure_ui_complete_excerpt(item.get('safety') or '', max_chars=800)
        if item_title:
            before_lines.append(item_title)
        if instruction:
            before_lines.append(instruction)
        if safety:
            before_lines.append(('Safety: ' if is_en else 'Sicurezza: ') + safety)
    personnel = _runtime.procedure_ui_complete_excerpt(model.get('personnel') or '', max_chars=320)
    safety_level = _runtime.procedure_ui_complete_excerpt(model.get('safety_level') or '', max_chars=140)
    if personnel:
        before_lines.append(('Qualified personnel: ' if is_en else 'Personale qualificato: ') + personnel)
    if safety_level:
        before_lines.append(('Safety level: ' if is_en else 'Livello di sicurezza: ') + safety_level)
    if before_lines:
        parts.append(('Before starting' if is_en else 'Prima di iniziare') + '\n' + '\n'.join(before_lines))
    step_blocks: list[str] = []
    for idx, item in enumerate(model.get('steps') or [], start=1):
        display_no = _runtime.safe_int(item.get('display_number'), idx)
        title_line = _runtime.procedure_ui_clean(item.get('title') or '') or (f'Step {display_no}' if is_en else f'Passaggio {display_no}')
        instruction = _runtime.procedure_ui_complete_excerpt(item.get('instruction') or '', max_chars=1500)
        safety = _runtime.procedure_ui_complete_excerpt(item.get('safety') or '', max_chars=800)
        lines = [f'{display_no}. {title_line}']
        if instruction:
            lines.append(instruction)
        if safety:
            lines.append(('Attention: ' if is_en else 'Attenzione: ') + safety)
        step_blocks.append('\n'.join(lines))
    if step_blocks:
        parts.append(('Procedure' if is_en else 'Procedura') + '\n' + '\n\n'.join(step_blocks))
    final_blocks: list[str] = []
    for item in model.get('final_checks') or []:
        title_line = _runtime.procedure_ui_clean(item.get('title') or '')
        instruction = _runtime.procedure_ui_complete_excerpt(item.get('instruction') or '', max_chars=1500)
        safety = _runtime.procedure_ui_complete_excerpt(item.get('safety') or '', max_chars=800)
        if title_line:
            final_blocks.append(title_line)
        if instruction:
            final_blocks.append(instruction)
        if safety:
            final_blocks.append(('Attention: ' if is_en else 'Attenzione: ') + safety)
    if final_blocks:
        parts.append(('Final check' if is_en else 'Verifica finale') + '\n' + '\n'.join(final_blocks))
    notes = [_runtime.procedure_ui_complete_excerpt(x, max_chars=700) for x in model.get('manual_notes') or [] if _runtime.procedure_ui_complete_excerpt(x, max_chars=700)]
    if notes:
        parts.append(('Manual support' if is_en else 'Supporto dal manuale') + '\n' + '\n'.join((f'- {x}' for x in notes)))
    return '\n\n'.join((x for x in parts if str(x or '').strip())).strip()

def assistant_ui_escape(value: Any, *, _runtime: ResponsePresentationRuntime) -> str:
    return html.escape(str(value or ''), quote=True)

def procedure_ui_model_to_html(model: dict, *, links: list[dict], response_language: str, _runtime: ResponsePresentationRuntime) -> str:
    """Render a clean, ChatGPT-like procedure body.

    Bubble keeps LINK and FONTI below this element. The answer body therefore uses
    simple typography, native ordered lists and only restrained callouts. It has no
    outer border, background or shadow, so the existing Bubble answer container remains
    the only visual frame.
    """
    if not isinstance(model, dict) or model.get('kind') != 'procedure':
        return ''
    is_en = str(response_language or 'it').strip().lower().startswith('en')
    title = assistant_ui_escape(model.get('title') or ('Operating procedure' if is_en else 'Procedura operativa'), _runtime=_runtime)
    summary = assistant_ui_escape(model.get('summary') or '', _runtime=_runtime)
    chunks: list[str] = ['<article data-mm-answer="procedure" data-mm-render="' + assistant_ui_escape(_runtime.assistant_ui_render_version, _runtime=_runtime) + '" style="box-sizing:border-box;width:100%;font-family:Inter,-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif;font-size:14px;line-height:1.62;color:#1f2937;overflow-wrap:anywhere;padding:2px 2px 1px 2px;">', '<h2 style="margin:0 0 8px 0;font-size:18px;line-height:1.35;font-weight:760;color:#111827;">' + title + '</h2>']
    if summary:
        chunks.append('<p style="margin:0 0 18px 0;color:#4b5563;">' + summary + '</p>')
    before_items = model.get('before') or []
    personnel = str(model.get('personnel') or '').strip()
    safety_level = str(model.get('safety_level') or '').strip()
    if before_items or personnel or safety_level:
        heading = 'Before starting' if is_en else 'Prima di iniziare'
        chunks.append('<section style="margin:0 0 20px 0;padding:11px 14px;border-left:3px solid #d97706;background:#fffbeb;border-radius:0 7px 7px 0;"><h3 style="margin:0 0 7px 0;font-size:14px;font-weight:760;color:#92400e;">' + assistant_ui_escape(heading, _runtime=_runtime) + '</h3>')
        if personnel:
            chunks.append('<p style="margin:3px 0;color:#78350f;"><strong>' + assistant_ui_escape('Qualified personnel' if is_en else 'Personale qualificato', _runtime=_runtime) + ':</strong> ' + assistant_ui_escape(personnel, _runtime=_runtime) + '</p>')
        if safety_level:
            chunks.append('<p style="margin:3px 0;color:#991b1b;"><strong>' + assistant_ui_escape('Safety level' if is_en else 'Livello di sicurezza', _runtime=_runtime) + ':</strong> ' + assistant_ui_escape(safety_level, _runtime=_runtime) + '</p>')
        for item in before_items:
            item_title = assistant_ui_escape(item.get('title') or '', _runtime=_runtime)
            instruction = assistant_ui_escape(item.get('instruction') or '', _runtime=_runtime)
            safety = assistant_ui_escape(item.get('safety') or '', _runtime=_runtime)
            if item_title:
                chunks.append('<p style="margin:7px 0 2px 0;font-weight:700;color:#78350f;">' + item_title + '</p>')
            if instruction:
                chunks.append('<p style="margin:2px 0;color:#78350f;">' + instruction + '</p>')
            if safety:
                chunks.append('<p style="margin:6px 0 0 0;color:#991b1b;"><strong>' + assistant_ui_escape('Attention' if is_en else 'Attenzione', _runtime=_runtime) + ':</strong> ' + safety + '</p>')
        chunks.append('</section>')
    steps = model.get('steps') or []
    if steps:
        heading = 'Procedure' if is_en else 'Procedura'
        chunks.append('<section style="margin:0 0 21px 0;">')
        chunks.append('<h3 style="margin:0 0 9px 0;font-size:15px;font-weight:760;color:#111827;">' + assistant_ui_escape(heading, _runtime=_runtime) + '</h3>')
        chunks.append('<ol style="margin:0;padding-left:22px;color:#374151;">')
        for idx, item in enumerate(steps, start=1):
            display_no = _runtime.safe_int(item.get('display_number'), idx)
            item_title = assistant_ui_escape(item.get('title') or (f'Step {display_no}' if is_en else f'Passaggio {display_no}'), _runtime=_runtime)
            instruction = assistant_ui_escape(item.get('instruction') or '', _runtime=_runtime)
            safety = assistant_ui_escape(item.get('safety') or '', _runtime=_runtime)
            chunks.append(f'<li value="{display_no}" style="margin:0 0 12px 0;padding-left:4px;">')
            chunks.append('<p style="margin:0 0 3px 0;font-weight:720;color:#111827;">' + item_title + '</p>')
            if instruction:
                chunks.append('<p style="margin:0;color:#374151;">' + instruction + '</p>')
            if safety:
                chunks.append('<p style="margin:5px 0 0 0;color:#991b1b;"><strong>' + assistant_ui_escape('Attention' if is_en else 'Attenzione', _runtime=_runtime) + ':</strong> ' + safety + '</p>')
            chunks.append('</li>')
        chunks.append('</ol></section>')
    final_checks = model.get('final_checks') or []
    if final_checks:
        heading = 'Final check' if is_en else 'Verifica finale'
        chunks.append('<section style="margin:0 0 19px 0;padding:11px 14px;border-left:3px solid #16a34a;background:#f0fdf4;border-radius:0 7px 7px 0;"><h3 style="margin:0 0 7px 0;font-size:14px;font-weight:760;color:#166534;">' + assistant_ui_escape(heading, _runtime=_runtime) + '</h3>')
        for item in final_checks:
            title_line = assistant_ui_escape(item.get('title') or '', _runtime=_runtime)
            instruction = assistant_ui_escape(item.get('instruction') or '', _runtime=_runtime)
            safety = assistant_ui_escape(item.get('safety') or '', _runtime=_runtime)
            if title_line:
                chunks.append('<p style="margin:4px 0 2px 0;font-weight:700;color:#166534;">' + title_line + '</p>')
            if instruction:
                chunks.append('<p style="margin:2px 0;color:#166534;">' + instruction + '</p>')
            if safety:
                chunks.append('<p style="margin:6px 0 0 0;color:#991b1b;"><strong>' + assistant_ui_escape('Attention' if is_en else 'Attenzione', _runtime=_runtime) + ':</strong> ' + safety + '</p>')
        chunks.append('</section>')
    notes = [str(x or '').strip() for x in model.get('manual_notes') or [] if str(x or '').strip()]
    if notes:
        heading = 'Manual support' if is_en else 'Supporto dal manuale'
        chunks.append('<section style="margin:0 0 8px 0;">')
        chunks.append('<h3 style="margin:0 0 7px 0;font-size:14px;font-weight:760;color:#111827;">' + assistant_ui_escape(heading, _runtime=_runtime) + '</h3>')
        chunks.append('<ul style="margin:0;padding-left:20px;color:#4b5563;">')
        for note in notes[:2]:
            chunks.append('<li style="margin:0 0 6px 0;padding-left:2px;">' + assistant_ui_escape(note, _runtime=_runtime) + '</li>')
        chunks.append('</ul></section>')
    chunks.append('</article>')
    rendered = ''.join(chunks)
    return rendered if len(rendered) <= _runtime.assistant_ui_max_html_chars else ''

def assistant_ui_inline_markup(value: Any, *, _runtime: ResponsePresentationRuntime) -> str:
    """Render a tiny safe subset of Markdown-like inline formatting.

    All input is escaped first; only bold and inline-code markers are converted.
    The model can therefore never inject arbitrary HTML, scripts or links.
    """
    escaped = assistant_ui_escape(value, _runtime=_runtime)
    escaped = re.sub('\\*\\*([^*\\n][^*\\n]*?)\\*\\*', lambda m: '<strong style="font-weight:750;color:#111827;">' + m.group(1).strip() + '</strong>', escaped)
    escaped = re.sub('`([^`\\n]+)`', lambda m: '<code style="padding:1px 5px;border-radius:5px;background:#f1f5f9;color:#0f172a;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;">' + m.group(1).strip() + '</code>', escaped)
    return escaped

def assistant_ui_section_kind(label: str, *, _runtime: ResponsePresentationRuntime) -> str:
    norm = _runtime.normalize_unicode(str(label or '')).lower()
    norm = re.sub('[^a-zà-öø-ÿ0-9]+', ' ', norm)
    norm = re.sub('\\s+', ' ', norm).strip()
    if not norm:
        return ''
    if any((x in norm for x in ['verifiche da eseguire', 'controlli consigliati', 'verifiche', 'controlli', 'checks to perform', 'recommended checks', 'checks'])):
        return 'checks'
    if any((x in norm for x in ['soluzione indicata', 'soluzione consigliata', 'intervento consigliato', 'soluzione', 'recommended solution', 'solution'])):
        return 'solution'
    if any((x in norm for x in ['nota tecnica', 'note tecniche', 'technical note', 'technical notes'])):
        return 'technical_note'
    if any((x in norm for x in ['causa probabile', 'causa più probabile', 'cause probabili', 'probable cause', 'likely cause'])):
        return 'cause'
    if any((x in norm for x in ['problema rilevato', 'problema', 'sintomo', 'problem', 'symptom'])):
        return 'problem'
    if any((x in norm for x in ['attenzione', 'avvertenza', 'nota di sicurezza', 'warning', 'safety note', 'attention'])):
        return 'safety'
    if any((x in norm for x in ['in sintesi', 'sintesi', 'summary', 'risposta', 'answer'])):
        return 'summary'
    return ''

def assistant_ui_extract_labeled_line(line: str, *, _runtime: ResponsePresentationRuntime) -> tuple[str, str, str]:
    """Return (section_kind, visible_label, remainder) for labelled answer lines."""
    work = str(line or '').strip()
    work = re.sub('^\\d{1,3}[.)]\\s+', '', work).strip()
    m = re.match('^\\*\\*\\s*([^*\\n:]{2,90})\\s*:\\s*\\*\\*\\s*(.*)$', work)
    if not m:
        m = re.match('^\\*\\*\\s*([^*\\n]{2,90})\\s*\\*\\*\\s*:\\s*(.*)$', work)
    if not m:
        m_plain = re.match('^([^:\\n]{2,90})\\s*:\\s*(.*)$', work)
        if m_plain and assistant_ui_section_kind(m_plain.group(1), _runtime=_runtime):
            m = m_plain
    if not m:
        return ('', '', '')
    label = re.sub('\\s+', ' ', str(m.group(1) or '')).strip()
    kind = assistant_ui_section_kind(label, _runtime=_runtime)
    if not kind:
        return ('', '', '')
    return (kind, label, str(m.group(2) or '').strip())

def assistant_ui_split_inline_numbered(value: str, *, _runtime: ResponsePresentationRuntime) -> list[str]:
    text = str(value or '').strip()
    if not text:
        return []
    matches = list(re.finditer('(?:^|\\s)(\\d{1,3})[.)]\\s+', text))
    if not matches:
        return [text]
    out: list[str] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        item = text[start:end].strip(' \t\n-•')
        if item:
            out.append(item)
    return out

def assistant_ui_render_numbered_cards(items: list[str], *, is_en: bool, _runtime: ResponsePresentationRuntime) -> str:
    """Render a native ordered list while preserving explicit source numbers.

    Items may be plain strings or internal ``{"number": n, "text": ...}`` records.
    The latter prevents the browser from restarting every separated ordered-list
    fragment at 1 when labelled detail lines split a procedure answer into sections.
    """
    clean_items: list[tuple[int | None, str]] = []
    for raw in items or []:
        if isinstance(raw, dict):
            text = str(raw.get("text") or "").strip()
            number = _runtime.safe_int(raw.get("number"), 0)
            clean_items.append((number if number > 0 else None, text)) if text else None
        else:
            text = str(raw or "").strip()
            clean_items.append((None, text)) if text else None
    if not clean_items:
        return ""
    chunks = ['<ol style="margin:0;padding-left:22px;color:#374151;">']
    for number, text in clean_items:
        value_attr = f' value="{number}"' if number is not None else ""
        chunks.append(
            f'<li{value_attr} style="margin:0 0 10px 0;padding-left:4px;">'
            '<p style="margin:0;">'
            + assistant_ui_inline_markup(text, _runtime=_runtime)
            + '</p></li>'
        )
    chunks.append("</ol>")
    return "".join(chunks)

def assistant_ui_sentence_has_any(value: str, markers: list[str], *, _runtime: ResponsePresentationRuntime) -> bool:
    low = _runtime.normalize_unicode(str(value or '')).lower()
    return any((marker in low for marker in markers))

def assistant_ui_promote_unlabelled_sections(sections: list[dict], *, _runtime: ResponsePresentationRuntime) -> list[dict]:
    """Infer visual roles only for genuinely unlabelled prose.

    Explicit headings, bullets and numbered lists are already a deliberate structure
    produced by the grounded answer. Earlier versions flattened those sections into
    an inferred checks block and silently dropped the real checklist. This function
    is intentionally conservative and lossless.
    """
    if not sections:
        return sections
    explicit_kinds = {str(row.get('kind') or 'body').strip().lower() for row in sections if isinstance(row, dict)}
    if any((kind != 'body' for kind in explicit_kinds)):
        return sections
    if any((str(row.get('kind') or '') in {'problem', 'cause', 'checks', 'solution', 'technical_note', 'safety'} for row in sections if isinstance(row, dict))):
        return sections
    body_paragraphs: list[str] = []
    for row in sections:
        if not isinstance(row, dict):
            continue
        body_paragraphs.extend((str(x or '').strip() for x in row.get('paragraphs') or [] if str(x or '').strip()))
    candidates = body_paragraphs
    if len(candidates) < 2:
        return sections
    cause_markers = ['causa', 'probabil', 'verosimil', 'dovut', 'riconduc', 'associat', 'superat', 'likely', 'probable', 'due to', 'caused by', 'associated with', 'overload']
    solution_markers = ['se i controlli', 'se le verifiche', 'intervento', 'soluzione', 'sostitu', 'selezionare', "verificare l'idoneità", 'verificare l’idoneità', 'contattare', 'ripristinare', 'regolare', 'if the checks', 'recommended action', 'solution', 'replace', 'select', 'contact', 'restore', 'adjust']
    note_markers = ['nota tecnica', 'documento', 'manuale', 'rapporto', 'relazione', 'technical note', 'the document', 'manual']
    first_is_cause = assistant_ui_sentence_has_any(candidates[0], cause_markers, _runtime=_runtime)
    last_is_solution = assistant_ui_sentence_has_any(candidates[-1], solution_markers, _runtime=_runtime)
    last_is_note = assistant_ui_sentence_has_any(candidates[-1], note_markers, _runtime=_runtime) and (not last_is_solution)
    out: list[dict] = []
    left = list(candidates)
    if first_is_cause:
        out.append({'kind': 'cause', 'label': '', 'paragraphs': [left.pop(0)], 'items': []})
    tail_note = ''
    tail_solution = ''
    if left and last_is_note:
        tail_note = left.pop()
    elif left and last_is_solution:
        tail_solution = left.pop()
    if left:
        if len(left) == 1 and (not first_is_cause) and (not tail_solution) and (not tail_note):
            out.append({'kind': 'body', 'label': '', 'paragraphs': left, 'items': []})
        else:
            out.append({'kind': 'checks', 'label': '', 'paragraphs': [], 'items': left})
    if tail_solution:
        out.append({'kind': 'solution', 'label': '', 'paragraphs': [tail_solution], 'items': []})
    if tail_note:
        out.append({'kind': 'technical_note', 'label': '', 'paragraphs': [tail_note], 'items': []})
    return out or sections

def assistant_ui_normalize_markdown_tables(text: str, *, _runtime: ResponsePresentationRuntime) -> str:
    """Convert Markdown tables to headings/bullets before safe HTML rendering."""
    lines = str(text or '').replace('\r', '\n').split('\n')
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if '|' in line and i + 1 < len(lines) and re.match('^\\s*\\|?\\s*:?-{3,}', lines[i + 1].strip()):
            headers = [x.strip() for x in line.strip('|').split('|')]
            i += 2
            rows: list[list[str]] = []
            while i < len(lines) and '|' in lines[i]:
                rows.append([x.strip() for x in lines[i].strip().strip('|').split('|')])
                i += 1
            for row in rows:
                if not any(row):
                    continue
                title = row[0] if row else ''
                if title:
                    out.append(f'**{title}**')
                for idx, value in enumerate(row[1:], start=1):
                    if not value:
                        continue
                    label = headers[idx] if idx < len(headers) and headers[idx] else f'Campo {idx}'
                    out.append(f'- **{label}:** {value}')
                out.append('')
            continue
        out.append(lines[i])
        i += 1
    return '\n'.join(out)

def assistant_ui_root_cause_text(resp: dict, *, response_language: str, _runtime: ResponsePresentationRuntime) -> str:
    """Canonical plain-text representation of the same structured Root Cause body."""
    if not isinstance(resp, dict):
        return ''
    is_en = str(response_language or 'it').lower().startswith('en')
    status = str(resp.get('status') or 'answered').strip().lower()
    summary = str(resp.get('problem_summary') or resp.get('symptom') or '').strip()
    causes = [c for c in resp.get('possible_causes') or [] if isinstance(c, dict)]
    recommended = _runtime.unique_non_empty_strings(resp.get('recommended_next_checks') or [], limit=10)
    if status != 'answered' or not causes:
        return summary
    lines = ['Probable causes' if is_en else 'Cause probabili']
    if summary:
        lines += ['', ('Problem: ' if is_en else 'Problema: ') + summary]
    used_checks: set[str] = set()
    for idx, cause in enumerate(causes, start=1):
        rank = _runtime.safe_int(cause.get('rank'), idx)
        label = ('Most likely cause' if rank == 1 else f'Possible cause {rank}') if is_en else 'Causa più probabile' if rank == 1 else f'Causa possibile {rank}'
        cause_text = str(cause.get('cause') or '').strip()
        why = str(cause.get('why') or '').strip()
        checks = _runtime.unique_non_empty_strings(cause.get('checks') or [], limit=6)
        if not cause_text:
            continue
        lines += ['', label, cause_text]
        if why:
            lines += [('Why: ' if is_en else 'Perché: ') + why]
        if checks:
            lines += ['Checks' if is_en else 'Controlli consigliati']
            for n, check in enumerate(checks, start=1):
                lines.append(f'{n}. {check}')
                used_checks.add(re.sub('\\s+', ' ', _runtime.normalize_unicode(check).lower()).strip())
    extra = [x for x in recommended if re.sub('\\s+', ' ', _runtime.normalize_unicode(x).lower()).strip() not in used_checks]
    if extra:
        lines += ['', 'Additional priority checks' if is_en else 'Ulteriori controlli prioritari']
        lines += [f'{n}. {x}' for n, x in enumerate(extra[:6], start=1)]
    return '\n'.join(lines).strip()

def assistant_ui_generic_html(answer: str, *, links: list[dict], citations: Optional[list[dict]]=None, response_language: str, status: str='answered', _runtime: ResponsePresentationRuntime) -> str:
    """Render ASK with the same visual hierarchy used by Root Cause.

    This function changes presentation only. The canonical answer text, citations,
    links and Bubble LINK/FONTI sections remain untouched. Every source line is
    escaped before the restrained inline formatting is applied.
    """
    is_en = str(response_language or 'it').strip().lower().startswith('en')
    clean = assistant_ui_normalize_markdown_tables(_runtime.strip_inline_citation_markers_for_display(str(answer or '')), _runtime=_runtime).strip()
    if not clean:
        return ''
    status_low = str(status or 'answered').strip().lower()
    sections: list[dict] = []
    current: Optional[dict] = None

    def new_section(kind: str, label: str='') -> dict:
        row = {'kind': kind or 'body', 'label': str(label or '').strip(), 'paragraphs': [], 'items': []}
        sections.append(row)
        return row
    raw_lines = [x.rstrip() for x in clean.replace('\r', '\n').split('\n')]
    for raw in raw_lines:
        line = str(raw or '').strip()
        if not line:
            continue
        kind, label, remainder = assistant_ui_extract_labeled_line(line, _runtime=_runtime)
        if kind:
            current = new_section(kind, label)
            if remainder:
                if kind == 'checks':
                    current['items'].extend(assistant_ui_split_inline_numbered(remainder, _runtime=_runtime))
                else:
                    current['paragraphs'].append(remainder)
            continue
        numbered = re.match(r'^(\d{1,3})[.)]\s+(.+)$', line)
        bullet = re.match('^[-•]\\s+(.+)$', line)
        if numbered:
            number = int(numbered.group(1))
            value = numbered.group(2).strip()
            if current is None or current.get('kind') not in {'checks', 'list'}:
                current = new_section('list', '')
            current['items'].append({'number': number, 'text': value})
            continue
        if bullet:
            value = bullet.group(1).strip()
            if current is None or current.get('kind') not in {'checks', 'list', 'bullets'}:
                current = new_section('bullets', '')
            current['items'].append(value)
            continue
        bold_heading = re.match('^\\*\\*([^*\\n]{2,90})\\*\\*\\s*$', line)
        if bold_heading:
            current = new_section('heading', bold_heading.group(1).strip())
            continue
        if current is None:
            current = new_section('body', '')
        current['paragraphs'].append(line)
    sections = assistant_ui_promote_unlabelled_sections(sections, _runtime=_runtime)
    article: list[str] = ['<article data-mm-answer="generic" data-mm-render="' + assistant_ui_escape(_runtime.assistant_ask_ui_render_version, _runtime=_runtime) + '" data-mm-visual="root-parity-v1" style="box-sizing:border-box;width:100%;font-family:Inter,-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif;font-size:14px;line-height:1.62;color:#1f2937;overflow-wrap:anywhere;padding:2px 2px 1px 2px;">', '<h2 style="margin:0 0 8px 0;font-size:18px;line-height:1.35;font-weight:760;color:#111827;">' + assistant_ui_escape('Answer' if is_en else 'Risposta', _runtime=_runtime) + '</h2>']
    if status_low != 'answered':
        if status_low in {'no_sources', 'needs_clarification'}:
            tone, background, border = ('#92400e', '#fffbeb', '#d97706')
        else:
            tone, background, border = ('#991b1b', '#fef2f2', '#dc2626')
        article.append('<section style="margin:0 0 18px 0;padding:10px 14px;border-left:3px solid ' + border + ';background:' + background + ';border-radius:0 7px 7px 0;color:' + tone + ';"><p style="margin:0;font-size:12px;line-height:1.4;font-weight:760;letter-spacing:.02em;text-transform:uppercase;color:' + tone + ';">' + assistant_ui_escape('Result' if is_en else 'Esito', _runtime=_runtime) + '</p></section>')
    first_body = True
    for section in sections:
        kind = str(section.get('kind') or 'body')
        label = str(section.get('label') or '').strip()
        paragraphs = [str(x or '').strip() for x in section.get('paragraphs') or [] if str(x or '').strip()]
        items: list[Any] = []
        for raw_item in section.get('items') or []:
            if isinstance(raw_item, dict):
                item_text = str(raw_item.get('text') or '').strip()
                if item_text:
                    items.append({'number': _runtime.safe_int(raw_item.get('number'), 0), 'text': item_text})
            else:
                item_text = str(raw_item or '').strip()
                if item_text:
                    items.append(item_text)
        if kind == 'cause':
            heading = label or ('Probable cause' if is_en else 'Causa probabile')
            article.append('<section style="margin:0 0 20px 0;padding-left:14px;border-left:3px solid #2563eb;"><p style="margin:0 0 4px 0;font-size:12px;line-height:1.4;font-weight:760;letter-spacing:.02em;text-transform:uppercase;color:#1d4ed8;">' + assistant_ui_escape(heading, _runtime=_runtime) + '</p>')
            for idx, p in enumerate(paragraphs):
                tag = 'h3' if idx == 0 else 'p'
                style = 'margin:0 0 7px 0;font-size:15px;line-height:1.45;font-weight:760;color:#111827;' if idx == 0 else 'margin:0 0 7px 0;color:#4b5563;'
                article.append(f'<{tag} style="{style}">' + assistant_ui_inline_markup(p, _runtime=_runtime) + f'</{tag}>')
            article.append('</section>')
        elif kind == 'problem':
            heading = label or ('Context' if is_en else 'Contesto')
            article.append('<section style="margin:0 0 20px 0;padding-left:14px;border-left:3px solid #cbd5e1;"><p style="margin:0 0 5px 0;font-size:12px;line-height:1.4;font-weight:760;letter-spacing:.02em;text-transform:uppercase;color:#475569;">' + assistant_ui_escape(heading, _runtime=_runtime) + '</p>')
            for p in paragraphs:
                article.append('<p style="margin:0 0 7px 0;color:#4b5563;">' + assistant_ui_inline_markup(p, _runtime=_runtime) + '</p>')
            article.append('</section>')
        elif kind == 'checks':
            heading = label or ('Checks to perform' if is_en else 'Verifiche da eseguire')
            article.append('<section style="margin:0 0 20px 0;padding-left:14px;border-left:3px solid #2563eb;"><p style="margin:0 0 7px 0;font-size:12px;line-height:1.4;font-weight:760;letter-spacing:.02em;text-transform:uppercase;color:#1d4ed8;">' + assistant_ui_escape(heading, _runtime=_runtime) + '</p>' + assistant_ui_render_numbered_cards(items, is_en=is_en, _runtime=_runtime))
            for p in paragraphs:
                article.append('<p style="margin:8px 0 0 0;color:#4b5563;">' + assistant_ui_inline_markup(p, _runtime=_runtime) + '</p>')
            article.append('</section>')
        elif kind == 'solution':
            heading = label or ('Recommended action' if is_en else 'Intervento consigliato')
            article.append('<section style="margin:0 0 18px 0;padding:10px 14px;border-left:3px solid #16a34a;background:#f0fdf4;border-radius:0 7px 7px 0;"><p style="margin:0 0 5px 0;font-size:12px;line-height:1.4;font-weight:760;letter-spacing:.02em;text-transform:uppercase;color:#166534;">' + assistant_ui_escape(heading, _runtime=_runtime) + '</p>')
            for p in paragraphs:
                article.append('<p style="margin:3px 0;color:#166534;">' + assistant_ui_inline_markup(p, _runtime=_runtime) + '</p>')
            article.append('</section>')
        elif kind == 'technical_note':
            heading = label or ('Technical note' if is_en else 'Nota tecnica')
            article.append('<section style="margin:0 0 18px 0;padding-left:14px;border-left:3px solid #cbd5e1;"><p style="margin:0 0 5px 0;font-size:12px;line-height:1.4;font-weight:760;letter-spacing:.02em;text-transform:uppercase;color:#475569;">' + assistant_ui_escape(heading, _runtime=_runtime) + '</p>')
            for p in paragraphs:
                article.append('<p style="margin:3px 0;color:#4b5563;">' + assistant_ui_inline_markup(p, _runtime=_runtime) + '</p>')
            article.append('</section>')
        elif kind == 'safety':
            heading = label or ('Attention' if is_en else 'Attenzione')
            article.append('<section style="margin:0 0 18px 0;padding:10px 14px;border-left:3px solid #dc2626;background:#fef2f2;border-radius:0 7px 7px 0;"><p style="margin:0 0 5px 0;font-size:12px;line-height:1.4;font-weight:760;letter-spacing:.02em;text-transform:uppercase;color:#991b1b;">' + assistant_ui_escape(heading, _runtime=_runtime) + '</p>')
            for p in paragraphs:
                article.append('<p style="margin:3px 0;color:#991b1b;">' + assistant_ui_inline_markup(p, _runtime=_runtime) + '</p>')
            article.append('</section>')
        elif kind == 'summary':
            heading = label or ('Summary' if is_en else 'In sintesi')
            article.append('<section style="margin:0 0 20px 0;padding-left:14px;border-left:3px solid #2563eb;"><p style="margin:0 0 5px 0;font-size:12px;line-height:1.4;font-weight:760;letter-spacing:.02em;text-transform:uppercase;color:#1d4ed8;">' + assistant_ui_escape(heading, _runtime=_runtime) + '</p>')
            for p in paragraphs:
                article.append('<p style="margin:0 0 7px 0;color:#374151;">' + assistant_ui_inline_markup(p, _runtime=_runtime) + '</p>')
            article.append('</section>')
        elif kind == 'list':
            heading = label or ('Steps' if is_en else 'Passaggi')
            article.append('<section style="margin:0 0 20px 0;padding-left:14px;border-left:3px solid #2563eb;"><p style="margin:0 0 7px 0;font-size:12px;line-height:1.4;font-weight:760;letter-spacing:.02em;text-transform:uppercase;color:#1d4ed8;">' + assistant_ui_escape(heading, _runtime=_runtime) + '</p>' + assistant_ui_render_numbered_cards(items, is_en=is_en, _runtime=_runtime) + '</section>')
        elif kind == 'bullets':
            heading = label or ('Key points' if is_en else 'Punti principali')
            article.append('<section style="margin:0 0 20px 0;padding-left:14px;border-left:3px solid #2563eb;"><p style="margin:0 0 7px 0;font-size:12px;line-height:1.4;font-weight:760;letter-spacing:.02em;text-transform:uppercase;color:#1d4ed8;">' + assistant_ui_escape(heading, _runtime=_runtime) + '</p><ul style="margin:0;padding-left:20px;color:#374151;">')
            for item in items:
                item_text = str(item.get('text') or '').strip() if isinstance(item, dict) else str(item or '').strip()
                if item_text:
                    article.append('<li style="margin:0 0 8px 0;padding-left:2px;">' + assistant_ui_inline_markup(item_text, _runtime=_runtime) + '</li>')
            article.append('</ul></section>')
        elif kind == 'heading':
            article.append('<section style="margin:0 0 20px 0;padding-left:14px;border-left:3px solid #2563eb;"><p style="margin:0 0 7px 0;font-size:12px;line-height:1.4;font-weight:760;letter-spacing:.02em;text-transform:uppercase;color:#1d4ed8;">' + assistant_ui_inline_markup(label, _runtime=_runtime) + '</p>')
            for p in paragraphs:
                article.append('<p style="margin:0 0 7px 0;color:#374151;">' + assistant_ui_inline_markup(p, _runtime=_runtime) + '</p>')
            if items:
                article.append(assistant_ui_render_numbered_cards(items, is_en=is_en, _runtime=_runtime))
            article.append('</section>')
        else:
            for idx, p in enumerate(paragraphs):
                looks_title = first_body and idx == 0 and (len(p) <= 105) and (not re.search('[.!?;]$', p))
                if looks_title:
                    article.append('<h3 style="margin:0 0 9px 0;font-size:15px;line-height:1.45;font-weight:760;color:#111827;">' + assistant_ui_inline_markup(p, _runtime=_runtime) + '</h3>')
                else:
                    color = '#4b5563' if first_body and idx == 0 else '#374151'
                    article.append('<p style="margin:0 0 11px 0;color:' + color + ';">' + assistant_ui_inline_markup(p, _runtime=_runtime) + '</p>')
                first_body = False
        if kind != 'heading':
            first_body = False
    article.append('</article>')
    rendered = ''.join(article)
    return rendered if len(rendered) <= _runtime.assistant_ui_max_html_chars else ''

def assistant_ui_root_cause_html(resp: dict, *, response_language: str, _runtime: ResponsePresentationRuntime) -> str:
    """Render validated Root Cause fields using the same clean visual language as ASK.

    LINK and FONTI remain separate Bubble sections. This function renders only the
    diagnostic answer body and never adds anchors, source labels or scores.
    """
    if not isinstance(resp, dict):
        return ''
    is_en = str(response_language or 'it').strip().lower().startswith('en')
    status = str(resp.get('status') or 'answered').strip().lower()
    summary = str(resp.get('problem_summary') or resp.get('symptom') or '').strip()
    causes = [c for c in resp.get('possible_causes') or [] if isinstance(c, dict)]
    recommended = _runtime.unique_non_empty_strings(resp.get('recommended_next_checks') or [], limit=10)
    article: list[str] = ['<article data-mm-answer="root-cause" data-mm-render="' + assistant_ui_escape(_runtime.assistant_ui_render_version, _runtime=_runtime) + '" style="box-sizing:border-box;width:100%;font-family:Inter,-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif;font-size:14px;line-height:1.62;color:#1f2937;overflow-wrap:anywhere;padding:2px 2px 1px 2px;">']
    if status != 'answered' or not causes:
        title = 'Analysis unavailable' if is_en else 'Analisi non disponibile'
        fallback = summary or ('I cannot find enough machine evidence to propose reliable probable causes.' if is_en else 'Non trovo evidenze sufficienti della macchina per proporre cause probabili affidabili.')
        article.append('<h2 style="margin:0 0 9px 0;font-size:18px;line-height:1.35;font-weight:760;color:#111827;">' + assistant_ui_escape(title, _runtime=_runtime) + '</h2>')
        article.append('<p style="margin:0;color:#4b5563;">' + assistant_ui_inline_markup(fallback, _runtime=_runtime) + '</p>')
        article.append('</article>')
        rendered = ''.join(article)
        return rendered if len(rendered) <= _runtime.assistant_ui_max_html_chars else ''
    title = 'Probable causes' if is_en else 'Cause probabili'
    article.append('<h2 style="margin:0 0 8px 0;font-size:18px;line-height:1.35;font-weight:760;color:#111827;">' + assistant_ui_escape(title, _runtime=_runtime) + '</h2>')
    if summary:
        problem_label = 'Problem' if is_en else 'Problema'
        article.append('<p style="margin:0 0 18px 0;color:#4b5563;"><strong style="color:#111827;">' + assistant_ui_escape(problem_label, _runtime=_runtime) + ':</strong> ' + assistant_ui_inline_markup(summary, _runtime=_runtime) + '</p>')
    flattened_checks: list[str] = []
    for idx, cause in enumerate(causes, start=1):
        rank = _runtime.safe_int(cause.get('rank'), idx)
        cause_text = str(cause.get('cause') or '').strip()
        why = str(cause.get('why') or '').strip()
        checks = _runtime.unique_non_empty_strings(cause.get('checks') or [], limit=6)
        flattened_checks.extend(checks)
        if not cause_text:
            continue
        if rank == 1:
            rank_label = 'Most likely cause' if is_en else 'Causa più probabile'
            border = '#2563eb'
            heading_color = '#1d4ed8'
        else:
            rank_label = f'Possible cause {rank}' if is_en else f'Causa possibile {rank}'
            border = '#cbd5e1'
            heading_color = '#334155'
        article.append('<section style="margin:0 0 20px 0;padding-left:14px;border-left:3px solid ' + border + ';"><p style="margin:0 0 4px 0;font-size:12px;line-height:1.4;font-weight:760;letter-spacing:.02em;text-transform:uppercase;color:' + heading_color + ';">' + assistant_ui_escape(rank_label, _runtime=_runtime) + '</p><h3 style="margin:0 0 7px 0;font-size:15px;line-height:1.45;font-weight:760;color:#111827;">' + assistant_ui_inline_markup(cause_text, _runtime=_runtime) + '</h3>')
        if why:
            why_label = 'Why' if is_en else 'Perché'
            article.append('<p style="margin:0 0 9px 0;color:#4b5563;"><strong style="color:#374151;">' + assistant_ui_escape(why_label, _runtime=_runtime) + ':</strong> ' + assistant_ui_inline_markup(why, _runtime=_runtime) + '</p>')
        if checks:
            checks_label = 'Checks' if is_en else 'Controlli consigliati'
            article.append('<p style="margin:0 0 6px 0;font-weight:720;color:#111827;">' + assistant_ui_escape(checks_label, _runtime=_runtime) + '</p>')
            article.append('<ol style="margin:0;padding-left:21px;color:#374151;">')
            for check in checks:
                article.append('<li style="margin:0 0 7px 0;padding-left:2px;">' + assistant_ui_inline_markup(check, _runtime=_runtime) + '</li>')
            article.append('</ol>')
        article.append('</section>')
    seen_checks = {re.sub('\\s+', ' ', _runtime.normalize_unicode(x or '').lower()).strip() for x in flattened_checks if str(x or '').strip()}
    extra_checks = [x for x in recommended if re.sub('\\s+', ' ', _runtime.normalize_unicode(x or '').lower()).strip() not in seen_checks]
    if extra_checks:
        final_label = 'Additional priority checks' if is_en else 'Ulteriori controlli prioritari'
        article.append('<section style="margin:0 0 18px 0;padding:10px 14px;border-left:3px solid #16a34a;background:#f0fdf4;border-radius:0 7px 7px 0;"><h3 style="margin:0 0 7px 0;font-size:14px;font-weight:760;color:#166534;">' + assistant_ui_escape(final_label, _runtime=_runtime) + '</h3><ol style="margin:0;padding-left:21px;color:#166534;">')
        for check in extra_checks[:6]:
            article.append('<li style="margin:0 0 6px 0;padding-left:2px;">' + assistant_ui_inline_markup(check, _runtime=_runtime) + '</li>')
        article.append('</ol></section>')
    article.append('</article>')
    rendered = ''.join(article)
    return rendered if len(rendered) <= _runtime.assistant_ui_max_html_chars else ''

def assistant_ui_normalize_url_for_key(value: str, *, _runtime: ResponsePresentationRuntime) -> str:
    url = str(value or '').strip()
    if not url:
        return ''
    url = re.sub('#page=\\d+.*$', '', url, flags=re.IGNORECASE)
    return url.rstrip('/')

def assistant_ui_dedupe_links(items: list[dict], *, max_items: int, _runtime: ResponsePresentationRuntime) -> list[dict]:
    """Remove duplicate Bubble links without changing their structure or buttons.

    Documents are unique per file+page; structured sources are unique per Bubble
    object. The first item keeps the existing retrieval priority and display label.
    """
    out: list[dict] = []
    seen: set[tuple] = set()
    for item in items or []:
        if not isinstance(item, dict):
            continue
        bdid = str(item.get('bubble_document_id') or '').strip()
        source_type = str(item.get('source_type') or _runtime.source_type_from_document_id(bdid)).strip().lower()
        role = str(item.get('evidence_role') or source_type or 'document').strip().lower()
        url = str(item.get('url') or '').strip()
        base = assistant_ui_normalize_url_for_key(url, _runtime=_runtime)
        page = _runtime.safe_int(item.get('page_from'), 0)
        is_structured = bool(item.get('is_structured_source')) or source_type in _runtime.structured_source_types or role in {'procedure', 'step', 'ps', 'md_photo', 'md_video', 'structured'}
        if is_structured:
            key = ('structured', bdid or base or str(item.get('source_id') or '').strip())
        else:
            key = ('document', base or bdid, page)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= max(1, int(max_items or 1)):
            break
    return out

def assistant_ui_dedupe_citations(items: list[dict], *, max_items: int, _runtime: ResponsePresentationRuntime) -> list[dict]:
    """Compact the visible FONTI list while preserving the existing data contract."""
    out: list[dict] = []
    seen: set[tuple] = set()
    for item in items or []:
        if not isinstance(item, dict):
            continue
        bdid = str(item.get('bubble_document_id') or '').strip()
        source_type = str(item.get('source_type') or _runtime.source_type_from_document_id(bdid)).strip().lower()
        role = str(item.get('evidence_role') or source_type or 'document').strip().lower()
        p1 = _runtime.safe_int(item.get('page_from'), 0)
        p2 = _runtime.safe_int(item.get('page_to'), p1)
        is_structured = bool(item.get('is_structured_source')) or source_type in _runtime.structured_source_types or role in {'procedure', 'step', 'ps', 'md_photo', 'md_video', 'structured'}
        key = ('structured', bdid) if is_structured else ('document', bdid, p1, p2)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= max(1, int(max_items or 1)):
            break
    return out

def assistant_ui_visible_text_from_html(value: str, *, _runtime: ResponsePresentationRuntime) -> str:
    raw = re.sub('(?is)<(?:script|style)\\b[^>]*>.*?</(?:script|style)>', ' ', str(value or ''))
    raw = re.sub('(?s)<[^>]+>', ' ', raw)
    raw = html.unescape(raw)
    return re.sub('\\s+', ' ', raw).strip()

def assistant_ui_canonical_tokens(value: str, *, _runtime: ResponsePresentationRuntime) -> list[str]:
    raw = html.unescape(str(value or ''))
    raw = re.sub('(?m)^\\s*\\d{1,3}[.)]\\s+', ' ', raw)
    raw = re.sub('(?m)^\\s*[-•☐]\\s*', ' ', raw)
    normalized = _runtime.normalize_unicode(raw).casefold()
    return re.findall('[a-z0-9]+(?:[.,][0-9]+)?', normalized)

def assistant_ui_token_coverage(reference: str, candidate: str, *, _runtime: ResponsePresentationRuntime) -> float:
    """Multiset token recall of the canonical text in the rendered visible text."""
    from collections import Counter
    ref = Counter(assistant_ui_canonical_tokens(reference, _runtime=_runtime))
    if not ref:
        return 1.0
    got = Counter(assistant_ui_canonical_tokens(candidate, _runtime=_runtime))
    matched = sum((min(count, got.get(token, 0)) for token, count in ref.items()))
    return matched / max(1, sum(ref.values()))

def assistant_ui_lossless_html(answer: str, *, response_language: str, status: str='answered', _runtime: ResponsePresentationRuntime) -> str:
    """Lossless ASK fallback using the same visual tokens as Root Cause."""
    clean = assistant_ui_normalize_markdown_tables(_runtime.strip_inline_citation_markers_for_display(str(answer or '')), _runtime=_runtime).strip()
    if not clean:
        return ''
    is_en = str(response_language or 'it').strip().lower().startswith('en')
    status_low = str(status or 'answered').strip().lower()
    chunks = ['<article data-mm-answer="lossless" data-mm-render="' + assistant_ui_escape(_runtime.assistant_ask_ui_render_version, _runtime=_runtime) + '" data-mm-visual="root-parity-v1" style="box-sizing:border-box;width:100%;font-family:Inter,-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif;font-size:14px;line-height:1.62;color:#1f2937;overflow-wrap:anywhere;padding:2px 2px 1px 2px;">', '<h2 style="margin:0 0 8px 0;font-size:18px;line-height:1.35;font-weight:760;color:#111827;">' + assistant_ui_escape('Answer' if is_en else 'Risposta', _runtime=_runtime) + '</h2>']
    if status_low != 'answered':
        if status_low in {'no_sources', 'needs_clarification'}:
            tone, background, border = ('#92400e', '#fffbeb', '#d97706')
        else:
            tone, background, border = ('#991b1b', '#fef2f2', '#dc2626')
        chunks.append('<section style="margin:0 0 18px 0;padding:10px 14px;border-left:3px solid ' + border + ';background:' + background + ';border-radius:0 7px 7px 0;color:' + tone + ';"><p style="margin:0;font-size:12px;line-height:1.4;font-weight:760;letter-spacing:.02em;text-transform:uppercase;color:' + tone + ';">' + assistant_ui_escape('Result' if is_en else 'Esito', _runtime=_runtime) + '</p></section>')
    list_kind = ''
    section_open = False

    def open_section(label: str) -> None:
        nonlocal section_open
        if section_open:
            close_section()
        chunks.append('<section style="margin:0 0 20px 0;padding-left:14px;border-left:3px solid #2563eb;"><p style="margin:0 0 7px 0;font-size:12px;line-height:1.4;font-weight:760;letter-spacing:.02em;text-transform:uppercase;color:#1d4ed8;">' + assistant_ui_inline_markup(label, _runtime=_runtime) + '</p>')
        section_open = True

    def close_list() -> None:
        nonlocal list_kind
        if list_kind:
            chunks.append(f'</{list_kind}>')
            list_kind = ''

    def close_section() -> None:
        nonlocal section_open
        close_list()
        if section_open:
            chunks.append('</section>')
            section_open = False
    for raw in clean.replace('\r', '\n').split('\n'):
        line = str(raw or '').strip()
        if not line:
            close_section()
            continue
        heading = re.fullmatch('\\*\\*(.+?)\\*\\*', line)
        numbered = re.match(r'^(\d{1,3})[.)]\s+(.+)$', line)
        bullet = re.match('^[-•]\\s+(.+)$', line)
        if heading:
            open_section(heading.group(1).strip())
        elif numbered:
            if not section_open:
                open_section('Steps' if is_en else 'Passaggi')
            if list_kind != 'ol':
                close_list()
                list_kind = 'ol'
                chunks.append('<ol style="margin:0;padding-left:21px;color:#374151;">')
            explicit_number = int(numbered.group(1))
            chunks.append(f'<li value="{explicit_number}" style="margin:0 0 7px 0;padding-left:2px;">' + assistant_ui_inline_markup(numbered.group(2).strip(), _runtime=_runtime) + '</li>')
        elif bullet:
            if not section_open:
                open_section('Key points' if is_en else 'Punti principali')
            if list_kind != 'ul':
                close_list()
                list_kind = 'ul'
                chunks.append('<ul style="margin:0;padding-left:20px;color:#374151;">')
            chunks.append('<li style="margin:0 0 8px 0;padding-left:2px;">' + assistant_ui_inline_markup(bullet.group(1).strip(), _runtime=_runtime) + '</li>')
        else:
            close_list()
            chunks.append('<p style="margin:0 0 10px 0;color:#374151;">' + assistant_ui_inline_markup(line, _runtime=_runtime) + '</p>')
    close_section()
    chunks.append('</article>')
    return ''.join(chunks)

def assistant_ui_finalize_response(resp: dict, *, language: str='it', _runtime: ResponsePresentationRuntime) -> dict:
    """Create one canonical answer and prove that the rendered body is lossless."""
    if not isinstance(resp, dict):
        return resp
    out = dict(resp)
    if isinstance(out.get('rg_links'), list):
        out['rg_links'] = assistant_ui_dedupe_links(out.get('rg_links') or [], max_items=max(1, int(_runtime.ask_ui_structured_max_links or _runtime.ask_ui_max_links or 14)), _runtime=_runtime)
    if isinstance(out.get('citations'), list):
        out['citations'] = assistant_ui_dedupe_citations(out.get('citations') or [], max_items=max(1, int(_runtime.ask_ui_structured_max_citations or _runtime.ask_ui_max_citations or 14)), _runtime=_runtime)
    status = str(out.get('status') or 'answered').strip().lower()
    effective_mode = str(out.get('effective_mode') or '').strip().lower()
    ui_model = out.get('_assistant_ui_model')
    is_procedure_model = isinstance(ui_model, dict) and str(ui_model.get('kind') or '').strip().lower() == 'procedure'
    is_native_root_cause = effective_mode == 'root_cause' and (isinstance(out.get('possible_causes'), list) or 'problem_summary' in out or isinstance(out.get('recommended_next_checks'), list))
    if is_native_root_cause:
        canonical_text = assistant_ui_root_cause_text(out, response_language=language, _runtime=_runtime)
        out['answer'] = canonical_text
        rendered = assistant_ui_root_cause_html(out, response_language=language, _runtime=_runtime) if canonical_text else ''
        mode = 'root_cause'
    else:
        canonical_text = str(out.get('answer') or '').strip()
        out['answer'] = canonical_text
        if canonical_text and is_procedure_model:
            rendered = procedure_ui_model_to_html(
                ui_model,
                links=out.get('rg_links') if isinstance(out.get('rg_links'), list) else [],
                response_language=language,
                _runtime=_runtime,
            )
            mode = 'procedure'
        else:
            rendered = assistant_ui_generic_html(
                canonical_text,
                links=out.get('rg_links') if isinstance(out.get('rg_links'), list) else [],
                citations=out.get('citations') if isinstance(out.get('citations'), list) else [],
                response_language=language,
                status=status,
                _runtime=_runtime,
            ) if canonical_text else ''
            mode = 'ask'
    visible = assistant_ui_visible_text_from_html(rendered, _runtime=_runtime)
    coverage = assistant_ui_token_coverage(canonical_text, visible, _runtime=_runtime) if canonical_text else 1.0
    renderer_fallback_used = False
    if canonical_text and coverage < 0.97:
        lossless = assistant_ui_lossless_html(canonical_text, response_language=language, status=status, _runtime=_runtime)
        lossless_visible = assistant_ui_visible_text_from_html(lossless, _runtime=_runtime)
        lossless_coverage = assistant_ui_token_coverage(canonical_text, lossless_visible, _runtime=_runtime)
        if lossless and lossless_coverage >= coverage:
            rendered = lossless
            visible = lossless_visible
            coverage = lossless_coverage
            renderer_fallback_used = True
    canonical_passed = bool(not canonical_text or coverage >= 0.97)
    out.pop('_assistant_ui_model', None)
    if rendered and canonical_passed:
        out['answer_html'] = rendered
        out['answer_format'] = 'html'
    else:
        out.pop('answer_html', None)
        out['answer_format'] = 'text'
    selected_render_version = (
        _runtime.assistant_ask_ui_render_version
        if renderer_fallback_used
        else _runtime.assistant_ui_render_version
        if mode in {'root_cause', 'procedure'}
        else _runtime.assistant_ask_ui_render_version
    )
    out['answer_render_version'] = selected_render_version
    meta = dict(out.get('meta') or {})
    meta.update({'answer_render_version': selected_render_version, 'answer_html_safe_template': True, 'answer_html_body_only': True, 'answer_html_contains_sources': False, 'answer_html_mode': mode, 'canonical_final_answer': canonical_passed, 'canonical_text_chars': len(canonical_text), 'answer_html_token_coverage': round(float(coverage), 6), 'answer_html_renderer_fallback_used': renderer_fallback_used})
    if not canonical_passed:
        meta['cacheable'] = False
        meta['semantic_cacheable'] = False
        meta['canonical_failure_reason'] = 'answer_html_token_coverage_below_0.97'
    out['meta'] = meta
    return out

def format_structured_procedure_answer_for_ui(*, structured_citations: list[dict], manual_support_citations: list[dict], grounded_points: list[dict], response_language: str, q: str='', _runtime: ResponsePresentationRuntime) -> str:
    model = build_structured_procedure_ui_model(structured_citations=structured_citations, manual_support_citations=manual_support_citations, grounded_points=grounded_points, response_language=response_language, q=q, _runtime=_runtime)
    return procedure_ui_model_to_text(model, response_language=response_language, _runtime=_runtime)
