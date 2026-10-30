"""Numeric dimensions from the existing, bounded PDF cell serialization.

This is representation parsing, not semantic entailment or evidence authority.
It never edits source text, converts values, or assigns a unit across columns.
"""
import json
import re


_LINE = re.compile(r'^- \[PDF_TABLE_(HEADER|RECORD) rows=(\d+)(?:-(\d+))? columns=(\d+)\] (.*)$')
_COL = re.compile(r'c(\d+)(?:-(\d+))?\s+')
_ROW = re.compile(r'r(\d+)(?:-(\d+))?=')
_DECODER = json.JSONDecoder()


def _cells(payload, base, first, last, columns):
    cells, pos = [], 0
    while pos < len(payload):
        col = _COL.match(payload, pos)
        if not col:
            return None
        left, right = int(col[1]), int(col[2] or col[1])
        pos = col.end()
        label = None
        if payload[pos:pos + 1] == '"':
            label, pos = _DECODER.raw_decode(payload, pos)
            if type(label) is not str or payload[pos:pos + 1] != ' ':
                return None
            pos += 1
        row = _ROW.match(payload, pos)
        if not row:
            return None
        top, bottom = int(row[1]), int(row[2] or row[1])
        start = row.end()
        value, pos = _DECODER.raw_decode(payload, start)
        if (type(value) is not str or not 1 <= left <= right <= columns
                or not first <= top <= bottom <= last):
            return None
        if any(left <= c['right'] and c['left'] <= right
               and top <= c['bottom'] and c['top'] <= bottom for c in cells):
            return None
        cells.append(dict(left=left, right=right, top=top, bottom=bottom,
                          label=label, value=value, span=(base + start, base + pos)))
        if pos == len(payload):
            break
        if payload[pos:pos + 3] != ' | ':
            return None
        pos += 3
    return cells


def numeric_cells(text, *, unit_token, number_atom, normalize_unit):
    """Return typed pairs, owned numeric spans and conflicting-cell spans.

    Repeated cell labels are self-contained. Otherwise inheritance requires a
    contiguous table, identical column count and an explicit unit-only heading
    relationship. Prose, row restart, malformed layout and unassigned text end
    that relationship. Only whole numeric/range/comparison cells are projected.
    """
    unit = re.compile(rf'(?<![\w])(?P<unit>{unit_token})(?![\w])', re.IGNORECASE)
    prefix = re.compile(rf'^\s*\[\s*(?P<unit>{unit_token})\s*\]\s*', re.IGNORECASE)
    number = re.compile(number_atom)
    numeric = re.compile(
        rf'^\s*(?:(?:minore di|maggiore di|inferiore a|superiore a|less than|greater than|'
        rf'below|above|under|over|da|from|[<>≤≥]=?)\s*)?'
        rf'(?P<first>{number_atom})(?:\s*(?:to|a|and|e|[-–—÷])\s*'
        rf'(?P<last>{number_atom}))?\s*(?P<unit>{unit_token})?\s*$', re.IGNORECASE)

    def heading(value):
        matches = list(unit.finditer(value))
        dimensions = {normalize_unit(m['unit']) for m in matches}
        # A heading with a numeric value is not a unit declaration. Digits in
        # a unit such as mm2 or min-1 are excluded from this check.
        remainder, end = [], 0
        for match in matches:
            remainder.append(value[end:match.start()]); end = match.end()
        remainder.append(value[end:])
        if len(dimensions) != 1 or number.search(''.join(remainder)):
            return None
        return matches[0]['unit']

    typed, owned, blocked = [], [], []
    headings, count, previous = {}, 0, 0
    offset = 0
    for raw_line in str(text or '').splitlines(keepends=True):
        line = raw_line.rstrip('\r\n')
        match = _LINE.fullmatch(line)
        if not match:
            if line.strip():
                headings, count, previous = {}, 0, 0
            offset += len(raw_line)
            continue
        role, first, last, columns = match[1], int(match[2]), int(match[3] or match[2]), int(match[4])
        if not 1 <= first <= last <= 100000 or not 1 <= columns <= 256:
            headings, count, previous = {}, 0, 0
            offset += len(raw_line)
            continue
        try:
            cells = _cells(match[5], offset + match.start(5), first, last, columns)
        except (ValueError, TypeError):
            cells = None
        if not cells:
            headings, count, previous = {}, 0, 0
            offset += len(raw_line)
            continue
        if first == 1 or columns != count or first != previous + 1:
            headings = {}
        count, previous = columns, last
        # HEADER is produced only for proven geometric/typographic headings.
        # A plain first RECORD can also explicitly declare dimensions, as in
        # "Oil temperature °C" / "Interval (h)"; it must contain no quantities.
        header = (role == 'HEADER' or (first == last == 1
                  and all(not number.search(unit.sub('', c['value'])) for c in cells)))
        if header:
            proposed, ambiguous = {}, set()
            for cell in cells:
                dimension = heading(cell['value'])
                if cell['left'] != cell['right']:
                    ambiguous.update(range(cell['left'], cell['right'] + 1))
                elif dimension:
                    col = cell['left']
                    if col in proposed and normalize_unit(proposed[col]) != normalize_unit(dimension):
                        ambiguous.add(col)
                    proposed[col] = dimension
            headings = {col: value for col, value in proposed.items() if col not in ambiguous}
        else:
            for cell in cells:
                if cell['left'] != cell['right']:
                    continue
                inherited = headings.get(cell['left'])
                label_conflict = False
                if cell['label'] is not None:
                    local = heading(cell['label'])
                    if inherited and local and normalize_unit(inherited) != normalize_unit(local):
                        label_conflict = True
                    # An explicit label is authoritative for its own cell;
                    # missing/ambiguous units cannot borrow a prior heading.
                    inherited = local
                if not inherited:
                    continue
                body = cell['value'].strip()
                before = prefix.match(body)
                if before:
                    body = body[before.end():]
                values = numeric.fullmatch(body)
                if not values:
                    continue
                explicit = [x for x in (before['unit'] if before else None, values['unit']) if x]
                if label_conflict or any(normalize_unit(x) != normalize_unit(inherited) for x in explicit):
                    blocked.append(cell['span'])
                    continue
                owned.append(cell['span'])
                typed.extend((values[k], inherited) for k in ('first', 'last') if values[k])
        offset += len(raw_line)
    return typed, owned, blocked
