"""Numeric dimensions from the existing, bounded PDF cell serialization.

This is representation parsing, not semantic entailment or evidence authority.
It never edits source text or converts values. Cross-column dimensions require
an explicit, unmerged label/value/unit row; proximity alone is not ownership.
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


def decoded_value_cells(text):
    """Return literal value/span pairs only from valid serialized table rows.

    Each value is an independent parser view. It never joins adjacent cells,
    inherits a heading, or changes the admitted source serialization.
    """
    result, offset = [], 0
    for raw_line in str(text or '').splitlines(keepends=True):
        match = _LINE.fullmatch(raw_line.rstrip('\r\n'))
        if match:
            first, last, columns = int(match[2]), int(match[3] or match[2]), int(match[4])
            if 1 <= first <= last <= 100000 and 1 <= columns <= 256:
                try:
                    cells = _cells(match[5], offset + match.start(5), first, last, columns)
                except (ValueError, TypeError):
                    cells = None
                result.extend((cell['value'], cell['span']) for cell in (cells or ()))
        offset += len(raw_line)
    return result


def numeric_cells(text, *, unit_token, number_atom, normalize_unit):
    """Return typed pairs, owned numeric spans and conflicting-cell spans.

    Repeated cell labels are self-contained. Otherwise inheritance requires a
    contiguous table, identical column count and an explicit unit-only heading
    relationship. Prose, row restart, malformed layout and unassigned text end
    that relationship. A separate unit cell is admitted only in a complete
    three-column label/value/unit row. Whole numeric/range/comparison cells and
    bounded numeric alternatives are projected; footnotes stay in source text
    for independent semantic review, rather than becoming numeric permission.
    """
    unit = re.compile(rf'(?<![\w])(?P<unit>{unit_token})(?![\w])', re.IGNORECASE)
    prefix = re.compile(rf'^\s*\[\s*(?P<unit>{unit_token})\s*\]\s*', re.IGNORECASE)
    number = re.compile(number_atom)
    numeric = re.compile(
        rf'^\s*(?:(?:minore di|maggiore di|inferiore a|superiore a|less than|greater than|'
        rf'below|above|under|over|da|from|[<>≤≥]=?)\s*)?'
        rf'(?P<first>{number_atom})(?:\s*(?:to|a|and|e|[-–—÷])\s*'
        rf'(?P<last>{number_atom}))?\s*(?P<unit>{unit_token})?\s*$', re.IGNORECASE)
    pure_unit = re.compile(rf'^\s*(?P<bracket>\[\s*)?(?P<unit>{unit_token})'
                           rf'(?(bracket)\s*\])\s*$', re.IGNORECASE)
    option = re.compile(rf'^\s*(?P<value>{number_atom})\s*(?:[*†‡])?\s*$')
    bracketed_unit = re.compile(rf'\[\s*(?P<unit>{unit_token})\s*\]', re.IGNORECASE)

    def quantities(body):
        before = prefix.match(body)
        body = body[before.end():] if before else body
        values = numeric.fullmatch(body)
        if values:
            return ([values[k] for k in ('first', 'last') if values[k]],
                    [x for x in (before['unit'] if before else None, values['unit']) if x])
        # Slash alternatives require whitespace on both sides. Neither
        # arithmetic (160/2, 160*2) nor partly numeric designations (500WF)
        # establish a dimensional value. A single footnote marker is literal
        # source notation; it does not remove or approve its qualification.
        parts = re.split(r'\s+/\s+', body)
        if not 1 <= len(parts) <= 64:
            return None
        parsed = [option.fullmatch(part) for part in parts]
        if not all(parsed) or (len(parts) == 1 and not re.search(r'[*†‡]\s*$', body)):
            return None
        return ([part['value'] for part in parsed], [before['unit']] if before else [])

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
            row_units = {}
            # Require the entire canonical row, exact geometry and a unit-only
            # third cell. A neighbouring word, merged cell or another row/table
            # can never lend its unit to a bare number. Bracketed dimensions in
            # the row label are additional consistency checks, not inference.
            if (columns == 3 and first == last and len(cells) == 3
                    and all(c['left'] == c['right'] and c['top'] == c['bottom'] == first
                            for c in cells)):
                by_column = {c['left']: c for c in cells}
                if set(by_column) == {1, 2, 3}:
                    label, value, unit_cell = (by_column[i] for i in (1, 2, 3))
                    dimension = pure_unit.fullmatch(unit_cell['value'])
                    # nm and N m share the legacy normalizer. Do not newly
                    # infer that ambiguous dimension from separate row cells.
                    if (dimension and normalize_unit(dimension['unit']) != 'nm'
                            and re.search(r'[^\W\d_]', label['value'], re.UNICODE)
                            and not pure_unit.fullmatch(label['value'])):
                        row_units[value['left']] = (dimension['unit'],
                            [m['unit'] for m in bracketed_unit.finditer(label['value'])])
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
                row_dimension = row_units.get(cell['left'])
                if row_dimension:
                    row_unit, label_units = row_dimension
                    if ((inherited and normalize_unit(inherited) != normalize_unit(row_unit))
                            or any(normalize_unit(x) != normalize_unit(row_unit) for x in label_units)):
                        label_conflict = True
                    inherited = row_unit
                if not inherited:
                    continue
                body = cell['value'].strip()
                values = quantities(body)
                if values is None:
                    continue
                numbers, explicit = values
                if label_conflict or any(normalize_unit(x) != normalize_unit(inherited) for x in explicit):
                    blocked.append(cell['span'])
                    continue
                owned.append(cell['span'])
                typed.extend((raw, inherited) for raw in numbers)
        offset += len(raw_line)
    return typed, owned, blocked
