"""Deterministic ruled-table geometry; no inferred values or domain knowledge."""
from __future__ import annotations

from collections import defaultdict
import math
import json
import re
import unicodedata


class PdfTableLayoutError(ValueError):
    """A text-bearing table cannot safely be projected into row/column relations."""


def _rect(value):
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__") or len(value) != 4:
        raise PdfTableLayoutError("table_rectangle_invalid")
    result = tuple(float(x) for x in value)
    if not all(math.isfinite(x) for x in result) or result[0] >= result[2] or result[1] >= result[3]:
        raise PdfTableLayoutError("table_rectangle_invalid")
    return result


def _inside(rect, x, y):
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def _overlap(a, b):
    return min(a[2], b[2]) - max(a[0], b[0]) > .5 and min(a[3], b[3]) - max(a[1], b[1]) > .5


def _contains(outer, inner):
    return (outer[0] <= inner[0] + .5 and outer[1] <= inner[1] + .5
            and outer[2] >= inner[2] - .5 and outer[3] >= inner[3] - .5)


# These failures describe uncertain relationships in a detector candidate, not
# unreadable PDF text. Quarantine their text instead of fabricating a table or
# making an unrelated page fail. Invalid rectangles and detector/runtime errors
# remain fatal: there is no reliable region to quarantine in those cases.
_QUARANTINABLE_LAYOUT_ERRORS = frozenset({
    "table_boundary_ambiguous", "table_grid_dimensions_ambiguous",
    "table_grid_extent_ambiguous", "table_cells_overlap",
    "table_grid_has_unresolved_cells", "table_word_cell_ambiguous",
})


def _unassigned_region(bbox, words):
    members = [(i, w) for i, w in enumerate(words) if _overlap(bbox, w[:4])]
    lines = defaultdict(list)
    for _, word in members:
        lines[word[5], word[6]].append(word)
    # Repeat the warning for each physical line so it survives page cleaning and
    # chunk boundaries. No inferred rows, columns, units or diagram connections.
    parts = [
        '- [PDF_TABLE_UNASSIGNED relationship=ambiguous] '
        'Text in an unresolved PDF layout; no row/column relationship or diagram '
        'connection is asserted: '
        + json.dumps(" ".join(_word_text(line).split()), ensure_ascii=False)
        for line in sorted(lines.values(), key=lambda line: (min(w[1] for w in line), min(w[0] for w in line)))
    ]
    return bbox, "\n\n".join(parts), {i for i, _ in members}


def _bounds(values):
    groups = []
    for value in sorted(values):
        if not groups or value - groups[-1][0] > .5:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [sum(group) / len(group) for group in groups]


def _coordinate(values, value):
    found = [i for i, boundary in enumerate(values) if abs(boundary - value) <= .5]
    if len(found) != 1:
        raise PdfTableLayoutError("table_boundary_ambiguous")
    return found[0]


def _word_text(words):
    lines = defaultdict(list)
    for word in words:
        lines[(word[5], word[6])].append(word)
    ordered = sorted(lines.values(), key=lambda row: (min(w[1] for w in row), min(w[0] for w in row)))
    return "\n".join(" ".join(str(w[4]) for w in sorted(row, key=lambda w: (w[0], w[7]))) for row in ordered)


def _text_key(value):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))


def _table(table, words, spans=(), images=()):
    bbox = _rect(table.bbox)
    cells = sorted({_rect(cell) for cell in table.cells if cell is not None}, key=lambda c: (c[1], c[0]))
    if not cells:
        return None
    # A bounding frame containing other rectangles is not a partition into
    # cells. It commonly encloses technical drawings and page footers.
    if any(all(cell[0] <= other[0]+.5 and cell[1] <= other[1]+.5
               and cell[2] >= other[2]-.5 and cell[3] >= other[3]-.5
               for other in cells) for cell in cells) and len(cells) > 1:
        return None
    xs = _bounds([v for cell in cells for v in (cell[0], cell[2])])
    ys = _bounds([v for cell in cells for v in (cell[1], cell[3])])
    # Ignore empty ruled drawings/frames. Text-bearing grids retain their exact
    # geometry even when their subject is not a conventional data table.
    members = [(i, w) for i, w in enumerate(words) if _inside(bbox, (w[0]+w[2])/2, (w[1]+w[3])/2)]
    occupied_text_cells = [cell for cell in cells if any(
        any(ch.isalnum() for ch in str(w[4]))
        and _inside(cell, (w[0]+w[2])/2, (w[1]+w[3])/2) for _, w in members)]
    if (len(occupied_text_cells) < 3
            or len(_bounds([cell[0] for cell in occupied_text_cells])) < 2
            or len(_bounds([cell[1] for cell in occupied_text_cells])) < 2):
        return None
    if len(xs) != table.col_count + 1 or len(ys) != table.row_count + 1:
        raise PdfTableLayoutError("table_grid_dimensions_ambiguous")
    if (abs(xs[0]-bbox[0]) > .5 or abs(xs[-1]-bbox[2]) > .5
            or abs(ys[0]-bbox[1]) > .5 or abs(ys[-1]-bbox[3]) > .5):
        raise PdfTableLayoutError("table_grid_extent_ambiguous")
    grid, origins = {}, []
    for index, cell in enumerate(cells):
        left,right = _coordinate(xs,cell[0]),_coordinate(xs,cell[2])
        top,bottom = _coordinate(ys,cell[1]),_coordinate(ys,cell[3])
        if left >= right or top >= bottom:
            raise PdfTableLayoutError("table_span_invalid")
        for row in range(top,bottom):
            for col in range(left,right):
                if (row,col) in grid:
                    raise PdfTableLayoutError("table_cells_overlap")
                grid[row,col] = index
        origins.append((top,left,bottom-top,right-left))
    if len(grid) != table.row_count * table.col_count:
        holes = [(xs[col],ys[row],xs[col+1],ys[row+1])
                 for row in range(table.row_count) for col in range(table.col_count)
                 if (row,col) not in grid]
        image_area = sum(max(0,min(box[2],bbox[2])-max(box[0],bbox[0]))
                         * max(0,min(box[3],bbox[3])-max(box[1],bbox[1])) for box in images)
        if image_area > .05*(bbox[2]-bbox[0])*(bbox[3]-bbox[1]) and not any(
            _inside(hole,(w[0]+w[2])/2,(w[1]+w[3])/2) for hole in holes for _,w in members
        ):
            # Irregular picture-panel layouts are not tabular partitions.
            return None
        raise PdfTableLayoutError("table_grid_has_unresolved_cells")
    by_cell = defaultdict(list)
    unassigned = []
    owned = set()
    for index, word in members:
        center = ((word[0]+word[2])/2,(word[1]+word[3])/2)
        matches = [i for i,cell in enumerate(cells) if _inside(cell,*center)]
        if len(matches) != 1:
            raise PdfTableLayoutError("table_word_cell_ambiguous")
        cell = cells[matches[0]]
        if word[0] < cell[0]-1.5 or word[1] < cell[1]-1.5 or word[2] > cell[2]+1.5 or word[3] > cell[3]+1.5:
            # A PDF can contain occluded or overprinted text crossing a rule.
            # Preserve it explicitly without inventing its cell relationship.
            unassigned.append(word)
            owned.add(index)
            continue
        by_cell[matches[0]].append(word)
        owned.add(index)
    header = getattr(table,"header",None)
    header_rows = 0
    if header is not None and not getattr(header,"external",True):
        rect = _rect(header.bbox)
        if abs(rect[1]-ys[0]) <= .5:
            matches = [i for i,value in enumerate(ys) if abs(value-rect[3]) <= .5]
            if len(matches) == 1:
                # PyMuPDF assumes the first row is a header when it finds no
                # external heading. Require independent typographic contrast;
                # otherwise keep all rows as data with explicit column numbers.
                proposed = matches[0]
                heading = [s for s in spans if s.get("text", "").strip()
                           and _inside(bbox, (s["bbox"][0]+s["bbox"][2])/2,
                                       (s["bbox"][1]+s["bbox"][3])/2)
                           and s["bbox"][1] < ys[proposed]]
                body = [s for s in spans if s.get("text", "").strip()
                        and _inside(bbox, (s["bbox"][0]+s["bbox"][2])/2,
                                    (s["bbox"][1]+s["bbox"][3])/2)
                        and s["bbox"][1] >= ys[proposed]]
                if len({grid[0,col] for col in range(table.col_count)}) >= 2 and heading and all(s.get("flags", 0) & 16 for s in heading) and any(
                    not s.get("flags", 0) & 16 for s in body
                ) and all(top+height <= proposed for top,_,height,_ in origins if top < proposed):
                    header_rows = proposed

    def quoted(text):
        return json.dumps(" ".join(text.split()), ensure_ascii=False)

    text_by_cell = {i: _word_text(by_cell[i]) for i in range(len(cells))}
    labels = {}
    if header_rows:
        for col in range(table.col_count):
            indices = sorted({grid[row,col] for row in range(header_rows)})
            labels[col] = " / ".join(text_by_cell[i] for i in indices if text_by_cell[i])

    # A rowspan connects the corresponding physical rows into one atomic record.
    # Emit each original cell once, with its exact span, rather than filling down
    # missing cells. Bullet records are preserved by the existing sentence splitter
    # and each carries its column labels, so a chunk never needs a prior header.
    parts = []
    start = 0
    while start < table.row_count:
        end = start + 1
        while True:
            expanded = max([end] + [top+height for top,_,height,_ in origins if start <= top < end])
            if expanded == end:
                break
            end = expanded
        fields = []
        for index,(row,col,rowspan,colspan) in enumerate(origins):
            if not start <= row < end:
                continue
            name = f"c{col+1}" if colspan == 1 else f"c{col+1}-{col+colspan}"
            if labels and start >= header_rows:
                name += " " + quoted(" / ".join(labels[c] for c in range(col,col+colspan)))
            position = f"r{row+1}" if rowspan == 1 else f"r{row+1}-{row+rowspan}"
            fields.append(f"{name} {position}={quoted(text_by_cell[index])}")
        rows = str(start+1) if end == start+1 else f"{start+1}-{end}"
        role = "HEADER" if header_rows and end <= header_rows else "RECORD"
        parts.append(f"- [PDF_TABLE_{role} rows={rows} columns={table.col_count}] " + " | ".join(fields))
        start = end
    if unassigned:
        lines = defaultdict(list)
        for word in unassigned:
            lines[word[5],word[6]].append(word)
        for line in lines.values():
            possible = [i for i,cell in enumerate(cells) if any(_overlap(cell,w[:4]) for w in line)]
            rows = sorted({r+1 for i in possible for r in range(origins[i][0],origins[i][0]+origins[i][2])})
            cols = sorted({c+1 for i in possible for c in range(origins[i][1],origins[i][1]+origins[i][3])})
            parts.append('- [PDF_TABLE_UNASSIGNED relationship=ambiguous '
                         + 'candidate_rows='+','.join(map(str,rows))+' candidate_columns='+','.join(map(str,cols))
                         + '] Text crosses a cell boundary; no row/column relationship is asserted: '
                         + quoted(_word_text(line)))
    return bbox,"\n\n".join(parts),owned


def page_text_with_tables(page, blocks):
    """Return text with explicit cells, or None for the unchanged block fallback.

    PyMuPDF 1.24.9 is pinned by the application. Older injected page doubles with
    no table API retain their historical behavior. A failed real table detector
    remains fatal. Ambiguous candidate geometry is preserved as explicitly
    unassigned text, never projected into invented row/column relationships.
    """
    detector = getattr(page,"find_tables",None)
    if not callable(detector):
        return None
    rotation = getattr(page, "rotation", 0)
    try:
        # get_text uses unrotated PDF coordinates. The pinned detector can use
        # the displayed rotation instead; align both coordinate spaces before
        # assigning words to cells, then restore the in-memory page metadata.
        try:
            if rotation:
                page.set_rotation(0)
            tables = detector(strategy="lines_strict").tables
        finally:
            if rotation:
                page.set_rotation(rotation)
    except Exception as exc:
        raise PdfTableLayoutError("table_detection_failed") from exc
    candidates = [table for table in tables if table.row_count >= 2 and table.col_count >= 2]
    if not candidates:
        return None
    words = page.get_text("words") or []
    rich_blocks = page.get_text("dict").get("blocks", [])
    spans = [span for block in rich_blocks if block.get("type") == 0
             for line in block.get("lines", []) for span in line.get("spans", [])]
    images = [block["bbox"] for block in rich_blocks if block.get("type") == 1]
    accepted, ambiguous = [], []
    for table in candidates:
        try:
            result = _table(table,words,spans,images)
        except PdfTableLayoutError as exc:
            if str(exc) not in _QUARANTINABLE_LAYOUT_ERRORS:
                raise
            ambiguous.append(_rect(table.bbox))
            continue
        if result is None:
            continue
        if any(result == old for old, _ in accepted):
            continue
        if any(_overlap(result[0],old[0]) for old, _ in accepted):
            raise PdfTableLayoutError("table_regions_overlap")
        accepted.append((result, table))
    unresolved = []
    for bbox in ambiguous:
        # Underlines and boxed text can create a spurious subgrid inside a real
        # cell. The validated parent already owns its words and establishes only
        # that cell relationship; do not invent the contradictory child grid.
        region_words = _unassigned_region(bbox, words)[2]
        if any(region_words <= result[2] and any(
                cell is not None and _contains(_rect(cell), bbox)
                for cell in table.cells) for result, table in accepted):
            continue
        unresolved.append(bbox)
    # An uncertain region crossing a verified table cannot leave half of that
    # table's claims published. Quarantine the union, including all its words.
    # Merge overlapping uncertain regions to preserve each word exactly once.
    regions = [(result, False) for result, _ in accepted]
    regions += [(_unassigned_region(bbox, words), True) for bbox in unresolved]
    changed = True
    while changed:
        changed = False
        for i, (left, left_uncertain) in enumerate(regions):
            for j in range(i + 1, len(regions)):
                right, right_uncertain = regions[j]
                if not (left_uncertain or right_uncertain) or not (
                        _overlap(left[0], right[0]) or left[2] & right[2]):
                    continue
                a, b = left[0], right[0]
                union = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                regions[i] = (_unassigned_region(union, words), True)
                regions.pop(j)
                changed = True
                break
            if changed:
                break
    accepted = [_unassigned_region(result[0], words) if uncertain else result
                for result, uncertain in regions]
    if not accepted:
        return None
    owned = set().union(*(item[2] for item in accepted))
    items = [(bbox[1],bbox[0],text) for bbox,text,_ in accepted]
    for block in blocks:
        text = str(block[4] or "").strip()
        if not text:
            continue
        block_id = block[5]
        all_words = [(i,w) for i,w in enumerate(words) if w[5] == block_id]
        removed = [(i,w) for i,w in all_words if i in owned]
        if not removed:
            items.append((float(block[1]),float(block[0]),text))
            continue
        # No text can disappear just because a PDF block spans a table edge.
        original_word_order = sorted((w for _,w in all_words),key=lambda w:(w[6],w[7]))
        if _text_key(text) != _text_key("".join(str(w[4]) for w in original_word_order)):
            raise PdfTableLayoutError("table_block_text_not_fully_mapped")
        remaining = [w for i,w in all_words if i not in owned]
        if remaining:
            items.append((min(w[1] for w in remaining),min(w[0] for w in remaining),_word_text(remaining)))
    return "\n\n".join(text for _,_,text in sorted(items,key=lambda item:(item[0],item[1]))).strip()
