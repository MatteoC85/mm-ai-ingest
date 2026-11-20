"""Lossless request-local review labels. The original packet remains authoritative."""
from copy import deepcopy
from dataclasses import dataclass
import json
import re


_LABEL = re.compile(r"@[sft][0-9]+", re.IGNORECASE)
_SOURCE_HEADER = re.compile(r"^\[([^\]\r\n]+)\] source_type=", re.MULTILINE)
INSTRUCTIONS = (
    " In this review, source headers and coverage tables use short labels. Cite the @s "
    "labels in citation fields, including closure evidence. Follow the exact obligation "
    "text in FACET_LABELS and ANSWER_TYPE_LABELS, and report its @f or @t label in the "
    "corresponding coverage fields. Describe newly discovered omissions in full. "
    "Keep all labels out of the answer, edits and evidence quotes."
)
CHECKLIST_POSITIONS = (
    " Write each required checklist item's exact text once in expected_list_items. "
    "In covered_list_items and missing_list_items, give only the 1-based positions "
    "of independently checked items in expected_list_items. Example: if the first "
    "item is covered and the second missing, return [1] and [2]."
)


def _aliases(prefix, values):
    return tuple((value, f"@{prefix}{i}") for i, value in enumerate(dict.fromkeys(values), 1))


@dataclass(frozen=True, slots=True)
class FiniteReviewWire:
    sources: str
    source_labels: tuple[tuple[str, str], ...]
    facet_labels: tuple[tuple[str, str], ...]
    type_labels: tuple[tuple[str, str], ...]
    checklist_positions: bool = False

    def coverage(self, required_types, hard_requirements, required_facets, facet_contracts):
        facets = dict(self.facet_labels)
        types = dict(self.type_labels)
        fields = {
            "FACET_LABELS": {alias: label for label, alias in self.facet_labels},
            "ANSWER_TYPE_LABELS": {alias: label for label, alias in self.type_labels},
            "REQUIRED_ANSWER_TYPES": [types[label] for label in required_types],
            "HARD_REQUIREMENTS": [types[label] for label in hard_requirements],
            "REQUIRED_FACETS": [facets[label] for label in required_facets],
            "FACET_CONTRACTS": [
                {**item, "facet": facets[item["facet"]], "answer_type": types[item["answer_type"]]}
                for item in facet_contracts
            ],
        }
        return "".join(f"{key}: {json.dumps(value, ensure_ascii=False)}\n"
                       for key, value in fields.items())

    def schema(self, original):
        result = deepcopy(original)
        fields = result["schema"]["properties"]
        source_ids = [label for _, label in self.source_labels]
        # Runtime also accepts a known canonical ID; it carries exactly the
        # same evidence identity. The provider is shown only the short label.
        citation_field = fields["citation_ids"]["items"]
        if "enum" in citation_field:
            citation_field["enum"] = source_ids
        covered_facets = fields["covered_facets"]["items"]
        if "enum" in covered_facets:
            facets = dict(self.facet_labels)
            covered_facets["enum"] = [facets[label] for label in covered_facets["enum"]]
        if "operation_boundary" in fields:
            closure = fields["operation_boundary"]["properties"]["changes"]["items"]["properties"]
            closure["closing_source_citation_id"]["enum"] = source_ids
        if self.checklist_positions:
            maximum = fields["expected_list_items"]["maxItems"]
            for name in ("covered_list_items", "missing_list_items"):
                fields[name]["items"] = {"type": "integer", "minimum": 1, "maximum": maximum}
        return result

    def decode(self, reply):
        out = deepcopy(reply)
        if self.checklist_positions:
            expected = out.get("expected_list_items")
            if type(expected) is not list or any(type(item) is not str for item in expected):
                raise ValueError("finite review expected checklist has invalid type")
            seen = set()
            for name in ("covered_list_items", "missing_list_items"):
                positions = out.get(name)
                if type(positions) is not list:
                    raise ValueError("finite review checklist positions have invalid type")
                resolved = []
                for position in positions:
                    if type(position) is not int or not 1 <= position <= len(expected) or position in seen:
                        raise ValueError("finite review checklist position is invalid")
                    seen.add(position)
                    resolved.append(expected[position - 1])
                out[name] = resolved
        sources, facets, types = (
            {alias: value for value, alias in pairs}
            for pairs in (self.source_labels, self.facet_labels, self.type_labels)
        )
        source_ids = set(sources.values())

        def source(value):
            if type(value) is not str:
                raise ValueError("finite review citation has invalid type")
            canonical = sources.get(value, value)
            if canonical not in source_ids:
                raise ValueError("finite review citation is outside the emitted packet")
            return canonical

        for names, aliases in (
            (("covered_facets", "missing_facets"), facets),
            (("covered_answer_types", "missing_answer_types"), types),
            (("citation_ids",), None),
        ):
            for name in names:
                values = out.get(name)
                if type(values) is not list or any(type(value) is not str for value in values):
                    raise ValueError("finite review coverage has invalid type")
                out[name] = [source(value) if aliases is None else aliases.get(value, value)
                             for value in values]
        boundary = out.get("operation_boundary")
        if boundary is not None:
            changes = boundary.get("changes") if type(boundary) is dict else None
            if type(changes) is not list or any(type(change) is not dict for change in changes):
                raise ValueError("finite review closure has invalid type")
            for change in changes:
                change["closing_source_citation_id"] = source(change.get("closing_source_citation_id"))
        if type(out.get("reason")) is str:
            # The reason is diagnostic prose, not coverage, proof, or citation
            # authority. Resolve known labels here without exposing source IDs.
            labels = {**facets, **types, **dict.fromkeys(sources, "cited source")}
            out["reason"] = _LABEL.sub(lambda match: labels.get(match[0], match[0]), out["reason"])
        # A label outside its declared field is never evidence, prose or a quote.
        if _LABEL.search(json.dumps(out, ensure_ascii=False)):
            raise ValueError("finite review label is unknown or out of position")
        return out


def prepare(sources, *, citation_ids, facets, types, question, answer, checklist_positions=False):
    """Replace the displayed identity of each unique, already-rendered header."""
    if not citation_ids or len(set(citation_ids)) != len(citation_ids):
        return None
    if any(_LABEL.search(text) for text in (sources, question, answer, *facets, *types)):
        return None
    headers = list(_SOURCE_HEADER.finditer(sources))
    if not headers or headers[0].start() != 0 or tuple(h[1] for h in headers) != tuple(citation_ids):
        return None
    source_labels = _aliases("s", citation_ids)
    parts = []
    start = 0
    for header, (_, label) in zip(headers, source_labels, strict=True):
        parts.extend((sources[start:header.start(1)], label))
        start = header.end(1)
    parts.append(sources[start:])
    return FiniteReviewWire("".join(parts), source_labels, _aliases("f", facets), _aliases("t", types),
                           checklist_positions=checklist_positions)
