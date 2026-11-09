"""Shared ASK wording rule; it does not exempt user premises from grounding."""

NUMERIC_PREMISE_CORRECTION_INSTRUCTION = (
    " When correcting an unsupported numeric premise in QUESTION, state the "
    "source-supported value and its unit, entity and applicability explicitly. "
    "Refer to the unsupported input as 'the proposed value' (or the corresponding "
    "quantity, in RESPONSE_LANGUAGE), without repeating its numeric literal. "
    "Do not silently accept, convert or validate an unsupported premise; explain "
    "whether it conflicts with the documented bound or cannot be established. "
    "This wording rule does not authorize any fact absent from SOURCES."
)
