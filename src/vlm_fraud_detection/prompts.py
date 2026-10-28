"""Prompt templates for VLM-based document fraud detection.

``build_text_prompt`` is the main entry point.  It returns the full text
portion of the prompt following the structure described in Section III-E of
the paper:

  1. System instruction
  2. Few-shot examples (each with label + image marker)
  3. Optional chain-of-thought instructions
  4. Query document header
  5. Response format
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# System instruction  (Appendix B of the paper)
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = """\
You are a document fraud detection expert working for a financial institution.

Examine each document carefully for signs of tampering, including:
- Inconsistent fonts or typography
- Altered or misaligned text fields
- Suspicious whitespace or cut-and-paste boundaries
- Irregular image compression artifacts
- Mismatched colors, stamps, or logos
- Unrealistic numerical values (amounts, dates, IDs)

Classify the final query document as "Authentic" or "Fraudulent".\
"""

# ---------------------------------------------------------------------------
# Response format (structured JSON as in Appendix B)
# ---------------------------------------------------------------------------

RESPONSE_FORMAT = """\
Respond ONLY in valid JSON with this exact structure:
{
  "fraud_label":          "Authentic" or "Fraudulent",
  "fraud_probability":    <float between 0.0 and 1.0>,
  "suspicious_elements":  [<string>, ...],
  "visual_observations":  [<string>, ...],
  "explanation":          "<one-paragraph reasoning>"
}\
"""

# ---------------------------------------------------------------------------
# Optional chain-of-thought addition (Section III-E)
# ---------------------------------------------------------------------------

COT_INSTRUCTION = """\
Before providing your JSON answer, reason step by step:
1. Describe the visual and textual features you observe in the query document.
2. Compare them to the authentic and fraudulent examples above.
3. Identify any anomalies or inconsistencies.
4. State your classification decision and confidence.\
"""

# ---------------------------------------------------------------------------
# Legacy templates (kept for backwards compatibility)
# ---------------------------------------------------------------------------

FRAUD_DETECTION_PROMPT_TEMPLATE = """\
You are an expert in document fraud detection for financial and insurance institutions.

Analyze the document image provided and determine whether it shows signs of fraud.
Look for visual inconsistencies, tampering, altered fields, unrealistic values, and other suspicious patterns.

Use the examples above as legitimate non-fraud references if provided.

Tasks:
1. Analyze the visual appearance of the document (image).
2. Identify suspicious elements (inconsistencies, altered fields, unrealistic values).
3. Assess the likelihood of fraud.
4. Provide a clear explanation based on visual analysis.

Respond ONLY in valid JSON with the following structure:
{
  "fraud_probability": float (0 to 1),
  "fraud_label": "fraud" or "legitimate",
  "suspicious_elements": [string],
  "visual_observations": [string],
  "explanation": string
}\
"""

FRAUD_FEWSHOT_INSTRUCTIONS = """\
The images provided above are examples of legitimate documents without fraud.
When evaluating the target document, use these non-fraud examples as a reference for normal appearance.\
"""

# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_text_prompt(
    examples=None,
    chain_of_thought: bool = False,
) -> str:
    """Build the complete text portion of the few-shot prompt.

    Image parts are interleaved by the detector *before* each example marker,
    so this function only produces the text labels and instructions.

    Args:
        examples: List of LabeledExample objects already embedded in the
                  prompt by the detector, or None for zero-shot.
        chain_of_thought: Whether to include step-by-step reasoning instructions.

    Returns:
        A multi-line string to be appended after all image parts.
    """
    parts = [SYSTEM_INSTRUCTION, ""]

    if examples:
        parts.append("=== Reference Examples ===")
        for idx, ex in enumerate(examples, start=1):
            label_str = "Authentic" if ex.label == "authentic" else "Fraudulent"
            parts.append(f"[Image of Example {idx} shown above]")
            parts.append(f"Example {idx} label: {label_str}")
            parts.append("")
        parts.append("=== End of Examples ===")
        parts.append("")

    if chain_of_thought:
        parts.append(COT_INSTRUCTION)
        parts.append("")

    parts.append("=== Query Document ===")
    parts.append("[The query document image is shown above]")
    parts.append("Classify the query document.")
    parts.append("")
    parts.append(RESPONSE_FORMAT)

    return "\n".join(parts)
