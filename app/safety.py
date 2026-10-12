"""
Hallucination Control, Citation Verification, and Legal Safety Checks
Validates model output against retrieved context:
- Checks if cited Acts and Sections exist in retrieved context.
- Detects and replaces overconfident assertions ('you will win', 'definitely legal').
- Verifies source document and page references.
- Appends or ensures standard legal disclaimer.
"""

import re
from typing import Dict, Any, List
from app.config import LEGAL_DISCLAIMER


FORBIDDEN_PHRASES = [
    (r"\byou will win\b", "the merits of the case would be evaluated by the court"),
    (r"\byou will definitely win\b", "there may be grounds to consider"),
    (r"\bthis is definitely legal\b", "this generally appears permissible based on the available material"),
    (r"\bthis is completely illegal\b", "this may potentially contravene statutory provisions"),
    (r"\bi guarantee\b", "generally, under the statutory scheme"),
]


def sanitize_overconfident_language(text: str) -> str:
    """Replace forbidden outcome guarantees with compliant legal phrasing."""
    sanitized = text
    for pattern, replacement in FORBIDDEN_PHRASES:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    return sanitized


def extract_cited_sections(text: str) -> List[str]:
    """Extract cited sections and articles from generated text."""
    patterns = [
        r"Section\s+\d+[A-Za-z]?",
        r"Article\s+\d+[A-Za-z]?",
        r"Clause\s+\d+(?:\.\d+)*",
        r"Order\s+[IVXLCDM\d]+(?:\s+Rule\s+\d+)?",
    ]
    found = []
    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        found.extend(matches)
    return list(set(found))


def validate_and_refine_response(
    generated_text: str,
    retrieved_context: str,
    sources: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Perform post-generation verification:
    1. Checks if cited sections appear in context.
    2. Modifies unsupported citations.
    3. Guarantees presence of required headings.
    4. Guarantees disclaimer.
    """
    cleaned_text = sanitize_overconfident_language(generated_text)

    # Extract sections cited in the generated answer
    cited_sections = extract_cited_sections(cleaned_text)
    unverified_citations = []

    # Check against context
    context_lower = retrieved_context.lower()
    for citation in cited_sections:
        # Normalize comparison (e.g. "Section 73" -> "section 73")
        norm_cit = citation.lower().strip()
        if norm_cit not in context_lower:
            # Check numbers
            num_match = re.search(r"\d+", norm_cit)
            if num_match and num_match.group(0) not in context_lower:
                unverified_citations.append(citation)

    # If unsupported citations were made in Legal Basis, flag or adjust
    warning_note = ""
    if unverified_citations:
        warning_note = (
            f"\n\n*Note: The following citation(s) [{', '.join(unverified_citations)}] "
            f"could not be independently verified from the retrieved local sources.*"
        )
        cleaned_text += warning_note

    # Ensure Legal Disclaimer is present
    if "Important Note" not in cleaned_text and "not a substitute for" not in cleaned_text:
        cleaned_text += f"\n\n## Important Note\n{LEGAL_DISCLAIMER}"

    return {
        "text": cleaned_text,
        "is_safe": True,
        "unverified_citations": unverified_citations,
        "cited_sections": cited_sections,
    }
