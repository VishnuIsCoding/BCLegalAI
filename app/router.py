"""
Query Router for Indian Legal AI
Categorizes incoming queries into:
- GREETING: Conversational greetings, introductions, and help queries (hello, hi, what can you do, etc.)
- LEGAL_QUESTION: Direct Indian legal questions (Acts, sections, jurisprudence, disputes, rights, procedures)
- DOCUMENT_ANALYSIS: Inquiries about uploaded documents (contracts, petitions, notices)
- IMAGE_ANALYSIS: Inquiries about uploaded images or OCR scans
- GENERAL_QUESTION: Explicitly non-legal queries (code writing, weather, sports scores, trivia, recipes)
"""

import re
from typing import Tuple, Optional


class QueryCategory:
    LEGAL_QUESTION = "LEGAL_QUESTION"
    DOCUMENT_ANALYSIS = "DOCUMENT_ANALYSIS"
    IMAGE_ANALYSIS = "IMAGE_ANALYSIS"
    GENERAL_QUESTION = "GENERAL_QUESTION"
    GREETING = "GREETING"


# Conversational greeting & introductory patterns
GREETING_PATTERNS = [
    r"^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening|day)|namaste|namaskar|pranam|yo|sup)(\s+(there|assistant|bot|team|ai|sir|ma'am|everyone|friend))?[\s!.]*$",
    r"^(who\s+are\s+you|what\s+can\s+you\s+do|help|how\s+can\s+you\s+help|what\s+is\s+this|introduce\s+yourself|tell\s+me\s+about\s+yourself)[\s?!.]*$",
    r"^(can\s+you\s+help\s+me|how\s+can\s+you\s+help\s+me|how\s+are\s+you|start|menu|options|capabilities)[\s?!.]*$",
]

# Strictly non-legal indicators (code, weather, sports, cooking, jokes, etc.)
NON_LEGAL_KEYWORDS = [
    r"\b(write|create|debug|code|script|program)\b.*\b(python|java|c\+\+|javascript|html|css|sql|rust|php|bash)\b",
    r"\b(weather|temperature|forecast|rain|humidity)\b",
    r"\b(cricket|football|soccer|ipl|world cup score|who won the match|match score)\b",
    r"\b(tell me a joke|funny joke|riddle|sing a song|recipe|how to cook|baking)\b",
    r"\b(box office collection|celebrity gossip|horoscope|zodiac sign|movie|actor|actress|cinema|film|singer)\b",
    r"\b(solve math|2\s*\+\s*2|calculate \d+\s*[\+\-\*\/]\s*\d+)\b",
]

# Expansive Legal Terms and Concepts in Indian Jurisprudence
LEGAL_KEYWORDS = [
    # Core statutes & legal concepts
    "law", "legal", "act", "section", "article", "court", "judge", "advocate", "lawyer",
    "contract", "breach", "agreement", "consideration", "bail", "anticipatory bail",
    "fir", "police", "cognizable", "non-cognizable", "complaint", "charge", "accused", "victim", "trial",
    "consumer", "consumer forum", "unfair trade", "deficiency", "compensation", "damages", "refund",
    "constitution", "fundamental right", "writ", "habeas corpus", "mandamus", "certiorari", "quowarranto",
    "notice", "legal notice", "statutory", "jurisdiction", "affidavit", "plaint", "written statement",
    "bns", "bnss", "bsa", "ipc", "crpc", "cpc", "negotiable instruments", "cheque bounce", "cheque", "dishonour",
    "injunction", "arbitration", "specific relief", "limitation", "property", "lease", "rent", "landlord", "tenant",
    "evict", "eviction", "deposit", "security deposit",
    "deed", "power of attorney", "tort", "criminal", "civil", "penal", "offence", "offense",
    "clause", "indemnity", "termination clause", "confidentiality", "nda", "employment agreement",
    "severability", "arbitration clause", "stamp duty", "registration", "vakalatnama",
    # Disputes, money, fraud, crime
    "cheat", "cheated", "cheating", "fraud", "scam", "stolen", "theft", "extortion", "blackmail",
    "salary", "wages", "unpaid salary", "employer", "employee", "fired", "terminated", "termination",
    "resignation", "severance", "notice period", "provident fund", "pf", "epf", "gratuity",
    "loan", "debt", "recovery", "default", "bank", "guarantor", "surety", "mortgage", "pledge",
    "arrest", "jail", "custody", "remand", "interrogation", "handcuff", "harassment", "threat",
    "assault", "battery", "murder", "culpable homicide", "accident", "hit and run", "challan", "fine",
    # Corporate & Commercial
    "company", "companies act", "director", "board of directors", "shareholder", "corporate veil",
    "ibc", "insolvency", "bankruptcy", "nclt", "nclat", "moratorium", "liquidation",
    "patent", "trademark", "copyright", "infringement", "intellectual property", "ipr", "passing off",
    "gst", "income tax", "tax", "taxation", "assessment", "itat", "customs",
    # Family, Matrimonial & Succession
    "hindu succession", "ancestral property", "succession", "inheritance", "will", "probate", "coparcener",
    "partition", "succession certificate", "gift deed", "sale deed",
    "hindu marriage act", "divorce", "maintenance", "alimony", "custody", "child custody", "section 125",
    "domestic violence", "dv act", "498a", "dowry", "posh", "sexual harassment", "pocso", "ndps", "pmla",
    "money laundering", "cybercrime", "it act", "defamation", "libel", "slander",
    # Real estate & Consumer
    "rera", "builder", "flat", "possession", "transfer of property", "adverse possession", "easement",
    "builder buyer", "encumbrance", "registry",
    # Writs & Supreme Court
    "special leave petition", "slp", "curative petition", "review petition", "quash", "quashing",
    "section 482", "contempt of court", "perjury", "remand", "chargesheet", "motor vehicles", "mact",
    "settled law", "precedent", "supreme court", "high court", "sessions court", "magistrate court",
    # Rights & Legal queries phrasing
    "sue", "can i sue", "file a case", "file a suit", "is it legal", "is it illegal", "my rights",
    "punishment", "penalty", "liable", "liability", "remedy", "grounds", "procedure to file",
    "who is responsible", "can police", "can landlord", "can employer", "can company", "what to do"
]

# Specialized topics requiring high-intelligence fallback
SPECIALIZED_DOMAINS = [
    (r"\b(company|companies act|director|board of directors|shareholder|corporate veil|nclt|nclat|ibc|insolvency|bankruptcy|moratorium|liquidation)\b", "Corporate Law & Insolvency (IBC/Companies Act)"),
    (r"\b(patent|trademark|copyright|infringement|intellectual property|ipr|passing off)\b", "Intellectual Property Rights (IPR)"),
    (r"\b(gst|income tax|tax evasion|tax assessment|itat|input tax credit|customs duty)\b", "Taxation & Fiscal Law"),
    (r"\b(hindu succession|ancestral property|coparcener|will|probate|succession certificate|hindu marriage|divorce|alimony|maintenance|section 125|custody)\b", "Family, Matrimonial & Succession Law"),
    (r"\b(498a|domestic violence|dv act|dowry|pocso|ndps|pmla|money laundering|uapa|prevention of corruption)\b", "Specialized Penal & Anti-Harassment Statutes"),
    (r"\b(rera|builder buyer|possession of flat|transfer of property|adverse possession|easementary rights)\b", "Real Estate (RERA) & Property Jurisprudence"),
    (r"\b(special leave petition|slp|curative petition|review petition|quash.*482|article 226|article 32|basic structure|contempt of court)\b", "Constitutional Writs & Appellate Supreme Court Practice"),
    (r"\b(arbitral award|section 34|section 9 arbitration|section 11|commercial court|interim relief)\b", "Advanced Arbitration & Commercial Remedies"),
    (r"\b(motor vehicles act|mact|accident claim|hit and run claim)\b", "Motor Accidents Claims (MACT)"),
    (r"\b(industrial dispute|gratuity act|provident fund|epf|retrenchment|posh act|severance)\b", "Labor & Workplace Harassment (POSH) Law"),
    (r"\b(cheque dishonour|section 138|ni act|negotiable instruments)\b", "Negotiable Instruments Act & Commercial Crime"),
]


def classify_query(
    query: str,
    has_uploaded_doc: bool = False,
    has_uploaded_image: bool = False,
) -> Tuple[str, str]:
    """
    Classify user input into one of five categories:
    - GREETING: Conversational hello/hi/help
    - IMAGE_ANALYSIS: Query about uploaded image
    - DOCUMENT_ANALYSIS: Query about uploaded document
    - GENERAL_QUESTION: Explicit non-legal topic (code, cooking, sports)
    - LEGAL_QUESTION: Indian legal questions, rights, statutory inquiries, or dispute scenarios
    """
    cleaned = (query or "").strip()
    query_lower = cleaned.lower()

    if not cleaned:
        return QueryCategory.GREETING, "Empty query treated as greeting"

    # 1. Check for conversational greetings & introductions
    for pattern in GREETING_PATTERNS:
        if re.match(pattern, query_lower):
            return QueryCategory.GREETING, "Conversational greeting or introduction"

    # 2. Check for explicit non-legal patterns (coding, weather, sports scores, recipes)
    for pattern in NON_LEGAL_KEYWORDS:
        if re.search(pattern, query_lower):
            return QueryCategory.GENERAL_QUESTION, "Matches non-legal query pattern"

    # 3. Check if user is referencing an uploaded image
    if has_uploaded_image:
        image_triggers = ["image", "picture", "photo", "scan", "ocr", "receipt", "document in image", "what does this say"]
        if any(t in query_lower for t in image_triggers) or len(cleaned) < 5:
            return QueryCategory.IMAGE_ANALYSIS, "User querying uploaded image"

    # 4. Check if user is referencing an uploaded document
    if has_uploaded_doc:
        doc_triggers = [
            "this agreement", "this contract", "this document", "this notice", "clause",
            "section in this", "page", "summarize this", "uploaded", "what does this mean",
            "explain this", "parties", "termination", "liability in this", "terms of this"
        ]
        if any(t in query_lower for t in doc_triggers) or "this" in query_lower or len(cleaned.split()) <= 6:
            return QueryCategory.DOCUMENT_ANALYSIS, "User querying uploaded document"

    # 5. Check for legal keywords or legal phrasing
    if any(re.search(rf"\b{re.escape(k)}\b", query_lower) for k in LEGAL_KEYWORDS):
        return QueryCategory.LEGAL_QUESTION, "Indian legal question identified from keywords"

    # 6. Check common dispute stems (landlord, salary, cheating, police, fine, arrested, rights)
    dispute_stems = [
        "dispute", "rights", "rule", "statute", "sue", "arrest", "fine", "jail", "punish",
        "order", "decree", "appeal", "tribunal", "petition", "liability", "can i claim",
        "remedy", "grounds", "procedure", "how to file", "is it legal", "is it illegal",
        "what are my rights", "who is responsible", "can police", "can someone", "can my",
        "what action", "what can i do", "what to do", "legal advice", "legal help", "legal action"
    ]
    if any(stem in query_lower for stem in dispute_stems):
        return QueryCategory.LEGAL_QUESTION, "Legal query identified from dispute phrasing"

    # 7. Default to LEGAL_QUESTION: in a legal AI assistant, treat all remaining user questions as legal inquiries
    return QueryCategory.LEGAL_QUESTION, "Standard query routed to Indian legal assistant"


def is_difficult_legal_query(
    query: str,
    retrieval_result: Optional[dict] = None,
    min_confidence: float = 0.35,
) -> Tuple[bool, str]:
    """
    Determines if a legal query is difficult/complex and should be routed
    to the high-intelligence Cloud Fallback (Google Gemini / Groq).

    Evaluation dimensions:
    1. Knowledge Base Coverage: Low similarity or 0 chunks from local RAG.
    2. Specialized Domains: Legal topics outside the 6 basic local text files.
    3. Structural Complexity: Multi-part scenario, comparative analysis, drafting, or lengthy query.
    """
    cleaned = query.strip()
    query_lower = cleaned.lower()
    words = cleaned.split()

    # Dimension 1: RAG context quality (if provided)
    if retrieval_result is not None:
        sources = retrieval_result.get("sources", [])
        if not sources or not retrieval_result.get("has_sources", False):
            return True, "No verified legal provisions found in local knowledge base"
        max_score = max((s.get("score", 0.0) for s in sources), default=0.0)
        if max_score < min_confidence:
            return True, f"Low local knowledge base match confidence ({max_score:.2f} < {min_confidence:.2f})"

    # Dimension 2: Specialized Indian Law domains
    for pattern, domain_name in SPECIALIZED_DOMAINS:
        if re.search(pattern, query_lower):
            return True, f"Specialized Indian legal domain: {domain_name}"

    # Dimension 3: Structural Complexity & Query Depth
    if len(words) > 35:
        return True, "High structural complexity / multi-fact scenario"

    comparative_patterns = [
        r"\b(difference between|compare|versus|vs\.?|distinguish between)\b",
        r"\b(pros and cons|advantages and disadvantages|which is better)\b",
    ]
    for cp in comparative_patterns:
        if re.search(cp, query_lower):
            return True, "Comparative legal analysis query"

    drafting_patterns = [
        r"\b(draft|template|format|sample clause|write a notice|draft a petition)\b",
    ]
    for dp in drafting_patterns:
        if re.search(dp, query_lower):
            return True, "Legal drafting or template request"

    multi_issue_indicators = [
        " and also ", " furthermore ", " in addition ", " what if ",
        " moreover ", " along with ", " secondly "
    ]
    if any(ind in query_lower for ind in multi_issue_indicators):
        return True, "Multi-issue legal scenario"

    return False, "Standard Indian legal query suitable for local Qwen 1.5B"
