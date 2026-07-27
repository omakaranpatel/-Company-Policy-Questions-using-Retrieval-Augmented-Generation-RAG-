"""Rule-based query expansion for policy terminology."""

POLICY_SYNONYMS: dict[str, list[str]] = {
    "reimbursement": ["expense report", "refund", "pay back", "repayment"],
    "travel": ["trip", "flight", "lodging", "hotel", "mileage"],
    "per diem": ["daily allowance", "meal allowance", "daily limit"],
    "receipt": ["proof of purchase", "invoice", "documentation"],
    "approval": ["authorize", "sign-off", "manager approval"],
    "wellness": ["gym", "fitness", "mental health stipend"],
    "remote": ["work from home", "home office", "telecommute"],
    "corporate card": ["company credit card", "business card"],
    "deadline": ["time limit", "submission window", "due date"],
}


def expand_query(query: str) -> str:
    """Append domain synonyms to improve recall without changing user intent."""
    lower = query.lower()
    extras: list[str] = []
    for term, synonyms in POLICY_SYNONYMS.items():
        if term in lower:
            extras.extend(synonyms[:2])
    if not extras:
        return query
    return f"{query} {' '.join(extras)}"
