from app.domain import CAUSE_NAMES, RECOMMENDATIONS, FaultCause


def recommendations_for(cause: FaultCause) -> list[str]:
    """Return the original project's deterministic troubleshooting actions."""
    return list(RECOMMENDATIONS[cause])


def cause_name(cause: FaultCause) -> str:
    return CAUSE_NAMES[cause]
