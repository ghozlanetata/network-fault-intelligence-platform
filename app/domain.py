from enum import StrEnum


class PredictionStatus(StrEnum):
    FAULT = "FAULT"
    NORMAL = "NORMAL"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class InvestigationStatus(StrEnum):
    DRAFT = "DRAFT"
    IN_PROGRESS = "IN_PROGRESS"
    PENDING_VERIFICATION = "PENDING_VERIFICATION"
    VERIFIED = "VERIFIED"


class VerificationStatus(StrEnum):
    PENDING = "PENDING"
    VERIFIED = "VERIFIED"
    UNRESOLVED = "UNRESOLVED"
    UNKNOWN = "UNKNOWN"


class ResolutionStatus(StrEnum):
    OPEN = "OPEN"
    RESOLVED = "RESOLVED"
    UNRESOLVED = "UNRESOLVED"


class FaultCause(StrEnum):
    ED = "ED"
    CH = "CH"
    II = "II"
    TLHO = "TLHO"
    RP = "RP"
    EU = "EU"
    NORMAL = "Normal"


CAUSE_NAMES = {
    FaultCause.ED: "Excessive Downtilt",
    FaultCause.CH: "Coverage Hole",
    FaultCause.II: "Inter-System Interference",
    FaultCause.TLHO: "Too Late HandOver",
    FaultCause.RP: "Reduction of Cell Power",
    FaultCause.EU: "Excessive Uptilt",
    FaultCause.NORMAL: "Normal",
}

RECOMMENDATIONS = {
    FaultCause.ED: [
        "Check Bandwidth Capacity Configuration",
        "Check Frequency Configuration",
        "Check Tilt Configuration",
        "Check Load Balance",
    ],
    FaultCause.EU: [
        "Check Bandwidth Capacity Configuration",
        "Check Frequency Configuration",
        "Check Tilt Configuration",
        "Check Load Balance",
        "Check Synchronization of Cell",
    ],
    FaultCause.TLHO: ["Check Handover Execution"],
    FaultCause.II: [
        "Check Planification of Sites",
        "Check Frequency Configuration",
        "Check Coverage",
    ],
    FaultCause.CH: [
        "Check Power Configuration",
        "Check Cell Range",
        "Check Tilt Configuration",
    ],
    FaultCause.RP: ["Check if RRU is Faulty", "Check Licence", "Check Power on Site"],
    FaultCause.NORMAL: [],
}
