"""
Policy and governance rules for high-risk support actions.
"""

from dataclasses import dataclass
from typing import Dict, List


HIGH_RISK_CATEGORIES = [
    "pii_exposure",
    "fraud_risk",
    "account_takeover",
    "prompt_injection",
    "legal_threat",
    "medical_safety",
]


@dataclass
class GovernanceDecision:
    decision: str
    reasons: List[str]
    required_evidence: List[str]


def detect_high_risk_flags(text: str) -> List[str]:
    t = text.lower()
    flags = []
    if any(x in t for x in ["aadhaar", "pan", "credit card", "cvv", "password"]):
        flags.append("pii_exposure")
    if any(x in t for x in ["fraud", "chargeback", "scam", "stolen"]):
        flags.append("fraud_risk")
    if any(x in t for x in ["account hacked", "unauthorized login", "sim swap"]):
        flags.append("account_takeover")
    if any(x in t for x in ["ignore previous instructions", "system prompt", "jailbreak"]):
        flags.append("prompt_injection")
    if any(x in t for x in ["legal notice", "consumer court", "lawyer"]):
        flags.append("legal_threat")
    if any(x in t for x in ["critical patient", "emergency treatment", "life threatening"]):
        flags.append("medical_safety")
    return flags


def governance_gate(action_type: str, risk_flags: List[str], evidence: List[str]) -> GovernanceDecision:
    evidence_set = set(evidence)
    reasons: List[str] = []
    required: List[str] = []

    if "prompt_injection" in risk_flags:
        reasons.append("Prompt-injection signal detected.")
        if "policy_reference" not in evidence_set:
            return GovernanceDecision("block", reasons, ["policy_reference"])
        return GovernanceDecision("human_review_required", reasons, ["policy_reference"])

    if "fraud_risk" in risk_flags and "fraud_check_id" not in evidence_set:
        reasons.append("Fraud risk without fraud evidence.")
        required.append("fraud_check_id")

    if "account_takeover" in risk_flags and "kyc_verified" not in evidence_set:
        reasons.append("Account takeover flow needs KYC verification.")
        required.append("kyc_verified")

    if "pii_exposure" in risk_flags and action_type == "resolve":
        reasons.append("Direct resolution blocked for PII exposure.")
        if "pii_redaction_proof" not in evidence_set:
            return GovernanceDecision("human_review_required", reasons, ["pii_redaction_proof"])

    if "legal_threat" in risk_flags:
        reasons.append("Legal threat requires legal hold.")
        return GovernanceDecision("legal_hold", reasons, ["policy_reference"])

    if required:
        return GovernanceDecision("human_review_required", reasons, sorted(set(required)))

    return GovernanceDecision("allow", ["No governance blockers."], [])

