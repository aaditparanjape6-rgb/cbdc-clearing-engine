"""
Regulatory Compliance Engine — Module 5
=========================================
Simulates rule-based AML/KYC and cross-border regulatory constraints.

This module models the compliance checks required by:
  - FATF (Financial Action Task Force) AML/CFT recommendations
  - BIS cross-border payment guidelines
  - Jurisdiction-specific CBDC regulations

Compliance gates enforced:
  1. Transaction threshold screening (large-value reporting)
  2. Corridor restriction rules (embargoed/high-risk jurisdictions)
  3. AML/KYC counterparty risk scoring
  4. Velocity controls (transaction frequency limits)
  5. Sanctions screening (OFAC-style simplified simulation)

Output: ComplianceDecision with status, flags, and penalty score.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations and Constants
# ---------------------------------------------------------------------------

class ComplianceStatus(Enum):
    APPROVED    = "APPROVED"
    FLAGGED     = "FLAGGED_FOR_REVIEW"
    BLOCKED     = "BLOCKED"


class RiskFlag(Enum):
    LARGE_VALUE           = "LARGE_VALUE_TRANSFER"
    HIGH_RISK_CORRIDOR    = "HIGH_RISK_CORRIDOR"
    AML_ALERT             = "AML_COUNTERPARTY_ALERT"
    SANCTIONS_HIT         = "SANCTIONS_SCREENING_HIT"
    VELOCITY_BREACH       = "VELOCITY_LIMIT_BREACH"
    RESTRICTED_JURISDICTION = "RESTRICTED_JURISDICTION"
    KYC_INCOMPLETE        = "KYC_INCOMPLETE"
    UNUSUAL_ROUTING       = "UNUSUAL_ROUTING_PATTERN"


# Large-value reporting thresholds (USD equivalent) per jurisdiction
REPORTING_THRESHOLDS = {
    "USD": 10_000,     # US FinCEN CTR threshold
    "EUR": 10_000,     # EU AMLD threshold
    "GBP":  9_000,     # UK threshold (lower)
    "INR": 50_000,     # India PMLA threshold (≈INR 5 lakh at 100 INR/USD)
    "SGD": 20_000,     # MAS threshold
    "CNY": 50_000,     # PBoC reporting threshold
    "AED": 55_000,     # UAE threshold (AED 200K / 3.67)
    "DEFAULT": 10_000, # Fallback
}

# Corridors flagged as high-risk or restricted (simplified simulation)
# Format: frozenset({src, tgt}) — bidirectional restriction
HIGH_RISK_CORRIDORS: Set[frozenset] = {
    frozenset({"CNY", "USD"}),   # capital flow restrictions, OFAC considerations
    frozenset({"CNY", "EUR"}),   # EU trade friction
    frozenset({"CNY", "GBP"}),   # UK-China tensions
    frozenset({"CNY", "SGD"}),   # monitored corridor
}

# Corridors requiring enhanced due diligence (EDD) but not blocked
EDD_CORRIDORS: Set[frozenset] = {
    frozenset({"INR", "AED"}),   # hawala risk monitoring
    frozenset({"AED", "CNY"}),   # UAE-China trade monitoring
    frozenset({"SGD", "CNY"}),   # SE Asia-China flows
}

# Sanctioned/embargoed jurisdictions (simplified — not CNY in this sim)
# Real system would integrate OFAC SDN list, EU consolidated list, etc.
SANCTIONED_ENTITIES: Set[str] = {
    "ENTITY_BLOCKED_001",
    "ENTITY_BLOCKED_002",
}

# Velocity limits: max transaction count per 24h per entity
VELOCITY_LIMITS = {
    "standard":  50,
    "premium":  200,
    "institutional": 1000,
}

# Penalty scores by rule violation (additive, cap at 1.0)
PENALTY_WEIGHTS = {
    RiskFlag.LARGE_VALUE:              0.05,
    RiskFlag.HIGH_RISK_CORRIDOR:       0.30,
    RiskFlag.AML_ALERT:                0.40,
    RiskFlag.SANCTIONS_HIT:            1.00,  # always blocks
    RiskFlag.VELOCITY_BREACH:          0.25,
    RiskFlag.RESTRICTED_JURISDICTION:  0.50,
    RiskFlag.KYC_INCOMPLETE:           0.20,
    RiskFlag.UNUSUAL_ROUTING:          0.15,
}

BLOCK_THRESHOLD = 0.80     # penalty score ≥ 0.80 → BLOCKED
REVIEW_THRESHOLD = 0.20    # penalty score ≥ 0.20 → FLAGGED


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class TransactionProfile:
    """Describes the parties and characteristics of a transfer."""
    tx_id: str
    sender_id: str                  # institution/entity identifier
    receiver_id: str
    source_currency: str            # e.g., "INR"
    destination_currency: str       # e.g., "USD"
    amount_usd_equivalent: float    # USD equivalent of transfer
    routing_path: List[str]         # full routing path (e.g., ["INR","SGD","USD"])
    purpose_code: str = "TRADE"     # TRADE, INVEST, REMITTANCE, etc.
    kyc_verified: bool = True
    entity_type: str = "institutional"   # retail / institutional / central_bank
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ComplianceDecision:
    """Result of compliance screening."""
    tx_id: str
    status: ComplianceStatus
    flags: List[RiskFlag]
    penalty_score: float            # 0.0 (clean) to 1.0 (blocked)
    required_actions: List[str]     # list of actions required (e.g., "Submit SAR")
    review_deadline_hours: Optional[float]  # None if approved immediately
    notes: str = ""

    def is_approved(self) -> bool:
        return self.status == ComplianceStatus.APPROVED

    def to_dict(self) -> Dict:
        return {
            "tx_id": self.tx_id,
            "status": self.status.value,
            "flags": [f.value for f in self.flags],
            "penalty_score": round(self.penalty_score, 3),
            "required_actions": self.required_actions,
            "review_deadline_hours": self.review_deadline_hours,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Compliance Engine
# ---------------------------------------------------------------------------

class ComplianceEngine:
    """
    Rule-based compliance screening engine for CBDC cross-border transactions.

    Designed to simulate a tiered compliance framework:
      Tier 1: Automated instant checks (sanctions, thresholds, KYC flags)
      Tier 2: Rule-based scoring → review queue if threshold exceeded
      Tier 3: Human review queue (simulated as FLAGGED status)
    """

    def __init__(self):
        # In-memory velocity tracker: {entity_id: [timestamp, ...]}
        self._velocity_log: Dict[str, List[datetime]] = {}
        # Audit log
        self.audit_log: List[Dict] = []

    def screen(self, txn: TransactionProfile) -> ComplianceDecision:
        """
        Run full compliance screening pipeline for a transaction.
        Returns a ComplianceDecision.
        """
        flags: List[RiskFlag] = []
        penalty = 0.0
        required_actions: List[str] = []

        # --- Rule 1: Sanctions Screening ---
        if self._check_sanctions(txn.sender_id) or self._check_sanctions(txn.receiver_id):
            flags.append(RiskFlag.SANCTIONS_HIT)
            penalty += PENALTY_WEIGHTS[RiskFlag.SANCTIONS_HIT]
            required_actions.append("Submit OFAC/sanctions report and freeze transaction")

        # --- Rule 2: Restricted Jurisdiction Check ---
        path_pair_restrictions = self._check_restricted_corridors(txn.routing_path)
        if path_pair_restrictions:
            flags.append(RiskFlag.RESTRICTED_JURISDICTION)
            penalty += PENALTY_WEIGHTS[RiskFlag.RESTRICTED_JURISDICTION]
            required_actions.append(
                f"Restricted corridor detected: {path_pair_restrictions}. "
                "Regulatory approval required."
            )

        # --- Rule 3: High-Risk Corridor (EDD required) ---
        edd_flag = self._check_high_risk_corridor(txn.source_currency, txn.destination_currency)
        if edd_flag:
            flags.append(RiskFlag.HIGH_RISK_CORRIDOR)
            penalty += PENALTY_WEIGHTS[RiskFlag.HIGH_RISK_CORRIDOR]
            required_actions.append("Enhanced Due Diligence (EDD) documentation required")

        # --- Rule 4: Large Value Threshold ---
        threshold = REPORTING_THRESHOLDS.get(txn.source_currency,
                    REPORTING_THRESHOLDS["DEFAULT"])
        if txn.amount_usd_equivalent > threshold:
            flags.append(RiskFlag.LARGE_VALUE)
            penalty += PENALTY_WEIGHTS[RiskFlag.LARGE_VALUE]
            required_actions.append(
                f"Currency Transaction Report (CTR) required "
                f"(amount ${txn.amount_usd_equivalent:,.0f} exceeds ${threshold:,.0f} threshold)"
            )

        # --- Rule 5: KYC Completeness ---
        if not txn.kyc_verified:
            flags.append(RiskFlag.KYC_INCOMPLETE)
            penalty += PENALTY_WEIGHTS[RiskFlag.KYC_INCOMPLETE]
            required_actions.append("KYC documentation incomplete — obtain before processing")

        # --- Rule 6: AML Pattern Analysis ---
        aml_hit = self._aml_heuristic_check(txn)
        if aml_hit:
            flags.append(RiskFlag.AML_ALERT)
            penalty += PENALTY_WEIGHTS[RiskFlag.AML_ALERT]
            required_actions.append("File Suspicious Activity Report (SAR) — AML trigger")

        # --- Rule 7: Velocity Controls ---
        velocity_breach = self._check_velocity(txn)
        if velocity_breach:
            flags.append(RiskFlag.VELOCITY_BREACH)
            penalty += PENALTY_WEIGHTS[RiskFlag.VELOCITY_BREACH]
            required_actions.append(
                f"Transaction velocity limit exceeded for {txn.sender_id}"
            )

        # --- Rule 8: Unusual Routing Pattern ---
        if self._detect_unusual_routing(txn.routing_path, txn.source_currency, txn.destination_currency):
            flags.append(RiskFlag.UNUSUAL_ROUTING)
            penalty += PENALTY_WEIGHTS[RiskFlag.UNUSUAL_ROUTING]
            required_actions.append(
                "Unusual routing pattern flagged — verify business justification"
            )

        # --- Final Decision ---
        penalty = min(penalty, 1.0)

        if penalty >= BLOCK_THRESHOLD or RiskFlag.SANCTIONS_HIT in flags:
            status = ComplianceStatus.BLOCKED
            deadline = None
        elif penalty >= REVIEW_THRESHOLD:
            status = ComplianceStatus.FLAGGED
            # Review deadline: larger penalty = shorter deadline
            deadline = max(0.5, 24 * (1 - penalty))
        else:
            status = ComplianceStatus.APPROVED
            deadline = None

        decision = ComplianceDecision(
            tx_id=txn.tx_id,
            status=status,
            flags=flags,
            penalty_score=penalty,
            required_actions=required_actions,
            review_deadline_hours=deadline,
            notes=self._build_notes(txn, flags),
        )

        self._audit(txn, decision)
        return decision

    # ------------------------------------------------------------------
    # Individual Rule Implementations
    # ------------------------------------------------------------------

    def _check_sanctions(self, entity_id: str) -> bool:
        """Simplified sanctions screening (hash-based match)."""
        return entity_id in SANCTIONED_ENTITIES

    def _check_restricted_corridors(self, path: List[str]) -> str:
        """Check if any hop in the path crosses a restricted corridor."""
        for i in range(len(path) - 1):
            pair = frozenset({path[i], path[i+1]})
            if pair in HIGH_RISK_CORRIDORS:
                return f"{path[i]}↔{path[i+1]}"
        return ""

    def _check_high_risk_corridor(self, src: str, tgt: str) -> bool:
        """Check if the endpoint pair requires EDD."""
        return frozenset({src, tgt}) in EDD_CORRIDORS

    def _aml_heuristic_check(self, txn: TransactionProfile) -> bool:
        """
        Simplified AML pattern checks:
          - Round-number transactions (structuring indicator)
          - Very high amounts from retail entities
          - Remittance purpose with high amounts
        """
        amount = txn.amount_usd_equivalent
        # Round number structuring check (simplified)
        if amount > 9_000 and amount % 1000 == 0 and txn.entity_type == "retail":
            return True
        # Retail entity sending large amounts
        if txn.entity_type == "retail" and amount > 100_000:
            return True
        # Remittance with unusual amount
        if txn.purpose_code == "REMITTANCE" and amount > 500_000:
            return True
        return False

    def _check_velocity(self, txn: TransactionProfile) -> bool:
        """Track and enforce transaction velocity per sender."""
        now = txn.timestamp
        window_start = now - timedelta(hours=24)
        sender = txn.sender_id
        entity_type = txn.entity_type

        # Purge old entries
        if sender not in self._velocity_log:
            self._velocity_log[sender] = []
        self._velocity_log[sender] = [
            t for t in self._velocity_log[sender] if t > window_start
        ]
        # Check limit
        limit = VELOCITY_LIMITS.get(entity_type, VELOCITY_LIMITS["standard"])
        if len(self._velocity_log[sender]) >= limit:
            return True
        # Log this transaction
        self._velocity_log[sender].append(now)
        return False

    def _detect_unusual_routing(self, path: List[str], src: str, tgt: str) -> bool:
        """
        Flag routing patterns that seem economically unjustified:
          - More than 3 hops where a direct route exists
          - Routing through high-friction nodes unnecessarily
        """
        direct_exists = len(path) == 2  # direct route
        if len(path) > 4:
            return True  # excessive hops
        # Flag if CNY appears as intermediate hop in a non-CNY transaction
        if src != "CNY" and tgt != "CNY" and "CNY" in path[1:-1]:
            return True
        return False

    def _build_notes(self, txn: TransactionProfile, flags: List[RiskFlag]) -> str:
        if not flags:
            return f"Transaction {txn.tx_id}: Clean screening. Auto-approved."
        return (
            f"Transaction {txn.tx_id}: {len(flags)} compliance flag(s) raised. "
            f"Path: {' → '.join(txn.routing_path)}. "
            f"Amount: USD {txn.amount_usd_equivalent:,.0f}."
        )

    def _audit(self, txn: TransactionProfile, decision: ComplianceDecision):
        self.audit_log.append({
            "timestamp": txn.timestamp.isoformat(),
            "tx_id": txn.tx_id,
            "sender": txn.sender_id,
            "receiver": txn.receiver_id,
            "amount_usd": txn.amount_usd_equivalent,
            "path": "→".join(txn.routing_path),
            "status": decision.status.value,
            "penalty_score": decision.penalty_score,
            "flags": [f.value for f in decision.flags],
        })

    def get_audit_summary(self):
        import pandas as pd
        return pd.DataFrame(self.audit_log)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    engine = ComplianceEngine()

    # Test 1: Standard India→USA transfer
    txn1 = TransactionProfile(
        tx_id="TXN-001",
        sender_id="INST_HDFC_001",
        receiver_id="INST_JPMC_001",
        source_currency="INR",
        destination_currency="USD",
        amount_usd_equivalent=500_000,
        routing_path=["INR", "USD"],
        purpose_code="TRADE",
        kyc_verified=True,
        entity_type="institutional",
    )
    d1 = engine.screen(txn1)
    print("=== TXN-001: Standard INR→USD Trade ===")
    for k, v in d1.to_dict().items():
        print(f"  {k}: {v}")

    # Test 2: Large CNY transfer (should trigger restrictions)
    txn2 = TransactionProfile(
        tx_id="TXN-002",
        sender_id="INST_BOC_001",
        receiver_id="INST_CITI_001",
        source_currency="CNY",
        destination_currency="USD",
        amount_usd_equivalent=5_000_000,
        routing_path=["CNY", "USD"],
        purpose_code="INVEST",
        kyc_verified=True,
        entity_type="institutional",
    )
    d2 = engine.screen(txn2)
    print("\n=== TXN-002: Large CNY→USD Transfer ===")
    for k, v in d2.to_dict().items():
        print(f"  {k}: {v}")
