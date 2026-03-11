"""
Settlement Simulator — Module 6
=================================
Simulates atomic CBDC cross-border settlement with a full transaction lifecycle.

Settlement flow (mirrors ISO 20022 / mBridge PvP design):
  1. INITIATE     — Transaction submitted to clearing engine
  2. ROUTE        — Optimal path computed
  3. COMPLIANCE   — AML/KYC/regulatory screening
  4. LOCK         — Funds locked on all corridor legs (atomic)
  5. CONFIRM      — Final rate confirmation and counterparty acknowledgment
  6. SETTLE       — Atomic delivery vs. payment (DvP/PvP)
  7. RELEASE      — Settled funds released to beneficiary; liquidity recycled
  8. RECORD       — Transaction ledger entry and audit

Each transaction is either fully settled (atomic success) or fully reversed
(no partial settlements) — this is the fundamental CBDC safety guarantee.
"""

import uuid
import random
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settlement Status Lifecycle
# ---------------------------------------------------------------------------

class SettlementStatus(Enum):
    INITIATED   = "INITIATED"
    ROUTED      = "ROUTED"
    COMPLIANCE  = "COMPLIANCE_CHECKED"
    LOCKED      = "FUNDS_LOCKED"
    CONFIRMED   = "CONFIRMED"
    SETTLED     = "SETTLED"
    FAILED      = "FAILED"
    REJECTED    = "COMPLIANCE_REJECTED"


# ---------------------------------------------------------------------------
# Transaction Record
# ---------------------------------------------------------------------------

@dataclass
class SettlementRecord:
    """Complete settlement ledger entry for one cross-border CBDC transaction."""
    tx_id: str
    sender_institution: str
    receiver_institution: str
    source_currency: str
    destination_currency: str
    # Amounts
    send_amount_local: float        # In source currency
    receive_amount_local: float     # In destination currency
    amount_usd_equivalent: float
    # Route
    routing_path: List[str]
    effective_fx_rate: float
    # Costs
    total_cost_bps: float
    total_cost_usd: float
    # Timing
    settlement_latency_s: float
    initiated_at: datetime = field(default_factory=datetime.utcnow)
    settled_at: Optional[datetime] = None
    # Outcomes
    status: SettlementStatus = SettlementStatus.INITIATED
    compliance_status: str = ""
    compliance_penalty: float = 0.0
    fx_risk_score: float = 0.0
    composite_score: float = 0.0
    failure_reason: str = ""
    # Reference data
    purpose_code: str = "TRADE"

    def to_dict(self) -> Dict:
        return {
            "tx_id": self.tx_id,
            "sender": self.sender_institution,
            "receiver": self.receiver_institution,
            "source_currency": self.source_currency,
            "destination_currency": self.destination_currency,
            "send_amount_local": round(self.send_amount_local, 4),
            "receive_amount_local": round(self.receive_amount_local, 4),
            "amount_usd_m": round(self.amount_usd_equivalent / 1_000_000, 4),
            "routing_path": " → ".join(self.routing_path),
            "effective_fx_rate": round(self.effective_fx_rate, 6),
            "total_cost_bps": round(self.total_cost_bps, 2),
            "total_cost_usd": round(self.total_cost_usd, 2),
            "settlement_latency_s": round(self.settlement_latency_s, 1),
            "status": self.status.value,
            "compliance_status": self.compliance_status,
            "compliance_penalty": round(self.compliance_penalty, 3),
            "fx_risk_score": round(self.fx_risk_score, 4),
            "composite_score": round(self.composite_score, 4),
            "failure_reason": self.failure_reason,
            "initiated_at": self.initiated_at.isoformat(),
            "settled_at": self.settled_at.isoformat() if self.settled_at else None,
        }


# ---------------------------------------------------------------------------
# Settlement Engine
# ---------------------------------------------------------------------------

class SettlementSimulator:
    """
    End-to-end atomic settlement simulator.

    Integrates with:
      - ClearingEngine   (routing)
      - ComplianceEngine (AML/KYC)
      - LiquidityManager (fund locking)
    """

    def __init__(
        self,
        clearing_engine,
        compliance_engine,
        liquidity_manager,
        simulate_network_delay: bool = True,
    ):
        self.clearing = clearing_engine
        self.compliance = compliance_engine
        self.liquidity = liquidity_manager
        self.simulate_delay = simulate_network_delay
        # Ledger: all settlement records
        self.ledger: List[SettlementRecord] = []
        # Stats
        self.stats = {
            "total_initiated": 0,
            "total_settled": 0,
            "total_failed": 0,
            "total_rejected": 0,
            "total_volume_usd": 0.0,
            "total_cost_usd": 0.0,
        }

    def process_transaction(
        self,
        source_currency: str,
        destination_currency: str,
        send_amount_local: float,
        sender_id: str = "INST_DEFAULT",
        receiver_id: str = "INST_DEFAULT_RCV",
        purpose_code: str = "TRADE",
        kyc_verified: bool = True,
        entity_type: str = "institutional",
    ) -> SettlementRecord:
        """
        Process a complete settlement from initiation to completion.

        Args:
            source_currency:      ISO code of sending CBDC (e.g., "INR")
            destination_currency: ISO code of receiving CBDC (e.g., "USD")
            send_amount_local:    Amount in source currency units
            sender_id:            Sending institution identifier
            receiver_id:          Receiving institution identifier
            purpose_code:         TRADE / REMITTANCE / INVEST / OTHER
            kyc_verified:         Whether KYC is complete
            entity_type:          retail / institutional / central_bank

        Returns:
            SettlementRecord with full transaction details and outcome.
        """
        tx_id = f"CBDC-{uuid.uuid4().hex[:10].upper()}"
        self.stats["total_initiated"] += 1

        # ----------------------------------------------------------------
        # STEP 1: ROUTE — Find optimal clearing path
        # ----------------------------------------------------------------
        route = self.clearing.find_optimal_route(source_currency, destination_currency)

        if not route.feasible:
            return self._fail(
                tx_id, source_currency, destination_currency,
                send_amount_local, sender_id, receiver_id,
                SettlementStatus.FAILED,
                f"No route available: {route.failure_reason}",
                purpose_code,
            )

        # Convert send amount to USD equivalent for compliance/liquidity checks
        # Using the effective FX rate (compound through path)
        # For USD-denominated sizing: convert via INR/USD rate
        fx_rate_to_usd = route.effective_fx_rate if destination_currency == "USD" else 1.0
        if source_currency != "USD":
            # Approximate USD conversion using network edge
            src_edge = self.clearing.network.get_edge_data(source_currency, "USD")
            if src_edge:
                fx_rate_to_usd = src_edge["fx_rate"]
            else:
                # Use effective rate proxy
                fx_rate_to_usd = route.effective_fx_rate if destination_currency == "USD" else 0.01

        amount_usd = send_amount_local * fx_rate_to_usd
        amount_usd_m = amount_usd / 1_000_000

        # Received amount = send amount × effective compound FX rate
        receive_amount = send_amount_local * route.effective_fx_rate

        # ----------------------------------------------------------------
        # STEP 2: COMPLIANCE — Screen transaction
        # ----------------------------------------------------------------
        from compliance_engine import TransactionProfile

        txn_profile = TransactionProfile(
            tx_id=tx_id,
            sender_id=sender_id,
            receiver_id=receiver_id,
            source_currency=source_currency,
            destination_currency=destination_currency,
            amount_usd_equivalent=amount_usd,
            routing_path=route.path,
            purpose_code=purpose_code,
            kyc_verified=kyc_verified,
            entity_type=entity_type,
        )

        compliance_decision = self.compliance.screen(txn_profile)

        if compliance_decision.status.value == "BLOCKED":
            self.stats["total_rejected"] += 1
            return self._fail(
                tx_id, source_currency, destination_currency,
                send_amount_local, sender_id, receiver_id,
                SettlementStatus.REJECTED,
                f"Compliance blocked: {', '.join(compliance_decision.required_actions[:2])}",
                purpose_code,
                route=route, amount_usd=amount_usd,
                receive_amount=receive_amount,
                compliance_penalty=compliance_decision.penalty_score,
            )

        # ----------------------------------------------------------------
        # STEP 3: LIQUIDITY — Lock funds atomically
        # ----------------------------------------------------------------
        feasible, liq_reason = self.liquidity.check_path_liquidity(route.path, amount_usd_m)

        if not feasible:
            return self._fail(
                tx_id, source_currency, destination_currency,
                send_amount_local, sender_id, receiver_id,
                SettlementStatus.FAILED,
                f"Liquidity constraint: {liq_reason}",
                purpose_code,
                route=route, amount_usd=amount_usd,
                receive_amount=receive_amount,
                compliance_penalty=compliance_decision.penalty_score,
            )

        locked = self.liquidity.lock_path_funds(tx_id, route.path, amount_usd_m)
        if not locked:
            return self._fail(
                tx_id, source_currency, destination_currency,
                send_amount_local, sender_id, receiver_id,
                SettlementStatus.FAILED,
                "Fund locking failed (concurrent settlement conflict)",
                purpose_code,
                route=route, amount_usd=amount_usd,
                receive_amount=receive_amount,
                compliance_penalty=compliance_decision.penalty_score,
            )

        # ----------------------------------------------------------------
        # STEP 4: SETTLE — Atomic execution
        # ----------------------------------------------------------------
        # In production this would be a distributed two-phase commit (2PC)
        # across central bank nodes. Here we simulate with a success probability
        # based on number of hops (more hops → slightly more settlement risk).
        n_hops = len(route.path) - 1
        settlement_success_prob = 0.995 ** n_hops   # 99.5% per hop

        if random.random() > settlement_success_prob:
            # Rare network failure — rollback
            self.liquidity.release_path_funds(tx_id, route.path, amount_usd_m)
            return self._fail(
                tx_id, source_currency, destination_currency,
                send_amount_local, sender_id, receiver_id,
                SettlementStatus.FAILED,
                "Network settlement failure (transient — retry eligible)",
                purpose_code,
                route=route, amount_usd=amount_usd,
                receive_amount=receive_amount,
                compliance_penalty=compliance_decision.penalty_score,
            )

        # ----------------------------------------------------------------
        # STEP 5: RELEASE & RECORD
        # ----------------------------------------------------------------
        self.liquidity.release_path_funds(tx_id, route.path, amount_usd_m)

        # Update network graph liquidity
        for src, tgt in route.corridors:
            self.clearing.network.update_liquidity(src, tgt, -amount_usd_m)

        total_cost_usd = route.estimated_cost_usd_k * 1000

        record = SettlementRecord(
            tx_id=tx_id,
            sender_institution=sender_id,
            receiver_institution=receiver_id,
            source_currency=source_currency,
            destination_currency=destination_currency,
            send_amount_local=send_amount_local,
            receive_amount_local=receive_amount,
            amount_usd_equivalent=amount_usd,
            routing_path=route.path,
            effective_fx_rate=route.effective_fx_rate,
            total_cost_bps=route.total_cost_bps,
            total_cost_usd=total_cost_usd,
            settlement_latency_s=route.total_latency_s,
            status=SettlementStatus.SETTLED,
            compliance_status=compliance_decision.status.value,
            compliance_penalty=compliance_decision.penalty_score,
            fx_risk_score=route.fx_risk_score,
            composite_score=route.composite_score,
            settled_at=datetime.utcnow(),
            purpose_code=purpose_code,
        )

        self.ledger.append(record)
        self.stats["total_settled"] += 1
        self.stats["total_volume_usd"] += amount_usd
        self.stats["total_cost_usd"] += total_cost_usd

        return record

    def _fail(
        self, tx_id, src_ccy, dst_ccy, send_amt, sender, receiver,
        status, reason, purpose, route=None, amount_usd=0,
        receive_amount=0, compliance_penalty=0,
    ) -> SettlementRecord:
        record = SettlementRecord(
            tx_id=tx_id,
            sender_institution=sender,
            receiver_institution=receiver,
            source_currency=src_ccy,
            destination_currency=dst_ccy,
            send_amount_local=send_amt,
            receive_amount_local=receive_amount,
            amount_usd_equivalent=amount_usd,
            routing_path=route.path if route else [],
            effective_fx_rate=route.effective_fx_rate if route else 0,
            total_cost_bps=route.total_cost_bps if route else 0,
            total_cost_usd=0,
            settlement_latency_s=route.total_latency_s if route else 0,
            status=status,
            compliance_penalty=compliance_penalty,
            failure_reason=reason,
            purpose_code=purpose,
        )
        self.ledger.append(record)
        self.stats["total_failed"] += 1
        return record

    def get_ledger_df(self):
        import pandas as pd
        return pd.DataFrame([r.to_dict() for r in self.ledger])

    def get_stats_summary(self) -> Dict:
        total = self.stats["total_initiated"]
        settled = self.stats["total_settled"]
        return {
            **self.stats,
            "success_rate_pct": round(100 * settled / total, 2) if total else 0,
            "avg_cost_per_txn_usd": (
                round(self.stats["total_cost_usd"] / settled, 2) if settled else 0
            ),
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    from cbdc_network import CBDCNetwork
    from clearing_engine import ClearingEngine
    from compliance_engine import ComplianceEngine
    from liquidity_manager import LiquidityManager

    net = CBDCNetwork()
    engine = ClearingEngine(net, transaction_amount_usd_m=10.0)
    compliance = ComplianceEngine()
    liquidity = LiquidityManager()

    simulator = SettlementSimulator(engine, compliance, liquidity)

    print("=== Settling: INR 835,000,000 → USD (≈ $10M) ===\n")
    record = simulator.process_transaction(
        source_currency="INR",
        destination_currency="USD",
        send_amount_local=835_000_000,  # INR 83.5 Cr ≈ $10M
        sender_id="INST_HDFC_001",
        receiver_id="INST_JPMC_001",
        purpose_code="TRADE",
    )

    for k, v in record.to_dict().items():
        print(f"  {k:35s}: {v}")

    print(f"\n\n=== Settlement Statistics ===")
    for k, v in simulator.get_stats_summary().items():
        print(f"  {k}: {v}")
