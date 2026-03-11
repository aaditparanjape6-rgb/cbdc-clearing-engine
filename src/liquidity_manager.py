"""
Liquidity Manager — Module 4
==============================
Simulates liquidity pools at each CBDC node and bilateral corridor.

Financial rationale:
  - In a CBDC network, central banks pre-fund "liquidity pools" at settlement
    nodes, analogous to nostro/vostro account balances in correspondent banking.
  - Transactions must not deplete a corridor's liquidity below its minimum
    threshold (the "liquidity floor") — a safety buffer mandated by central bank
    operating rules (e.g., BIS PFMI Principle 7).
  - Intraday liquidity is recycled as settlements complete (PvP netting effect).

This module tracks liquidity at both node and corridor levels,
enforces thresholds, and triggers rerouting or queuing if needed.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class LiquidityPool:
    """
    Liquidity pool at a CBDC node (central bank settlement account).

    Thresholds inspired by BIS PFMI and mBridge design documents.
    """
    currency_code: str
    available_usd_m: float          # current available balance
    minimum_threshold_usd_m: float  # regulatory minimum (cannot go below)
    maximum_capacity_usd_m: float   # total pool capacity
    # Intraday recycling rate: fraction of settled amounts returned per hour
    recycling_rate: float = 0.85

    @property
    def utilization(self) -> float:
        """Fraction of capacity currently committed."""
        return 1.0 - (self.available_usd_m / self.maximum_capacity_usd_m)

    @property
    def headroom_usd_m(self) -> float:
        """Usable balance above minimum threshold."""
        return max(0.0, self.available_usd_m - self.minimum_threshold_usd_m)

    def can_fund(self, amount_usd_m: float) -> bool:
        return self.headroom_usd_m >= amount_usd_m

    def debit(self, amount_usd_m: float) -> bool:
        """Attempt to debit funds. Returns True if successful."""
        if not self.can_fund(amount_usd_m):
            return False
        self.available_usd_m -= amount_usd_m
        return True

    def credit(self, amount_usd_m: float):
        """Credit settled funds back (inbound settlement or recycling)."""
        self.available_usd_m = min(
            self.maximum_capacity_usd_m,
            self.available_usd_m + amount_usd_m
        )


@dataclass
class CorridorLiquidity:
    """
    Bilateral corridor liquidity pool (PvP settlement link).
    Separate from node pools — represents the joint liquidity lock
    between two central banks for a specific corridor.
    """
    source: str
    target: str
    available_usd_m: float
    minimum_threshold_usd_m: float
    peak_demand_usd_m: float = 0.0     # highest single-day demand seen
    total_settled_usd_m: float = 0.0   # cumulative settled volume

    @property
    def utilization(self) -> float:
        if self.available_usd_m <= 0:
            return 1.0
        baseline = self.available_usd_m + self.total_settled_usd_m
        return self.total_settled_usd_m / baseline if baseline > 0 else 0

    def can_settle(self, amount_usd_m: float) -> bool:
        headroom = self.available_usd_m - self.minimum_threshold_usd_m
        return headroom >= amount_usd_m

    def lock(self, amount_usd_m: float) -> bool:
        """Lock funds for pending settlement."""
        if not self.can_settle(amount_usd_m):
            return False
        self.available_usd_m -= amount_usd_m
        self.peak_demand_usd_m = max(self.peak_demand_usd_m, amount_usd_m)
        return True

    def release(self, amount_usd_m: float):
        """Release locked funds after settlement completion."""
        self.available_usd_m += amount_usd_m
        self.total_settled_usd_m += amount_usd_m


# ---------------------------------------------------------------------------
# Liquidity Manager
# ---------------------------------------------------------------------------

class LiquidityManager:
    """
    Central manager for all node and corridor liquidity pools.

    Responsibilities:
      1. Validate liquidity feasibility before routing.
      2. Lock/release funds during atomic settlement.
      3. Simulate intraday recycling and top-up events.
      4. Report liquidity metrics for dashboard.
    """

    # Node pool configurations (USD millions)
    NODE_POOLS = {
        "USD": {"available": 50_000, "minimum": 10_000, "capacity": 80_000},
        "EUR": {"available": 40_000, "minimum":  8_000, "capacity": 65_000},
        "GBP": {"available": 20_000, "minimum":  4_000, "capacity": 35_000},
        "INR": {"available": 15_000, "minimum":  3_000, "capacity": 25_000},
        "SGD": {"available":  8_000, "minimum":  1_500, "capacity": 15_000},
        "CNY": {"available": 25_000, "minimum":  5_000, "capacity": 45_000},
        "AED": {"available": 12_000, "minimum":  2_000, "capacity": 20_000},
    }

    # Corridor pool configurations (USD millions)
    CORRIDOR_POOLS = {
        ("INR", "USD"): {"available":  8_000, "minimum":  500},
        ("USD", "INR"): {"available":  8_000, "minimum":  500},
        ("USD", "EUR"): {"available": 15_000, "minimum": 1_000},
        ("EUR", "USD"): {"available": 15_000, "minimum": 1_000},
        ("USD", "GBP"): {"available": 12_000, "minimum":  800},
        ("GBP", "USD"): {"available": 12_000, "minimum":  800},
        ("USD", "SGD"): {"available":  6_000, "minimum":  400},
        ("SGD", "USD"): {"available":  6_000, "minimum":  400},
        ("USD", "CNY"): {"available":  5_000, "minimum":  600},
        ("CNY", "USD"): {"available":  5_000, "minimum":  600},
        ("USD", "AED"): {"available": 10_000, "minimum":  600},
        ("AED", "USD"): {"available": 10_000, "minimum":  600},
        ("EUR", "GBP"): {"available": 10_000, "minimum":  700},
        ("GBP", "EUR"): {"available": 10_000, "minimum":  700},
        ("INR", "SGD"): {"available":  4_000, "minimum":  300},
        ("SGD", "INR"): {"available":  4_000, "minimum":  300},
        ("AED", "INR"): {"available":  5_000, "minimum":  350},
        ("INR", "AED"): {"available":  5_000, "minimum":  350},
        ("EUR", "SGD"): {"available":  5_000, "minimum":  350},
        ("SGD", "EUR"): {"available":  5_000, "minimum":  350},
        ("CNY", "AED"): {"available":  3_000, "minimum":  400},
        ("AED", "CNY"): {"available":  3_000, "minimum":  400},
        ("INR", "EUR"): {"available":  4_500, "minimum":  300},
        ("EUR", "INR"): {"available":  4_500, "minimum":  300},
        ("SGD", "CNY"): {"available":  2_500, "minimum":  350},
        ("CNY", "SGD"): {"available":  2_500, "minimum":  350},
        ("GBP", "INR"): {"available":  3_500, "minimum":  250},
        ("INR", "GBP"): {"available":  3_500, "minimum":  250},
        ("AED", "EUR"): {"available":  4_000, "minimum":  300},
        ("EUR", "AED"): {"available":  4_000, "minimum":  300},
    }

    def __init__(self, shock_multiplier: float = 1.0):
        """
        Args:
            shock_multiplier: Scale factor for liquidity (< 1 = stress scenario).
        """
        self.node_pools: Dict[str, LiquidityPool] = {}
        self.corridor_pools: Dict[Tuple[str, str], CorridorLiquidity] = {}
        self._build_pools(shock_multiplier)
        self.events: List[Dict] = []   # audit trail of liquidity events

    def _build_pools(self, shock: float):
        for code, cfg in self.NODE_POOLS.items():
            self.node_pools[code] = LiquidityPool(
                currency_code=code,
                available_usd_m=cfg["available"] * shock,
                minimum_threshold_usd_m=cfg["minimum"],
                maximum_capacity_usd_m=cfg["capacity"],
            )
        for (src, tgt), cfg in self.CORRIDOR_POOLS.items():
            self.corridor_pools[(src, tgt)] = CorridorLiquidity(
                source=src, target=tgt,
                available_usd_m=cfg["available"] * shock,
                minimum_threshold_usd_m=cfg["minimum"],
            )

    # ------------------------------------------------------------------
    # Feasibility Checks
    # ------------------------------------------------------------------

    def check_path_liquidity(
        self, path: List[str], amount_usd_m: float
    ) -> Tuple[bool, str]:
        """
        Verify that all corridors and nodes along a path have sufficient liquidity.
        Returns (feasible, reason_if_not).
        """
        corridors = list(zip(path[:-1], path[1:]))
        for src, tgt in corridors:
            # Check corridor pool
            pool = self.corridor_pools.get((src, tgt))
            if pool and not pool.can_settle(amount_usd_m):
                return False, f"Insufficient corridor liquidity: {src}→{tgt} (need {amount_usd_m:.1f}M, headroom {max(0,pool.available_usd_m - pool.minimum_threshold_usd_m):.1f}M)"

            # Check source node pool
            src_pool = self.node_pools.get(src)
            if src_pool and not src_pool.can_fund(amount_usd_m):
                return False, f"Insufficient node liquidity at {src} (need {amount_usd_m:.1f}M, headroom {src_pool.headroom_usd_m:.1f}M)"

        return True, ""

    # ------------------------------------------------------------------
    # Settlement Lifecycle (Lock / Release)
    # ------------------------------------------------------------------

    def lock_path_funds(
        self, tx_id: str, path: List[str], amount_usd_m: float
    ) -> bool:
        """
        Atomically lock funds on all corridors in the path.
        If any corridor fails, roll back all previously locked corridors.
        """
        corridors = list(zip(path[:-1], path[1:]))
        locked: List[Tuple[str, str]] = []

        for src, tgt in corridors:
            pool = self.corridor_pools.get((src, tgt))
            node_pool = self.node_pools.get(src)

            corridor_ok = (pool is None) or pool.lock(amount_usd_m)
            node_ok = (node_pool is None) or node_pool.debit(amount_usd_m)

            if not (corridor_ok and node_ok):
                # Rollback
                for r_src, r_tgt in locked:
                    rp = self.corridor_pools.get((r_src, r_tgt))
                    rnp = self.node_pools.get(r_src)
                    if rp:
                        rp.release(amount_usd_m)
                    if rnp:
                        rnp.credit(amount_usd_m)
                self._log_event(tx_id, "LOCK_FAILED", path, amount_usd_m,
                                f"Failed at corridor {src}→{tgt}")
                return False
            locked.append((src, tgt))

        self._log_event(tx_id, "LOCKED", path, amount_usd_m)
        return True

    def release_path_funds(
        self, tx_id: str, path: List[str], amount_usd_m: float
    ):
        """
        Release (recycle) funds after successful settlement.
        The destination node receives the credit.
        """
        corridors = list(zip(path[:-1], path[1:]))
        for src, tgt in corridors:
            pool = self.corridor_pools.get((src, tgt))
            if pool:
                pool.release(amount_usd_m)
            # Credit destination node
            tgt_pool = self.node_pools.get(tgt)
            if tgt_pool:
                tgt_pool.credit(amount_usd_m * self.node_pools[src].recycling_rate
                                if src in self.node_pools else amount_usd_m)

        self._log_event(tx_id, "RELEASED", path, amount_usd_m)

    # ------------------------------------------------------------------
    # Stress Simulation
    # ------------------------------------------------------------------

    def apply_liquidity_shock(self, severity: float = 0.30):
        """
        Simulate a liquidity shock by draining a fraction of all pools.
        Used in stress testing scenarios.
        severity: fraction of available liquidity to remove (0–1).
        """
        for pool in self.node_pools.values():
            drain = pool.available_usd_m * severity
            pool.available_usd_m = max(pool.minimum_threshold_usd_m,
                                       pool.available_usd_m - drain)
        for cpool in self.corridor_pools.values():
            drain = cpool.available_usd_m * severity
            cpool.available_usd_m = max(cpool.minimum_threshold_usd_m,
                                        cpool.available_usd_m - drain)

    def simulate_intraday_recycling(self, hours: float = 1.0):
        """
        Simulate intraday liquidity recycling (PvP netting returns liquidity).
        In real systems this happens continuously; here we advance by `hours`.
        """
        for pool in self.node_pools.values():
            # Simple recycling model: fraction of locked amounts return
            recycled = pool.recycling_rate * pool.utilization * pool.maximum_capacity_usd_m * (hours / 8)
            pool.credit(recycled)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_node_liquidity_df(self) -> pd.DataFrame:
        rows = []
        for code, pool in self.node_pools.items():
            rows.append({
                "currency": code,
                "available_usd_m": pool.available_usd_m,
                "minimum_usd_m": pool.minimum_threshold_usd_m,
                "capacity_usd_m": pool.maximum_capacity_usd_m,
                "headroom_usd_m": pool.headroom_usd_m,
                "utilization_pct": pool.utilization * 100,
            })
        return pd.DataFrame(rows).set_index("currency")

    def get_corridor_liquidity_df(self) -> pd.DataFrame:
        rows = []
        for (src, tgt), pool in self.corridor_pools.items():
            rows.append({
                "corridor": f"{src}→{tgt}",
                "source": src,
                "target": tgt,
                "available_usd_m": pool.available_usd_m,
                "minimum_usd_m": pool.minimum_threshold_usd_m,
                "utilization_pct": pool.utilization * 100,
                "total_settled_usd_m": pool.total_settled_usd_m,
            })
        return pd.DataFrame(rows)

    def _log_event(self, tx_id: str, event_type: str, path: List[str],
                   amount: float, note: str = ""):
        self.events.append({
            "timestamp": datetime.utcnow().isoformat(),
            "tx_id": tx_id,
            "event": event_type,
            "path": "→".join(path),
            "amount_usd_m": amount,
            "note": note,
        })


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    lm = LiquidityManager()
    print("=== Node Liquidity Status ===")
    print(lm.get_node_liquidity_df().to_string())

    ok, reason = lm.check_path_liquidity(["INR", "USD"], 500)
    print(f"\nPath feasibility (INR→USD, $500M): {ok} | {reason}")

    ok2, reason2 = lm.check_path_liquidity(["INR", "USD"], 20_000)
    print(f"Path feasibility (INR→USD, $20B): {ok2} | {reason2}")
