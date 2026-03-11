"""
Clearing Engine — Module 2
===========================
Multi-objective routing engine for cross-border CBDC settlement.

Optimization objective:
    F = α·Cost + β·Risk + γ·Time + δ·CompliancePenalty

The engine finds the Pareto-optimal settlement path by converting the
multi-objective problem into a single composite weight per edge, then
applying Dijkstra's shortest-path algorithm on the directed graph.

This mirrors the approach used by ISO 20022-compliant payment networks
and proposed mBridge-style CBDC routing frameworks.
"""

import networkx as nx
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from cbdc_network import CBDCNetwork


# ---------------------------------------------------------------------------
# Routing Parameters
# ---------------------------------------------------------------------------

@dataclass
class RoutingWeights:
    """
    Multi-objective weighting coefficients (must sum to 1.0 for interpretability,
    though the engine normalizes internally).

    α (alpha) — cost sensitivity: weight on transaction cost (basis points)
    β (beta)  — risk sensitivity: weight on FX volatility / risk score
    γ (gamma) — time sensitivity: weight on settlement latency
    δ (delta) — compliance sensitivity: weight on regulatory friction penalty
    """
    alpha: float = 0.35   # cost
    beta: float  = 0.25   # FX risk
    gamma: float = 0.20   # settlement time
    delta: float = 0.20   # compliance/regulatory


@dataclass
class RouteResult:
    """Encapsulates a routing solution."""
    source: str
    destination: str
    path: List[str]
    corridors: List[Tuple[str, str]]
    # Absolute costs along the path
    total_cost_bps: float
    total_latency_s: float
    fx_risk_score: float
    compliance_penalty: float
    composite_score: float
    # Effective FX rate for the full path (product of corridor rates)
    effective_fx_rate: float
    # USD equivalent cost for a $1M transfer
    estimated_cost_usd_k: float = 0.0
    feasible: bool = True
    failure_reason: str = ""

    def to_dict(self) -> Dict:
        return {
            "Route": " → ".join(self.path),
            "Hops": len(self.path) - 1,
            "Total Cost (bps)": round(self.total_cost_bps, 2),
            "Latency (s)": round(self.total_latency_s, 1),
            "FX Risk Score": round(self.fx_risk_score, 4),
            "Compliance Penalty": round(self.compliance_penalty, 4),
            "Composite Score": round(self.composite_score, 4),
            "Effective FX Rate": round(self.effective_fx_rate, 6),
            "Est. Cost per $1M (USD k)": round(self.estimated_cost_usd_k, 2),
            "Feasible": self.feasible,
        }


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

# Normalization bounds derived from the CBDC network edge definition ranges.
# These convert raw values into [0, 1] scores for multi-objective comparison.
NORM = {
    "cost_bps": (5.0, 25.0),       # min/max transaction cost in bps
    "latency_s": (15.0, 120.0),    # min/max settlement latency
    "friction": (0.0, 1.0),        # regulatory friction already [0,1]
    "risk_score": (0.0, 1.0),      # FX risk score already [0,1]
}


def normalize(value: float, lo: float, hi: float) -> float:
    """Min-max normalize a value to [0, 1]."""
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


# ---------------------------------------------------------------------------
# Clearing Engine
# ---------------------------------------------------------------------------

class ClearingEngine:
    """
    Multi-objective clearing and routing engine.

    Workflow:
      1. Build a composite-weight graph from the CBDC network.
      2. Use Dijkstra to find the optimal path.
      3. Evaluate path metrics and return a RouteResult.
      4. Optionally enumerate K-best alternative routes.
    """

    def __init__(
        self,
        network: CBDCNetwork,
        weights: Optional[RoutingWeights] = None,
        fx_risk_scores: Optional[Dict[str, float]] = None,
        transaction_amount_usd_m: float = 1.0,
    ):
        """
        Args:
            network: The CBDC network graph.
            weights: Multi-objective weighting coefficients.
            fx_risk_scores: Dict of {pair: risk_score} from risk model.
                            If None, friction-based proxy is used.
            transaction_amount_usd_m: Transaction size in USD millions
                                       (affects liquidity feasibility check).
        """
        self.network = network
        self.weights = weights or RoutingWeights()
        self.fx_risk_scores = fx_risk_scores or {}
        self.tx_amount = transaction_amount_usd_m
        # Build the composite-weight graph once
        self._composite_graph = self._build_composite_graph()

    def _get_fx_risk(self, src: str, tgt: str) -> float:
        """
        Retrieve FX risk score for a corridor.
        Falls back to regulatory_friction as a proxy if model scores unavailable.
        """
        pair = f"{src}/{tgt}"
        if pair in self.fx_risk_scores:
            return self.fx_risk_scores[pair]
        # Proxy: higher friction corridors typically have higher FX risk
        edge = self.network.get_edge_data(src, tgt)
        return edge.get("regulatory_friction", 0.1) if edge else 0.1

    def _composite_weight(self, src: str, tgt: str) -> float:
        """
        Compute composite edge weight for routing:
            w = α·norm(cost) + β·norm(risk) + γ·norm(latency) + δ·norm(friction)
        """
        edge = self.network.get_edge_data(src, tgt)
        if not edge:
            return float("inf")

        # Liquidity feasibility check — skip edges with insufficient liquidity
        if edge["liquidity_usd_m"] < self.tx_amount:
            return float("inf")  # infeasible corridor

        w = self.weights

        cost_norm     = normalize(edge["transaction_cost_bps"],  *NORM["cost_bps"])
        latency_norm  = normalize(edge["settlement_latency_s"],  *NORM["latency_s"])
        friction_norm = normalize(edge["regulatory_friction"],   *NORM["friction"])
        risk_norm     = normalize(self._get_fx_risk(src, tgt),   *NORM["risk_score"])

        return (
            w.alpha * cost_norm
            + w.beta  * risk_norm
            + w.gamma * latency_norm
            + w.delta * friction_norm
        )

    def _build_composite_graph(self) -> nx.DiGraph:
        """Build a new DiGraph with composite weights for Dijkstra."""
        G = nx.DiGraph()
        for src, tgt in self.network.graph.edges():
            cw = self._composite_weight(src, tgt)
            if cw < float("inf"):
                G.add_edge(src, tgt, weight=cw)
        return G

    def refresh_graph(self):
        """Re-build composite graph (call after liquidity/rate updates)."""
        self._composite_graph = self._build_composite_graph()

    # ------------------------------------------------------------------
    # Primary Routing API
    # ------------------------------------------------------------------

    def find_optimal_route(self, source: str, destination: str) -> RouteResult:
        """
        Find the optimal settlement route using Dijkstra's algorithm
        on the composite-weight graph.

        Returns a RouteResult with full path analytics.
        """
        if source not in self.network.graph:
            return self._infeasible(source, destination, f"Source '{source}' not in network")
        if destination not in self.network.graph:
            return self._infeasible(source, destination, f"Destination '{destination}' not in network")

        try:
            path = nx.dijkstra_path(self._composite_graph, source, destination, weight="weight")
        except nx.NetworkXNoPath:
            return self._infeasible(source, destination, "No feasible path (liquidity/regulatory constraints)")
        except nx.NodeNotFound as e:
            return self._infeasible(source, destination, str(e))

        return self._evaluate_path(source, destination, path)

    def find_k_best_routes(self, source: str, destination: str, k: int = 3) -> List[RouteResult]:
        """
        Find K simple paths sorted by composite score (Yen's K-shortest paths concept).
        Useful for providing alternative routes for resilience.
        """
        try:
            all_paths = list(nx.shortest_simple_paths(
                self._composite_graph, source, destination, weight="weight"
            ))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [self._infeasible(source, destination, "No feasible path")]

        results = []
        for path in all_paths[:k]:
            results.append(self._evaluate_path(source, destination, path))
        return results

    def _evaluate_path(self, source: str, destination: str, path: List[str]) -> RouteResult:
        """Compute all metrics for a given routing path."""
        corridors = list(zip(path[:-1], path[1:]))

        total_cost_bps   = 0.0
        total_latency_s  = 0.0
        total_friction   = 0.0
        total_risk       = 0.0
        effective_fx     = 1.0
        composite        = 0.0

        w = self.weights

        for src, tgt in corridors:
            edge = self.network.get_edge_data(src, tgt)
            if not edge:
                return self._infeasible(source, destination, f"Missing corridor {src}→{tgt}")

            cost  = edge["transaction_cost_bps"]
            lat   = edge["settlement_latency_s"]
            fric  = edge["regulatory_friction"]
            risk  = self._get_fx_risk(src, tgt)
            fx    = edge["fx_rate"]

            total_cost_bps  += cost
            total_latency_s += lat
            total_friction  += fric
            total_risk      += risk
            effective_fx    *= fx  # compound FX conversion

            # Composite contribution
            composite += (
                w.alpha * normalize(cost, *NORM["cost_bps"])
                + w.beta  * normalize(risk, *NORM["risk_score"])
                + w.gamma * normalize(lat,  *NORM["latency_s"])
                + w.delta * normalize(fric, *NORM["friction"])
            )

        # Average risk and friction scores over hops
        n_hops = len(corridors)
        avg_risk    = total_risk    / n_hops if n_hops else 0
        avg_friction = total_friction / n_hops if n_hops else 0

        # Estimated USD cost for the transaction
        # cost_bps are applied to the notional; compound across hops
        estimated_cost_usd_k = self.tx_amount * 1000 * (total_cost_bps / 10_000)

        return RouteResult(
            source=source,
            destination=destination,
            path=path,
            corridors=corridors,
            total_cost_bps=total_cost_bps,
            total_latency_s=total_latency_s,
            fx_risk_score=avg_risk,
            compliance_penalty=avg_friction,
            composite_score=composite,
            effective_fx_rate=effective_fx,
            estimated_cost_usd_k=estimated_cost_usd_k,
            feasible=True,
        )

    def _infeasible(self, source: str, destination: str, reason: str) -> RouteResult:
        return RouteResult(
            source=source, destination=destination,
            path=[], corridors=[],
            total_cost_bps=0, total_latency_s=0,
            fx_risk_score=0, compliance_penalty=0,
            composite_score=float("inf"),
            effective_fx_rate=0,
            feasible=False,
            failure_reason=reason,
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    net = CBDCNetwork()
    engine = ClearingEngine(net, transaction_amount_usd_m=10.0)

    result = engine.find_optimal_route("INR", "USD")
    print("=== Optimal Route: INR → USD ===")
    for k, v in result.to_dict().items():
        print(f"  {k}: {v}")

    print("\n=== Top 3 Alternative Routes: INR → USD ===")
    for i, r in enumerate(engine.find_k_best_routes("INR", "USD", k=3), 1):
        print(f"\n  Route {i}: {' → '.join(r.path)}")
        print(f"    Cost: {r.total_cost_bps:.1f} bps | Latency: {r.total_latency_s:.0f}s | Score: {r.composite_score:.4f}")
