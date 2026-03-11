"""
CBDC Network Model — Module 1
==============================
Models the global CBDC ecosystem as a directed weighted graph.
Each node = a sovereign CBDC system; each edge = a bilateral settlement corridor.

Financial rationale:
  - Directed edges because FX rates and regulatory flows are asymmetric
    (e.g., India→USA may have different constraints than USA→India).
  - Edge weights encode the multi-dimensional cost of a cross-border transfer:
      cost, risk, time, and regulatory friction.
"""

import networkx as nx
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class CBDCNode:
    """Represents a sovereign CBDC system (central bank digital currency node)."""
    country: str
    currency: str
    currency_code: str
    # Central bank credibility score (0-1); affects risk weighting
    cb_credibility: float = 1.0
    # GDP (USD trillion) — proxy for systemic importance
    gdp_usd_trillion: float = 1.0
    # AML/CFT risk rating (FATF): 0=low, 1=medium, 2=high, 3=blacklist
    aml_risk_rating: int = 0
    # Daily CBDC transaction volume capacity (USD millions)
    daily_volume_capacity_usd_m: float = 10_000.0


@dataclass
class CorridorEdge:
    """
    Represents a bilateral CBDC settlement corridor.
    Each attribute is a cost dimension in the routing objective function.
    """
    source: str          # ISO currency code of sending CBDC
    target: str          # ISO currency code of receiving CBDC
    fx_rate: float       # Units of target per 1 unit of source
    # Transaction cost in basis points (1 bp = 0.01%)
    transaction_cost_bps: float = 10.0
    # Expected settlement latency in seconds (T+0 to T+2 range)
    settlement_latency_s: float = 30.0
    # Available liquidity in USD millions for this corridor
    liquidity_usd_m: float = 5_000.0
    # Regulatory friction score 0-1 (0=frictionless, 1=maximum friction)
    regulatory_friction: float = 0.1
    # Whether a direct CBDC bridge / PvP link exists (Payment vs Payment)
    direct_pvp_link: bool = True


# ---------------------------------------------------------------------------
# Network Builder
# ---------------------------------------------------------------------------

class CBDCNetwork:
    """
    Constructs and manages the global CBDC network graph.

    The graph is a MultiDiGraph to allow multiple parallel corridors
    (e.g., direct and via correspondent) between the same node pair.
    """

    # Node definitions: country, currency name, ISO code, credibility, GDP, AML rating
    NODES: List[CBDCNode] = [
        CBDCNode("India",     "Digital Rupee",     "INR", 0.85, 3.7,   0),
        CBDCNode("USA",       "Digital Dollar",    "USD", 0.99, 26.9,  0),
        CBDCNode("EU",        "Digital Euro",      "EUR", 0.97, 17.1,  0),
        CBDCNode("UK",        "Digital Pound",     "GBP", 0.95, 3.1,   0),
        CBDCNode("Singapore", "Digital SGD",       "SGD", 0.98, 0.5,   0),
        CBDCNode("China",     "Digital Yuan (e-CNY)","CNY", 0.80, 17.7, 1),
        CBDCNode("UAE",       "Digital Dirham",    "AED", 0.90, 0.5,   0),
    ]

    # Bilateral corridors (source, target, fx_rate, cost_bps, latency_s, liquidity_m, friction)
    # FX rates are approximate; simulation will overlay a dynamic model.
    EDGES: List[Tuple] = [
        # Corridor format: (src, tgt, fx, cost_bps, latency_s, liquidity_m, friction, pvp)
        ("INR", "USD", 0.01198, 12, 45,  8_000,  0.10, True),
        ("USD", "INR", 83.5,   10, 40,  8_000,  0.10, True),
        ("USD", "EUR", 0.920,   6,  20, 15_000,  0.05, True),
        ("EUR", "USD", 1.087,   6,  20, 15_000,  0.05, True),
        ("USD", "GBP", 0.790,   7,  22, 12_000,  0.05, True),
        ("GBP", "USD", 1.266,   7,  22, 12_000,  0.05, True),
        ("USD", "SGD", 1.340,   8,  25,  6_000,  0.05, True),
        ("SGD", "USD", 0.746,   8,  25,  6_000,  0.05, True),
        ("USD", "CNY", 7.240,  18,  60,  5_000,  0.35, False),
        ("CNY", "USD", 0.138,  18,  60,  5_000,  0.35, False),
        ("USD", "AED", 3.671,   5,  15, 10_000,  0.08, True),
        ("AED", "USD", 0.272,   5,  15, 10_000,  0.08, True),
        ("EUR", "GBP", 0.859,   6,  18, 10_000,  0.06, True),
        ("GBP", "EUR", 1.164,   6,  18, 10_000,  0.06, True),
        ("EUR", "SGD", 1.457,   9,  28,  5_000,  0.08, True),
        ("SGD", "EUR", 0.686,   9,  28,  5_000,  0.08, True),
        ("INR", "SGD", 0.01605, 10, 35,  4_000,  0.12, True),  # Mbridge-style
        ("SGD", "INR", 62.31,  10,  35,  4_000,  0.12, True),
        ("AED", "INR", 22.74,   8,  20,  5_000,  0.10, True),
        ("INR", "AED", 0.04397, 8,  20,  5_000,  0.10, True),
        ("CNY", "AED", 0.507,  15,  50,  3_000,  0.28, False),
        ("AED", "CNY", 1.972,  15,  50,  3_000,  0.28, False),
        ("INR", "EUR", 0.01106,14,  55,  4_500,  0.15, True),
        ("EUR", "INR", 90.44,  14,  55,  4_500,  0.15, True),
        ("SGD", "CNY", 5.403,  20,  70,  2_500,  0.32, False),
        ("CNY", "SGD", 0.185,  20,  70,  2_500,  0.32, False),
        ("GBP", "INR", 105.7,  13,  50,  3_500,  0.14, True),
        ("INR", "GBP", 0.00946,13, 50,  3_500,  0.14, True),
        ("AED", "EUR", 0.2504,  9,  30,  4_000,  0.09, True),
        ("EUR", "AED", 3.992,   9,  30,  4_000,  0.09, True),
    ]

    def __init__(self, dynamic_fx: Optional[Dict[str, float]] = None):
        """
        Args:
            dynamic_fx: Optional override of FX rates {pair: rate},
                        e.g. {"INR/USD": 0.0121}. Used for stress testing.
        """
        self.graph = nx.DiGraph()
        self.node_map: Dict[str, CBDCNode] = {}  # code → CBDCNode
        self._build_network(dynamic_fx)

    def _build_network(self, dynamic_fx: Optional[Dict[str, float]] = None):
        """Populate graph nodes and edges from static definitions."""

        # Add nodes
        for node in self.NODES:
            self.node_map[node.currency_code] = node
            self.graph.add_node(
                node.currency_code,
                country=node.country,
                currency=node.currency,
                cb_credibility=node.cb_credibility,
                gdp=node.gdp_usd_trillion,
                aml_rating=node.aml_risk_rating,
                daily_capacity=node.daily_volume_capacity_usd_m,
            )

        # Add edges
        for edge_def in self.EDGES:
            src, tgt, fx, cost_bps, lat_s, liq_m, friction, pvp = edge_def

            # Apply dynamic FX override if provided
            pair_key = f"{src}/{tgt}"
            if dynamic_fx and pair_key in dynamic_fx:
                fx = dynamic_fx[pair_key]

            # Composite weight for Dijkstra (primary: cost; details stored as attrs)
            # Weight = transaction_cost_bps (used for simple shortest path)
            self.graph.add_edge(
                src, tgt,
                fx_rate=fx,
                transaction_cost_bps=cost_bps,
                settlement_latency_s=lat_s,
                liquidity_usd_m=liq_m,
                regulatory_friction=friction,
                direct_pvp=pvp,
                weight=cost_bps,  # default weight for nx algorithms
            )

    def get_edge_data(self, src: str, tgt: str) -> Optional[Dict]:
        """Return edge attributes for a corridor."""
        return self.graph.get_edge_data(src, tgt)

    def update_liquidity(self, src: str, tgt: str, delta_usd_m: float):
        """Decrease (or increase) liquidity on a corridor after settlement."""
        data = self.get_edge_data(src, tgt)
        if data:
            data["liquidity_usd_m"] = max(0, data["liquidity_usd_m"] + delta_usd_m)

    def get_network_summary(self) -> pd.DataFrame:
        """Return edge attributes as a DataFrame for analysis/display."""
        rows = []
        for src, tgt, attrs in self.graph.edges(data=True):
            rows.append({
                "corridor": f"{src}→{tgt}",
                "source": src,
                "target": tgt,
                **attrs,
            })
        return pd.DataFrame(rows)

    def available_nodes(self) -> List[str]:
        return list(self.graph.nodes())

    def available_corridors(self) -> List[Tuple[str, str]]:
        return list(self.graph.edges())


# ---------------------------------------------------------------------------
# Quick Sanity Check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    net = CBDCNetwork()
    print(f"Nodes: {net.available_nodes()}")
    print(f"Edges: {len(net.available_corridors())}")
    print("\nNetwork Summary (first 5 corridors):")
    print(net.get_network_summary().head())
