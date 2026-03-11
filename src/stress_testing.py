"""
Stress Testing Engine — Module 7
==================================
Monte Carlo simulation framework to evaluate CBDC network resilience.

Stress scenarios modeled:
  S1: FX Volatility Shock     — sudden exchange rate moves (±5–25%)
  S2: Liquidity Drought       — corridor liquidity depleted by 20–70%
  S3: Network Congestion      — settlement latency spikes, corridor failures
  S4: Combined Systemic Shock — all three simultaneously

Each scenario runs N independent Monte Carlo iterations.
The engine reports:
  - Settlement success rate
  - Average total cost (bps)
  - Average latency (s)
  - Routing stability (% of trials using same path as baseline)
  - VaR-95 on cost (Value at Risk)
"""

import copy
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scenario Definitions
# ---------------------------------------------------------------------------

@dataclass
class StressScenario:
    """Parameters for a single stress test scenario."""
    name: str
    fx_shock_range: Tuple[float, float] = (0.0, 0.0)   # min/max % change in FX rates
    liquidity_drain: Tuple[float, float] = (0.0, 0.0)  # fraction of liquidity removed
    latency_multiplier: Tuple[float, float] = (1.0, 1.0)  # latency spike factor
    cost_multiplier: Tuple[float, float] = (1.0, 1.0)     # cost spike factor
    corridor_failure_prob: float = 0.0   # probability a single corridor goes offline
    description: str = ""


SCENARIOS = {
    "baseline": StressScenario(
        name="Baseline",
        description="Normal operating conditions — no stress applied",
    ),
    "fx_shock_mild": StressScenario(
        name="FX Shock (Mild)",
        fx_shock_range=(-0.05, 0.05),
        description="±5% FX rate shock — typical EM currency move",
    ),
    "fx_shock_severe": StressScenario(
        name="FX Shock (Severe)",
        fx_shock_range=(-0.25, 0.15),
        cost_multiplier=(1.5, 3.0),
        description="−25%/+15% FX shock — crisis-level currency dislocation",
    ),
    "liquidity_mild": StressScenario(
        name="Liquidity Drought (Mild)",
        liquidity_drain=(0.20, 0.35),
        description="20–35% liquidity drain — moderate funding stress",
    ),
    "liquidity_severe": StressScenario(
        name="Liquidity Drought (Severe)",
        liquidity_drain=(0.50, 0.70),
        description="50–70% liquidity drain — Lehman-scale funding crisis",
    ),
    "network_congestion": StressScenario(
        name="Network Congestion",
        latency_multiplier=(2.0, 5.0),
        cost_multiplier=(1.2, 2.0),
        corridor_failure_prob=0.10,
        description="Heavy congestion — 10% corridor failure, 2–5× latency",
    ),
    "systemic_shock": StressScenario(
        name="Systemic Crisis",
        fx_shock_range=(-0.20, 0.10),
        liquidity_drain=(0.40, 0.60),
        latency_multiplier=(2.0, 4.0),
        cost_multiplier=(1.5, 2.5),
        corridor_failure_prob=0.15,
        description="Full systemic crisis — all stress factors combined",
    ),
}


# ---------------------------------------------------------------------------
# Simulation Result
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Aggregated results from N Monte Carlo trials for one scenario."""
    scenario_name: str
    n_trials: int
    source: str
    destination: str
    amount_usd_m: float

    # Success metrics
    success_count: int = 0
    failure_count: int = 0
    success_rate: float = 0.0

    # Cost distribution (bps)
    cost_mean: float = 0.0
    cost_std: float = 0.0
    cost_var95: float = 0.0    # 95th percentile cost (VaR-95)
    cost_min: float = 0.0
    cost_max: float = 0.0

    # Latency distribution (seconds)
    latency_mean: float = 0.0
    latency_std: float = 0.0
    latency_p95: float = 0.0

    # Routing stability
    baseline_path: List[str] = field(default_factory=list)
    path_stability_pct: float = 0.0   # % trials using same path as baseline

    # Raw trial data
    trial_costs: List[float] = field(default_factory=list)
    trial_latencies: List[float] = field(default_factory=list)
    trial_paths: List[str] = field(default_factory=list)
    trial_success: List[bool] = field(default_factory=list)

    def summarize(self) -> Dict:
        return {
            "scenario": self.scenario_name,
            "n_trials": self.n_trials,
            "success_rate_%": round(self.success_rate * 100, 2),
            "cost_mean_bps": round(self.cost_mean, 2),
            "cost_std_bps": round(self.cost_std, 2),
            "cost_VaR95_bps": round(self.cost_var95, 2),
            "latency_mean_s": round(self.latency_mean, 1),
            "latency_p95_s": round(self.latency_p95, 1),
            "path_stability_%": round(self.path_stability_pct * 100, 2),
        }


# ---------------------------------------------------------------------------
# Monte Carlo Engine
# ---------------------------------------------------------------------------

class StressTestingEngine:
    """
    Monte Carlo stress testing for the CBDC clearing network.

    Each trial:
      1. Clones the network with scenario-adjusted parameters
      2. Runs the clearing engine to find optimal route
      3. Simulates settlement outcome
      4. Records metrics
    """

    def __init__(
        self,
        network_class,
        clearing_engine_class,
        routing_weights=None,
        n_trials: int = 1000,
        random_seed: int = 42,
    ):
        """
        Args:
            network_class:        CBDCNetwork class (not instance — instantiated per trial)
            clearing_engine_class: ClearingEngine class
            routing_weights:      RoutingWeights for ClearingEngine
            n_trials:             Number of Monte Carlo trials per scenario
            random_seed:          For reproducibility
        """
        self.NetworkClass = network_class
        self.ClearingClass = clearing_engine_class
        self.routing_weights = routing_weights
        self.n_trials = n_trials
        np.random.seed(random_seed)
        random.seed(random_seed)

    def run_scenario(
        self,
        scenario: StressScenario,
        source: str,
        destination: str,
        amount_usd_m: float,
    ) -> SimulationResult:
        """Run N Monte Carlo trials for a single stress scenario."""
        result = SimulationResult(
            scenario_name=scenario.name,
            n_trials=self.n_trials,
            source=source,
            destination=destination,
            amount_usd_m=amount_usd_m,
        )

        # Get baseline path
        baseline_net = self.NetworkClass()
        baseline_eng = self.ClearingClass(baseline_net, self.routing_weights, transaction_amount_usd_m=amount_usd_m)
        baseline_route = baseline_eng.find_optimal_route(source, destination)
        result.baseline_path = baseline_route.path
        baseline_path_str = "→".join(baseline_route.path)

        for trial in range(self.n_trials):
            try:
                success, cost_bps, latency_s, path_str = self._run_trial(
                    scenario, source, destination, amount_usd_m
                )
                result.trial_success.append(success)
                result.trial_paths.append(path_str)
                if success:
                    result.trial_costs.append(cost_bps)
                    result.trial_latencies.append(latency_s)
            except Exception as e:
                result.trial_success.append(False)
                result.trial_paths.append("")

        # Aggregate results
        result.success_count = sum(result.trial_success)
        result.failure_count = self.n_trials - result.success_count
        result.success_rate = result.success_count / self.n_trials

        if result.trial_costs:
            costs = np.array(result.trial_costs)
            lats  = np.array(result.trial_latencies)
            result.cost_mean   = float(np.mean(costs))
            result.cost_std    = float(np.std(costs))
            result.cost_var95  = float(np.percentile(costs, 95))
            result.cost_min    = float(np.min(costs))
            result.cost_max    = float(np.max(costs))
            result.latency_mean = float(np.mean(lats))
            result.latency_std  = float(np.std(lats))
            result.latency_p95  = float(np.percentile(lats, 95))

        # Path stability
        path_matches = sum(1 for p in result.trial_paths if p == baseline_path_str)
        result.path_stability_pct = path_matches / self.n_trials

        return result

    def _run_trial(
        self,
        scenario: StressScenario,
        source: str,
        destination: str,
        amount_usd_m: float,
    ) -> Tuple[bool, float, float, str]:
        """
        Single Monte Carlo trial.
        Returns (success, cost_bps, latency_s, path_str).
        """
        # Build perturbed network
        dynamic_fx = self._generate_fx_shock(scenario)
        net = self.NetworkClass(dynamic_fx=dynamic_fx)

        # Apply corridor failures (remove from graph)
        if scenario.corridor_failure_prob > 0:
            failed_corridors = []
            for src, tgt in list(net.graph.edges()):
                if random.random() < scenario.corridor_failure_prob:
                    failed_corridors.append((src, tgt))
            for src, tgt in failed_corridors:
                if net.graph.has_edge(src, tgt):
                    net.graph.remove_edge(src, tgt)

        # Apply liquidity drain
        liq_drain = 0.0
        if scenario.liquidity_drain != (0.0, 0.0):
            liq_drain = random.uniform(*scenario.liquidity_drain)
            # Reduce edge liquidity
            for src, tgt in net.graph.edges():
                data = net.graph[src][tgt]
                data["liquidity_usd_m"] = max(0, data["liquidity_usd_m"] * (1 - liq_drain))

        # Apply cost multiplier
        if scenario.cost_multiplier != (1.0, 1.0):
            mult = random.uniform(*scenario.cost_multiplier)
            for src, tgt in net.graph.edges():
                net.graph[src][tgt]["transaction_cost_bps"] *= mult

        # Apply latency multiplier
        if scenario.latency_multiplier != (1.0, 1.0):
            lat_mult = random.uniform(*scenario.latency_multiplier)
            for src, tgt in net.graph.edges():
                net.graph[src][tgt]["settlement_latency_s"] *= lat_mult

        # Run clearing engine on perturbed network
        eng = self.ClearingClass(net, self.routing_weights, transaction_amount_usd_m=amount_usd_m)
        route = eng.find_optimal_route(source, destination)

        if not route.feasible:
            return False, 0.0, 0.0, ""

        # Simulate settlement outcome (slight random failure for realism)
        n_hops = len(route.path) - 1
        success_prob = 0.995 ** n_hops
        if random.random() > success_prob:
            return False, 0.0, 0.0, ""

        path_str = "→".join(route.path)
        return True, route.total_cost_bps, route.total_latency_s, path_str

    def _generate_fx_shock(self, scenario: StressScenario) -> Optional[Dict[str, float]]:
        """Generate shocked FX rates for the network."""
        if scenario.fx_shock_range == (0.0, 0.0):
            return None

        from cbdc_network import CBDCNetwork
        base_net = CBDCNetwork()
        shocked_fx = {}

        for src, tgt, data in base_net.graph.edges(data=True):
            shock = random.uniform(*scenario.fx_shock_range)
            shocked_fx[f"{src}/{tgt}"] = data["fx_rate"] * (1 + shock)

        return shocked_fx

    def run_all_scenarios(
        self,
        source: str = "INR",
        destination: str = "USD",
        amount_usd_m: float = 10.0,
        scenario_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Run all (or selected) scenarios and return a summary DataFrame.
        """
        target = scenario_names or list(SCENARIOS.keys())
        summaries = []

        for name in target:
            if name not in SCENARIOS:
                print(f"  Unknown scenario: {name}")
                continue
            scenario = SCENARIOS[name]
            print(f"  Running scenario: {scenario.name} ({self.n_trials} trials)...")
            result = self.run_scenario(scenario, source, destination, amount_usd_m)
            summaries.append(result.summarize())
            print(f"    → Success: {result.success_rate*100:.1f}% | "
                  f"Cost: {result.cost_mean:.1f} bps | "
                  f"Latency: {result.latency_mean:.0f}s | "
                  f"Path stability: {result.path_stability_pct*100:.1f}%")

        return pd.DataFrame(summaries)

    def get_scenario_detail(
        self,
        scenario_name: str,
        source: str = "INR",
        destination: str = "USD",
        amount_usd_m: float = 10.0,
    ) -> SimulationResult:
        """Run and return full detail for a single named scenario."""
        scenario = SCENARIOS.get(scenario_name, SCENARIOS["baseline"])
        return self.run_scenario(scenario, source, destination, amount_usd_m)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    from cbdc_network import CBDCNetwork
    from clearing_engine import ClearingEngine, RoutingWeights

    print("=== CBDC Stress Testing Engine ===")
    print("Running 1000-trial Monte Carlo simulation for INR→USD...\n")

    engine = StressTestingEngine(
        network_class=CBDCNetwork,
        clearing_engine_class=ClearingEngine,
        n_trials=1000,
    )

    df = engine.run_all_scenarios("INR", "USD", amount_usd_m=10.0)

    print("\n=== Stress Test Summary ===")
    print(df.to_string(index=False))
    df.to_csv("../data/stress_test_results.csv", index=False)
    print("\nResults saved to data/stress_test_results.csv")
