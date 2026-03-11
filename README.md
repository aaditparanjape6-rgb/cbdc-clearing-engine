# CBDC Multi-Currency Clearing Engine

> A research-grade simulation platform for cross-border Central Bank Digital Currency (CBDC) settlement.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    CBDC Clearing Engine                         │
│                                                                 │
│  ┌─────────────┐   ┌──────────────┐   ┌────────────────────┐  │
│  │ CBDC Network│──▶│Clearing Eng. │──▶│Settlement Simulator│  │
│  │  (NetworkX) │   │(Dijkstra MO) │   │  (Atomic PvP)      │  │
│  └─────────────┘   └──────────────┘   └────────────────────┘  │
│         │                 │                      │              │
│  ┌──────▼──────┐   ┌──────▼──────┐   ┌──────────▼─────────┐  │
│  │  Liquidity  │   │  FX Risk    │   │  Compliance Engine  │  │
│  │  Manager    │   │  Model      │   │  (AML/KYC/FATF)     │  │
│  └─────────────┘   └──────────── ┘   └────────────────────┘  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │            Stress Testing Engine (Monte Carlo)          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │          Streamlit Dashboard (Plotly Interactive)       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
cbdc_engine/
│
├── data/
│   ├── generate_data.py      # Synthetic FX dataset generator (GBM)
│   └── fx_rates.csv          # Generated hourly FX rates (42 pairs × 365 days)
│
├── src/
│   ├── cbdc_network.py       # Module 1: Network graph (NetworkX DiGraph)
│   ├── clearing_engine.py    # Module 2: Multi-objective route optimizer
│   ├── risk_model.py         # Module 3: FX volatility + AR(1) risk model
│   ├── liquidity_manager.py  # Module 4: Node/corridor liquidity pools
│   ├── compliance_engine.py  # Module 5: AML/KYC/FATF rule engine
│   ├── settlement_simulator.py # Module 6: Atomic settlement lifecycle
│   └── stress_testing.py     # Module 7: Monte Carlo stress scenarios
│
├── models/
│   └── fx_model.pkl          # Persisted FX risk model
│
├── notebooks/
│   └── exploration.ipynb     # Jupyter analysis notebook
│
├── dashboard.py              # Module 8: Streamlit interactive dashboard
├── requirements.txt          # Python dependencies
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Data and Fit Models

```bash
cd data && python generate_data.py && cd ..
cd src && python risk_model.py && cd ..
```

### 3. Run Demo Settlement

```bash
cd src && python settlement_simulator.py
```

### 4. Run Stress Tests

```bash
cd src && python stress_testing.py
```

### 5. Launch Dashboard

```bash
streamlit run dashboard.py
```

---

## Core Algorithms

### Multi-Objective Routing
```
F = α·Cost + β·Risk + γ·Time + δ·CompliancePenalty
```
Each dimension is min-max normalized to [0,1] before weighting.
Dijkstra's algorithm finds the globally optimal path.

### FX Risk Model
- **Realized Volatility**: EWMA with 24h half-life, annualized
- **AR(1) Forecast**: 24-hour conditional variance forecast
- **Blend**: 60% EWMA realized + 40% AR(1) forecast
- **Output**: Risk score ∈ [0, 1], where 1 = 30% annualized vol

### Liquidity Management
- Node pools modeled as BIS PFMI Principle 7-compliant reserves
- Atomic 2-phase lock: all corridors locked before settlement executes
- Rollback on any failure (no partial settlement)

### Stress Testing
Monte Carlo scenarios randomize:
- FX shocks (Geometric Brownian Motion perturbations)
- Liquidity drains (uniform random drain fraction)
- Corridor failures (Bernoulli random failures)
- Latency/cost multipliers

---

## Sample Output: INR → USD Settlement

```
tx_id                  : CBDC-A1B2C3D4E5
source_currency        : INR
destination_currency   : USD
send_amount_local      : 835,000,000 INR
receive_amount_local   : 10,000,000 USD (approx)
routing_path           : INR → USD
effective_fx_rate      : 0.011976
total_cost_bps         : 12.0
total_cost_usd         : 1,200 USD
settlement_latency_s   : 45.0s
status                 : SETTLED
compliance_status      : APPROVED
fx_risk_score          : 0.0873
composite_score        : 0.1842
```

---

## CBDC Nodes Modeled

| Node      | Currency         | Code | GDP ($T) | AML Rating |
|-----------|-----------------|------|----------|------------|
| India     | Digital Rupee   | INR  | 3.7      | Low        |
| USA       | Digital Dollar  | USD  | 26.9     | Low        |
| EU        | Digital Euro    | EUR  | 17.1     | Low        |
| UK        | Digital Pound   | GBP  | 3.1      | Low        |
| Singapore | Digital SGD     | SGD  | 0.5      | Low        |
| China     | e-CNY           | CNY  | 17.7     | Medium     |
| UAE       | Digital Dirham  | AED  | 0.5      | Low        |

---

## References

- BIS: [Project mBridge](https://www.bis.org/publ/work1013.htm)
- BIS PFMI Principle 7: Liquidity Risk
- FATF Recommendations: AML/CFT for Virtual Assets
- ISO 20022: Universal Financial Industry Message Scheme
