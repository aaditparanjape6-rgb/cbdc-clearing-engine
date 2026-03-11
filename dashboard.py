"""
CBDC Clearing Engine Dashboard — Module 8
==========================================
Interactive Streamlit dashboard for policymakers and financial institutions.

Panels:
  1. Network Overview    — CBDC graph visualization
  2. Route Optimizer     — Interactive transaction routing
  3. FX Risk Heatmap     — Currency pair risk matrix
  4. Liquidity Monitor   — Node and corridor liquidity bars
  5. Settlement Ledger   — Real-time transaction log
  6. Stress Testing      — Scenario analysis results
"""

import sys
import os

# Ensure src modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import networkx as nx
import time
import uuid
from datetime import datetime

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CBDC Multi-Currency Clearing Engine",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .stMetric { background: #161b22; border-radius: 8px; padding: 10px; }
    .stMetric label { color: #8b949e !important; font-size: 12px !important; }
    .stMetric [data-testid="metric-container"] { color: #e6edf3; }
    .block-container { padding-top: 1rem; }
    h1, h2, h3 { color: #e6edf3; }
    .sidebar .sidebar-content { background-color: #161b22; }
    div[data-testid="stExpander"] { background: #161b22; border-radius: 8px; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Imports with error handling ───────────────────────────────────────────
@st.cache_resource(show_spinner="Initializing CBDC Network...")
def load_system():
    """Load and cache all system components."""
    from cbdc_network import CBDCNetwork
    from clearing_engine import ClearingEngine, RoutingWeights
    from compliance_engine import ComplianceEngine
    from liquidity_manager import LiquidityManager
    from settlement_simulator import SettlementSimulator
    from risk_model import FXRiskModel

    # Build network
    net = CBDCNetwork()

    # Try loading pre-fitted risk model; fit on-the-fly if not found
    risk_model = FXRiskModel("data/fx_rates.csv")
    try:
        if os.path.exists("models/fx_model.pkl"):
            risk_model = FXRiskModel.load("models/fx_model.pkl")
        else:
            with st.spinner("Fitting FX risk model (first run)..."):
                risk_model.fit()
                risk_model.save("models/fx_model.pkl")
    except Exception:
        risk_model.fit()

    fx_risk_scores = risk_model.get_risk_scores_dict()
    weights = RoutingWeights()
    engine = ClearingEngine(net, weights, fx_risk_scores, transaction_amount_usd_m=10.0)
    compliance = ComplianceEngine()
    liquidity = LiquidityManager()
    simulator = SettlementSimulator(engine, compliance, liquidity)

    return {
        "network": net,
        "engine": engine,
        "compliance": compliance,
        "liquidity": liquidity,
        "simulator": simulator,
        "risk_model": risk_model,
    }


# ── Colour Palette ────────────────────────────────────────────────────────
COLOURS = {
    "USD": "#2563EB", "EUR": "#7C3AED", "GBP": "#0D9488",
    "INR": "#D97706", "SGD": "#DC2626", "CNY": "#BE185D", "AED": "#059669",
}

COUNTRY_LABELS = {
    "USD": "🇺🇸 USA", "EUR": "🇪🇺 EU", "GBP": "🇬🇧 UK",
    "INR": "🇮🇳 India", "SGD": "🇸🇬 Singapore",
    "CNY": "🇨🇳 China", "AED": "🇦🇪 UAE",
}

# Pre-computed layout positions for the CBDC network graph
NODE_POSITIONS = {
    "USD": (0.5,  0.5),
    "EUR": (0.15, 0.7),
    "GBP": (0.15, 0.85),
    "INR": (0.85, 0.25),
    "SGD": (0.85, 0.75),
    "CNY": (0.65, 0.15),
    "AED": (0.35, 0.15),
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPER DRAWING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def draw_network_graph(net, highlight_path=None):
    """Render the CBDC network as an interactive Plotly figure."""
    G = net.graph
    pos = NODE_POSITIONS

    edge_traces = []
    for src, tgt, data in G.edges(data=True):
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        # Colour edges by regulatory friction
        friction = data.get("regulatory_friction", 0.1)
        colour = f"rgba({int(255*friction)}, {int(255*(1-friction))}, 80, 0.35)"

        # Check if this edge is on the highlighted path
        on_path = False
        if highlight_path:
            pairs = list(zip(highlight_path[:-1], highlight_path[1:]))
            on_path = (src, tgt) in pairs

        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(
                width=4 if on_path else 1.5,
                color="#F59E0B" if on_path else colour,
            ),
            hoverinfo="none",
            showlegend=False,
        ))

    # Node trace
    node_x, node_y, node_text, node_colour, node_size, hover_text = [], [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(COUNTRY_LABELS.get(node, node))
        node_colour.append(COLOURS.get(node, "#666"))
        on_path = highlight_path and node in highlight_path
        node_size.append(38 if on_path else 28)

        attrs = G.nodes[node]
        hover = (
            f"<b>{COUNTRY_LABELS.get(node, node)}</b><br>"
            f"Currency Code: {node}<br>"
            f"CB Credibility: {attrs.get('cb_credibility', 'N/A')}<br>"
            f"GDP: ${attrs.get('gdp', 0):.1f}T<br>"
            f"AML Rating: {'🟢 Low' if attrs.get('aml_rating',0)==0 else '🟡 Med' if attrs.get('aml_rating',0)==1 else '🔴 High'}<br>"
            f"Daily Capacity: ${attrs.get('daily_capacity',0):,.0f}M"
        )
        hover_text.append(hover)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=node_size, color=node_colour,
                    line=dict(color="#ffffff", width=2)),
        text=node_text,
        textposition="bottom center",
        textfont=dict(color="#e6edf3", size=11),
        hovertext=hover_text,
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=420,
    )
    return fig


def draw_fx_heatmap(risk_model):
    """Draw FX risk heatmap."""
    df = risk_model.get_risk_heatmap_data()
    if df.empty:
        return go.Figure()

    currencies = ["USD", "EUR", "GBP", "INR", "SGD", "CNY", "AED"]
    matrix = pd.DataFrame(index=currencies, columns=currencies, dtype=float)

    for _, row in df.iterrows():
        if row["source"] in currencies and row["target"] in currencies:
            matrix.loc[row["source"], row["target"]] = row["risk_score"]

    np.fill_diagonal(matrix.values, 0)
    matrix = matrix.fillna(0)

    fig = go.Figure(go.Heatmap(
        z=matrix.values,
        x=currencies, y=currencies,
        colorscale="RdYlGn_r",
        zmin=0, zmax=0.4,
        text=matrix.round(3).values,
        texttemplate="%{text}",
        hovertemplate="<b>%{y} → %{x}</b><br>Risk Score: %{z:.3f}<extra></extra>",
        colorbar=dict(title="Risk Score", tickfont=dict(color="#e6edf3"),
                      titlefont=dict(color="#e6edf3")),
    ))
    fig.update_layout(
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font=dict(color="#e6edf3"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=380,
        xaxis=dict(tickfont=dict(color="#e6edf3")),
        yaxis=dict(tickfont=dict(color="#e6edf3")),
    )
    return fig


def draw_liquidity_bars(liquidity):
    """Draw node liquidity utilization chart."""
    df = liquidity.get_node_liquidity_df().reset_index()
    colours = [COLOURS.get(c, "#666") for c in df["currency"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Available",
        x=df["currency"],
        y=df["available_usd_m"],
        marker_color=colours,
        text=df["available_usd_m"].apply(lambda x: f"${x:,.0f}M"),
        textposition="outside",
        textfont=dict(color="#e6edf3", size=10),
    ))
    fig.add_trace(go.Bar(
        name="Minimum Reserve",
        x=df["currency"],
        y=df["minimum_usd_m"],
        marker_color="rgba(255,255,255,0.12)",
        marker_line=dict(color="rgba(255,100,100,0.6)", width=2),
    ))
    fig.update_layout(
        barmode="overlay",
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font=dict(color="#e6edf3"),
        legend=dict(font=dict(color="#e6edf3"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
        yaxis=dict(title="USD Millions", tickfont=dict(color="#8b949e"),
                   gridcolor="#21262d"),
        xaxis=dict(tickfont=dict(color="#e6edf3")),
    )
    return fig


def draw_stress_results(df_stress):
    """Draw stress test comparison chart."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Success Rate %",
        x=df_stress["scenario"],
        y=df_stress["success_rate_%"],
        marker_color="#22c55e",
        yaxis="y",
        offsetgroup=1,
    ))
    fig.add_trace(go.Scatter(
        name="Cost VaR-95 (bps)",
        x=df_stress["scenario"],
        y=df_stress["cost_VaR95_bps"],
        marker=dict(size=10, color="#f59e0b"),
        line=dict(color="#f59e0b", width=2),
        mode="lines+markers",
        yaxis="y2",
    ))

    fig.update_layout(
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font=dict(color="#e6edf3"),
        legend=dict(font=dict(color="#e6edf3"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=30, b=100),
        height=380,
        xaxis=dict(tickangle=-30, tickfont=dict(color="#e6edf3")),
        yaxis=dict(title="Success Rate %", tickfont=dict(color="#22c55e"),
                   range=[0, 105], gridcolor="#21262d"),
        yaxis2=dict(title="Cost VaR-95 (bps)", overlaying="y", side="right",
                    tickfont=dict(color="#f59e0b")),
        title=dict(text="Stress Scenario Comparison", font=dict(color="#e6edf3")),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Load system
    system = load_system()
    net        = system["network"]
    engine     = system["engine"]
    compliance = system["compliance"]
    liquidity  = system["liquidity"]
    simulator  = system["simulator"]
    risk_model = system["risk_model"]

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🌐 CBDC Clearing Engine")
        st.markdown("*Multi-Currency Settlement Network*")
        st.divider()

        st.markdown("### 🔧 Routing Parameters")
        alpha = st.slider("α Cost Weight",     0.0, 1.0, 0.35, 0.05)
        beta  = st.slider("β FX Risk Weight",  0.0, 1.0, 0.25, 0.05)
        gamma = st.slider("γ Time Weight",     0.0, 1.0, 0.20, 0.05)
        delta = st.slider("δ Compliance Weight", 0.0, 1.0, 0.20, 0.05)

        total_w = alpha + beta + gamma + delta
        if abs(total_w - 1.0) > 0.01:
            st.warning(f"Weights sum to {total_w:.2f} (recommend 1.0)")

        st.divider()
        st.markdown("### 📊 System Status")
        stats = simulator.get_stats_summary()
        st.metric("Transactions", stats["total_initiated"])
        st.metric("Success Rate", f"{stats['success_rate_pct']}%")
        st.metric("Total Volume", f"${stats['total_volume_usd']/1e6:.1f}M")

        st.divider()
        st.caption("Built for research & demonstration purposes.")
        st.caption("Not for production use.")

    # ── Header ────────────────────────────────────────────────────────────
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown("# 🏦 Multi-Currency CBDC Clearing Engine")
        st.markdown("*Cross-Border Digital Currency Settlement Platform*")
    with col_h2:
        st.markdown(f"<div style='text-align:right;color:#8b949e;padding-top:20px'>"
                    f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</div>",
                    unsafe_allow_html=True)

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️ Network", "🔄 Route Optimizer", "📈 FX Risk",
        "💧 Liquidity", "⚡ Stress Tests"
    ])

    # ══════════════════════════════════════════════
    # TAB 1: NETWORK OVERVIEW
    # ══════════════════════════════════════════════
    with tab1:
        st.markdown("### Global CBDC Network Graph")
        st.markdown(
            "Nodes = sovereign CBDC systems. "
            "Edge width = corridor activity. "
            "Edge colour = regulatory friction (green=low, red=high)."
        )
        col1, col2 = st.columns([2, 1])
        with col1:
            fig_net = draw_network_graph(net)
            st.plotly_chart(fig_net, use_container_width=True)
        with col2:
            st.markdown("**Corridor Statistics**")
            df_corridors = net.get_network_summary()
            display_cols = ["corridor", "fx_rate", "transaction_cost_bps",
                            "settlement_latency_s", "liquidity_usd_m", "regulatory_friction"]
            st.dataframe(
                df_corridors[display_cols].style.background_gradient(
                    subset=["regulatory_friction"], cmap="RdYlGn_r"
                ),
                height=380,
                use_container_width=True,
            )

        # Network metrics
        st.divider()
        st.markdown("### Network Health Metrics")
        cols = st.columns(5)
        cols[0].metric("Total Nodes", net.graph.number_of_nodes())
        cols[1].metric("Active Corridors", net.graph.number_of_edges())
        avg_cost = np.mean([d["transaction_cost_bps"] for _, _, d in net.graph.edges(data=True)])
        avg_lat  = np.mean([d["settlement_latency_s"] for _, _, d in net.graph.edges(data=True)])
        avg_liq  = np.mean([d["liquidity_usd_m"] for _, _, d in net.graph.edges(data=True)])
        avg_fric = np.mean([d["regulatory_friction"] for _, _, d in net.graph.edges(data=True)])
        cols[2].metric("Avg. Cost", f"{avg_cost:.1f} bps")
        cols[3].metric("Avg. Latency", f"{avg_lat:.0f}s")
        cols[4].metric("Avg. Friction", f"{avg_fric:.3f}")

    # ══════════════════════════════════════════════
    # TAB 2: ROUTE OPTIMIZER
    # ══════════════════════════════════════════════
    with tab2:
        st.markdown("### Transaction Route Optimizer")
        currencies = net.available_nodes()
        currency_labels = {c: COUNTRY_LABELS.get(c, c) for c in currencies}

        col_form, col_result = st.columns([1, 2])

        with col_form:
            st.markdown("**Transaction Parameters**")
            src_ccy = st.selectbox(
                "Source CBDC",
                currencies,
                index=currencies.index("INR") if "INR" in currencies else 0,
                format_func=lambda x: currency_labels.get(x, x),
            )
            dst_ccy = st.selectbox(
                "Destination CBDC",
                [c for c in currencies if c != src_ccy],
                index=0,
                format_func=lambda x: currency_labels.get(x, x),
            )
            amount_local = st.number_input(
                f"Amount ({src_ccy})",
                min_value=1.0, value=835_000_000.0, step=1_000_000.0,
                format="%.0f",
            )
            purpose = st.selectbox(
                "Purpose",
                ["TRADE", "INVEST", "REMITTANCE", "OTHER"],
            )
            entity_type = st.selectbox(
                "Entity Type",
                ["institutional", "retail", "central_bank"],
            )
            kyc_ok = st.checkbox("KYC Verified", value=True)
            k_routes = st.slider("Show Top K Routes", 1, 5, 3)
            st.divider()
            settle_btn = st.button("🚀 Find Route & Settle", use_container_width=True,
                                   type="primary")

        with col_result:
            if settle_btn:
                if src_ccy == dst_ccy:
                    st.error("Source and destination cannot be the same.")
                else:
                    # Update engine weights from sidebar
                    from clearing_engine import RoutingWeights
                    new_weights = RoutingWeights(alpha=alpha, beta=beta,
                                                  gamma=gamma, delta=delta)
                    engine.weights = new_weights
                    engine.refresh_graph()

                    # Find route
                    with st.spinner("Computing optimal route..."):
                        route = engine.find_optimal_route(src_ccy, dst_ccy)

                    if not route.feasible:
                        st.error(f"❌ No feasible route: {route.failure_reason}")
                    else:
                        # Optimal route metrics
                        st.markdown("#### ✅ Optimal Route Found")
                        mcols = st.columns(4)
                        mcols[0].metric("Route", " → ".join(route.path))
                        mcols[1].metric("Total Cost", f"{route.total_cost_bps:.1f} bps")
                        mcols[2].metric("Latency", f"{route.total_latency_s:.0f}s")
                        mcols[3].metric("FX Risk Score", f"{route.fx_risk_score:.3f}")

                        mcols2 = st.columns(4)
                        mcols2[0].metric("FX Rate", f"{route.effective_fx_rate:.6f}")
                        mcols2[1].metric("Hops", len(route.path)-1)
                        mcols2[2].metric("Compliance Penalty", f"{route.compliance_penalty:.3f}")
                        mcols2[3].metric("Composite Score", f"{route.composite_score:.4f}")

                        # Settle transaction
                        with st.spinner("Processing settlement..."):
                            record = simulator.process_transaction(
                                source_currency=src_ccy,
                                destination_currency=dst_ccy,
                                send_amount_local=amount_local,
                                sender_id=f"INST_{src_ccy}_001",
                                receiver_id=f"INST_{dst_ccy}_001",
                                purpose_code=purpose,
                                kyc_verified=kyc_ok,
                                entity_type=entity_type,
                            )

                        status_icon = "✅" if record.status.value == "SETTLED" else "❌"
                        st.success(f"{status_icon} Settlement Status: **{record.status.value}**")

                        st.markdown("#### Settlement Details")
                        rec_dict = record.to_dict()
                        df_rec = pd.DataFrame([
                            {"Field": k, "Value": str(v)}
                            for k, v in rec_dict.items()
                        ])
                        st.dataframe(df_rec, use_container_width=True, height=340)

                        # Network graph with highlighted path
                        st.markdown("#### Route Visualization")
                        fig_route = draw_network_graph(net, highlight_path=route.path)
                        st.plotly_chart(fig_route, use_container_width=True)

                        # Alternative routes
                        st.markdown(f"#### Top {k_routes} Alternative Routes")
                        alt_routes = engine.find_k_best_routes(src_ccy, dst_ccy, k=k_routes)
                        df_alts = pd.DataFrame([r.to_dict() for r in alt_routes])
                        if not df_alts.empty:
                            display = ["Route", "Hops", "Total Cost (bps)", "Latency (s)",
                                       "FX Risk Score", "Composite Score", "Feasible"]
                            st.dataframe(df_alts[[c for c in display if c in df_alts.columns]],
                                         use_container_width=True)
            else:
                st.info("Configure transaction parameters and click **Find Route & Settle**.")

        # Ledger
        st.divider()
        st.markdown("### 📋 Settlement Ledger")
        ledger_df = simulator.get_ledger_df()
        if ledger_df.empty:
            st.info("No transactions processed yet.")
        else:
            st.dataframe(ledger_df, use_container_width=True, height=280)

    # ══════════════════════════════════════════════
    # TAB 3: FX RISK HEATMAP
    # ══════════════════════════════════════════════
    with tab3:
        st.markdown("### FX Risk Heatmap")
        st.markdown(
            "Risk score per currency pair derived from EWMA realized volatility "
            "blended with AR(1) 24-hour forecast. Range: 0 (low) → 1 (high)."
        )
        col_heat, col_tbl = st.columns([1.4, 1])
        with col_heat:
            fig_heat = draw_fx_heatmap(risk_model)
            st.plotly_chart(fig_heat, use_container_width=True)
        with col_tbl:
            st.markdown("**Top Risk Pairs**")
            df_risk = risk_model.get_risk_heatmap_data()
            if not df_risk.empty:
                df_risk_sorted = df_risk.sort_values("risk_score", ascending=False)
                st.dataframe(
                    df_risk_sorted[["pair", "risk_score", "vol_annualized",
                                    "vol_regime", "risk_tier"]].head(15),
                    use_container_width=True,
                    height=340,
                )

        # Volatility regime distribution
        st.divider()
        if not df_risk.empty:
            st.markdown("### Volatility Regime Distribution")
            regime_counts = df_risk["vol_regime"].value_counts().reset_index()
            regime_counts.columns = ["Regime", "Count"]
            colours_regime = {"low": "#22c55e", "medium": "#f59e0b", "high": "#ef4444"}
            fig_regime = px.pie(
                regime_counts, names="Regime", values="Count",
                color="Regime",
                color_discrete_map=colours_regime,
            )
            fig_regime.update_layout(
                paper_bgcolor="#161b22", font=dict(color="#e6edf3"),
                height=300,
            )
            st.plotly_chart(fig_regime, use_container_width=True)

    # ══════════════════════════════════════════════
    # TAB 4: LIQUIDITY MONITOR
    # ══════════════════════════════════════════════
    with tab4:
        st.markdown("### Liquidity Monitor")
        col_refresh = st.columns([4, 1])
        with col_refresh[1]:
            if st.button("🔄 Simulate Recycling"):
                liquidity.simulate_intraday_recycling(hours=1.0)
                st.success("Intraday recycling simulated (+1hr)")

        st.markdown("#### Node Liquidity Pools (USD Millions)")
        fig_liq = draw_liquidity_bars(liquidity)
        st.plotly_chart(fig_liq, use_container_width=True)

        # Node liquidity table
        df_node_liq = liquidity.get_node_liquidity_df().reset_index()
        st.dataframe(df_node_liq.style.background_gradient(
            subset=["utilization_pct"], cmap="RdYlGn_r"),
            use_container_width=True,
        )

        st.divider()
        st.markdown("#### Corridor Liquidity Status")
        df_corr_liq = liquidity.get_corridor_liquidity_df()
        # Colour by utilization
        st.dataframe(
            df_corr_liq.sort_values("utilization_pct", ascending=False)
                       .style.background_gradient(subset=["utilization_pct"], cmap="RdYlGn_r"),
            use_container_width=True,
            height=380,
        )

    # ══════════════════════════════════════════════
    # TAB 5: STRESS TESTING
    # ══════════════════════════════════════════════
    with tab5:
        st.markdown("### Monte Carlo Stress Testing Engine")
        st.markdown(
            "Evaluate CBDC network resilience under adverse scenarios. "
            "Each scenario runs N independent simulations with randomized shocks."
        )

        col_cfg, col_run = st.columns([1, 2])
        with col_cfg:
            st.markdown("**Stress Test Configuration**")
            stress_src = st.selectbox("Source", currencies,
                                      index=currencies.index("INR") if "INR" in currencies else 0,
                                      format_func=lambda x: currency_labels.get(x, x),
                                      key="stress_src")
            stress_dst = st.selectbox("Destination", currencies,
                                      index=currencies.index("USD") if "USD" in currencies else 1,
                                      format_func=lambda x: currency_labels.get(x, x),
                                      key="stress_dst")
            stress_amount = st.number_input("Amount (USD millions)", 1.0, 1000.0, 10.0)
            n_trials = st.select_slider("Monte Carlo Trials", [100, 500, 1000, 2000], value=1000)
            run_stress = st.button("▶ Run Stress Tests", use_container_width=True, type="primary")

        with col_run:
            if run_stress:
                from cbdc_network import CBDCNetwork
                from clearing_engine import ClearingEngine
                from stress_testing import StressTestingEngine

                with st.spinner(f"Running {n_trials}-trial Monte Carlo simulation..."):
                    stress_engine = StressTestingEngine(
                        network_class=CBDCNetwork,
                        clearing_engine_class=ClearingEngine,
                        n_trials=n_trials,
                    )
                    df_stress = stress_engine.run_all_scenarios(
                        source=stress_src,
                        destination=stress_dst,
                        amount_usd_m=stress_amount,
                    )

                st.success(f"✅ Completed {n_trials * len(df_stress)} total simulations")

                # Summary metrics
                st.markdown("#### Scenario Summary")
                st.dataframe(df_stress, use_container_width=True)

                # Visualize
                fig_stress = draw_stress_results(df_stress)
                st.plotly_chart(fig_stress, use_container_width=True)

                # Key insight callouts
                best = df_stress.loc[df_stress["success_rate_%"].idxmax()]
                worst = df_stress.loc[df_stress["success_rate_%"].idxmin()]
                c1, c2, c3 = st.columns(3)
                c1.metric("Most Resilient Scenario", best["scenario"],
                          f"{best['success_rate_%']:.1f}% success")
                c2.metric("Worst Scenario", worst["scenario"],
                          f"{worst['success_rate_%']:.1f}% success")
                baseline_row = df_stress[df_stress["scenario"] == "Baseline"]
                if not baseline_row.empty:
                    c3.metric("Baseline Success Rate",
                              f"{baseline_row['success_rate_%'].values[0]:.1f}%")
            else:
                st.info("Configure parameters and click **Run Stress Tests**.")

                # Show scenario descriptions
                from stress_testing import SCENARIOS
                st.markdown("#### Available Stress Scenarios")
                for name, sc in SCENARIOS.items():
                    with st.expander(f"📋 {sc.name}"):
                        st.write(sc.description)
                        c1, c2 = st.columns(2)
                        c1.markdown(f"**FX Shock:** {sc.fx_shock_range}")
                        c1.markdown(f"**Liquidity Drain:** {sc.liquidity_drain}")
                        c2.markdown(f"**Latency Multiplier:** {sc.latency_multiplier}")
                        c2.markdown(f"**Corridor Failure Prob:** {sc.corridor_failure_prob:.0%}")


# ── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
