import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Retail Investor Terminal — Multi-Asset Optimizer", page_icon="🤖📈", layout="wide")


# --- Premium light-slate corporate theme CSS ---
st.markdown("""
<style>
    :root { color-scheme: light; }
    body, .stApp { background-color: #f8fafc; color: #0f172a; }
    .block-container { padding-top: 1.6rem; padding-bottom: 1.6rem; }
    h1, h2, h3, h4 { font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, Arial; font-weight:600; letter-spacing:0.2px; color:#0f172a; }
    .stSidebar { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
    .card { background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:14px; color:#0f172a; }
    .metric-grid { display:flex; gap:18px; align-items:stretch; }
    .metric { flex:1; background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:18px; box-shadow:0 10px 15px -3px rgba(0,0,0,0.05); }
    .metric .label { color:#475569; font-size:13px; }
    .metric .value { font-size:22px; font-weight:700; margin-top:6px; }
    /* Hero header */
    .hero { padding:22px; border-radius:14px; margin-bottom:18px; }
    .hero-title { font-size:30px; font-weight:800; background: linear-gradient(90deg,#00b4d8,#0077b6,#7209b7); -webkit-background-clip:text; background-clip:text; color:transparent; }
    .hero-sub { color:#0f172a; margin-top:6px; }
    /* AI banner */
    .ai-banner { padding:3px; border-radius:12px; background: linear-gradient(90deg,#e6f6ff,#f3e9ff); }
    .ai-inner { background:#ffffff; border:1px solid rgba(226,232,240,0.8); padding:16px; border-radius:10px; }
    .ai-inner h4 { margin-top:0; }
    .small-muted { color:#475569; font-size:13px; }
    table.dataframe { background: transparent; }
</style>
""", unsafe_allow_html=True)


# --- Asset universe and modelling inputs (unchanged) ---
assets = [
    "Domestic Equities (Nifty 50)",
    "International Equities (S&P 500)",
    "Fixed Income (Corporate Debt)",
    "Gold",
    "Liquid Cash",
]

expected_returns = np.array([0.12, 0.10, 0.08, 0.07, 0.05])
volatilities = np.array([0.16, 0.18, 0.05, 0.12, 0.01])

correlation_matrix = pd.DataFrame(
    [
        [1.00, 0.62, 0.08, -0.12, 0.03],
        [0.62, 1.00, -0.04, -0.08, 0.02],
        [0.08, -0.04, 1.00, 0.05, 0.14],
        [-0.12, -0.08, 0.05, 1.00, 0.00],
        [0.03, 0.02, 0.14, 0.00, 1.00],
    ],
    index=assets,
    columns=assets,
)

covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix.to_numpy()


def optimize_weights(risk_profile: str) -> np.ndarray:
    base_weights = {
        "Conservative": np.array([0.16, 0.10, 0.28, 0.16, 0.30]),
        "Moderate": np.array([0.30, 0.20, 0.22, 0.14, 0.14]),
        "Aggressive": np.array([0.36, 0.26, 0.13, 0.16, 0.09]),
    }
    aversion = {"Conservative": 2.8, "Moderate": 1.6, "Aggressive": 0.8}[risk_profile]
    w = base_weights[risk_profile].astype(float)
    for _ in range(250):
        gradient = expected_returns - 2 * aversion * (covariance_matrix @ w)
        w = w + 0.02 * gradient
        w = np.clip(w, 0.05, 0.40)
        w = w / w.sum()
    return w


st.title("Retail Investor Terminal — Multi-Asset Optimizer")

# --- HERO ---
st.markdown('<div class="hero"><div class="hero-title">Retail Investor Terminal — Multi‑Asset Optimizer</div><div class="hero-sub">Premium allocation advice with an embedded AI strategist and investor-friendly presentation.</div></div>', unsafe_allow_html=True)

# short description
st.markdown("<div class='small-muted'>A friendly terminal that recommends diversified allocations and provides clear, actionable next steps.</div>", unsafe_allow_html=True)


# --- Sidebar inputs ---
with st.sidebar:
    st.header("Investor Inputs")
    investment = st.number_input("Investment Amount (INR)", min_value=10000, value=5000000, step=10000, format="%d")
    risk_profile = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"]) 
    st.markdown("---")
    st.subheader("Portfolio Logic")
    st.write("A risk-adjusted allocation engine balances return and volatility while preserving liquidity and diversification.")


# --- Core calculations (kept intact) ---
weights = optimize_weights(risk_profile)
allocation_inr = weights * investment
portfolio_return = float(weights @ expected_returns)
portfolio_volatility = float(np.sqrt(weights @ covariance_matrix @ weights))
portfolio_sharpe = (portfolio_return - 0.04) / portfolio_volatility if portfolio_volatility else 0.0


# --- Overview metrics section (luxury floating metric cards) ---
st.markdown("## Overview")
col1, col2 = st.columns([2,3])
with col1:
        # metric grid as HTML
        metric_html = f"""
        <div class='metric-grid'>
            <div class='metric'>
                <div class='label'>Total Capital</div>
                <div class='value'>₹{investment:,.0f}</div>
            </div>
            <div class='metric'>
                <div class='label'>Expected Annual Return</div>
                <div class='value'>{portfolio_return*100:.2f}%</div>
            </div>
            <div class='metric'>
                <div class='label'>Risk Level</div>
                <div class='value'>{risk_profile}</div>
            </div>
        </div>
        """
        st.markdown(metric_html, unsafe_allow_html=True)
with col2:
        # contextual note area
        st.markdown('<div class="card"><div class="small-muted">Use these metrics as a high-level snapshot. Expand the allocation table for exact rupee amounts per asset.</div></div>', unsafe_allow_html=True)



# --- Allocation visualization and table ---
st.markdown("## Recommended Allocation")
left, right = st.columns([1.6, 1])
alloc_df = pd.DataFrame({
    "Asset Class": assets,
    "Recommended %": (weights * 100).round(2),
    "Amount (INR)": allocation_inr.round(0).astype(int),
})

with left:
    # create color gradient per bar based on weight interpolation
    def hex_to_rgb(h):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    def rgb_to_hex(rgb):
        return '#%02x%02x%02x' % rgb

    start = hex_to_rgb('#00b4d8')
    mid = hex_to_rgb('#0077b6')
    end = hex_to_rgb('#7209b7')

    # map weights [0..1] to gradient between start->mid->end
    colors = []
    for w in weights:
        if w < 0.5:
            t = w / 0.5
            rgb = tuple(int(start[i] + (mid[i]-start[i])*t) for i in range(3))
        else:
            t = (w-0.5) / 0.5
            rgb = tuple(int(mid[i] + (end[i]-mid[i])*t) for i in range(3))
        colors.append(rgb_to_hex(rgb))

    import plotly.graph_objects as go
    fig = go.Figure()
    for i, row in alloc_df.iterrows():
        fig.add_trace(go.Bar(
            x=[row['Recommended %']],
            y=[row['Asset Class']],
            orientation='h',
            marker=dict(color=colors[i]),
            text=f"{row['Recommended %']:.1f}%",
            textposition='outside',
            hovertemplate=f"%{{y}}: {row['Recommended %']:.2f}% · ₹{row['Amount (INR)']:,}<extra></extra>",
        ))

    fig.update_layout(
        barmode='stack',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=10, t=10, b=10),
        xaxis=dict(title='Allocation (%)', showgrid=True, gridcolor='#e9eef2'),
        yaxis=dict(showgrid=False),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown('<div class="card"><h4 style="margin-top:0">Allocation Table</h4></div>', unsafe_allow_html=True)
    display_df = alloc_df.copy()
    display_df['Recommended %'] = display_df['Recommended %'].map(lambda v: f"{v:.1f}%")
    display_df['Amount (INR)'] = display_df['Amount (INR)'].map(lambda v: f"₹{v:,}")
    st.table(display_df.set_index('Asset Class'))



# --- Correlation matrix ---
st.markdown("## Correlation Matrix")
st.write("How the asset classes move relative to each other (diversification helps reduce portfolio volatility).")
fig_corr = px.imshow(correlation_matrix.round(2), text_auto=True, color_continuous_scale="Blues", aspect="auto")
fig_corr.update_layout(plot_bgcolor="#ffffff", paper_bgcolor="#f8fafc", font={"color":"#0f172a"}, margin=dict(l=20,r=20,t=10,b=10))
fig_corr.update_xaxes(showgrid=True, gridcolor="#e6e8eb", zeroline=False)
fig_corr.update_yaxes(showgrid=True, gridcolor="#e6e8eb", zeroline=False)
st.plotly_chart(fig_corr, use_container_width=True)



# --- AI Suggestion Engine (deterministic template-based) ---
st.markdown("## 🤖 AI Investment Strategist Copilot Advice")
st.markdown('<div class="ai-banner"><div class="ai-inner">', unsafe_allow_html=True)
st.subheader(f"Personalized Advice — {risk_profile} Profile")

# Build rationale and content
top_idx = int(np.argmax(weights))
top_asset = assets[top_idx]
top_weight = weights[top_idx]

st.markdown("**Strategic Rationale**")
st.write(f"The optimizer assigns a higher weight to **{top_asset}** ({top_weight*100:.1f}%) to align with the **{risk_profile}** objective.")
st.write(f"- Fixed Income ({weights[2]*100:.1f}%) provides income and downside protection during equity drawdowns.")
st.write(f"- Gold ({weights[3]*100:.1f}%) acts as an inflation hedge and risk-off diversifier.")
st.write(f"- Liquid Cash ({weights[4]*100:.1f}%) preserves optionality for rebalancing and opportunistic deployment.")

st.markdown("**Core Macroeconomic Benefits of This Mix**")
for b in [
    "Exposure to global growth via international equities reduces single-market concentration risk.",
    "Domestic equities capture local market upside and currency-aligned returns.",
    "Corporate debt lowers portfolio volatility and contributes steady income when yields are favorable.",
    "Gold tends to perform in inflationary or risk-off periods, offering downside protection.",
    "Cash ensures liquidity for tactical rebalancing and emergency needs without forced selling.",
]:
    st.write(f"- {b}")

st.markdown("**3-Step Next Steps Action Plan**")
st.write("1) Establish the core holdings at the recommended weights and set a rebalancing cadence (quarterly).")
st.write("2) Keep the suggested cash buffer intact to avoid forced selling during downturns.")
st.write("3) Review performance quarterly and adjust only if your personal risk budget changes.")

st.markdown("**Quick Notes**")
st.write(f"- Largest allocation: **{top_asset}** ({top_weight*100:.1f}%).")
st.write(f"- Portfolio expected return: **{portfolio_return*100:.2f}%** · Expected volatility: **{portfolio_volatility*100:.2f}%**")
st.markdown('</div></div>', unsafe_allow_html=True)
