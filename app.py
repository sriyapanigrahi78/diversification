import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Global Multi-Asset Portfolio Optimizer", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
        :root {
            color-scheme: light;
        }
        body, .stApp {
            background-color: #f8fafc;
            color: #0f172a;
            font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3, h4, h5 {
            font-weight: 600;
            letter-spacing: 0.2px;
            color: #0f172a;
            font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
        }
        /* Sidebar styling */
        .stSidebar {
            background-color: #ffffff;
            border-right: 1px solid #e2e8f0;
            color: #0f172a;
        }
        /* Card / metric styling */
        .stMetric, .card {
            background-color: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 12px;
            padding: 12px !important;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06) !important;
            color: #0f172a !important;
        }
        .card h4 { margin-top: 0; }
        /* Ensure dataframes have a subtle background */
        .stDataFrameContainer { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)

assets = ["Domestic Equities (Nifty 50)", "International Equities (S&P 500)", "Fixed Income (Corporate Debt)", "Gold", "Liquid Cash"]

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


st.title("Global Multi-Asset Portfolio Optimizer")
st.caption("A premium, dark-mode dashboard for diversified wealth building across global markets and defensive assets.")

with st.sidebar:
    st.header("Investor Inputs")
    investment = st.number_input("Investment Amount (INR)", min_value=10000, value=5000000, step=10000, format="%d")
    risk_profile = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"])
    st.markdown("---")
    st.subheader("Portfolio Logic")
    st.write("The optimizer balances return potential, volatility, and diversification using a risk-adjusted allocation engine.")

weights = optimize_weights(risk_profile)
allocation_inr = weights * investment
portfolio_return = float(weights @ expected_returns)
portfolio_volatility = float(np.sqrt(weights @ covariance_matrix @ weights))
portfolio_sharpe = (portfolio_return - 0.04) / portfolio_volatility if portfolio_volatility else 0.0

allocation_df = pd.DataFrame(
    {
        "Asset Class": assets,
        "Allocation Weight": weights * 100,
        "Allocation INR": allocation_inr,
    }
)
allocation_df = allocation_df.round({"Allocation Weight": 2, "Allocation INR": 2})

col1, col2, col3 = st.columns(3)
col1.metric("Expected Annual Return", f"{portfolio_return * 100:.2f}%")
col2.metric("Expected Volatility", f"{portfolio_volatility * 100:.2f}%")
col3.metric("Sharpe Lens", f"{portfolio_sharpe:.2f}")

st.markdown("### Asset Allocation")
left_col, right_col = st.columns([1.2, 0.95])
with left_col:
    fig = px.bar(
        allocation_df,
        x="Allocation Weight",
        y="Asset Class",
        orientation="h",
        text="Allocation Weight",
        color="Allocation Weight",
        color_continuous_scale="Viridis",
    )
    fig.update_traces(texttemplate="%{text:.1f} %", textposition="outside")
    fig.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#f8fafc",
        font={"color": "#0f172a"},
        margin=dict(l=20, r=20, t=10, b=10),
        coloraxis_showscale=False,
        xaxis=dict(showgrid=True, gridcolor="#e6e8eb", zeroline=False),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.markdown("<div class='card'><h4 style='margin-top:0'>Allocation Breakdown</h4></div>", unsafe_allow_html=True)
    st.dataframe(
        allocation_df.assign(Allocation_INR=allocation_df["Allocation INR"].map(lambda v: f"₹{v:,.0f}")),
        hide_index=True,
        use_container_width=True,
    )
    st.markdown(f"<div class='card'><strong>Cash Distribution:</strong> ₹{allocation_df.loc[allocation_df['Asset Class'] == 'Liquid Cash', 'Allocation INR'].iloc[0]:,.0f} reserved as liquid capital.</div>", unsafe_allow_html=True)

st.markdown("### Correlation Matrix")
st.write("The matrix below highlights the diversification benefit: most assets do not move in lockstep, which lowers aggregate portfolio risk.")
fig_corr = px.imshow(
    correlation_matrix.round(2),
    text_auto=True,
    color_continuous_scale="Blues",
    aspect="auto",
)
fig_corr.update_layout(
    plot_bgcolor="#ffffff",
    paper_bgcolor="#f8fafc",
    font={"color": "#0f172a"},
    margin=dict(l=20, r=20, t=10, b=10),
)
fig_corr.update_xaxes(showgrid=True, gridcolor="#e6e8eb", zeroline=False)
fig_corr.update_yaxes(showgrid=True, gridcolor="#e6e8eb", zeroline=False)
st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("### Strategic Advisory")
if risk_profile == "Conservative":
    cards = [
        ("Capital Preservation", "Your allocation leans into corporate debt and cash to buffer drawdowns while keeping a measured equity sleeve for inflation protection.", "#38bdf8"),
        ("Cash Buffer", "Holding a larger liquid reserve helps you stay flexible for emergencies, rebalancing, or opportunistic buying during market dips.", "#22c55e"),
        ("Execution Note", "Review the portfolio quarterly and rebalance if equity exposure drifts above the planned range.", "#f59e0b"),
    ]
elif risk_profile == "Moderate":
    cards = [
        ("Balanced Growth", "This profile combines both developed and domestic equity exposure with steady fixed income to smooth out market cycles.", "#818cf8"),
        ("Diversification Focus", "Gold and cash add ballast, helping the portfolio remain resilient if equity markets enter a volatile stretch.", "#34d399"),
        ("Execution Note", "A semi-annual rebalance is ideal to keep the allocation aligned with your long-term objective.", "#fbbf24"),
    ]
else:
    cards = [
        ("Growth Orientation", "The higher equity weight seeks stronger long-term compounding, while gold adds protection during risk-off phases.", "#fb7185"),
        ("Risk Tolerance", "This mix still preserves diversification, but it accepts more short-term volatility to pursue stronger growth.", "#f43f5e"),
        ("Execution Note", "Stay invested through market noise and revisit the strategy only when your risk budget materially changes.", "#f59e0b"),
    ]

for title, body, accent in cards:
    st.markdown(
        f"""
        <div class='card'>
            <h4 style='margin-top:0; color:{accent};'>{title}</h4>
            <p style='margin-bottom:0; color:#e2e8f0;'>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
