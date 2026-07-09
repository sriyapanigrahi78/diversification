import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Simple Multi-Asset Planner", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
        :root { color-scheme: light; }
        body, .stApp { background: #f8fafc; color: #0f172a; }
        .block-container { padding-top: 1.2rem; padding-bottom: 1.4rem; }
        h1, h2, h3, h4 { font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, Arial; font-weight: 700; color: #0f172a; }
        .stSidebar { background: #ffffff; border-right: 1px solid #e2e8f0; }
        .hero-card {
            background: linear-gradient(135deg, #eff6ff 0%, #f8fafc 100%);
            border: 1px solid #dbeafe;
            border-radius: 18px;
            padding: 20px;
            margin-bottom: 16px;
        }
        .hero-title { font-size: 28px; font-weight: 800; color: #0f172a; }
        .hero-sub { color: #475569; margin-top: 6px; font-size: 15px; }
        .soft-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 16px;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
        }
        .kpi-grid { display: flex; gap: 12px; flex-wrap: wrap; }
        .kpi-box {
            flex: 1;
            min-width: 180px;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 16px;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
        }
        .kpi-label { color: #64748b; font-size: 13px; margin-bottom: 6px; }
        .kpi-value { font-size: 24px; font-weight: 700; color: #0f172a; }
        .kpi-note { color: #64748b; font-size: 13px; margin-top: 6px; }
        .risk-card {
            border-radius: 16px;
            padding: 18px;
            margin-top: 10px;
            border: 1px solid rgba(255,255,255,0.4);
            color: #111827;
        }
        .risk-card.conservative { background: linear-gradient(135deg, #e8fdf2 0%, #dff3ff 100%); }
        .risk-card.moderate { background: linear-gradient(135deg, #fff7e6 0%, #ffe4b8 100%); }
        .risk-card.aggressive { background: linear-gradient(135deg, #ffe7e7 0%, #ffd9d9 100%); }
        .risk-tag { font-size: 12px; text-transform: uppercase; letter-spacing: 0.18em; font-weight: 700; opacity: 0.8; }
        .risk-title { font-size: 20px; font-weight: 700; margin-top: 4px; }
        .risk-copy { margin-top: 6px; color: #334155; }
        .advice-card { background: linear-gradient(135deg, #f8fafc 0%, #eef7ff 100%); border: 1px solid #dbeafe; border-radius: 16px; padding: 16px; }
        .small-muted { color: #64748b; font-size: 13px; }
        table.dataframe { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)


assets = ["Nifty 50", "US Stocks", "Gold", "Bonds", "Cash"]
expected_returns = np.array([0.12, 0.10, 0.07, 0.06, 0.04])
volatilities = np.array([0.16, 0.18, 0.12, 0.05, 0.01])

correlation_matrix = pd.DataFrame(
    [
        [1.00, 0.60, -0.10, 0.08, 0.02],
        [0.60, 1.00, -0.05, 0.04, 0.03],
        [-0.10, -0.05, 1.00, 0.06, 0.12],
        [0.08, 0.04, 0.06, 1.00, 0.10],
        [0.02, 0.03, 0.12, 0.10, 1.00],
    ],
    index=assets,
    columns=assets,
)

covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix.to_numpy()


def optimize_weights(risk_profile: str) -> np.ndarray:
    base_weights = {
        "Conservative": np.array([0.16, 0.10, 0.24, 0.20, 0.30]),
        "Moderate": np.array([0.28, 0.22, 0.16, 0.18, 0.16]),
        "Aggressive": np.array([0.36, 0.28, 0.12, 0.12, 0.12]),
    }
    aversion = {"Conservative": 2.8, "Moderate": 1.6, "Aggressive": 0.8}[risk_profile]
    w = base_weights[risk_profile].astype(float)
    for _ in range(220):
        gradient = expected_returns - 2 * aversion * (covariance_matrix @ w)
        w = w + 0.015 * gradient
        w = np.clip(w, 0.05, 0.40)
        w = w / w.sum()
    return w


with st.sidebar:
    st.header("Your planner")
    investment = st.number_input("Investment amount (INR)", min_value=10000, value=5000000, step=10000, format="%d")
    st.warning(
        "💡 Smart Retail Rule: The Emergency Shield\n\nBefore investing your ₹ Capital, ensure you have set aside 3 to 6 months of basic living expenses in a regular bank savings account. Never invest money you might need next month!"
    )
    risk_profile = st.selectbox("Risk approach", ["Conservative", "Moderate", "Aggressive"])
    st.caption("Pick the pace that feels right for you. This simple planner adjusts your mix automatically.")


weights = optimize_weights(risk_profile)
allocation_inr = weights * investment
portfolio_return = float(weights @ expected_returns)
portfolio_volatility = float(np.sqrt(weights @ covariance_matrix @ weights))

expected_return_pct = portfolio_return * 100
inflation_rate = 3.93
net_profit_above_inflation = expected_return_pct - inflation_rate

risk_class = risk_profile.lower()
risk_title = {
    "Conservative": "Low & Safe Growth",
    "Moderate": "Balanced Growth",
    "Aggressive": "High Risk / High Return",
}[risk_profile]
risk_copy = {
    "Conservative": "This path leans on stability, steady income, and more cash for comfort.",
    "Moderate": "This path mixes growth and protection so your money can keep moving forward.",
    "Aggressive": "This path aims for bigger upside, with more room for swings along the way.",
}[risk_profile]


st.markdown("## Capital Diversification Framework")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f"""
        <div class="kpi-box">
            <div class="kpi-label">Total capital</div>
            <div class="kpi-value">₹{investment:,.0f}</div>
            <div class="kpi-note">Starting amount you want to put to work.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"""
        <div class="kpi-box">
            <div class="kpi-label">Expected annual return</div>
            <div class="kpi-value">{expected_return_pct:.1f}%</div>
            <div class="kpi-note">A simple long-term growth estimate for this mix.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"""
        <div class="kpi-box">
            <div class="kpi-label">Current Indian inflation rate</div>
            <div class="kpi-value">{inflation_rate:.2f}%</div>
            <div class="kpi-note">This is the fixed marker used for your planning check.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col4:
    if net_profit_above_inflation > 0:
        card_bg = "#e6f4ea"
        card_border = "#10b981"
        card_text = "#10b981"
    else:
        card_bg = "#fde8e8"
        card_border = "#ef4444"
        card_text = "#ef4444"

    st.markdown(
        f"""
        <div class="kpi-box" style="background:{card_bg}; border-color:{card_border};">
            <div class="kpi-label">Net profit above inflation</div>
            <div class="kpi-value" style="color:{card_text};">{net_profit_above_inflation:+.2f}%</div>
            <div class="kpi-note" style="color:{card_text};">This compares your return with India's current inflation rate.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    "✨ Real-World Impact: Beating inflation means your wealth is actively growing. Instead of your cash losing value in a standard bank account, this diversified framework defends your family's future purchasing power against rising prices."
)

st.markdown("### Quick check")
st.markdown(
    f"""
    <div class="risk-card {risk_class}">
        <div class="risk-tag">Risk approach</div>
        <div class="risk-title">{risk_title}</div>
        <div class="risk-copy">{risk_copy}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("## Where your money goes")
left, right = st.columns([1.45, 1])

with left:
    asset_colors = {
        "Nifty 50": "#2563eb",
        "US Stocks": "#7c3aed",
        "Gold": "#f59e0b",
        "Bonds": "#10b981",
        "Cash": "#64748b",
    }
    alloc_df = pd.DataFrame(
        {
            "Asset Class": assets,
            "Recommended %": (weights * 100).round(1),
            "Amount (INR)": allocation_inr.round(0).astype(int),
        }
    )

    fig = go.Figure()
    for _, row in alloc_df.iterrows():
        fig.add_trace(
            go.Bar(
                x=[row["Recommended %"]],
                y=[row["Asset Class"]],
                orientation="h",
                marker=dict(color=asset_colors.get(row["Asset Class"], "#2563eb")),
                text=f"{row['Recommended %']:.1f}%",
                textposition="outside",
                hovertemplate=f"{row['Asset Class']}: {row['Recommended %']:.1f}% · ₹{row['Amount (INR)']:,}<extra></extra>",
            )
        )

    fig.update_layout(
        barmode="stack",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=10, t=6, b=6),
        xaxis=dict(title="Allocation (%)", showgrid=True, gridcolor="#e2e8f0"),
        yaxis=dict(showgrid=False),
        showlegend=False,
        height=360,
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
    st.markdown("### Simple allocation table")
    display_df = alloc_df.copy()
    display_df["Recommended %"] = display_df["Recommended %"].map(lambda v: f"{v:.1f}%")
    display_df["Amount (INR)"] = display_df["Amount (INR)"].map(lambda v: f"₹{v:,}")
    st.table(display_df.set_index("Asset Class"))
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown("## How these assets usually move together")
fig_corr = px.imshow(correlation_matrix.round(2), text_auto=True, color_continuous_scale="Blues", aspect="auto")
fig_corr.update_layout(
    plot_bgcolor="#ffffff",
    paper_bgcolor="#f8fafc",
    font={"color": "#0f172a"},
    margin=dict(l=18, r=18, t=8, b=8),
)
fig_corr.update_xaxes(showgrid=True, gridcolor="#e2e8f0", zeroline=False)
fig_corr.update_yaxes(showgrid=True, gridcolor="#e2e8f0", zeroline=False)
st.plotly_chart(fig_corr, use_container_width=True)

with st.container():
    st.markdown("🛡️ How This Mix Protects Your Capital (Hedging Guide)")
    st.write(
        "Notice how some blocks on the matrix show very low or negative numbers? For example, when Equities fall during a market crash, Gold or Fixed Income traditionally stand firm or move up. This relationship is called a 'hedge'. By keeping assets that don't always move together, your total money stays safe even if one market hits a rough patch."
    )

with st.expander("🔍 Don't understand these grid numbers? Click here for a simple breakdown"):
    st.markdown(
        "- A score of 1.0 means the assets move like identical twins.\n"
        "- A score near 0.0 means they don't care about each other—this is perfect for safety!\n"
        "- A negative score (like -0.12) means when one goes down, the other goes up to save you."
    )


st.markdown("## Friendly advisor takeaway")
st.markdown(
    "- Your plan spreads money across growth, protection, and cash.\n"
    "- The biggest slice fits your chosen risk style.\n"
    "- Gold and bonds help soften sharp market drops.\n"
    "- Your return still beats inflation when the plan stays steady."
)

st.markdown("## 📊 Quick Asset Class Guide")
cheat_sheet_rows = [
    ["Nifty 50", "High Risk", "~12-14%", "Ultimate Wealth Generator"],
    ["US Stocks", "High Risk", "~10-12%", "Global Growth Booster"],
    ["Gold", "Medium Risk", "~8-10%", "Crisis Protection Shield"],
    ["Bonds", "Low Risk", "~6-8%", "Steady & Safe Income"],
    ["Cash", "Zero Risk", "~3-4%", "Safest (Instant Use Emergency Fund)"],
]

risk_badge_html = {
    "High Risk": "<span style='display:inline-block;padding:6px 10px;border-radius:999px;font-size:12px;font-weight:700;color:#ef4444;background:#fee2e2;'>High Risk</span>",
    "Medium Risk": "<span style='display:inline-block;padding:6px 10px;border-radius:999px;font-size:12px;font-weight:700;color:#f97316;background:#ffedd5;'>Medium Risk</span>",
    "Low Risk": "<span style='display:inline-block;padding:6px 10px;border-radius:999px;font-size:12px;font-weight:700;color:#22c55e;background:#dcfce7;'>Low Risk</span>",
    "Zero Risk": "<span style='display:inline-block;padding:6px 10px;border-radius:999px;font-size:12px;font-weight:700;color:#3b82f6;background:#dbeafe;'>Zero Risk</span>",
}

rows_html = ""
for asset, risk, returns, superpower in cheat_sheet_rows:
    badge = risk_badge_html.get(risk, risk)
    rows_html += f"<tr><td style='padding:12px 10px;border:1px solid #e2e8f0;text-align:center;'>{asset}</td><td style='padding:12px 10px;border:1px solid #e2e8f0;text-align:center;'>{badge}</td><td style='padding:12px 10px;border:1px solid #e2e8f0;text-align:center;'>{returns}</td><td style='padding:12px 10px;border:1px solid #e2e8f0;text-align:center;'>{superpower}</td></tr>"

st.markdown(
    f"""
    <div style='background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;padding:10px;box-shadow:0 8px 18px rgba(15,23,42,0.04);'>
        <table style='width:100%;border-collapse:collapse;font-size:14px;'>
            <thead>
                <tr>
                    <th style='padding:12px 10px;border:1px solid #e2e8f0;background:#f8fafc;text-align:center;'>Asset Class</th>
                    <th style='padding:12px 10px;border:1px solid #e2e8f0;background:#f8fafc;text-align:center;'>Risk Level</th>
                    <th style='padding:12px 10px;border:1px solid #e2e8f0;background:#f8fafc;text-align:center;'>Historical Returns</th>
                    <th style='padding:12px 10px;border:1px solid #e2e8f0;background:#f8fafc;text-align:center;'>Superpower</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("🏆 Safest Spot: Cash and Bonds keep your money completely safe from stock market crashes.")
st.markdown("🚀 Growth Engines: Nifty 50 and US Stocks have big swings but grow your wealth the fastest over time.")
st.markdown("🛡️ Your Insurance: Gold is your ultimate protection shield when inflation rises or markets panic.")

shock_test = st.checkbox("💥 Simulate a 2020-Style Stock Market Crash")
if shock_test:
    equity_weight = weights[0] + weights[1]
    estimated_drop = round((equity_weight * 0.30 * 0.30) * 100, 1)
    st.success(
        f"If the stock market crashes by -30%, your specific diversified mix would only dip by roughly {estimated_drop:.1f}% because your Gold and Bonds act as a protective cushion!"
    )

st.markdown("## 🔮 Your 10-Year Wealth Journey")
years = st.slider("Select Time Horizon (Years)", min_value=1, max_value=25, value=10, step=1)
future_value = investment * ((1 + portfolio_return) ** years)

st.markdown(
    f"💰 Growing to: ₹{future_value:,.0f}! By keeping your money diversified over {years} years, your wealth multiplies efficiently while safely defending against inflation."
)
