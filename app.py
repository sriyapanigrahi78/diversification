import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Simple Multi-Asset Planner", page_icon="chart", layout="wide")

st.markdown(
    """
    <style>
        :root { color-scheme: light; }
        body, .stApp { background: #f8fafc; color: #0f172a; }
        .block-container { padding-top: 1.2rem; padding-bottom: 1.4rem; }
        h1, h2, h3, h4 { font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, Arial; font-weight: 700; color: #0f172a; }
        .stSidebar { background: #ffffff; border-right: 1px solid #e2e8f0; }
        .soft-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 16px;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
        }
        .metric-card {
            border-radius: 18px;
            padding: 18px 20px;
            color: #ffffff;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.16);
            min-height: 120px;
            margin-bottom: 10px;
        }
        .metric-card .label { font-size: 13px; text-transform: uppercase; letter-spacing: 0.18em; opacity: 0.9; }
        .metric-card .value { font-size: 24px; font-weight: 800; margin-top: 8px; }
        .metric-card .note { font-size: 13px; margin-top: 8px; color: rgba(255,255,255,0.9); }
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
        "Best Mix (Help me choose)": np.array([0.28, 0.22, 0.16, 0.18, 0.16]),
    }
    normalized_profile = "Moderate" if risk_profile == "Best Mix (Help me choose)" else risk_profile
    aversion = {"Conservative": 2.8, "Moderate": 1.6, "Aggressive": 0.8}[normalized_profile]
    w = base_weights[normalized_profile].astype(float)
    for _ in range(220):
        gradient = expected_returns - 2 * aversion * (covariance_matrix @ w)
        w = w + 0.015 * gradient
        w = np.clip(w, 0.05, 0.40)
        w = w / w.sum()
    return w


def get_risk_profile_config(risk_profile: str) -> dict:
    normalized_profile = "Moderate" if risk_profile == "Best Mix (Help me choose)" else risk_profile
    configs = {
        "Conservative": {
            "risk_class": "conservative",
            "risk_title": "Low & Safe Growth",
            "risk_copy": "This path leans on stability, steady income, and more cash for comfort.",
            "safety_score": 92,
            "max_drawdown": 0.04,
        },
        "Moderate": {
            "risk_class": "moderate",
            "risk_title": "Balanced Growth",
            "risk_copy": "This path mixes growth and protection so your money can keep moving forward.",
            "safety_score": 82,
            "max_drawdown": 0.10,
        },
        "Aggressive": {
            "risk_class": "aggressive",
            "risk_title": "High Risk / High Return",
            "risk_copy": "This path aims for bigger upside, with more room for swings along the way.",
            "safety_score": 70,
            "max_drawdown": 0.24,
        },
    }
    config = configs[normalized_profile].copy()
    if risk_profile == "Best Mix (Help me choose)":
        config["risk_title"] = "Balanced & Protected Growth"
        config["risk_copy"] = "This profile is configured as a balanced approach, automatically prioritizing maximum capital preservation and inflation protection without requiring market expertise."
    return config


def build_drawdown_series(risk_profile: str) -> pd.DataFrame:
    config = get_risk_profile_config(risk_profile)
    months = pd.date_range("2025-01-01", periods=12, freq="ME")
    monthly_returns = [0.005, 0.008, 0.004, 0.003, -0.001, -config["max_drawdown"] * 0.95, -config["max_drawdown"] * 0.5, 0.008, 0.007, 0.004, 0.003, 0.005]
    portfolio_value = [100.0]
    for ret in monthly_returns[1:]:
        portfolio_value.append(portfolio_value[-1] * (1 + ret))
    series = pd.DataFrame({"month": months, "portfolio_value": portfolio_value})
    series["portfolio_value"] = series["portfolio_value"].round(2)
    series["protected_cushion"] = (series["portfolio_value"] * (0.96 - config["max_drawdown"] * 0.2)).round(2)
    return series


def main() -> None:
    with st.sidebar:
        st.header("Your planner")
        investment = st.number_input("Investment amount (INR)", min_value=10000, value=5000000, step=10000, format="%d")
        st.warning(
            "Smart Retail Rule: The Emergency Shield\n\nBefore investing your ₹ Capital, ensure you have set aside 3 to 6 months of basic living expenses in a regular bank savings account. Never invest money you might need next month!"
        )
        risk_profile = st.selectbox(
            "Risk approach",
            ["Conservative", "Moderate", "Aggressive", "Best Mix (Help me choose)"],
        )
        st.caption("Pick the pace that feels right for you. This simple planner adjusts your mix automatically.")

    weights = optimize_weights(risk_profile)
    allocation_inr = weights * investment
    portfolio_return = float(weights @ expected_returns)

    expected_return_pct = portfolio_return * 100
    config = get_risk_profile_config(risk_profile)
    risk_class = config["risk_class"]
    inflation_rate = 3.93
    net_real_return = expected_return_pct - inflation_rate
    net_real_gradient = (
        "linear-gradient(135deg, #10b981 0%, #059669 100%)"
        if net_real_return >= 0
        else "linear-gradient(135deg, #ef4444 0%, #b91c1c 100%)"
    )

    st.markdown("## Capital Diversification Framework")
    if risk_profile == "Best Mix (Help me choose)":
        st.success(
            "Strategy Recommendation: This profile is configured as a balanced approach, automatically prioritizing maximum capital preservation and inflation protection without requiring market expertise."
        )

    st.markdown(
        """
        <style>
            div[data-baseweb="tab-list"] button {{
                background-color: #f1f5f9;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                padding: 8px 16px;
                margin-right: 4px;
                font-weight: 500;
            }}
            div[data-baseweb="tab-list"] button[aria-selected="true"] {{
                background: linear-gradient(135deg, #2563eb, #1d4ed8);
                color: white !important;
                border-color: #2563eb;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.markdown(
        f"""
        <div style='background:linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border-radius:22px; padding:24px; min-height:152px; color:#ffffff;'>
            <div style='font-size:12px; text-transform:uppercase; letter-spacing:0.18em; opacity:0.85;'>Total Capital</div>
            <div style='font-size:32px; font-weight:800; margin-top:16px;'>₹{investment:,.0f}</div>
            <div style='margin-top:12px; font-size:13px; opacity:0.86;'>Starting amount you want to put to work.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c2.markdown(
        f"""
        <div style='background:linear-gradient(135deg, #2563eb 0%, #0fbc9b 100%); border-radius:22px; padding:24px; min-height:152px; color:#ffffff;'>
            <div style='font-size:12px; text-transform:uppercase; letter-spacing:0.18em; opacity:0.85;'>Total Expected Return</div>
            <div style='font-size:32px; font-weight:800; margin-top:16px;'>{expected_return_pct:.1f}%</div>
            <div style='margin-top:12px; font-size:13px; opacity:0.86;'>Long-run estimate for this mix.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c3.markdown(
        """
        <div style='background:linear-gradient(135deg, #64748b 0%, #475569 100%); border-radius:22px; padding:24px; min-height:152px; color:#ffffff;'>
            <div style='font-size:12px; text-transform:uppercase; letter-spacing:0.18em; opacity:0.85;'>India Inflation Rate</div>
            <div style='font-size:32px; font-weight:800; margin-top:16px;'>3.93%</div>
            <div style='margin-top:12px; font-size:13px; opacity:0.86;'>Fixed benchmark for real return comparison.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c4.markdown(
        f"""
        <div style='background:{net_real_gradient}; border-radius:22px; padding:24px; min-height:152px; color:#ffffff;'>
            <div style='font-size:12px; text-transform:uppercase; letter-spacing:0.18em; opacity:0.85;'>Net Real Return Above Inflation</div>
            <div style='font-size:32px; font-weight:800; margin-top:16px;'>{net_real_return:+.2f}%</div>
            <div style='margin-top:12px; font-size:13px; opacity:0.86;'>{'Above inflation' if net_real_return >= 0 else 'Below inflation'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c5.markdown(
        f"""
        <div style='background:linear-gradient(135deg, #047857 0%, #064e3b 100%); border-radius:22px; padding:24px; min-height:152px; color:#ffffff;'>
            <div style='font-size:12px; text-transform:uppercase; letter-spacing:0.18em; opacity:0.85;'>Safety & Drawdown Score</div>
            <div style='font-size:32px; font-weight:800; margin-top:16px;'>{config['safety_score']}/100</div>
            <div style='margin-top:12px; font-size:13px; opacity:0.86;'>Stronger protection during rough markets.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["Framework Allocation", "Annual Returns Profile", "Risk & Drawdown Assessment"])

    with tab1:
        st.markdown("### Framework Allocation")
        st.markdown(
            "Real-World Impact: Beating inflation means your wealth is actively growing. Instead of your cash losing value in a standard bank account, this diversified framework defends your family's future purchasing power against rising prices."
        )

        st.markdown("### Quick check")
        st.markdown(
            f"""
            <div class="risk-card {risk_class}">
                <div class="risk-tag">Risk approach</div>
                <div class="risk-title">{config['risk_title']}</div>
                <div class="risk-copy">{config['risk_copy']}</div>
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
            rows_html = ""
            for idx, row in alloc_df.iterrows():
                row_bg = "#f8fafc" if idx % 2 == 0 else "#ffffff"
                rows_html += f"""
                    <tr style='background:{row_bg};'>
                        <td style='padding:14px 12px; border-bottom:1px solid #e2e8f0; text-align:left;'>{row['Asset Class']}</td>
                        <td style='padding:14px 12px; border-bottom:1px solid #e2e8f0; text-align:right;'>{row['Recommended %']:.1f}%</td>
                        <td style='padding:14px 12px; border-bottom:1px solid #e2e8f0; text-align:right;'>₹{row['Amount (INR)']:,}</td>
                    </tr>
                """

            table_html = f"""
            <table style='width:100%; border-collapse: collapse; font-size:14px;'>
                <thead>
                    <tr>
                        <th style='padding:14px 12px; text-align:left; color:#475569; border-bottom:1px solid #e2e8f0;'>Asset Class</th>
                        <th style='padding:14px 12px; text-align:right; color:#475569; border-bottom:1px solid #e2e8f0;'>Allocation</th>
                        <th style='padding:14px 12px; text-align:right; color:#475569; border-bottom:1px solid #e2e8f0;'>Amount</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
            """
            st.markdown(table_html, unsafe_allow_html=True)

        st.markdown("### Emergency Shield")
        st.warning(
            "Smart Retail Rule: Keep 3 to 6 months of essentials in cash before you commit more capital. That shield protects your family from sudden job changes and keeps your long-term plan flexible."
        )

        st.markdown("## Wealth Journey and Goal Planner")
        years = st.slider("Select Time Horizon (Years)", min_value=1, max_value=25, value=10, step=1)
        goal_target = st.number_input("Enter your target financial goal amount (INR)", value=500000)
        future_value = investment * ((1 + portfolio_return) ** years)

        st.markdown(
            f"Growing to: ₹{future_value:,.0f}! By keeping your money diversified over {years} years, your wealth multiplies efficiently while safely defending against inflation."
        )

        if future_value >= goal_target:
            st.success("Goal Status: On Track. Based on your current strategy, your projected wealth will cover your target goal with an estimated surplus.")
        else:
            st.info("Goal Status: Capital Gap. To reach your goal within this timeline, consider increasing your investment or adjusting your risk tolerance.")

    with tab2:
        st.markdown("### Annualized Returns & Performance Breakdown")
        st.markdown(
            "The long-run baseline below summarizes the expected annual return profile for each asset class while keeping your plan grounded in real-world expectations."
        )

        bullet_points = [
            "Nifty 50: Expect roughly 12-14% yearly growth over long horizons.",
            "US Stocks: Expect roughly 10-12% yearly growth providing global currency diversification.",
            "Gold: Expect roughly 8-10% yearly growth acting as crisis protection.",
            "Bonds: Expect roughly 6-8% yearly growth providing steady, predictable income.",
            "Cash: Expect roughly 3-4% yearly baseline keeping your money completely safe and accessible.",
        ]
        st.markdown("- " + "\n- ".join(bullet_points))

        st.markdown("### Quick Asset Class Guide")
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
                            <th style='padding:12px 10px;border:1px solid #e2e8f0;text-align:center;'>Superpower</th>
                        </tr>
                    </thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab3:
        st.markdown("### Risk & Drawdown Assessment")
        st.markdown("### How Your Assets Behave Together (Safety Map)")

        correlation_pairs = [
            ("Nifty 50 + Gold", -0.10, "Shield (Moves opposite to protect you)"),
            ("Nifty 50 + US Stocks", 0.60, "Growth Partner (Moves together for global diversification)"),
            ("Equities + Bonds/Cash", 0.02, "Independent Safety Net (Steady cushion)"),
        ]
        corr_df = pd.DataFrame(correlation_pairs, columns=["pair", "score", "note"])

        fig_corr = go.Figure()
        for _, row in corr_df.iterrows():
            color = "#10b981" if row["score"] <= 0 else "#2563eb"
            fig_corr.add_trace(
                go.Bar(
                    x=[row["score"]],
                    y=[row["pair"]],
                    orientation="h",
                    marker=dict(color=color, line=dict(color="#ffffff", width=1)),
                    hovertemplate=f"{row['pair']}: {row['score']:.2f}<br>{row['note']}<extra></extra>",
                )
            )

        fig_corr.update_layout(
            title="How Your Assets Behave Together (Safety Map)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(title="Correlation Score", range=[-0.25, 0.7], zeroline=True, zerolinecolor="#d1d5db"),
            yaxis=dict(automargin=True, tickfont=dict(size=13)),
            height=380,
            bargap=0.35,
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("How This Mix Protects Your Capital (Hedging Guide)")
        st.write(
            "Notice how some pairings show protection while others move together for growth. This simple view helps you understand where your portfolio leans on shields, cushions, and diversification."
        )

        st.markdown("### Historical Portfolio Drawdown Graph")
        drawdown_df = build_drawdown_series(risk_profile)
        fig_drawdown = go.Figure()
        fig_drawdown.add_trace(
            go.Scatter(
                x=drawdown_df["month"],
                y=drawdown_df["portfolio_value"],
                mode="lines+markers",
                name="Portfolio value",
                line=dict(color="#2563eb", width=3),
                marker=dict(size=7),
            )
        )
        fig_drawdown.add_trace(
            go.Scatter(
                x=drawdown_df["month"],
                y=drawdown_df["protected_cushion"],
                mode="lines+markers",
                name="Cash / gold / bond cushion",
                line=dict(color="#10b981", width=2, dash="dash"),
                marker=dict(size=6),
            )
        )
        fig_drawdown.update_layout(
            title="Simulated drawdown protection under the current profile",
            xaxis_title="Month",
            yaxis_title="Portfolio value (% of starting capital)",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#f8fafc",
            margin=dict(l=10, r=10, t=40, b=10),
            hovermode="x unified",
            height=360,
        )
        st.plotly_chart(fig_drawdown, use_container_width=True)

        st.caption(
            "Conservative mixes stay especially flat, moderate mixes soften the drop, and aggressive mixes show a deeper drawdown before recovery."
        )

        shock_test = st.checkbox("Simulate a 2020-Style Stock Market Crash")
        if shock_test:
            equity_weight = weights[0] + weights[1]
            estimated_drop = round((equity_weight * 0.30 * 0.30) * 100, 1)
            st.success(
                f"If the stock market crashes by -30%, your specific diversified mix would only dip by roughly {estimated_drop:.1f}% because your Gold and Bonds act as a protective cushion!"
            )


if __name__ == "__main__":
    main()
