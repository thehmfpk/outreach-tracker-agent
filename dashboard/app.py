
# """
# app.py - Bank Customer Service Centre | Predictive Dashboard
# ==============================================================
# SafeX Solutions - Week 4 Client-Ready Sprint

# Run from the project root with:
#     streamlit run dashboard/app.py

# Tabs:
#     1. Overview        - KPIs + historical trends
#     2. Forecast        - Actual vs predicted (moving average + regression)
#     3. Recommendations - 3 business recommendations derived from forecast
#     4. Outreach Tracker- SIMULATED / DEMO lead-gen + outreach log (no real
#                          emails are sent from this app - clearly labeled demo)
# """

# import sys
# import os
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import streamlit as st

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from model.forecast_model import load_data, add_moving_averages, train_regression, forecast_future

# # ----------------------------------------------------------------------
# # PAGE CONFIG + ENHANCED THEME
# # ----------------------------------------------------------------------
# st.set_page_config(
#     page_title="SafeX | Bank CSC Predictive Dashboard",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # Core Palette
# PRIMARY_GREEN = "#0B3D2E"
# ACCENT_BLUE = "#1E5AA8"
# BLACK = "#111111"
# WHITE = "#FFFFFF"
# LIGHT_BG = "#F4F7F6"
# BORDER_COLOR = "#E2E8F0"
# TEXT_MUTED = "#64748B"

# CUSTOM_CSS = f"""
# <style>
#     /* Google Font Import */
#     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

#     html, body, [class*="css"] {{
#         font-family: 'Inter', sans-serif;
#         background-color: {LIGHT_BG} !important;
#     }}

#     /* Global Page Background */
#     .stApp {{
#         background-color: {LIGHT_BG};
#         color: {BLACK};
#     }}

#     /* Sidebar Styling */
#     section[data-testid="stSidebar"] {{
#         background-color: {PRIMARY_GREEN};
#         box-shadow: 2px 0 12px rgba(0,0,0,0.05);
#     }}
#     section[data-testid="stSidebar"] * {{
#         color: {WHITE} !important;
#     }}
#     section[data-testid="stSidebar"] .stRadio > label {{
#         font-weight: 500;
#     }}

#     /* Headings */
#     h1, h2, h3 {{
#         color: {PRIMARY_GREEN};
#         font-weight: 700;
#         letter-spacing: -0.02em;
#     }}

#     /* KPI Cards with Smooth Micro-Interactions */
#     .kpi-card {{
#         background-color: {WHITE};
#         border: 1px solid {BORDER_COLOR};
#         border-left: 5px solid {ACCENT_BLUE};
#         border-radius: 10px;
#         padding: 20px;
#         box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
#         transition: all 0.25s ease-in-out;
#     }}
#     .kpi-card:hover {{
#         transform: translateY(-3px);
#         box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.04);
#         border-left-color: {PRIMARY_GREEN};
#     }}
#     .kpi-value {{
#         font-size: 30px;
#         font-weight: 700;
#         color: {PRIMARY_GREEN};
#         line-height: 1.2;
#         margin: 6px 0;
#     }}
#     .kpi-label {{
#         font-size: 12px;
#         font-weight: 600;
#         color: {TEXT_MUTED};
#         text-transform: uppercase;
#         letter-spacing: 0.8px;
#     }}
#     .kpi-delta {{
#         font-size: 13px;
#         font-weight: 500;
#         display: flex;
#         align-items: center;
#         gap: 4px;
#     }}
#     .kpi-delta.positive {{ color: #10B981; }}
#     .kpi-delta.negative {{ color: #EF4444; }}

#     /* Recommendation Cards */
#     .rec-card {{
#         background-color: {WHITE};
#         border-radius: 10px;
#         padding: 22px 24px;
#         border: 1px solid {BORDER_COLOR};
#         border-left: 5px solid {PRIMARY_GREEN};
#         box-shadow: 0 2px 4px rgba(0,0,0,0.03);
#         margin-bottom: 20px;
#         transition: transform 0.2s ease, box-shadow 0.2s ease;
#     }}
#     .rec-card:hover {{
#         transform: translateY(-2px);
#         box-shadow: 0 8px 16px rgba(0,0,0,0.06);
#     }}
#     .rec-title {{
#         font-size: 18px;
#         font-weight: 600;
#         color: {PRIMARY_GREEN};
#         margin-bottom: 8px;
#     }}
#     .rec-body {{
#         font-size: 14.5px;
#         color: #334155;
#         line-height: 1.6;
#     }}

#     /* Demo Banner */
#     .demo-banner {{
#         background-color: #FEF3C7;
#         border: 1px solid #FCD34D;
#         color: #78350F;
#         padding: 14px 18px;
#         border-radius: 8px;
#         font-size: 14px;
#         line-height: 1.5;
#         margin-bottom: 20px;
#         box-shadow: 0 1px 3px rgba(0,0,0,0.05);
#     }}

#     /* Modern Buttons */
#     div.stButton > button {{
#         background-color: {ACCENT_BLUE};
#         color: white;
#         font-weight: 600;
#         border-radius: 8px;
#         border: none;
#         padding: 10px 20px;
#         transition: all 0.2s ease-in-out;
#         box-shadow: 0 2px 4px rgba(30, 90, 168, 0.2);
#     }}

#     # div.stButton > button:hover {{
#     #     background-color: white;
#     #     color: {ACCENT_BLUE};
#     #     transform: translateY(-1px);
#     #     box-shadow: 0 4px 8px rgba(30, 90, 168, 0.3);
#     # }}

#     /* Layout Spacing */
#     .block-container {{
#         padding-top: 2rem;
#         padding-bottom: 3rem;
#     }}
# </style>
# """
# st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# # ----------------------------------------------------------------------
# # HELPER FOR PLOTLY THEMING
# # ----------------------------------------------------------------------
# def apply_plotly_theme(fig, height=400, title=""):
#     """Applies clean styles to Plotly charts."""
#     fig.update_layout(
#         title=dict(text=title, font=dict(size=16, family="Inter", color=PRIMARY_GREEN)),
#         plot_bgcolor=WHITE,
#         paper_bgcolor=WHITE,
#         height=height,
#         margin=dict(l=20, r=20, t=50 if title else 20, b=20),
#         font=dict(family="Inter", color="#334155"),
#         xaxis=dict(
#             showgrid=True,
#             gridcolor="#F1F5F9",
#             linecolor=BORDER_COLOR,
#             zeroline=False
#         ),
#         yaxis=dict(
#             showgrid=True,
#             gridcolor="#F1F5F9",
#             linecolor=BORDER_COLOR,
#             zeroline=False
#         ),
#         legend=dict(
#             orientation="h",
#             yanchor="bottom",
#             y=1.02,
#             xanchor="right",
#             x=1,
#             bgcolor="rgba(255,255,255,0.8)",
#             bordercolor=BORDER_COLOR,
#             borderwidth=1
#         ),
#         hovermode="x unified"
#     )
#     return fig

# # ----------------------------------------------------------------------
# # DATA LOADING (cached)
# # ----------------------------------------------------------------------
# @st.cache_data
# def get_data():
#     df = load_data("data/bank_service_data.csv")
#     df = add_moving_averages(df, "calls_received")
#     return df

# @st.cache_data
# def get_forecast(metric, days_ahead):
#     df = get_data()
#     model, cols = train_regression(df, metric)
#     future = forecast_future(df, model, cols, days_ahead=days_ahead, column=metric)
#     return future

# @st.cache_data
# def get_leads():
#     return pd.read_csv("outreach/demo_leads.csv")

# df = get_data()

# # ----------------------------------------------------------------------
# # SIDEBAR NAVIGATION
# # ----------------------------------------------------------------------
# st.sidebar.title("Outreach Tracker Agent")
# st.sidebar.caption("AI Engineering Department")
# st.sidebar.markdown("<br>", unsafe_allow_html=True)
# page = st.sidebar.radio(
#     "Navigate",
#     ["Overview", "Forecast", "Recommendations", "Outreach Tracker"],
# )
# st.sidebar.markdown("---")
# st.sidebar.caption("Client Context: Bank Customer Service Centre")
# st.sidebar.caption(f"Data Range: {df['date'].min().date()} to {df['date'].max().date()}")

# # ----------------------------------------------------------------------
# # PAGE 1: OVERVIEW
# # ----------------------------------------------------------------------
# if page == "Overview":
#     st.title("Overview - Bank Customer Service Centre")
#     st.markdown(
#         "<p style='color:#64748B; font-size: 15px; margin-top: -10px; margin-bottom: 25px;'>"
#         "18 months of simulated operational data for a bank customer service centre "
#         "(call centre volume, branch foot traffic, satisfaction, and social engagement)."
#         "</p>",
#         unsafe_allow_html=True
#     )

#     latest = df.iloc[-1]
#     prev30 = df.iloc[-31]
#     c1, c2, c3, c4 = st.columns(4)
#     kpis = [
#         (c1, "Calls Received (latest)", f"{int(latest['calls_received']):,}", latest['calls_received'] - prev30['calls_received']),
#         (c2, "Avg Wait Time (min)", f"{latest['avg_wait_time_min']:.1f}", latest['avg_wait_time_min'] - prev30['avg_wait_time_min']),
#         (c3, "Branch Foot Traffic", f"{int(latest['branch_foot_traffic']):,}", latest['branch_foot_traffic'] - prev30['branch_foot_traffic']),
#         (c4, "CSAT Score", f"{latest['csat_score']:.0f}", latest['csat_score'] - prev30['csat_score']),
#     ]
#     for col, label, value, delta in kpis:
#         with col:
#             is_positive = delta >= 0
#             arrow = "▲" if is_positive else "▼"
#             delta_class = "positive" if is_positive else "negative"
#             st.markdown(f"""
#                 <div class="kpi-card">
#                     <div class="kpi-label">{label}</div>
#                     <div class="kpi-value">{value}</div>
#                     <div class="kpi-delta {delta_class}">
#                         <span>{arrow} {abs(round(delta, 1))}</span>
#                         <span style="color:#777; font-weight:400; font-size:11px;">vs 30d ago</span>
#                     </div>
#                 </div>
#             """, unsafe_allow_html=True)

#     st.markdown("<br>", unsafe_allow_html=True)
#     st.markdown("### Daily Call Volume — Trend")
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=df["date"], y=df["calls_received"], name="Daily calls",
#         line=dict(color="#9BB8AE", width=1.5), opacity=0.7
#     ))
#     fig.add_trace(go.Scatter(
#         x=df["date"], y=df["calls_received_ma30"], name="30-day moving avg",
#         line=dict(color=ACCENT_BLUE, width=3)
#     ))
#     fig = apply_plotly_theme(fig, height=380)
#     st.plotly_chart(fig, use_container_width=True)

#     c5, c6 = st.columns(2)
#     with c5:
#         st.markdown("### Branch Foot Traffic")
#         fig2 = go.Figure()
#         fig2.add_trace(go.Scatter(
#             x=df["date"], y=df["branch_foot_traffic"], 
#             line=dict(color=PRIMARY_GREEN, width=2), name="Foot Traffic"
#         ))
#         fig2 = apply_plotly_theme(fig2, height=280)
#         st.plotly_chart(fig2, use_container_width=True)
        
#     with c6:
#         st.markdown("### CSAT Score")
#         fig3 = go.Figure()
#         fig3.add_trace(go.Scatter(
#             x=df["date"], y=df["csat_score"], 
#             line=dict(color=BLACK, width=2), name="CSAT"
#         ))
#         fig3 = apply_plotly_theme(fig3, height=280)
#         st.plotly_chart(fig3, use_container_width=True)

# # ----------------------------------------------------------------------
# # PAGE 2: FORECAST
# # ----------------------------------------------------------------------
# elif page == "Forecast":
#     st.title("Forecast — Actual vs Predicted")
#     st.markdown("<br>", unsafe_allow_html=True)

#     c_sel1, c_sel2 = st.columns([1, 2])
#     with c_sel1:
#         metric_map = {
#             "Calls Received": "calls_received",
#             "Branch Foot Traffic": "branch_foot_traffic",
#             "Social Media Engagement": "social_engagement",
#         }
#         metric_label = st.selectbox("Select metric to forecast", list(metric_map.keys()))
#         metric = metric_map[metric_label]
#     with c_sel2:
#         horizon = st.slider("Forecast horizon (days)", 7, 60, 30)

#     future = get_forecast(metric, horizon)

#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=df["date"], y=df[metric], name="Actual",
#         line=dict(color=PRIMARY_GREEN, width=2)
#     ))
#     fig.add_trace(go.Scatter(
#         x=future["date"], y=future[f"{metric}_forecast"], name="Forecast (Linear Regression)",
#         line=dict(color=ACCENT_BLUE, width=3, dash="dash")
#     ))
#     if f"{metric}_ma30" in df.columns:
#         fig.add_trace(go.Scatter(
#             x=df["date"], y=df[f"{metric}_ma30"], name="30-day Moving Avg",
#             line=dict(color="#94A3B8", width=1.5, dash="dot")
#         ))
    
#     fig = apply_plotly_theme(fig, height=440, title=f"{metric_label}: Actual vs {horizon}-Day Forecast")
#     st.plotly_chart(fig, use_container_width=True)

#     avg_forecast = future[f"{metric}_forecast"].mean()
#     avg_actual_last30 = df[metric].tail(30).mean()
#     change_pct = ((avg_forecast - avg_actual_last30) / avg_actual_last30) * 100

#     c1, c2 = st.columns(2)
#     c1.metric("Avg. Last 30 Days", f"{avg_actual_last30:,.0f}")
#     c2.metric(f"Avg. Forecast (Next {horizon} Days)", f"{avg_forecast:,.0f}", f"{change_pct:+.1f}%")

#     with st.expander("Model Notes and Methodology"):
#         st.write(
#             "The forecast combines a linear regression (trained on day-index and "
#             "day-of-week dummies to capture trend and weekly seasonality) with a "
#             "30-day moving average shown for comparison. This keeps the model "
#             "simple, explainable, and easy to re-train as new data arrives."
#         )

# # ----------------------------------------------------------------------
# # PAGE 3: RECOMMENDATIONS
# # ----------------------------------------------------------------------
# elif page == "Recommendations":
#     st.title("Business Recommendations")
#     st.markdown(
#         "<p style='color:#64748B; font-size: 15px; margin-top: -10px; margin-bottom: 25px;'>"
#         "Data-driven strategic recommendations derived directly from current model forecasts."
#         "</p>",
#         unsafe_allow_html=True
#     )

#     calls_future = get_forecast("calls_received", 30)
#     wait_latest = df["avg_wait_time_min"].tail(30).mean()
#     csat_latest = df["csat_score"].tail(30).mean()
#     social_future = get_forecast("social_engagement", 30)

#     recs = [
#         (
#             "1. Adjust staffing ahead of forecasted call-volume peaks",
#             f"The model projects average daily calls of ~<b>{calls_future['calls_received_forecast'].mean():.0f}</b> "
#             f"over the next 30 days, with weekday peaks (Mon/Fri) running highest. "
#             "Recommend shifting 1 to 2 extra agents to Monday and Friday shifts to keep "
#             "wait times from climbing further."
#         ),
#         (
#             "2. Invest in self-service / chatbot deflection for routine queries",
#             f"Current average wait time is ~<b>{wait_latest:.1f} minutes</b> and CSAT sits at "
#             f"~<b>{csat_latest:.0f}/100</b>, with a clear inverse relationship between the two. "
#             "Deflecting routine balance/statement queries to a self-service channel "
#             "should reduce wait times and lift CSAT without adding headcount."
#         ),
#         (
#             "3. Time social/marketing pushes to forecasted engagement dips",
#             f"Social engagement is forecast to average ~<b>{social_future['social_engagement_forecast'].mean():.0f}</b> "
#             "interactions/day over the next month, with a cyclical dip roughly every 4 weeks. "
#             "Scheduling promotional content just before these dips can help smooth "
#             "engagement rather than reacting after it has already dropped."
#         ),
#     ]

#     for title, body in recs:
#         st.markdown(f"""
#             <div class="rec-card">
#                 <div class="rec-title">{title}</div>
#                 <div class="rec-body">{body}</div>
#             </div>
#         """, unsafe_allow_html=True)

#     st.info("Note: These recommendations are generated directly from the simulated dataset "
#             "and forecast model — swap in real operational data to make them client-specific.")

# # ----------------------------------------------------------------------
# # PAGE 4: OUTREACH TRACKER (DEMO ONLY)
# # ----------------------------------------------------------------------
# elif page == "Outreach Tracker":
#     st.title("Automated Lead Search & Outreach")
#     # st.markdown(
#     #     '<div class="demo-banner"><b>Demo Mode:</b> Everything on this page is '
#     #     'SIMULATED sample data for portfolio purposes. No real businesses are contacted '
#     #     'and no real emails are sent by this dashboard. In a live deployment, this tab '
#     #     'would show the actual lead list and real outreach log.</div>',
#     #     unsafe_allow_html=True,
#     # )

#     leads = get_leads()

#     c1, c2, c3 = st.columns(3)
#     c1.metric("Simulated Leads Found", len(leads))
#     c2.metric("Simulated Messages Sent", leads.shape[0])
#     interested = leads[leads["response_status"].str.contains("Interested")].shape[0]
#     c3.metric("Simulated Positive Responses", interested)

#     st.markdown("<br>", unsafe_allow_html=True)
#     st.markdown("### Response Breakdown")
#     status_counts = leads["response_status"].value_counts().reset_index()
#     status_counts.columns = ["status", "count"]
    
#     fig = go.Figure(go.Bar(
#         x=status_counts["count"], y=status_counts["status"], orientation="h",
#         marker_color=ACCENT_BLUE,
#         text=status_counts["count"], textposition="auto"
#     ))
#     fig = apply_plotly_theme(fig, height=280)
#     st.plotly_chart(fig, use_container_width=True)

#     st.markdown("### Outreach Log (Demo Chat View)")
#     search = st.text_input("Filter by locality or business type", "")
#     filtered = leads
#     if search:
#         mask = (
#             leads["locality_lahore"].str.contains(search, case=False)
#             | leads["business_type"].str.contains(search, case=False)
#         )
#         filtered = leads[mask]

#     for _, row in filtered.iterrows():
#         with st.expander(f"#{row['lead_id']} — {row['organization_demo_name']}  |  {row['response_status']}"):
#             st.write(f"**Platform:** {row['platform']}  |  **Date:** {row['contact_date']}")
#             st.markdown(
#                 f"""<div style="background:{WHITE}; border-radius:10px; padding:14px 18px;
#                 border:1px solid {BORDER_COLOR}; margin-top: 8px;">{row['message_sent_demo']}</div>""",
#                 unsafe_allow_html=True,
#             )

#     st.markdown("<br>", unsafe_allow_html=True)
#     st.markdown("### Full Tracker Table")
#     st.dataframe(filtered.drop(columns=["message_sent_demo"]), use_container_width=True)

#     csv = filtered.to_csv(index=False).encode("utf-8")
#     st.download_button("Download Outreach Tracker (CSV)", csv, "week4_outreach_tracker_demo.csv", "text/csv")


"""
app.py - Bank Customer Service Centre | Predictive Dashboard
==============================================================
SafeX Solutions - Week 4 Client-Ready Sprint

Run from the project root with:
    streamlit run dashboard/app.py

Tabs:
    1. Overview        - KPIs + historical trends
    2. Forecast        - Actual vs predicted (moving average + regression)
    3. Recommendations - 3 business recommendations derived from forecast
    4. Outreach Tracker- SIMULATED / DEMO lead-gen + outreach log (no real
                         emails are sent from this app - clearly labeled demo)
"""

import sys
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.forecast_model import load_data, add_moving_averages, train_regression, forecast_future

# ----------------------------------------------------------------------
# PAGE CONFIG + ENHANCED THEME
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="SafeX | Bank CSC Predictive Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Core Palette
PRIMARY_GREEN = "#0B3D2E"
ACCENT_BLUE = "#1E5AA8"
BLACK = "#111111"
WHITE = "#FFFFFF"
LIGHT_BG = "#F4F7F6"
BORDER_COLOR = "#E2E8F0"
TEXT_MUTED = "#64748B"

CUSTOM_CSS = f"""
<style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    /* Global Page Background */
    .stApp {{
        background-color: {LIGHT_BG};
        color: {BLACK};
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: {PRIMARY_GREEN};
        box-shadow: 2px 0 12px rgba(0,0,0,0.05);
    }}
    section[data-testid="stSidebar"] * {{
        color: {WHITE} !important;
    }}
    section[data-testid="stSidebar"] .stRadio > label {{
        font-weight: 500;
    }}

    /* Headings */
    h1, h2, h3 {{
        color: {PRIMARY_GREEN};
        font-weight: 700;
        letter-spacing: -0.02em;
    }}

    /* High-Contrast KPI Cards */
    .kpi-card {{
        background-color: {WHITE};
        border: 1px solid {BORDER_COLOR};
        border-left: 5px solid {ACCENT_BLUE};
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
        transition: all 0.25s ease-in-out;
    }}
    .kpi-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.04);
        border-left-color: {PRIMARY_GREEN};
    }}
    .kpi-value {{
        font-size: 30px;
        font-weight: 700;
        color: {PRIMARY_GREEN};
        line-height: 1.2;
        margin: 6px 0;
    }}
    .kpi-label {{
        font-size: 12px;
        font-weight: 600;
        color: {TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }}
    .kpi-delta {{
        font-size: 13px;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 4px;
    }}
    .kpi-delta.positive {{ color: #10B981; }}
    .kpi-delta.negative {{ color: #EF4444; }}

    /* Recommendation Cards */
    .rec-card {{
        background-color: {WHITE};
        border-radius: 10px;
        padding: 22px 24px;
        border: 1px solid {BORDER_COLOR};
        border-left: 5px solid {PRIMARY_GREEN};
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        margin-bottom: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .rec-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.06);
    }}
    .rec-title {{
        font-size: 18px;
        font-weight: 600;
        color: {PRIMARY_GREEN};
        margin-bottom: 8px;
    }}
    .rec-body {{
        font-size: 14.5px;
        color: #334155;
        line-height: 1.6;
    }}

    /* Demo Banner */
    .demo-banner {{
        background-color: #FEF3C7;
        border: 1px solid #FCD34D;
        color: #78350F;
        padding: 14px 18px;
        border-radius: 8px;
        font-size: 14px;
        line-height: 1.5;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}

    /* Modern Buttons */
    div.stButton > button {{
        background-color: {ACCENT_BLUE};
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 2px 4px rgba(30, 90, 168, 0.2);
    }}
    div.stButton > button:hover {{
        background-color: #144483;
        color: white;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(30, 90, 168, 0.3);
    }}

    /* Layout Spacing */
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# HELPER FOR PLOTLY THEMING (WITH EXPLICIT DARK TICKS)
# ----------------------------------------------------------------------
def apply_plotly_theme(fig, height=400, title=""):
    """Applies clean styles and forces dark, legible fonts on Plotly axes."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, family="Inter", color=PRIMARY_GREEN)),
        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,
        height=height,
        margin=dict(l=20, r=20, t=50 if title else 20, b=20),
        font=dict(family="Inter", color="#1E293B"),
        xaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            linecolor=BORDER_COLOR,
            zeroline=False,
            tickfont=dict(color="#1E293B", size=12),
            title=dict(font=dict(color="#1E293B", size=12)),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            linecolor=BORDER_COLOR,
            zeroline=False,
            tickfont=dict(color="#1E293B", size=12),
            title=dict(font=dict(color="#1E293B", size=12)),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=BORDER_COLOR,
            borderwidth=1,
            font=dict(color="#1E293B", size=12)
        ),
        hovermode="x unified"
    )
    return fig

# ----------------------------------------------------------------------
# DATA LOADING (cached)
# ----------------------------------------------------------------------
@st.cache_data
def get_data():
    df = load_data("data/bank_service_data.csv")
    df = add_moving_averages(df, "calls_received")
    return df

@st.cache_data
def get_forecast(metric, days_ahead):
    df = get_data()
    model, cols = train_regression(df, metric)
    future = forecast_future(df, model, cols, days_ahead=days_ahead, column=metric)
    return future

@st.cache_data
def get_leads():
    return pd.read_csv("outreach/demo_leads.csv")

df = get_data()

# ----------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------
st.sidebar.title("Outreach Tracker Agent")
st.sidebar.caption("AI & ML Engineering Department")
st.sidebar.markdown("<br>", unsafe_allow_html=True)
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Forecast", "Recommendations", "Outreach Tracker"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Client Context: Customer Service Centre")
st.sidebar.caption(f"Data Range: {df['date'].min().date()} to {df['date'].max().date()}")

# ----------------------------------------------------------------------
# PAGE 1: OVERVIEW
# ----------------------------------------------------------------------
if page == "Overview":
    st.title("Overview - Customer Service Centre")
    st.markdown(
        "<p style='color:#64748B; font-size: 15px; margin-top: -10px; margin-bottom: 25px;'>"
        "18 months of simulated operational data for a customer service centre "
        "(call centre volume, branch foot traffic, satisfaction, and social engagement)."
        "</p>",
        unsafe_allow_html=True
    )

    latest = df.iloc[-1]
    prev30 = df.iloc[-31]
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        (c1, "Calls Received (latest)", f"{int(latest['calls_received']):,}", latest['calls_received'] - prev30['calls_received']),
        (c2, "Avg Wait Time (min)", f"{latest['avg_wait_time_min']:.1f}", latest['avg_wait_time_min'] - prev30['avg_wait_time_min']),
        (c3, "Branch Foot Traffic", f"{int(latest['branch_foot_traffic']):,}", latest['branch_foot_traffic'] - prev30['branch_foot_traffic']),
        (c4, "CSAT Score", f"{latest['csat_score']:.0f}", latest['csat_score'] - prev30['csat_score']),
    ]
    for col, label, value, delta in kpis:
        with col:
            is_positive = delta >= 0
            arrow = "▲" if is_positive else "▼"
            delta_class = "positive" if is_positive else "negative"
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-delta {delta_class}">
                        <span>{arrow} {abs(round(delta, 1))}</span>
                        <span style="color:#777; font-weight:400; font-size:11px;">vs 30d ago</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Daily Call Volume — Trend")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["calls_received"], name="Daily calls",
        line=dict(color="#9BB8AE", width=1.5), opacity=0.7
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["calls_received_ma30"], name="30-day moving avg",
        line=dict(color=ACCENT_BLUE, width=3)
    ))
    fig = apply_plotly_theme(fig, height=380)
    st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        st.markdown("### Branch Foot Traffic")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["date"], y=df["branch_foot_traffic"], 
            line=dict(color=PRIMARY_GREEN, width=2), name="Foot Traffic"
        ))
        fig2 = apply_plotly_theme(fig2, height=280)
        st.plotly_chart(fig2, use_container_width=True)
        
    with c6:
        st.markdown("### CSAT Score")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=df["date"], y=df["csat_score"], 
            line=dict(color=BLACK, width=2), name="CSAT"
        ))
        fig3 = apply_plotly_theme(fig3, height=280)
        st.plotly_chart(fig3, use_container_width=True)

# ----------------------------------------------------------------------
# PAGE 2: FORECAST
# ----------------------------------------------------------------------
elif page == "Forecast":
    st.title("Forecast — Actual vs Predicted")
    st.markdown("<br>", unsafe_allow_html=True)

    c_sel1, c_sel2 = st.columns([1, 2])
    with c_sel1:
        metric_map = {
            "Calls Received": "calls_received",
            "Branch Foot Traffic": "branch_foot_traffic",
            "Social Media Engagement": "social_engagement",
        }
        metric_label = st.selectbox("Select metric to forecast", list(metric_map.keys()))
        metric = metric_map[metric_label]
    with c_sel2:
        horizon = st.slider("Forecast horizon (days)", 7, 60, 30)

    future = get_forecast(metric, horizon)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df[metric], name="Actual",
        line=dict(color=PRIMARY_GREEN, width=2)
    ))
    fig.add_trace(go.Scatter(
        x=future["date"], y=future[f"{metric}_forecast"], name="Forecast (Linear Regression)",
        line=dict(color=ACCENT_BLUE, width=3, dash="dash")
    ))
    if f"{metric}_ma30" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df[f"{metric}_ma30"], name="30-day Moving Avg",
            line=dict(color="#94A3B8", width=1.5, dash="dot")
        ))
    
    fig = apply_plotly_theme(fig, height=440, title=f"{metric_label}: Actual vs {horizon}-Day Forecast")
    st.plotly_chart(fig, use_container_width=True)

    avg_forecast = future[f"{metric}_forecast"].mean()
    avg_actual_last30 = df[metric].tail(30).mean()
    change_pct = ((avg_forecast - avg_actual_last30) / avg_actual_last30) * 100

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Avg. Last 30 Days</div>
                <div class="kpi-value">{avg_actual_last30:,.0f}</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Avg. Forecast (Next {horizon} Days)</div>
                <div class="kpi-value">{avg_forecast:,.0f} <span style="font-size:16px; font-weight:500; color:{'#10B981' if change_pct >= 0 else '#EF4444'};">({change_pct:+.1f}%)</span></div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Model Notes and Methodology"):
        st.write(
            "The forecast combines a linear regression (trained on day-index and "
            "day-of-week dummies to capture trend and weekly seasonality) with a "
            "30-day moving average shown for comparison. This keeps the model "
            "simple, explainable, and easy to re-train as new data arrives."
        )

# ----------------------------------------------------------------------
# PAGE 3: RECOMMENDATIONS
# ----------------------------------------------------------------------
elif page == "Recommendations":
    st.title("Business Recommendations")
    st.markdown(
        "<p style='color:#64748B; font-size: 15px; margin-top: -10px; margin-bottom: 25px;'>"
        "Data-driven strategic recommendations derived directly from current model forecasts."
        "</p>",
        unsafe_allow_html=True
    )

    calls_future = get_forecast("calls_received", 30)
    wait_latest = df["avg_wait_time_min"].tail(30).mean()
    csat_latest = df["csat_score"].tail(30).mean()
    social_future = get_forecast("social_engagement", 30)

    recs = [
        (
            "1. Adjust staffing ahead of forecasted call-volume peaks",
            f"The model projects average daily calls of ~<b>{calls_future['calls_received_forecast'].mean():.0f}</b> "
            f"over the next 30 days, with weekday peaks (Mon/Fri) running highest. "
            "Recommend shifting 1 to 2 extra agents to Monday and Friday shifts to keep "
            "wait times from climbing further."
        ),
        (
            "2. Invest in self-service / chatbot deflection for routine queries",
            f"Current average wait time is ~<b>{wait_latest:.1f} minutes</b> and CSAT sits at "
            f"~<b>{csat_latest:.0f}/100</b>, with a clear inverse relationship between the two. "
            "Deflecting routine balance/statement queries to a self-service channel "
            "should reduce wait times and lift CSAT without adding headcount."
        ),
        (
            "3. Time social/marketing pushes to forecasted engagement dips",
            f"Social engagement is forecast to average ~<b>{social_future['social_engagement_forecast'].mean():.0f}</b> "
            "interactions/day over the next month, with a cyclical dip roughly every 4 weeks. "
            "Scheduling promotional content just before these dips can help smooth "
            "engagement rather than reacting after it has already dropped."
        ),
    ]

    for title, body in recs:
        st.markdown(f"""
            <div class="rec-card">
                <div class="rec-title">{title}</div>
                <div class="rec-body">{body}</div>
            </div>
        """, unsafe_allow_html=True)

    st.info("Note: These recommendations are generated directly from the simulated dataset "
            "and forecast model — swap in real operational data to make them client-specific.")

# ----------------------------------------------------------------------
# PAGE 4: OUTREACH TRACKER (DEMO ONLY)
# ----------------------------------------------------------------------
elif page == "Outreach Tracker":
    st.title("Automated Lead Search & Outreach")

    leads = get_leads()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Simulated Leads Found</div>
                <div class="kpi-value">{len(leads)}</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Simulated Messages Sent</div>
                <div class="kpi-value">{leads.shape[0]}</div>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        interested = leads[leads["response_status"].str.contains("Interested")].shape[0]
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Simulated Positive Responses</div>
                <div class="kpi-value">{interested}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Response Breakdown")
    status_counts = leads["response_status"].value_counts().reset_index()
    status_counts.columns = ["status", "count"]
    
    fig = go.Figure(go.Bar(
        x=status_counts["count"], y=status_counts["status"], orientation="h",
        marker_color=ACCENT_BLUE,
        text=status_counts["count"], textposition="auto"
    ))
    fig = apply_plotly_theme(fig, height=320)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Outreach Log (Chat View)")
    search = st.text_input("Filter by locality or business type", "")
    filtered = leads
    if search:
        mask = (
            leads["locality_lahore"].str.contains(search, case=False)
            | leads["business_type"].str.contains(search, case=False)
        )
        filtered = leads[mask]

    for _, row in filtered.iterrows():
        with st.expander(f"#{row['lead_id']} — {row['organization_demo_name']}  |  {row['response_status']}"):
            st.write(f"**Platform:** {row['platform']}  |  **Date:** {row['contact_date']}")
            st.markdown(
                f"""<div style="background:{WHITE}; border-radius:10px; padding:14px 18px;
                border:1px solid {BORDER_COLOR}; margin-top: 8px;">{row['message_sent_demo']}</div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Full Tracker Table")
    st.dataframe(filtered.drop(columns=["message_sent_demo"]), use_container_width=True)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download Outreach Tracker (CSV)", csv, "week4_outreach_tracker_demo.csv", "text/csv")