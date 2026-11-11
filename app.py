"""Advanced ML Model Comparison Dashboard.
Engineered for leakage-safe benchmarking, rigorous evaluation, and production-grade UI aesthetics.
Run: streamlit run app.py
"""
from __future__ import annotations

import html
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix

from ml_engine import BUILTIN_DATASETS, evaluate_models, load_builtin

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ModelBenchmark AI | Comparative ML Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# ADVANCED CUSTOM CSS & DESIGN SYSTEM
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1440px;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: #0d121f;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
    padding-left: 1.25rem;
    padding-right: 1.25rem;
}

/* Header Navbar Banner */
.hero-navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.25rem 1.75rem;
    background: linear-gradient(135deg, rgba(19, 27, 46, 0.85) 0%, rgba(15, 23, 42, 0.7) 100%);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
}

.hero-brand {
    display: flex;
    align-items: center;
    gap: 14px;
}

.hero-logo-box {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: linear-gradient(135deg, #6366f1 0%, #3b82f6 50%, #06b6d4 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);
}

.hero-title {
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #ffffff;
    margin: 0;
    line-height: 1.2;
}

.hero-subtitle {
    font-size: 0.82rem;
    color: #94a3b8;
    margin: 0;
    font-weight: 500;
}

.hero-pills {
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
}

.pill-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #c7d2fe;
}

.pill-badge-emerald {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #6ee7b7;
}

.pill-badge-amber {
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.3);
    color: #fcd34d;
}

/* Step indicator badges in sidebar */
.step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 20px;
    height: 20px;
    border-radius: 6px;
    background: #6366f1;
    color: white;
    font-size: 0.7rem;
    font-weight: 700;
    margin-right: 8px;
}

.sidebar-section-title {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    font-weight: 700;
    margin-top: 1.25rem;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
}

/* Fix Streamlit default metrics for high-contrast dark theme */
[data-testid='stMetric'] {
    background: #131b2e !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    padding: 18px 20px !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

[data-testid='stMetric']:hover {
    border-color: rgba(99, 102, 241, 0.4) !important;
    transform: translateY(-2px);
}

[data-testid='stMetricLabel'] {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

[data-testid='stMetricValue'] {
    color: #f8fafc !important;
    font-size: 1.85rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
}

/* Custom Executive Stat Card */
.kpi-card {
    background: #131b2e;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.25rem 1.4rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
}

.kpi-card:hover {
    border-color: rgba(99, 102, 241, 0.4);
    transform: translateY(-3px);
    box-shadow: 0 12px 30px rgba(99, 102, 241, 0.15);
}

.kpi-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: #94a3b8;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.5rem;
}

.kpi-value {
    font-size: 1.95rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.03em;
    line-height: 1.2;
}

.kpi-sub {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 0.4rem;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Champion Gold Highlight Card */
.champion-banner {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(245, 158, 11, 0.12) 100%);
    border: 1px solid rgba(245, 158, 11, 0.4);
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.5rem;
    position: relative;
    box-shadow: 0 10px 30px -10px rgba(245, 158, 11, 0.2);
}

.champion-inner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
}

.champion-info {
    display: flex;
    align-items: center;
    gap: 18px;
}

.champion-icon-box {
    width: 54px;
    height: 54px;
    border-radius: 14px;
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    box-shadow: 0 8px 20px rgba(245, 158, 11, 0.35);
}

.champion-title {
    font-size: 0.85rem;
    text-transform: uppercase;
    font-weight: 800;
    letter-spacing: 0.08em;
    color: #fbbf24;
    margin: 0;
}

.champion-name {
    font-size: 1.6rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.02em;
    margin: 2px 0 0 0;
}

.champion-stats {
    display: flex;
    gap: 24px;
    align-items: center;
}

.champion-stat-item {
    text-align: right;
}

.champion-stat-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    color: #94a3b8;
    font-weight: 600;
    letter-spacing: 0.04em;
}

.champion-stat-val {
    font-size: 1.5rem;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -0.02em;
}

/* Styled Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #0f172a;
    padding: 6px;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.06);
}

.stTabs [data-baseweb="tab"] {
    height: 42px;
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 600;
    font-size: 0.88rem;
    padding: 0 16px;
    border: none !important;
    background: transparent;
    transition: all 0.2s ease;
}

.stTabs [aria-selected="true"] {
    background: #1e293b !important;
    color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
}

/* Modern Primary Button */
.stButton>button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em !important;
    padding: 0.65rem 1.25rem !important;
    box-shadow: 0 4px 18px rgba(99, 102, 241, 0.35) !important;
    transition: all 0.2s ease !important;
}

.stButton>button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 22px rgba(99, 102, 241, 0.5) !important;
    border-color: rgba(255, 255, 255, 0.3) !important;
}

/* Section Subheaders */
.section-heading {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.15rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 1.5rem 0 1rem 0;
    letter-spacing: -0.02em;
}

.section-heading span.badge-num {
    background: rgba(99, 102, 241, 0.2);
    color: #818cf8;
    width: 26px;
    height: 26px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
}

/* Dataframe & Tables styling */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    overflow: hidden;
}

/* Info & helper callouts */
.callout-box {
    background: rgba(15, 23, 42, 0.6);
    border-left: 4px solid #6366f1;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    margin: 12px 0;
    font-size: 0.85rem;
    color: #cbd5e1;
}

.callout-box strong {
    color: #ffffff;
}

/* Footer note */
.footer-note {
    text-align: center;
    color: #64748b;
    font-size: 0.78rem;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PLOTLY MODERN DARK THEME HELPER
# -----------------------------------------------------------------------------
def get_plotly_layout(height: int = 400) -> dict:
    """Returns a unified, clean modern dark layout for Plotly figures."""
    return dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(19, 27, 46, 0.4)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#94a3b8", size=12),
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.06)",
            linecolor="rgba(255, 255, 255, 0.12)",
            tickfont=dict(color="#94a3b8"),
            zerolinecolor="rgba(255, 255, 255, 0.1)",
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.06)",
            linecolor="rgba(255, 255, 255, 0.12)",
            tickfont=dict(color="#94a3b8"),
            zerolinecolor="rgba(255, 255, 255, 0.1)",
        ),
        legend=dict(
            bgcolor="rgba(19, 27, 46, 0.8)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1,
            font=dict(color="#e2e8f0"),
        ),
    )


PALETTE = ["#6366f1", "#10b981", "#3b82f6", "#f59e0b", "#ec4899", "#8b5cf6"]


# -----------------------------------------------------------------------------
# SIDEBAR: RIGOROUS EXPERIMENT CONFIGURATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom: 1.25rem;">
        <div style="width:34px; height:34px; border-radius:8px; background:linear-gradient(135deg, #6366f1, #3b82f6); display:flex; align-items:center; justify-content:center; font-size:18px;">⚡</div>
        <div>
            <div style="font-weight:800; font-size:1.05rem; color:#fff; line-height:1.1;">BenchLab Pro</div>
            <div style="font-size:0.72rem; color:#94a3b8;">Leakage-Safe ML Studio</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title"><span class="step-badge">1</span> Dataset Source</div>', unsafe_allow_html=True)
    source = st.radio(
        "Source",
        ["Built-in dataset", "Upload CSV"],
        label_visibility="collapsed",
        horizontal=False,
    )

    file = None
    if source == "Built-in dataset":
        dataset_name = st.selectbox("Select Benchmark Dataset", list(BUILTIN_DATASETS))
        df, target_default, task_default = load_builtin(dataset_name)
        task = task_default
        target = target_default
        st.markdown(f"""
        <div class="callout-box">
            <strong>Active Target:</strong> <code>{html.escape(str(target))}</code><br>
            <strong>Inferred Task:</strong> <span class="pill-badge pill-badge-emerald" style="padding:2px 8px; margin-top:4px;">{task.title()}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        file = st.file_uploader("Upload CSV Dataset", type=["csv"], help="Select a clean CSV dataset.")
        if file is None:
            st.info("Upload a CSV file to begin automated evaluation.")
            st.stop()
        try:
            df = pd.read_csv(file)
        except Exception as exc:
            st.error(f"Cannot read CSV file: {exc}")
            st.stop()
        if df.empty or len(df.columns) < 2:
            st.error("The dataset must include at least one feature and one target column.")
            st.stop()

        target = st.selectbox("Target (Label) Column", list(df.columns), index=len(df.columns) - 1)
        task = st.radio("Problem Type", ["classification", "regression"], horizontal=True)

    st.markdown('<div class="sidebar-section-title"><span class="step-badge">2</span> Split & Validation</div>', unsafe_allow_html=True)
    test_pct = st.slider("Holdout Test Split (%)", min_value=10, max_value=40, value=25, step=5,
                         help="Percentage of observations reserved exclusively for unseen testing.")
    seed = st.number_input("Deterministic Random Seed", min_value=0, max_value=10000, value=42,
                           help="Ensures exact algorithmic reproducibility across model runs.")

    st.markdown('<div class="sidebar-section-title"><span class="step-badge">3</span> Execution</div>', unsafe_allow_html=True)
    run_btn = st.button("🚀 Train & Benchmark Models", type="primary", use_container_width=True)

    st.markdown("""
    <div style="margin-top: 1.5rem; padding: 12px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; font-size: 0.72rem; color: #94a3b8; line-height: 1.5;">
        <span style="color:#10b981; font-weight:700;">● Leakage-Safe Guarantee:</span> Preprocessing transforms (imputation, standard scaling, one-hot encoding) are fitted strictly on training slices only.
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# HERO NAVBAR
# -----------------------------------------------------------------------------
missing_total = int(df.isna().sum().sum())
n_rows, n_cols = len(df), len(df.columns)

st.markdown(f"""
<div class="hero-navbar">
    <div class="hero-brand">
        <div class="hero-logo-box">⚡</div>
        <div>
            <h1 class="hero-title">ML Model Comparison & Benchmark Suite</h1>
            <p class="hero-subtitle">Rigorous cross-algorithmic performance evaluation across unified train/test holdouts</p>
        </div>
    </div>
    <div class="hero-pills">
        <span class="pill-badge pill-badge-emerald">● Pipeline Ready</span>
        <span class="pill-badge">{n_rows:,} Rows</span>
        <span class="pill-badge">{n_cols} Features</span>
        <span class="pill-badge pill-badge-amber">{task.title()}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# DATASET QUICK HEALTH METRICS
# -----------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label"><span>Dataset Records</span> <span>📋</span></div>
        <div class="kpi-value">{n_rows:,}</div>
        <div class="kpi-sub">Total sample count</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label"><span>Feature Dimensions</span> <span>📐</span></div>
        <div class="kpi-value">{n_cols}</div>
        <div class="kpi-sub">Columns including target</div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label"><span>Missing Cells</span> <span>🔍</span></div>
        <div class="kpi-value" style="color: {'#10b981' if missing_total == 0 else '#f59e0b'};">{missing_total:,}</div>
        <div class="kpi-sub">Auto-imputed in pipeline</div>
    </div>
    """, unsafe_allow_html=True)
with c4:
    test_rows_est = int(np.ceil(n_rows * (test_pct / 100)))
    train_rows_est = n_rows - test_rows_est
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label"><span>Train / Holdout Split</span> <span>⚖️</span></div>
        <div class="kpi-value">{100 - test_pct}/{test_pct}%</div>
        <div class="kpi-sub">{train_rows_est:,} train · {test_rows_est:,} test rows</div>
    </div>
    """, unsafe_allow_html=True)


file_key = f"{file.name}_{file.size}" if (file is not None and hasattr(file, "name")) else "builtin"
config_key = (
    source,
    dataset_name if source == "Built-in dataset" else file_key,
    target,
    task,
    test_pct,
    int(seed),
)

if run_btn:
    with st.spinner("Executing rigorous leakage-safe training pipelines across 6 algorithms..."):
        try:
            metrics_df, models, outputs, test_n, errors = evaluate_models(
                df, target, task, test_pct / 100, int(seed)
            )
            st.session_state["comparison"] = (config_key, metrics_df, models, outputs, test_n, errors)
            st.toast("🎉 Benchmark completed successfully!", icon="✅")
        except Exception as exc:
            st.session_state.pop("comparison", None)
            st.error(f"Execution Error: {exc}")

saved = st.session_state.get("comparison")
if not saved or saved[0] != config_key:
    st.markdown("""
    <div style="background: rgba(19, 27, 46, 0.7); border: 1px dashed rgba(99, 102, 241, 0.4); border-radius: 16px; padding: 3rem 2rem; text-align: center; margin-top: 2rem;">
        <div style="font-size: 3rem; margin-bottom: 0.75rem;">🚀</div>
        <h3 style="color: #ffffff; font-weight: 700; margin-bottom: 0.5rem;">Ready to Benchmark Algorithms</h3>
        <p style="color: #94a3b8; max-width: 600px; margin: 0 auto 1.5rem auto; font-size: 0.95rem;">
            Click <strong>"Train & Benchmark Models"</strong> in the sidebar to train 6 machine learning architectures on identical train/test splits with standardized metrics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🔎 Preview Raw Dataset & Schema", expanded=False):
        st.dataframe(df.head(15), use_container_width=True)
    st.stop()
else:
    # Unpack benchmark results
    _, scores, models, predictions, test_n, errors = saved
    best_model = scores.iloc[0]

    # -----------------------------------------------------------------------------
    # EXECUTIVE CHAMPION CALLOUT
    # -----------------------------------------------------------------------------
    if task == "classification":
        stat_primary_label = "Top Accuracy"
        stat_primary_val = f"{best_model['Accuracy']:.2%}"
        stat_secondary_label = "F1-Score"
        stat_secondary_val = f"{best_model['F1 Score']:.2%}"
        stat_tertiary_label = "Error Rate"
        stat_tertiary_val = f"{best_model['Error Rate']:.2%}"
    else:
        stat_primary_label = "Lowest RMSE"
        stat_primary_val = f"{best_model['RMSE']:.3f}"
        stat_secondary_label = "MAE"
        stat_secondary_val = f"{best_model['MAE']:.3f}"
        stat_tertiary_label = "R² Score"
        stat_tertiary_val = f"{best_model['R2 Score']:.3f}"

    st.markdown(f"""
    <div class="champion-banner">
        <div class="champion-inner">
            <div class="champion-info">
                <div class="champion-icon-box">🏆</div>
                <div>
                    <p class="champion-title">Benchmarked Champion Model</p>
                    <h2 class="champion-name">{html.escape(str(best_model['Model']))}</h2>
                    <div style="font-size: 0.8rem; color: #cbd5e1; margin-top: 4px;">
                        Outperformed all competitors on holdout split of <strong>{test_n}</strong> unseen test samples
                    </div>
                </div>
            </div>
            <div class="champion-stats">
                <div class="champion-stat-item">
                    <div class="champion-stat-label">{stat_primary_label}</div>
                    <div class="champion-stat-val" style="color: #34d399;">{stat_primary_val}</div>
                </div>
                <div class="champion-stat-item">
                    <div class="champion-stat-label">{stat_secondary_label}</div>
                    <div class="champion-stat-val" style="color: #60a5fa;">{stat_secondary_val}</div>
                </div>
                <div class="champion-stat-item">
                    <div class="champion-stat-label">{stat_tertiary_label}</div>
                    <div class="champion-stat-val" style="color: #f87171;">{stat_tertiary_val}</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------------------------------------------------------
    # MAIN DASHBOARD TABS
    # -----------------------------------------------------------------------------
    tab_leaderboard, tab_charts, tab_deepdive, tab_data, tab_export = st.tabs([
        "🏆 Leaderboard & Rankings",
        "📈 Visual Analytics",
        "🔬 Diagnostic Deep-Dive",
        "📁 Dataset Explorer",
        "📑 Executive Report & Export",
    ])

    # -----------------------------------------------------------------------------
    # TAB 1: LEADERBOARD & RANKINGS
    # -----------------------------------------------------------------------------
    with tab_leaderboard:
        st.markdown('<div class="section-heading"><span class="badge-num">1</span> Comprehensive Model Leaderboard</div>', unsafe_allow_html=True)
        st.caption("All models evaluated against the identical leakage-safe test split. Models ranked by primary performance metric.")

        display_df = scores.copy()
        medals = ["🥇 1st", "🥈 2nd", "🥉 3rd"] + [f"{i}th" for i in range(4, len(display_df) + 1)]
        display_df.insert(0, "Rank", medals[:len(display_df)])

        format_dict = {}
        if task == "classification":
            for col in ["Accuracy", "Error Rate", "Precision", "Recall", "F1 Score"]:
                if col in display_df.columns:
                    format_dict[col] = "{:.2%}"
        else:
            for col in ["MAE", "MSE", "RMSE", "R2 Score"]:
                if col in display_df.columns:
                    format_dict[col] = "{:.4f}"

        st.dataframe(
            display_df.style.format(format_dict),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown('<div class="section-heading"><span class="badge-num">2</span> Multi-Metric Algorithmic Comparison</div>', unsafe_allow_html=True)

        if task == "classification":
            compare_metrics = [c for c in ["Accuracy", "Precision", "Recall", "F1 Score"] if c in scores.columns]
            melted = scores.melt(id_vars=["Model"], value_vars=compare_metrics, var_name="Metric", value_name="Score")

            fig_multi = px.bar(
                melted,
                x="Model",
                y="Score",
                color="Metric",
                barmode="group",
                text_auto=".1%",
                color_discrete_sequence=PALETTE,
            )
            fig_multi.update_layout(
                **get_plotly_layout(height=420),
                yaxis_tickformat=".0%",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_multi, use_container_width=True)
        else:
            compare_metrics = [c for c in ["MAE", "RMSE"] if c in scores.columns]
            melted = scores.melt(id_vars=["Model"], value_vars=compare_metrics, var_name="Metric", value_name="Error")

            fig_multi = px.bar(
                melted,
                x="Model",
                y="Error",
                color="Metric",
                barmode="group",
                text_auto=".3f",
                color_discrete_sequence=PALETTE,
            )
            fig_multi.update_layout(
                    height=420,
                    margin=dict(l=20, r=20, t=40, b=20), # Add your preferred layout defaults here
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

    # -----------------------------------------------------------------------------
    # TAB 2: VISUAL ANALYTICS
    # -----------------------------------------------------------------------------
    with tab_charts:
        st.markdown('<div class="section-heading"><span class="badge-num">1</span> Interactive Metric Inspector</div>', unsafe_allow_html=True)

        c_metric, c_sort = st.columns([3, 1])
        available_metrics = [c for c in scores.columns if c != "Model"]
        with c_metric:
            selected_metric = st.selectbox("Select Target Metric to Visualize", available_metrics)
        with c_sort:
            sort_order = st.radio("Sort Direction", ["Descending", "Ascending"], horizontal=True)

        chart_df = scores.sort_values(selected_metric, ascending=(sort_order == "Ascending"))

        is_pct = task == "classification" and selected_metric in ["Accuracy", "Error Rate", "Precision", "Recall", "F1 Score"]
        format_str = ".2%" if is_pct else ".4f"

        fig_bar = px.bar(
            chart_df,
            x="Model",
            y=selected_metric,
            color=selected_metric,
            text_auto=format_str,
            color_continuous_scale="Viridis" if "Error" in selected_metric or selected_metric in ["MAE", "MSE", "RMSE"] else "Tealgrn",
        )
        layout_cfg = get_plotly_layout(height=450)
        if is_pct:
            layout_cfg["yaxis_tickformat"] = ".0%"
        fig_bar.update_layout(**layout_cfg, coloraxis_showscale=False)
        fig_bar.update_traces(
            marker_line_color="rgba(255,255,255,0.2)",
            marker_line_width=1,
            textfont_size=13,
            textposition="outside",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # -----------------------------------------------------------------------------
    # TAB 3: DIAGNOSTIC DEEP-DIVE
    # -----------------------------------------------------------------------------
    with tab_deepdive:
        st.markdown('<div class="section-heading"><span class="badge-num">1</span> Algorithmic Diagnostics & Predictions</div>', unsafe_allow_html=True)

        inspect_model_name = st.selectbox("Select Model for In-Depth Diagnostics", scores["Model"].tolist())
        actual = predictions[inspect_model_name]["actual"]
        predicted = predictions[inspect_model_name]["predicted"]

        if task == "classification":
            labels = sorted(set(actual.tolist()) | set(predicted.tolist()), key=str)
            cm = confusion_matrix(actual, predicted, labels=labels)

            col_cm1, col_cm2 = st.columns([3, 2])
            with col_cm1:
                st.markdown("##### Confusion Matrix Heatmap")
                cm_fig = px.imshow(
                    cm,
                    x=[str(x) for x in labels],
                    y=[str(y) for y in labels],
                    text_auto=True,
                    color_continuous_scale="Blues",
                    labels=dict(x="Predicted Class", y="Actual Ground Truth", color="Observations"),
                )
                cm_fig.update_layout(**get_plotly_layout(height=380))
                st.plotly_chart(cm_fig, use_container_width=True)

            with col_cm2:
                st.markdown("##### Holdout Prediction Statistics")
                total_cases = len(actual)
                correct_cases = int(np.sum(actual == predicted))
                incorrect_cases = total_cases - correct_cases

                st.markdown(f"""
                <div style="background: #131b2e; border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 1.25rem; margin-top: 10px;">
                    <div style="font-size: 0.8rem; text-transform: uppercase; color: #94a3b8; font-weight:700;">Prediction Breakdown</div>
                    <div style="display:flex; justify-content:space-between; margin-top: 14px; padding-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.06);">
                        <span style="color: #6ee7b7; font-weight: 600;">✓ Correct Classifications</span>
                        <strong style="color: #fff;">{correct_cases} ({correct_cases/total_cases:.1%})</strong>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-top: 8px; padding-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.06);">
                        <span style="color: #f87171; font-weight: 600;">✗ Misclassifications</span>
                        <strong style="color: #fff;">{incorrect_cases} ({incorrect_cases/total_cases:.1%})</strong>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-top: 8px;">
                        <span style="color: #94a3b8;">Total Unseen Test Cases</span>
                        <strong style="color: #fff;">{total_cases}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Interactive sample predictions table
            st.markdown("##### Prediction Samples Inspector")
            inspect_df = pd.DataFrame({
                "Sample ID": range(1, len(actual) + 1),
                "Actual Ground Truth": actual,
                "Model Prediction": predicted,
                "Match Status": ["✓ Correct" if a == p else "✗ Misclassified" for a, p in zip(actual, predicted)],
            })

            view_filter = st.radio("Filter Samples", ["Show All", "Misclassified Only", "Correct Only"], horizontal=True)
            if view_filter == "Misclassified Only":
                inspect_df = inspect_df[inspect_df["Match Status"] == "✗ Misclassified"]
            elif view_filter == "Correct Only":
                inspect_df = inspect_df[inspect_df["Match Status"] == "✓ Correct"]

            st.dataframe(inspect_df.head(50), hide_index=True, use_container_width=True)

        else:
            # Regression diagnostics
            reg_c1, reg_c2 = st.columns(2)
            with reg_c1:
                st.markdown("##### Actual vs. Predicted Correlation")
                scatter_fig = px.scatter(
                    x=actual,
                    y=predicted,
                    labels={"x": "Actual Ground Truth", "y": "Model Prediction"},
                    opacity=0.8,
                )
                min_val = float(min(np.min(actual), np.min(predicted)))
                max_val = float(max(np.max(actual), np.max(predicted)))
                scatter_fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode="lines",
                        name="Ideal Fit (y = x)",
                        line=dict(color="#10b981", dash="dash", width=2),
                    )
                )
                scatter_fig.update_layout(**get_plotly_layout(height=400))
                st.plotly_chart(scatter_fig, use_container_width=True)

            with reg_c2:
                st.markdown("##### Residual Error Distribution")
                residuals = actual - predicted
                hist_fig = px.histogram(
                    residuals,
                    nbins=25,
                    labels={"value": "Residual (Actual - Predicted)", "count": "Frequency"},
                    color_discrete_sequence=["#6366f1"],
                )
                hist_fig.update_layout(**get_plotly_layout(height=400))
                st.plotly_chart(hist_fig, use_container_width=True)

    # -----------------------------------------------------------------------------
    # TAB 4: DATASET EXPLORER
    # -----------------------------------------------------------------------------
    with tab_data:
        st.markdown('<div class="section-heading"><span class="badge-num">1</span> Dataset Schema & Summary Statistics</div>', unsafe_allow_html=True)

        sub_t1, sub_t2 = st.tabs(["Data Preview", "Statistical Summary"])
        with sub_t1:
            st.dataframe(df.head(100), use_container_width=True)
        with sub_t2:
            st.dataframe(df.describe(include="all").T, use_container_width=True)

        # Missing values breakdown
        st.markdown('<div class="section-heading"><span class="badge-num">2</span> Missing Values & Integrity Check</div>', unsafe_allow_html=True)
        null_counts = df.isna().sum()
        null_df = pd.DataFrame({"Feature": null_counts.index, "Missing Count": null_counts.values})
        null_df["Missing %"] = (null_df["Missing Count"] / len(df)) * 100

        if null_df["Missing Count"].sum() == 0:
            st.success("✓ Clean dataset: Zero missing cells detected across all columns.")
        else:
            st.dataframe(
                null_df[null_df["Missing Count"] > 0].style.format({"Missing %": "{:.2f}%"}),
                use_container_width=True,
                hide_index=True,
            )

    # -----------------------------------------------------------------------------
    # TAB 5: EXECUTIVE REPORT & EXPORT
    # -----------------------------------------------------------------------------
    with tab_export:
        st.markdown('<div class="section-heading"><span class="badge-num">1</span> Export Benchmarking Results</div>', unsafe_allow_html=True)
        st.caption("Generate publication-ready comparison reports and machine-readable metric datasets.")

        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            csv_data = scores.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Metrics (CSV)",
                data=csv_data,
                file_name=f"benchmark_metrics_{task}_{int(seed)}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with exp_col2:
            table_rows_html = "".join(
                "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
                for row in scores.round(4).itertuples(index=False, name=None)
            )
            table_headers_html = "".join(f"<th>{html.escape(str(c))}</th>" for c in scores.columns)

            html_report = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>ML Benchmark Executive Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 40px; color: #1e293b; line-height: 1.6; }}
    .header {{ border-bottom: 2px solid #6366f1; padding-bottom: 16px; margin-bottom: 24px; }}
    h1 {{ margin: 0 0 6px 0; color: #0f172a; font-size: 24px; }}
    .meta {{ color: #64748b; font-size: 13px; }}
    .champion-card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-left: 6px solid #f59e0b; border-radius: 8px; padding: 16px; margin: 20px 0; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 14px; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 10px 14px; text-align: left; }}
    th {{ background: #f1f5f9; font-weight: 600; color: #334155; }}
    tr:nth-child(even) {{ background: #f8fafc; }}
    .footer {{ margin-top: 32px; font-size: 12px; color: #94a3b8; border-top: 1px solid #e2e8f0; padding-top: 12px; }}
    @media print {{ button {{ display: none; }} }}
</style>
</head>
<body>
<div class="header">
    <h1>Machine Learning Model Comparison & Benchmark Report</h1>
    <div class="meta">
        <strong>Generated:</strong> {datetime.now():%Y-%m-%d %H:%M:%S} &nbsp;|&nbsp;
        <strong>Task:</strong> {html.escape(task.title())} &nbsp;|&nbsp;
        <strong>Target:</strong> {html.escape(str(target))} &nbsp;|&nbsp;
        <strong>Records:</strong> {n_rows:,} &nbsp;|&nbsp;
        <strong>Holdout Split:</strong> {test_pct}% &nbsp;|&nbsp;
        <strong>Seed:</strong> {seed}
    </div>
</div>

<div class="champion-card">
    <div style="font-size: 12px; text-transform: uppercase; font-weight: bold; color: #b45309;">Rank #1 Top Performer</div>
    <div style="font-size: 20px; font-weight: bold; color: #0f172a; margin-top: 4px;">{html.escape(str(best_model['Model']))}</div>
    <div style="margin-top: 6px; font-size: 13px; color: #475569;">
        Achieved premier ranking based on <strong>{stat_primary_label} ({stat_primary_val})</strong> over {test_n} holdout samples.
    </div>
</div>

<h3>Benchmark Leaderboard</h3>
<table>
    <thead><tr>{table_headers_html}</tr></thead>
    <tbody>{table_rows_html}</tbody>
</table>

<h3 style="margin-top: 28px;">Methodology & Safeguards</h3>
<p style="font-size: 13px; color: #475569;">
    All algorithms evaluated synchronously under an identical stratified/random split partition.
    Preprocessing (imputation, numerical scaling, and one-hot encoding) was executed strictly inside scikit-learn
    Pipelines fit solely on training data partitions to prevent data leakage.
</p>

<div class="footer">
    Report generated by ModelBenchmark AI Engine · Leakage-Safe Machine Learning Studio
</div>
</body>
</html>"""

            st.download_button(
                "⬇️ Download Executive Report (HTML / Printable PDF)",
                data=html_report.encode("utf-8"),
                file_name=f"benchmark_report_{task}_{int(seed)}.html",
                mime="text/html",
                use_container_width=True,
            )

    if errors:
        with st.expander("⚠️ Pipeline Training Notices / Skipped Models"):
            for name, msg in errors.items():
                st.warning(f"**{name}**: {msg}")

st.markdown("""
<div class="footer-note">
    ModelBenchmark AI · Leakage-Safe Machine Learning Studio · Engineered for Rigorous Algorithmic Evaluation
</div>
""", unsafe_allow_html=True)
