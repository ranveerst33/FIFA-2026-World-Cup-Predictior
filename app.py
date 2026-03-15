"""
FIFA 2026 World Cup Prediction - Streamlit Web Application
============================================================
Interactive web app with player search, team predictions,
custom Playing XI builder, and injury impact analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ====================================================================
# PAGE CONFIG
# ====================================================================
st.set_page_config(
    page_title="FIFA 2026 World Cup Predictor ⚽",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====================================================================
# CUSTOM CSS - Dark FIFA theme with gold accents
# ====================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Outfit:wght@300;400;600;700;800&display=swap');

    * { font-family: 'Outfit', 'Rajdhani', sans-serif; box-sizing: border-box; }

    /* ================================================================
       GLOBAL APP BACKGROUND
    ================================================================ */
    .stApp {
        background: linear-gradient(160deg, #060c18 0%, #0d1523 55%, #07111d 100%);
        min-height: 100vh;
    }

    /* ================================================================
       MAIN CONTENT TEXT - White by default on dark bg
    ================================================================ */
    .main .block-container p,
    .main .block-container span,
    .main .block-container li,
    .main .block-container label,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] li {
        color: #e8f0fe !important;
        font-weight: 500;
    }

    /* Widget labels */
    .stSelectbox > label,
    .stMultiSelect > label,
    .stTextInput > label,
    .stSlider > label,
    .stCheckbox > label,
    .stRadio > label {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* ================================================================
       HEADERS - FIFA Gold Style
    ================================================================ */
    h1 {
        color: #f0c040 !important;
        font-weight: 800 !important;
        font-family: 'Rajdhani', sans-serif !important;
        letter-spacing: 1px;
        text-transform: uppercase;
        text-shadow: 0 0 30px rgba(240,192,64,0.3), 0 2px 4px rgba(0,0,0,0.5);
    }
    h2 {
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px;
    }
    h3 {
        color: #e0e8ff !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px;
    }

    /* ================================================================
       SIDEBAR - Deep Navy FIFA Style
    ================================================================ */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #04080f 0%, #0a1628 60%, #091220 100%) !important;
        border-right: 3px solid #f0c040;
        box-shadow: 4px 0 20px rgba(240,192,64,0.1);
    }

    /* Nuclear option: every text type in sidebar → white */
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] p { font-weight: 600 !important; font-size: 1.05rem !important; }

    section[data-testid="stSidebar"] div[role="radiogroup"] > div {
        border-radius: 10px;
        padding: 6px 10px;
        margin-bottom: 6px;
        transition: background 0.2s ease;
        border: 1px solid transparent;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] > div:hover {
        background: rgba(240, 192, 64, 0.15) !important;
        border-color: rgba(240,192,64,0.3);
    }

    /* ================================================================
       SELECTBOX - Dark popup dropdown with WHITE text
    ================================================================ */
    /* The select box container */
    [data-baseweb="select"] > div {
        background-color: #0d1a2d !important;
        border: 1px solid rgba(240,192,64,0.4) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }

    /* Selected value text inside selectbox */
    [data-baseweb="select"] span,
    [data-baseweb="select"] div[class*="ValueContainer"] span,
    [data-baseweb="select"] div[class*="singleValue"],
    [data-baseweb="select"] input {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* The DROPDOWN POPUP (most important fix) */
    [data-baseweb="popover"],
    [data-baseweb="popover"] div,
    [data-baseweb="menu"],
    [data-baseweb="menu"] div,
    ul[data-baseweb="menu"],
    ul[data-baseweb="menu"] li {
        background-color: #0d1a2d !important;
        color: #ffffff !important;
    }

    /* Every option in the dropdown list */
    [role="option"],
    [role="option"] div,
    [role="option"] span,
    li[role="option"],
    li[role="option"] span {
        background-color: #0d1a2d !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        transition: background 0.15s ease !important;
    }

    [role="option"]:hover,
    li[role="option"]:hover {
        background-color: rgba(240, 192, 64, 0.2) !important;
        color: #f0c040 !important;
    }

    /* Focused / selected option */
    [aria-selected="true"][role="option"],
    [data-highlighted="true"][role="option"] {
        background-color: rgba(240, 192, 64, 0.25) !important;
        color: #f0c040 !important;
    }

    /* ================================================================
       MULTISELECT
    ================================================================ */
    [data-baseweb="tag"] {
        background: linear-gradient(135deg, #1a3a5c, #0d2340) !important;
        border: 1px solid rgba(240,192,64,0.5) !important;
        border-radius: 8px !important;
    }
    [data-baseweb="tag"] span { color: #f0c040 !important; font-weight: 700 !important; }

    /* ================================================================
       CARDS - FIFA match card style
    ================================================================ */
    .stat-card {
        background: linear-gradient(135deg, rgba(13,26,45,0.95), rgba(7,17,29,0.98));
        border: 1px solid rgba(240,192,64,0.25);
        border-left: 4px solid #f0c040;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 8px 0;
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    }
    .stat-card:hover {
        transform: translateX(4px);
        border-left-color: #ffd700;
        box-shadow: 0 4px 24px rgba(240,192,64,0.18);
    }
    .stat-value {
        font-size: 2.4em;
        font-weight: 800;
        background: linear-gradient(135deg, #f0c040, #ffd700, #e8a800);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Rajdhani', sans-serif;
        margin: 0;
        line-height: 1.1;
    }
    .stat-label {
        color: #a0b4cc !important;
        font-size: 0.78em;
        font-weight: 700;
        margin-top: 6px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* ================================================================
       PLAYER CARD - FIFA Ultimate Team style
    ================================================================ */
    .player-card {
        background: linear-gradient(160deg, #1a2e4a 0%, #0d1a2d 50%, #071220 100%);
        border: 2px solid rgba(240,192,64,0.4);
        border-radius: 20px;
        padding: 32px 24px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05);
        position: relative;
        overflow: hidden;
    }
    .player-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 50% 0%, rgba(240,192,64,0.08) 0%, transparent 60%);
        pointer-events: none;
    }
    .player-name {
        font-size: 2em;
        font-weight: 800;
        color: #ffffff !important;
        margin-bottom: 8px;
        text-shadow: 0 0 20px rgba(240,192,64,0.3);
        font-family: 'Rajdhani', sans-serif;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .player-detail {
        color: #c8d8e8 !important;
        font-size: 1em;
        margin: 6px 0;
        font-weight: 500;
    }

    /* ================================================================
       GROUP HEADER - Gold Banner
    ================================================================ */
    .group-header {
        background: linear-gradient(90deg, #c8960c, #f0c040, #c8960c);
        color: #060c18;
        padding: 14px 24px;
        border-radius: 10px 10px 0 0;
        font-weight: 800;
        font-size: 1.3em;
        text-align: center;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-family: 'Rajdhani', sans-serif;
    }

    /* ================================================================
       METRICS - Score board style
    ================================================================ */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #0d1a2d, #091220);
        border: 1px solid rgba(240,192,64,0.3);
        border-top: 3px solid #f0c040;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    }
    [data-testid="stMetricValue"] {
        color: #f0c040 !important;
        font-weight: 800 !important;
        font-size: 2rem !important;
        font-family: 'Rajdhani', sans-serif !important;
    }
    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    [data-testid="stMetricDelta"] { font-weight: 700 !important; }

    /* ================================================================
       BUTTONS - FIFA Action button
    ================================================================ */
    .stButton > button {
        background: linear-gradient(135deg, #c8960c, #f0c040, #c8960c) !important;
        background-size: 200% auto !important;
        color: #060c18 !important;
        font-weight: 800 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 14px 28px !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(240,192,64,0.2) !important;
    }
    .stButton > button:hover {
        background-position: right center !important;
        box-shadow: 0 6px 25px rgba(240,192,64,0.5) !important;
        transform: translateY(-2px) !important;
        letter-spacing: 2px !important;
    }

    /* ================================================================
       TABS - FIFA navigation style
    ================================================================ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(13,26,45,0.6);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(240,192,64,0.15);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #a0b4cc !important;
        font-weight: 700;
        letter-spacing: 0.5px;
        padding: 10px 20px;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #ffffff !important; }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1a3a5c, #0d2340) !important;
        color: #f0c040 !important;
        border-bottom: 2px solid #f0c040 !important;
    }

    /* ================================================================
       BADGES
    ================================================================ */
    .injury-badge {
        background: linear-gradient(135deg, #c0392b, #922b21);
        color: #ffffff !important;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.82em;
        font-weight: 700;
        display: inline-block;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        box-shadow: 0 2px 8px rgba(192,57,43,0.4);
    }
    .healthy-badge {
        background: linear-gradient(135deg, #1e8449, #145a32);
        color: #ffffff !important;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.82em;
        font-weight: 700;
        display: inline-block;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        box-shadow: 0 2px 8px rgba(30,132,73,0.4);
    }

    /* ================================================================
       DATAFRAMES
    ================================================================ */
    .stDataFrame { border-radius: 12px; overflow: hidden; border: 1px solid rgba(240,192,64,0.2); }
    .stDataFrame thead th { background-color: #0d2340 !important; color: #f0c040 !important; font-weight: 700; }
    .stDataFrame tbody tr:hover { background-color: rgba(240,192,64,0.05) !important; }

    /* ================================================================
       EXPANDERS
    ================================================================ */
    [data-testid="stExpander"] {
        border: 1px solid rgba(240,192,64,0.2) !important;
        border-radius: 12px !important;
        background: rgba(13, 26, 45, 0.6) !important;
    }
    [data-testid="stExpander"] summary { color: #e0e8ff !important; font-weight: 700 !important; }

    /* ================================================================
       DIVIDERS & MISC
    ================================================================ */
    hr { border-color: rgba(240,192,64,0.25) !important; margin: 20px 0 !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Success/Warning/Info boxes */
    [data-testid="stAlert"] { border-radius: 10px !important; border-left-width: 4px !important; }

    /* Text inputs */
    .stTextInput input {
        background: #0d1a2d !important;
        border: 1px solid rgba(240,192,64,0.3) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0a0f1a; }
    ::-webkit-scrollbar-thumb { background: rgba(240,192,64,0.4); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #f0c040; }

</style>
""", unsafe_allow_html=True)


# ====================================================================
# LOAD ENGINE (cached)
# ====================================================================
@st.cache_resource
def load_engine():
    from prediction_engine import PredictionEngine
    engine = PredictionEngine()
    engine.load()
    return engine


@st.cache_data
def load_eval_results():
    path = os.path.join(MODELS_DIR, "evaluation_results.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_feature_importances():
    path = os.path.join(MODELS_DIR, "feature_importances.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


# ====================================================================
# HELPER FUNCTIONS
# ====================================================================
def format_value(val):
    """Format market value to human readable string."""
    if val >= 1_000_000:
        return f"€{val/1_000_000:.1f}M"
    elif val >= 1_000:
        return f"€{val/1_000:.0f}K"
    else:
        return f"€{val:.0f}"


def get_flag(team_name):
    """Return the national flag emoji for a given team."""
    flags = {
        "Argentina": "🇦🇷", "Brazil": "🇧🇷", "France": "🇫🇷", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", 
        "Belgium": "🇧🇪", "Portugal": "🇵🇹", "Netherlands": "🇳🇱", "Spain": "🇪🇸", 
        "Italy": "🇮🇹", "Croatia": "🇭🇷", "United States": "🇺🇸", "USA": "🇺🇸", 
        "Mexico": "🇲🇽", "Uruguay": "🇺🇾", "Switzerland": "🇨🇭", "Colombia": "🇨🇴", 
        "Germany": "🇩🇪", "Senegal": "🇸🇳", "Japan": "🇯🇵", "Morocco": "🇲🇦", 
        "Denmark": "🇩🇰", "Iran": "🇮🇷", "South Korea": "🇰🇷", "Korea, South": "🇰🇷",
        "Poland": "🇵🇱", "Serbia": "🇷🇸", "Nigeria": "🇳🇬", "Algeria": "🇩🇿", 
        "Egypt": "🇪🇬", "Ghana": "🇬🇭", "Cameroon": "🇨🇲", "Australia": "🇦🇺", 
        "Canada": "🇨🇦", "Ecuador": "🇪🇨", "Peru": "🇵🇪", "Chile": "🇨🇱", 
        "Ivory Coast": "🇨🇮", "Cote d'Ivoire": "🇨🇮", "Saudi Arabia": "🇸🇦", 
        "Tunisia": "🇹🇳", "Wales": "🏴󠁧󠁢󠁷󠁬󠁳󠁿", "Costa Rica": "🇨🇷", 
        "Sweden": "🇸🇪", "Ukraine": "🇺🇦", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿"
    }
    return flags.get(team_name, "🏁")


def clean_name(name):
    """Clean common encoding issues in player names."""
    if not isinstance(name, str): return name
    mapping = {
        'Ã«': 'ë', 'Ã£': 'ã', 'Ã¡': 'á', 'Ã³': 'ó', 
        'Ã©': 'é', 'Ãº': 'ú', 'Ã­': 'í', 'Ã±': 'ñ',
        'Ã': 'í', 'Ã§': 'ç', 'Ã': 'ô'
    }
    for k, v in mapping.items():
        name = name.replace(k, v)
    return name


def create_probability_chart(proba_dict, title="Match Prediction"):
    """Create a horizontal bar chart for match probabilities."""
    labels = list(proba_dict.keys())
    values = [v * 100 for v in proba_dict.values()]
    colors = ["#16a34a", "#eab308", "#dc2626"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels,
        x=values,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color="rgba(255,255,255,0.1)", width=1),
        ),
        text=[f"{v:.1f}%" for v in values],
        textposition="auto",
        textfont=dict(color="white", size=14, family="Outfit"),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#d4af37", size=18, family="Outfit")),
        xaxis=dict(title="Probability (%)", range=[0, 100],
                   color="#94a3b8", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(color="#e2e8f0"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=250,
        margin=dict(l=10, r=10, t=50, b=30),
    )
    return fig


def create_group_standings_chart(standings):
    """Create a bar chart of group standings points."""
    teams = [s["team"] for s in standings]
    points = [s["points"] for s in standings]
    colors = ["#d4af37" if s["qualifies"] else "#475569" for s in standings]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=teams, y=points,
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.1)", width=1)),
        text=[f"{p:.1f}" for p in points],
        textposition="auto",
        textfont=dict(color="white", size=14, family="Outfit"),
    ))
    fig.update_layout(
        yaxis=dict(title="Expected Points", color="#94a3b8",
                   gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(color="#e2e8f0"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(l=10, r=10, t=20, b=30),
    )
    return fig


# ====================================================================
# SIDEBAR NAVIGATION
# ====================================================================
with st.sidebar:
    st.markdown("""
        <div style="text-align:center; padding: 20px 0;">
            <div style="font-size: 3em;">⚽</div>
            <div style="font-size: 1.5em; font-weight: 800; color: #d4af37; margin-top: 8px;">
                FIFA 2026
            </div>
            <div style="color: #94a3b8; font-size: 0.9em; margin-top:4px;">
                World Cup Predictor
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Home", "🌟 Finals Prediction", "⚽ Player Explorer", "🏆 Team Predictions",
         "👥 Custom XI", "🏥 Injury Impact", "📊 Model Performance"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("""
        <div style="color: #64748b; font-size: 0.75em; text-align: center; padding: 10px;">
            Powered by Machine Learning<br>
            Random Forest • XGBoost • GBM<br>
            Logistic Regression • Ensemble
        </div>
    """, unsafe_allow_html=True)


# Load engine
try:
    engine = load_engine()
    ENGINE_LOADED = True
except Exception as e:
    ENGINE_LOADED = False
    engine_error = str(e)


# ====================================================================
# PAGE: HOME
# ====================================================================
if page == "🏠 Home":
    st.markdown("""
        <div style="text-align: center; padding: 40px 0 20px 0;">
            <h1 style="font-size: 3em; margin-bottom: 0;">⚽ FIFA 2026 World Cup Predictor</h1>
            <p style="color: #94a3b8; font-size: 1.2em; margin-top: 8px;">
                AI-Powered Match Predictions • USA 🇺🇸 • Mexico 🇲🇽 • Canada 🇨🇦
            </p>
        </div>
    """, unsafe_allow_html=True)

    if not ENGINE_LOADED:
        st.error(f"⚠️ Engine failed to load: {engine_error}")
        st.info("Please run `python model_trainer.py` first to train the models.")
        st.stop()

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-value">48</div>
                <div class="stat-label">Teams</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-value">104</div>
                <div class="stat-label">Matches</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-value">12</div>
                <div class="stat-label">Groups</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-value">964</div>
                <div class="stat-label">Historical Matches Analyzed</div>
            </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Top teams by ELO
    st.markdown("### 🏅 Top Teams by ELO Rating")
    sorted_elo = sorted(engine.dp.elo_ratings.items(), key=lambda x: x[1], reverse=True)[:15]

    fig = go.Figure()
    teams = [t[0] for t in sorted_elo]
    elos = [t[1] for t in sorted_elo]
    fig.add_trace(go.Bar(
        x=teams, y=elos,
        marker=dict(
            color=elos,
            colorscale=[[0, "#1e3a5f"], [0.5, "#d4af37"], [1, "#f4d03f"]],
            line=dict(color="rgba(255,255,255,0.1)", width=1),
        ),
        text=[f"{e:.0f}" for e in elos],
        textposition="auto",
        textfont=dict(color="white", size=11, family="Outfit"),
    ))
    fig.update_layout(
        yaxis=dict(title="ELO Rating", color="#94a3b8", gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(color="#e2e8f0", tickangle=-45),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=400,
        margin=dict(l=10, r=10, t=20, b=80),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Quick match predictor
    st.divider()
    st.markdown("### ⚡ Quick Match Predictor")

    teams_2026 = engine.dp.get_2026_teams()
    all_teams = sorted(engine.dp.elo_ratings.keys())

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("🏠 Home Team", all_teams, index=all_teams.index("Brazil") if "Brazil" in all_teams else 0)
    with col2:
        away_team = st.selectbox("✈️ Away Team", all_teams, index=all_teams.index("Germany") if "Germany" in all_teams else 1)

    if st.button("🔮 Predict Match", use_container_width=True):
        result = engine.predict_match(home_team, away_team)
        st.plotly_chart(create_probability_chart(result["probabilities"], f"{home_team} vs {away_team}"),
                       use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prediction", result["prediction"])
        with col2:
            st.metric(f"{home_team} ELO", f"{result['elo_home']:.0f}")
        with col3:
            st.metric(f"{away_team} ELO", f"{result['elo_away']:.0f}")


# ====================================================================
# PAGE: FINALS PREDICTION
# ====================================================================
elif page == "🌟 Finals Prediction":
    st.markdown("# 🌟 Finals Prediction")
    st.markdown("*Simulate the entire World Cup and trace the path to glory.*")

    if not ENGINE_LOADED:
        st.error("Engine not loaded. Please train models first.")
        st.stop()
        
    if st.button("🏆 Predict Entire World Cup", use_container_width=True):
        with st.spinner("Simulating Group Stages and Knockout Bracket using ML Engine..."):
            tournament_result = engine.simulate_tournament()
            st.session_state["tournament_result"] = tournament_result
            st.session_state["selected_match"] = None
            
    if "tournament_result" in st.session_state:
        res = st.session_state["tournament_result"]
        bracket = res["knockout_bracket"]
        
        st.divider()
        st.markdown(f"<h2 style='text-align: center; color: #f0c040;'>👑 Tournament Winner: {bracket['Winner']} 👑</h2>", unsafe_allow_html=True)
        st.divider()
        
        # Build bracket layout using tabs for different rounds to save space
        rounds = ["Round of 32", "Round of 16", "Quarterfinals", "Semifinals", "Final"]
        tabs = st.tabs(rounds)
        
        for i, r_name in enumerate(rounds):
            with tabs[i]:
                # 2 columns for layout
                col1, col2 = st.columns(2)
                for j, match in enumerate(bracket[r_name]):
                    target_col = col1 if j % 2 == 0 else col2
                    with target_col:
                        home = match['home']
                        away = match['away']
                        score = match['score']
                        winner = match['winner']
                        
                        btn_label = f"⚽ {home} {score} {away}"
                        if st.button(btn_label, key=f"{r_name}_{home}_{away}", use_container_width=True):
                            st.session_state["selected_match"] = match
                        st.caption(f"↳ **{winner}** advances")
                        st.markdown("<hr style='margin: 8px 0; border: 0.5px solid rgba(240,192,64,0.15);'>", unsafe_allow_html=True)

        # Match Details View
        if st.session_state.get("selected_match"):
            match = st.session_state["selected_match"]
            stats = match["stats"]
            
            st.divider()
            st.markdown(f"<h3 style='text-align: center;'>📋 Match Setup: {match['home']} vs {match['away']}</h3>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align: center; color: #f0c040;'>{match['score']}</h2>", unsafe_allow_html=True)
            
            home_s = "<br>".join(stats["scorers"]["home"]) if stats["scorers"]["home"] else "<i>None</i>"
            away_s = "<br>".join(stats["scorers"]["away"]) if stats["scorers"]["away"] else "<i>None</i>"
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="stat-card" style="text-align: center;">
                    <h4 style="color: #ffffff; margin-bottom: 15px;">{match['home']}</h4>
                    <p style="margin: 5px 0;"><b>Possession:</b> {stats['possession']['home']}%</p>
                    <p style="margin: 5px 0;"><b>Shots:</b> {stats['shots']['home']}</p>
                    <p style="margin: 5px 0; color: #fbbf24;"><b>Yellow Cards:</b> {stats['yellow_cards']['home']}</p>
                    <p style="margin: 5px 0; color: #ef4444;"><b>Red Cards:</b> {stats['red_cards']['home']}</p>
                    <hr style="margin: 10px 0; border: 0.5px solid rgba(255,255,255,0.1);">
                    <p style="margin: 5px 0;"><b>⚽ Goalscorers</b><br>{home_s}</p>
                </div>
                """, unsafe_allow_html=True)
                
            with c2:
                st.markdown(f"""
                <div class="stat-card" style="text-align: center;">
                    <h4 style="color: #ffffff; margin-bottom: 15px;">{match['away']}</h4>
                    <p style="margin: 5px 0;"><b>Possession:</b> {stats['possession']['away']}%</p>
                    <p style="margin: 5px 0;"><b>Shots:</b> {stats['shots']['away']}</p>
                    <p style="margin: 5px 0; color: #fbbf24;"><b>Yellow Cards:</b> {stats['yellow_cards']['away']}</p>
                    <p style="margin: 5px 0; color: #ef4444;"><b>Red Cards:</b> {stats['red_cards']['away']}</p>
                    <hr style="margin: 10px 0; border: 0.5px solid rgba(255,255,255,0.1);">
                    <p style="margin: 5px 0;"><b>⚽ Goalscorers</b><br>{away_s}</p>
                </div>
                """, unsafe_allow_html=True)

# ====================================================================
# PAGE: PLAYER EXPLORER
# ====================================================================
elif page == "⚽ Player Explorer":
    st.markdown("# ⚽ Player Explorer")
    st.markdown("*Use the searchable dropdown below to find any player (Start typing a name!)*")

    if not ENGINE_LOADED:
        st.error("Engine not loaded. Please train models first.")
        st.stop()

    # Load top 5000 players to act as auto-complete list
    top_players_list = engine.get_top_players(limit=5000)
    
    # Create searchable format for the dropdown
    options = {
        f"{r['name']} ({r['nationality']} - {r['position']} - {r['club']})": r 
        for r in top_players_list
    }

    # Streamlit selectbox natively supports type-to-search (autocomplete)
    selected_name = st.selectbox(
        "🔍 Search Player (Type name to filter)", 
        list(options.keys()), 
        index=0,
        help="Start typing a player's name. Top 5000 players worldwide are loaded here."
    )

    if selected_name:
        selected_player = options[selected_name]

        # Get full stats
        stats = engine.get_player_stats(selected_player["name"].split(" (")[0])

        if "error" in stats:
            st.error(stats["error"])
        else:
            st.divider()

            # Player card
            col1, col2 = st.columns([1, 2])
            with col1:
                flag = get_flag(stats['nationality'])
                st.markdown(f"""
                    <div class="player-card">
                        <div style="font-size: 4em; margin-bottom: 10px;">⚽</div>
                        <div class="player-name">{clean_name(stats['name'])}</div>
                        <div class="player-detail">{flag} {stats['nationality']}</div>
                        <div class="player-detail">📍 {stats['position']}</div>
                        <div class="player-detail">🏟️ {stats['current_club']}</div>
                            <div class="player-detail">🦶 {stats['foot']}</div>
                            <div class="player-detail">📏 {stats['height']}cm</div>
                            <div class="player-detail">🎂 Age at WC: {stats['age_at_wc2026']}</div>
                            <div class="player-detail" style="margin-top: 10px;">
                                💰 {format_value(stats['market_value'])}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                with col2:
                    # National team stats
                    st.markdown("### 🏆 National Team Career")
                    for ns in stats["national_team_stats"]:
                        status = "🟢 Active" if ns["career_state"] == "CURRENT_NATIONAL_PLAYER" else "🔴 Former"
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Caps", ns["matches"])
                        with col_b:
                            st.metric("Goals", ns["goals"])
                        with col_c:
                            st.metric("Status", status)

                    # Injury history
                    st.markdown("### 🏥 Injury History")
                    if stats["recent_injuries"]:
                        injury_df = pd.DataFrame(stats["recent_injuries"])
                        st.dataframe(injury_df, use_container_width=True, hide_index=True)
                        st.caption(f"Total injuries: {stats['total_injuries']} | Total days missed: {stats['total_days_missed']}")
                    else:
                        st.markdown('<span class="healthy-badge">✓ No recorded injuries</span>', unsafe_allow_html=True)

                # Team predictions
                st.divider()
                st.markdown(f"### 🌍 {stats['nationality']} – FIFA 2026 Predictions")

                nationality = stats["nationality"]
                groups = engine.dp.get_2026_groups()

                team_group = None
                team_name_in_group = None
                for grp, teams in groups.items():
                    for t in teams:
                        if nationality.lower() in t.lower() or t.lower() in nationality.lower():
                            team_group = grp
                            team_name_in_group = t
                            break

                if team_group:
                    group_result = engine.predict_group(team_group)
                    if "standings" in group_result:
                        st.markdown(f"**{team_group}** Standings:")
                        st.plotly_chart(create_group_standings_chart(group_result["standings"]),
                                       use_container_width=True)

                        standings_df = pd.DataFrame(group_result["standings"])
                        standings_df = standings_df[["position", "team", "points", "w", "d", "l", "gf", "ga", "gd", "qualifies"]]
                        st.dataframe(standings_df, use_container_width=True, hide_index=True)

                        if group_result["matches"]:
                            st.markdown("**Match-by-Match Predictions:**")
                            for match in group_result["matches"]:
                                with st.expander(f"🏟️ {match['home_team']} vs {match['away_team']} → {match['prediction']}"):
                                    st.plotly_chart(
                                        create_probability_chart(match["probabilities"]),
                                        use_container_width=True,
                                    )
                    else:
                        st.info(f"{nationality} group predictions: {group_result}")
                else:
                    st.info(f"{nationality} is not in a confirmed 2026 group or may be in a playoff spot.")


# ====================================================================
# PAGE: TEAM PREDICTIONS
# ====================================================================
elif page == "🏆 Team Predictions":
    st.markdown("# 🏆 Team Predictions")
    st.markdown("*Select a team to see their group stage simulation and knockout path*")

    if not ENGINE_LOADED:
        st.error("Engine not loaded. Please train models first.")
        st.stop()

    # Group selection
    groups = engine.dp.get_2026_groups()
    selected_group = st.selectbox("Select Group", sorted(groups.keys()))

    if selected_group:
        group_result = engine.predict_group(selected_group)

        if "error" in group_result:
            st.error(group_result["error"])
        else:
            st.markdown(f"""
                <div class="group-header">
                    {selected_group} – FIFA 2026 World Cup
                </div>
            """, unsafe_allow_html=True)

            # Standings chart
            st.plotly_chart(create_group_standings_chart(group_result["standings"]),
                           use_container_width=True)

            # Standings table
            standings_df = pd.DataFrame(group_result["standings"])
            standings_df.columns = ["Pos", "Team", "Pts", "W", "D", "L", "GF", "GA", "GD", "Qualifies"]
            st.dataframe(standings_df, use_container_width=True, hide_index=True)

            # Qualified teams
            qualified = group_result.get("qualified_teams", [])
            if qualified:
                st.success(f"🎉 Predicted to qualify: **{', '.join(qualified)}**")

            # Match predictions
            st.divider()
            st.markdown("### 📋 Match Predictions")

            for match in group_result.get("matches", []):
                col1, col2, col3 = st.columns([2, 3, 2])
                with col1:
                    st.markdown(f"**{match['home_team']}**")
                    st.caption(f"ELO: {match['elo_home']:.0f}")
                with col2:
                    st.plotly_chart(
                        create_probability_chart(match["probabilities"],
                                               f"→ {match['prediction']} ({match['confidence']:.0%})"),
                        use_container_width=True,
                    )
                with col3:
                    st.markdown(f"**{match['away_team']}**")
                    st.caption(f"ELO: {match['elo_away']:.0f}")
                st.divider()

    # All groups overview
    st.divider()
    st.markdown("### 🌍 All Groups Overview")
    if st.button("🔄 Simulate All Groups"):
        all_results = engine.predict_all_groups()
        for grp_name, grp_result in sorted(all_results.items()):
            if "standings" in grp_result:
                with st.expander(f"📋 {grp_name}", expanded=False):
                    standings_df = pd.DataFrame(grp_result["standings"])
                    st.dataframe(standings_df[["position", "team", "points", "w", "d", "l", "qualifies"]],
                                use_container_width=True, hide_index=True)
                    qualified = grp_result.get("qualified_teams", [])
                    if qualified:
                        st.success(f"Qualify: {', '.join(qualified)}")


# ====================================================================
# PAGE: CUSTOM XI
# ====================================================================
elif page == "👥 Custom XI":
    st.markdown("# 👥 Build Your Custom Playing XI")
    st.markdown("*Select 11 players and predict the match outcome*")

    if not ENGINE_LOADED:
        st.error("Engine not loaded. Please train models first.")
        st.stop()

    col1, col2 = st.columns([2, 1])

    with col1:
        # Use dropdown for better UX
        teams_2026 = engine.dp.get_2026_teams()
        team_name = st.selectbox("🏴 Select Your Team", teams_2026, index=teams_2026.index("Portugal") if "Portugal" in teams_2026 else 0)

    with col2:
        opponent = st.selectbox("⚔️ Select Opponent", teams_2026, index=teams_2026.index("France") if "France" in teams_2026 else 1)

    if team_name:
        players = engine.get_team_players(team_name)

        if not players:
            st.warning(f"No players found for {team_name}. Try a different country name.")
        else:
            st.markdown(f"### 📋 {team_name} Squad ({len(players)} players)")

            # Create player selection
            player_options = {
                f"{p['name'].split(' (')[0]} - {p['position']} ({p['club']}) - {p['caps']} caps": p["player_id"]
                for p in players[:50]  # Limit to top 50 by value
            }

            selected = st.multiselect(
                "Select 11 players for your XI:",
                list(player_options.keys()),
                max_selections=11,
            )

            if len(selected) == 11:
                selected_ids = [player_options[s] for s in selected]

                if st.button("🔮 Analyze XI vs " + opponent, use_container_width=True):
                    result = engine.analyze_custom_xi(selected_ids, opponent)

                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.divider()
                        st.markdown(f"### 📊 {result['your_team']} XI vs {result['opponent']}")

                        # XI stats
                        xi = result["xi_stats"]
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Caps", xi["total_caps"])
                        with col2:
                            st.metric("Total Goals", xi["total_goals"])
                        with col3:
                            st.metric("Squad Value", format_value(xi["total_market_value"]))
                        with col4:
                            st.metric("Strength Ratio", f"{result['strength_ratio']:.2f}")

                        # Prediction
                        st.markdown("### 🔮 Match Prediction (Custom XI)")
                        st.plotly_chart(
                            create_probability_chart(result["prediction"],
                                                   f"Your XI vs {result['opponent']}"),
                            use_container_width=True,
                        )

                        # Base vs adjusted
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Base Prediction (full squad):**")
                            for k, v in result["base_prediction"].items():
                                st.write(f"  {k}: {v:.1%}")
                        with col2:
                            st.markdown("**Adjusted Prediction (your XI):**")
                            for k, v in result["prediction"].items():
                                st.write(f"  {k}: {v:.1%}")

                        # Position breakdown
                        st.markdown("### ⚽ Position Breakdown")
                        positions = xi.get("positions", {})
                        if positions:
                            fig = go.Figure(data=[go.Pie(
                                labels=list(positions.keys()),
                                values=list(positions.values()),
                                hole=0.4,
                                marker=dict(colors=["#d4af37", "#16a34a", "#3b82f6", "#ef4444"]),
                                textfont=dict(color="white", family="Outfit"),
                            )])
                            fig.update_layout(
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)",
                                height=300,
                                font=dict(color="#e2e8f0"),
                            )
                            st.plotly_chart(fig, use_container_width=True)
            elif len(selected) > 0:
                st.info(f"Selected {len(selected)}/11 players. Need {11 - len(selected)} more.")

            # Show full squad reference
            with st.expander("📋 Full Squad Reference"):
                squad_df = pd.DataFrame(players[:50])
                squad_df["market_value"] = squad_df["market_value"].apply(format_value)
                st.dataframe(squad_df[["name", "position", "club", "caps", "goals", "market_value", "age"]],
                            use_container_width=True, hide_index=True)


# ====================================================================
# PAGE: INJURY IMPACT
# ====================================================================
elif page == "🏥 Injury Impact":
    st.markdown("# 🏥 Injury Impact Analyzer")
    st.markdown("*See how losing key players affects your team's World Cup predictions*")

    if not ENGINE_LOADED:
        st.error("Engine not loaded. Please train models first.")
        st.stop()

    all_2026_teams = engine.dp.get_2026_teams()
    team_name = st.selectbox("🏴 Select National Team", all_2026_teams, index=all_2026_teams.index("Brazil") if "Brazil" in all_2026_teams else 0)

    if team_name:
        players = engine.get_team_players(team_name)

        if not players:
            st.warning(f"No players found for {team_name}")
        else:
            players = engine.get_team_players(team_name)
            
            # Create selection dictionary for all players in squad
            player_options = {
                f"{p['name']} - {p['position']} (Val: {format_value(p['market_value'])}, Caps: {p['caps']})"
                : p["player_id"]
                for p in players
            }

            injured_ids = st.multiselect(
                "🚩 Select Injured / Unavailable Players:",
                list(player_options.keys()),
                help="Prediction results will adjust based on the lost value and experience of these players."
            )
            
            selected_ids = [player_options[p] for p in injured_ids]

            if st.button("🏥 Analyze Injury Impact", use_container_width=True):
                result = engine.analyze_injury_impact(team_name, selected_ids)

                if "error" in result:
                    st.error(result["error"])
                else:
                    st.divider()

                    # Impact metrics
                    impact = result["squad_impact"]
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Squad Value Loss", format_value(impact["value_loss"]),
                                 delta=f"-{impact['value_loss_percentage']:.1f}%", delta_color="inverse")
                    with col2:
                        st.metric("Caps Lost", impact["caps_lost"])
                    with col3:
                        st.metric("Goals Lost", impact["goals_lost"])
                    with col4:
                        st.metric("Healthy Squad Size", result["healthy_squad_size"])

                    # Show injured players
                    st.markdown("### 🤕 Injured Players")
                    for p in result["injured_players"]:
                        st.markdown(f"""
                            <div class="stat-card" style="border-color: rgba(220, 38, 38, 0.4);">
                                <span class="injury-badge">INJURED</span>
                                <span style="color: #e2e8f0; font-weight: 600; margin-left: 12px;">
                                    {p['name'].split(' (')[0]}
                                </span>
                                <span style="color: #94a3b8; margin-left: 8px;">
                                    {p['position']} | {p['caps']} caps | {p['goals']} goals | {format_value(p['market_value'])}
                                </span>
                            </div>
                        """, unsafe_allow_html=True)

                    # Group predictions comparison
                    if result.get("group") and result.get("group_predictions"):
                        st.divider()
                        st.markdown(f"### 🛡️ Rival & Group Impact (Comparison)")
                        
                        # Get baseline (no injuries)
                        baseline = engine.predict_group(result["group"])
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown("**Standings (Full Strength)**")
                            st.plotly_chart(create_group_standings_chart(baseline["standings"]), use_container_width=True, key="base_chart")
                            st.dataframe(pd.DataFrame(baseline["standings"])[["team", "points"]], use_container_width=True, hide_index=True)
                        
                        with col_b:
                            st.markdown(f"**Standings (With {team_name} Injuries)**")
                            st.plotly_chart(create_group_standings_chart(result["group_predictions"]["standings"]), use_container_width=True, key="inj_chart")
                            st.dataframe(pd.DataFrame(result["group_predictions"]["standings"])[["team", "points"]], use_container_width=True, hide_index=True)

                        # Detailed rival analysis
                        st.markdown(f"#### ⚔️ Rival Performance Advantage")
                        b_pts = {s['team']: s['points'] for s in baseline['standings']}
                        i_pts = {s['team']: s['points'] for s in result['group_predictions']['standings']}
                        
                        for r_team in b_pts:
                            if r_team != team_name:
                                diff = i_pts[r_team] - b_pts[r_team]
                                if diff > 0.05:
                                    st.info(f"📈 **{r_team}** gains **+{diff:.2f}** expected points due to your team's absences!")
                                elif diff < -0.05:
                                    st.warning(f"📉 **{r_team}** also loses **{diff:.2f}** points (lower match quality impact).")
            else:
                # Show player list with injury histories
                st.markdown("### 📋 Squad Overview")
                for i, p in enumerate(players[:20]):
                    injury_count = 0
                    days_missed = 0
                    badge = f'<span class="healthy-badge">Low Risk</span>'
                    st.markdown(f"""
                        <div class="stat-card">
                            <span style="color: #ffffff; font-weight: 700; font-size:1.05em;">
                                {i+1}. {clean_name(p['name'])}
                            </span>
                            <span style="color: #e2e8f0; margin-left: 10px;">
                                {p.get('position','?')} | {p.get('caps',0)} caps | {format_value(p.get('market_value',0))}
                            </span>
                            <span style="float: right;">{badge}</span>
                        </div>
                    """, unsafe_allow_html=True)


# ====================================================================
# PAGE: MODEL PERFORMANCE
# ====================================================================
elif page == "📊 Model Performance":
    st.markdown("# 📊 Model Performance")
    st.markdown("*Evaluation metrics, confusion matrices, and feature importance*")

    eval_results = load_eval_results()
    feature_imp = load_feature_importances()

    if not eval_results:
        st.warning("No evaluation results found. Please run `python model_trainer.py` first.")
        st.stop()

    # Model comparison
    st.markdown("### 📈 Model Comparison")

    model_names = list(eval_results.keys())
    metrics_data = []
    for model in model_names:
        metrics_data.append({
            "Model": model,
            "Accuracy": eval_results[model]["accuracy"],
            "Precision": eval_results[model]["precision"],
            "Recall": eval_results[model]["recall"],
            "F1-Score": eval_results[model]["f1_score"],
            "CV Mean": eval_results[model].get("cv_mean", 0),
        })

    metrics_df = pd.DataFrame(metrics_data)

    # Bar chart comparison
    fig = go.Figure()
    colors = ["#d4af37", "#16a34a", "#3b82f6", "#ef4444", "#8b5cf6"]
    for i, metric in enumerate(["Accuracy", "Precision", "Recall", "F1-Score"]):
        fig.add_trace(go.Bar(
            name=metric,
            x=metrics_df["Model"],
            y=metrics_df[metric],
            marker_color=colors[i],
            text=metrics_df[metric].apply(lambda x: f"{x:.3f}"),
            textposition="auto",
            textfont=dict(color="white", size=10, family="Outfit"),
        ))

    fig.update_layout(
        barmode="group",
        yaxis=dict(title="Score", range=[0, 1.05], color="#94a3b8",
                   gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(color="#e2e8f0"),
        legend=dict(font=dict(color="#e2e8f0")),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=450,
        margin=dict(l=10, r=10, t=20, b=80),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Metrics table
    st.dataframe(metrics_df.set_index("Model").style.format("{:.4f}"),
                 use_container_width=True)

    # Confusion matrices
    st.divider()
    st.markdown("### 🎯 Confusion Matrices")

    labels = ["Home Win", "Draw", "Away Win"]
    cols = st.columns(min(len(model_names), 3))
    for i, model in enumerate(model_names):
        with cols[i % 3]:
            cm = np.array(eval_results[model]["confusion_matrix"])
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale=[[0, "#0f172a"], [1, "#d4af37"]],
                text=cm,
                texttemplate="%{text}",
                textfont=dict(color="white", size=14),
                showscale=False,
            ))
            fig.update_layout(
                title=dict(text=model, font=dict(color="#d4af37", size=14, family="Outfit")),
                xaxis=dict(title="Predicted", color="#94a3b8"),
                yaxis=dict(title="Actual", color="#94a3b8"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=300,
                margin=dict(l=10, r=10, t=40, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    if feature_imp:
        st.divider()
        st.markdown("### 🔑 Feature Importance")

        selected_model = st.selectbox("Select model:", list(feature_imp.keys()))
        if selected_model:
            imp = feature_imp[selected_model]
            imp_sorted = dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))

            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=list(imp_sorted.keys()),
                x=list(imp_sorted.values()),
                orientation="h",
                marker=dict(
                    color=list(imp_sorted.values()),
                    colorscale=[[0, "#1e3a5f"], [1, "#d4af37"]],
                    line=dict(color="rgba(255,255,255,0.1)", width=1),
                ),
                text=[f"{v:.4f}" for v in imp_sorted.values()],
                textposition="auto",
                textfont=dict(color="white", size=10, family="Outfit"),
            ))
            fig.update_layout(
                xaxis=dict(title="Importance", color="#94a3b8",
                           gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(color="#e2e8f0", autorange="reversed"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=600,
                margin=dict(l=10, r=10, t=20, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Cross-validation
    st.divider()
    st.markdown("### 📊 Cross-Validation Results")
    cv_data = []
    for model in model_names:
        cv_mean = eval_results[model].get("cv_mean")
        cv_std = eval_results[model].get("cv_std")
        if cv_mean is not None:
            cv_data.append({
                "Model": model,
                "CV Mean Accuracy": cv_mean,
                "CV Std": cv_std,
                "Range": f"{cv_mean - cv_std:.4f} - {cv_mean + cv_std:.4f}",
            })

    if cv_data:
        cv_df = pd.DataFrame(cv_data)
        st.dataframe(cv_df.set_index("Model"), use_container_width=True)
    else:
        st.info("Cross-validation results not available.")
