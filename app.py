"""
================================================================================
SMS SHIELD · AI-POWERED CYBERSECURITY SMS THREAT DETECTION PLATFORM
================================================================================

HOW TO RUN THIS APPLICATION:
--------------------------------------------------------------------------------
1. Open PowerShell or Command Prompt in the project root directory:
   cd "c:\\Users\\mkc\\Main Projects\\spam-classifier"

2. Activate the Python virtual environment:
   Windows (PowerShell):   .\\.venv\\Scripts\\Activate.ps1
   Windows (CMD):          .\\.venv\\Scripts\\activate.bat
   macOS / Linux:          source .venv/bin/activate

3. Launch the Streamlit application:
   streamlit run app.py

   Alternative direct command:
   python -m streamlit run app.py --server.port 8501

4. Open your web browser at:
   http://localhost:8501

ARCHITECTURE SUMMARY:
--------------------------------------------------------------------------------
- Pipeline: Raw Text -> clean_text() -> TF-IDF Vectorizer -> LinearSVC Classifier
- Preprocessing: Lowercasing, punctuation stripping, English stopwords, Porter stemming
- Features: 6,959 unique unigram tokens from training set (TfidfVectorizer)
- Classifier: Linear Support Vector Classifier (LinearSVC, C=1.0, squared-hinge loss)
- Test Set Performance: 98.26% Accuracy | 97.48% Precision | 88.55% Recall
================================================================================
"""

# ==============================================================================
# PART 1: LIBRARY IMPORTS
# ==============================================================================

# os: Used for cross-platform filesystem path navigation (e.g., locating saved model files).
import os

# string: Provides string.punctuation used in our character translation table to strip symbols.
import string

# joblib: Scientific Python serialization library used to load our trained scikit-learn models.
import joblib

# nltk: Natural Language Toolkit providing linguistic stopwords and stemming algorithms.
import nltk

# stopwords: Curated list of high-frequency English grammatical words ('the', 'is', 'at') to remove.
from nltk.corpus import stopwords

# PorterStemmer: Implementation of the Porter Stemming Algorithm to reduce words to linguistic roots.
from nltk.stem.porter import PorterStemmer

# streamlit: The reactive Python web framework powering the user interface.
import streamlit as st


# ==============================================================================
# PART 2: STREAMLIT PAGE CONFIGURATION
# ==============================================================================

# st.set_page_config MUST be the very first Streamlit command called in the script.
# - page_title: Browser tab label
# - page_icon: Browser tab favicon (Shield icon)
# - layout: 'centered' keeps content centered in a balanced SaaS column
# - initial_sidebar_state: 'collapsed' presents a clean dashboard without default sidebars
st.set_page_config(
    page_title="SMS Shield · Intelligent Threat Detection",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ==============================================================================
# PART 3: ADVANCED CYBERSECURITY DESIGN SYSTEM (DARK THEME CSS)
# ==============================================================================

# Custom CSS injected directly into Streamlit's frontend DOM.
# Visual Design System:
# - Background: Deep midnight navy / near-black (#0B0F17)
# - Surfaces: Layered dark charcoal and slate (#111827, #161F30, #1E293B)
# - Primary Accent: Electric violet (#8B5CF6 / #7C3AED)
# - Secondary Accent: Luminous cyan (#06B6D4 / #22D3EE)
# - SPAM Threat: Sophisticated coral-red (#EF4444 on #1F1315)
# - HAM Legitimate: Fresh emerald-green (#10B981 on #0B1E16)
# - Typography: Plus Jakarta Sans / Inter with JetBrains Mono for technical telemetry
st.markdown("""
<style>
/* ── Google Fonts: Inter & Plus Jakarta Sans & JetBrains Mono ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

/* ── Global Theme Overrides ── */
:root {
    --primary-color: #8B5CF6 !important;
}

html, body, [class*="css"], [class*="st-"] {
    font-family: 'Plus Jakarta Sans', 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* ── Hide Streamlit default chrome ── */
#MainMenu { visibility: hidden; display: none !important; }
footer { visibility: hidden; display: none !important; }
header { visibility: hidden; display: none !important; }
.stDeployButton { display: none !important; }
div[data-testid="stToolbar"] { display: none !important; }
div[data-testid="stDecoration"] { display: none !important; }
div[data-testid="stStatusWidget"] { display: none !important; }

/* ── Page Background & Sizing (Spacious 820px layout for desktop presence) ── */
.stApp {
    background-color: #0B0F17 !important;
    background-image:
        radial-gradient(circle at 50% 0%, rgba(139, 92, 246, 0.08) 0%, transparent 60%),
        radial-gradient(circle at 100% 30%, rgba(6, 182, 212, 0.04) 0%, transparent 50%) !important;
    color: #F8FAFC !important;
}

.block-container {
    max-width: 820px !important;
    padding-top: 1.75rem !important;
    padding-bottom: 4rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
}

/* ── Navigation Header ── */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.07);
    margin-bottom: 1.75rem;
}
.brand-group {
    display: flex;
    align-items: center;
    gap: 0.85rem;
}
.brand-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.25), rgba(6, 182, 212, 0.15));
    border: 1px solid rgba(139, 92, 246, 0.45);
    border-radius: 10px;
    font-size: 1.25rem;
    box-shadow: 0 0 15px rgba(139, 92, 246, 0.2);
    flex-shrink: 0;
}
.brand-title {
    font-size: 1.35rem;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: -0.025em;
    line-height: 1.2;
}
.brand-sub {
    font-size: 0.78rem;
    color: #94A3B8;
    font-weight: 500;
    letter-spacing: 0.02em;
}

/* ── Status Indicator (Pulsing Live Radar Badge) ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.3rem 0.75rem;
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.25);
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    color: #34D399;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.pulse-dot {
    width: 7px;
    height: 7px;
    background-color: #10B981;
    border-radius: 50%;
    box-shadow: 0 0 8px #10B981;
    animation: radarPulse 2s infinite ease-in-out;
}
@keyframes radarPulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.6); }
    70% { transform: scale(1.1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

/* ── Hero Section ── */
.hero-box {
    margin-bottom: 1.75rem;
}
.hero-badge-row {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 0.65rem;
    flex-wrap: wrap;
}
.cyber-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.72rem;
    font-weight: 600;
    color: #A78BFA;
    background: rgba(139, 92, 246, 0.1);
    border: 1px solid rgba(139, 92, 246, 0.25);
    padding: 0.2rem 0.55rem;
    border-radius: 6px;
    letter-spacing: 0.03em;
}
.hero-title {
    font-size: 2.35rem;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: -0.035em;
    line-height: 1.15;
    margin-bottom: 0.6rem;
}
.hero-title-gradient {
    background: linear-gradient(135deg, #F8FAFC 20%, #A78BFA 65%, #22D3EE 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 1.02rem;
    font-weight: 400;
    color: #94A3B8;
    line-height: 1.6;
    margin: 0;
    max-width: 90%;
}

/* ── Model Performance Telemetry Strip ── */
.metrics-strip {
    display: flex;
    background: rgba(17, 24, 39, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    margin-bottom: 0.45rem;
    backdrop-filter: blur(12px);
}
.metric-item {
    flex: 1;
    padding: 1.1rem 1.25rem;
    text-align: center;
    border-right: 1px solid rgba(255, 255, 255, 0.07);
}
.metric-item:last-child {
    border-right: none;
}
.metric-val {
    font-size: 1.65rem;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: -0.03em;
    font-family: 'JetBrains Mono', monospace;
}
.metric-val-cyan   { color: #22D3EE; }
.metric-val-violet { color: #C084FC; }
.metric-val-emerald{ color: #34D399; }
.metric-lbl {
    font-size: 0.74rem;
    font-weight: 700;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 3px;
}
.metric-note {
    font-size: 0.76rem;
    color: #64748B;
    text-align: center;
    margin-bottom: 1.85rem;
}

/* ── Demonstration Scenario Suite ── */
.library-card {
    background: #111827;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.25rem 1.45rem;
    margin-bottom: 1.65rem;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}
.library-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.85rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}
.library-title {
    font-size: 0.82rem;
    font-weight: 800;
    color: #F8FAFC;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.library-counter {
    font-size: 0.74rem;
    font-weight: 700;
    color: #A78BFA;
    background: rgba(139, 92, 246, 0.12);
    border: 1px solid rgba(139, 92, 246, 0.3);
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    letter-spacing: 0.03em;
}
.scenario-preview {
    background: #0B0F17;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 0.75rem 0.95rem;
    font-size: 0.88rem;
    color: #CBD5E1;
    line-height: 1.55;
    margin-top: 0.5rem;
    margin-bottom: 0.75rem;
    font-style: italic;
    border-left: 3px solid #8B5CF6;
}

/* ── Selectbox Overrides (Dark Cybersecurity Dropdown) ── */
div[data-testid="stSelectbox"] {
    margin-bottom: 0.35rem !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background-color: #0B0F17 !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 8px !important;
    color: #F8FAFC !important;
    font-size: 0.92rem !important;
    transition: all 0.15s ease !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"]:hover {
    border-color: #8B5CF6 !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within {
    border-color: #8B5CF6 !important;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.25) !important;
}
div[data-testid="stSelectbox"] svg {
    fill: #94A3B8 !important;
}

/* Dropdown popover menu styling */
div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
    background-color: #111827 !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    color: #F8FAFC !important;
}
li[role="option"] {
    color: #E2E8F0 !important;
}
li[role="option"]:hover, li[role="option"][aria-selected="true"] {
    background-color: rgba(139, 92, 246, 0.2) !important;
    color: #F8FAFC !important;
}

/* ── Streamlit Tabs (Electric Violet Underline Segmented Control) ── */
div[data-testid="stTabs"] button[data-baseweb="tab"] {
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    color: #64748B !important;
    padding: 0.5rem 1.15rem !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
}
div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
    color: #A78BFA !important;
}
div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
    color: #F8FAFC !important;
    border-bottom-color: #8B5CF6 !important;
}

/* ── Workspace Header Bar ── */
.workspace-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.55rem;
}
.workspace-label {
    font-size: 0.84rem;
    font-weight: 800;
    color: #CBD5E1;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* ── Action Pill Buttons (Secondary Surface Buttons) ── */
div[data-testid="stHorizontalBlock"] {
    gap: 0.55rem !important;
    margin-bottom: 0.65rem !important;
}
div[data-testid="stHorizontalBlock"] button {
    background-color: #161F30 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    color: #E2E8F0 !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    padding: 0.45rem 0.85rem !important;
    line-height: 1.4 !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2) !important;
}
div[data-testid="stHorizontalBlock"] button:hover {
    border-color: #8B5CF6 !important;
    color: #FFFFFF !important;
    background-color: rgba(139, 92, 246, 0.15) !important;
    box-shadow: 0 0 10px rgba(139, 92, 246, 0.25) !important;
}

/* ── Message Textarea (Dark Slate Surface, Crisp White Text, Violet Focus) ── */
div[data-testid="stTextArea"] {
    background: transparent !important;
    margin-bottom: 0.3rem !important;
}
div[data-testid="stTextAreaRootElement"] {
    background-color: #111827 !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 10px !important;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.25) !important;
    transition: all 0.15s ease !important;
}
div[data-testid="stTextAreaRootElement"]:hover {
    border-color: rgba(255, 255, 255, 0.2) !important;
}
div[data-testid="stTextAreaRootElement"]:focus-within {
    border-color: #8B5CF6 !important;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.25), inset 0 2px 4px rgba(0, 0, 0, 0.25) !important;
    outline: none !important;
}
div[data-testid="stTextAreaRootElement"] textarea {
    background-color: #111827 !important;
    color: #F8FAFC !important;
    caret-color: #22D3EE !important;
    font-size: 0.96rem !important;
    line-height: 1.6 !important;
    font-family: inherit !important;
    border: none !important;
    outline: none !important;
    padding: 0.95rem 1.05rem !important;
}
div[data-testid="stTextAreaRootElement"] textarea::placeholder {
    color: #64748B !important;
}

/* ── Context Counter Row ── */
.counter-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.78rem;
    color: #64748B;
    margin-bottom: 0.95rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Primary Action Button (Electric Violet Glow CTA) ── */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #7C3AED 0%, #6366F1 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 10px !important;
    color: #FFFFFF !important;
    font-size: 0.98rem !important;
    font-weight: 700 !important;
    padding: 0.75rem 1.45rem !important;
    letter-spacing: 0.01em !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    box-shadow: 0 4px 16px rgba(124, 58, 237, 0.35) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #6D28D9 0%, #4F46E5 100%) !important;
    box-shadow: 0 6px 22px rgba(124, 58, 237, 0.5) !important;
    transform: translateY(-1.5px) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:active {
    transform: translateY(0) !important;
}

/* ── Prediction Result Experience (Highlight of the Platform) ── */
@keyframes threatReveal {
    from { opacity: 0; transform: translateY(-6px); }
    to   { opacity: 1; transform: translateY(0); }
}

.result-card {
    border-radius: 12px;
    padding: 1.45rem 1.7rem;
    margin-top: 1.5rem;
    animation: threatReveal 0.25s ease-out;
}

/* SPAM THREAT CARD (Dark Crimson Surface with Coral Glow) */
.result-card-spam {
    background: linear-gradient(135deg, rgba(31, 19, 21, 0.95), rgba(42, 18, 21, 0.95));
    border: 1px solid rgba(239, 68, 68, 0.4);
    box-shadow: 0 8px 30px rgba(239, 68, 68, 0.18);
}
/* LEGITIMATE SAFE CARD (Dark Emerald Surface with Green Glow) */
.result-card-ham {
    background: linear-gradient(135deg, rgba(11, 30, 22, 0.95), rgba(15, 41, 30, 0.95));
    border: 1px solid rgba(16, 185, 129, 0.4);
    box-shadow: 0 8px 30px rgba(16, 185, 129, 0.18);
}

.result-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.65rem;
}
.verdict-badge-spam {
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    font-size: 1.35rem;
    font-weight: 800;
    color: #F87171;
    letter-spacing: -0.02em;
}
.verdict-badge-ham {
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    font-size: 1.35rem;
    font-weight: 800;
    color: #34D399;
    letter-spacing: -0.02em;
}
.signal-tag {
    font-size: 0.78rem;
    font-weight: 700;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    letter-spacing: 0.04em;
    font-family: 'JetBrains Mono', monospace;
}
.signal-tag-spam {
    color: #FCA5A5;
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.4);
}
.signal-tag-ham {
    color: #6EE7B7;
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.4);
}
.result-text {
    font-size: 0.96rem;
    line-height: 1.6;
    margin-bottom: 1.05rem;
}
.result-text-spam { color: #FCA5A5; }
.result-text-ham  { color: #A7F3D0; }

.signal-row {
    display: flex;
    align-items: center;
    gap: 0.95rem;
    padding-top: 0.95rem;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
}
.signal-label {
    font-size: 0.78rem;
    font-weight: 700;
    color: #94A3B8;
    white-space: nowrap;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.signal-track {
    flex: 1;
    height: 7px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 999px;
    overflow: hidden;
}
.signal-fill-spam {
    height: 100%;
    background: linear-gradient(90deg, #EF4444, #F87171);
    border-radius: 999px;
    box-shadow: 0 0 10px rgba(239, 68, 68, 0.5);
}
.signal-fill-ham {
    height: 100%;
    background: linear-gradient(90deg, #10B981, #34D399);
    border-radius: 999px;
    box-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
}

/* ── Technical Diagnostics Expander ── */
details[data-testid="stExpander"] {
    background-color: #111827 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    margin-top: 1.15rem !important;
}
details[data-testid="stExpander"] summary {
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    color: #94A3B8 !important;
}
details[data-testid="stExpander"] summary:hover {
    color: #C084FC !important;
}

.token-label {
    font-size: 0.74rem;
    font-weight: 700;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.45rem;
}
.token-container {
    background: #0B0F17;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 0.75rem 0.95rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.84rem;
    color: #38BDF8;
    word-break: break-all;
    line-height: 1.6;
    margin-bottom: 0.95rem;
}
.score-box {
    font-size: 0.82rem;
    color: #94A3B8;
    line-height: 1.6;
    padding: 0.75rem 0.95rem;
    background: #0B0F17;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
}
.score-box strong { color: #F8FAFC; }
.score-box code { color: #22D3EE; background: rgba(6, 182, 212, 0.1); padding: 0.1rem 0.35rem; border-radius: 4px; }

/* ── 5-Step Pipeline Sequence ── */
.sequence-card {
    background: #111827;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.45rem 1.7rem;
    margin-top: 2rem;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}
.sequence-head {
    font-size: 0.86rem;
    font-weight: 800;
    color: #F8FAFC;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 1.15rem;
    padding-bottom: 0.65rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    display: flex;
    align-items: center;
    gap: 0.45rem;
}
.pipeline-steps {
    display: flex;
    flex-direction: column;
    gap: 0.85rem;
}
.step-row {
    display: flex;
    align-items: flex-start;
    gap: 0.9rem;
    padding: 0.75rem 0.95rem;
    background: #0B0F17;
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
}
.step-num {
    font-size: 0.74rem;
    font-weight: 800;
    color: #8B5CF6;
    background: rgba(139, 92, 246, 0.15);
    border: 1px solid rgba(139, 92, 246, 0.3);
    padding: 0.2rem 0.5rem;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    flex-shrink: 0;
}
.step-title {
    font-size: 0.86rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 0.2rem;
}
.step-desc {
    font-size: 0.82rem;
    color: #94A3B8;
    line-height: 1.5;
}

/* ── Minimalist SaaS Footer ── */
.app-footer {
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(255, 255, 255, 0.07);
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.78rem;
    color: #64748B;
}
.footer-left {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Responsive Mobile Layout Media Queries ── */
@media (max-width: 640px) {
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    .hero-title {
        font-size: 1.85rem;
    }
    .metrics-strip {
        flex-direction: column;
    }
    .metric-item {
        border-right: none;
        border-bottom: 1px solid rgba(255, 255, 255, 0.07);
        padding: 0.85rem;
    }
    .metric-item:last-child {
        border-bottom: none;
    }
    .app-footer {
        flex-direction: column;
        gap: 0.75rem;
        text-align: center;
    }
}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# PART 4: TEXT PREPROCESSING PIPELINE (NLP)
# ==============================================================================

# Ensure the NLTK English stop words corpus is downloaded and cached locally.
nltk.download('stopwords', quiet=True)

# Instantiate the Porter Stemmer at module level for reuse.
stemmer = PorterStemmer()

# Load the set of standard English stopwords from NLTK.
# Using a Python set provides O(1) lookup time when checking 'word not in stop_words'.
stop_words = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """
    Cleans and normalizes raw SMS text for machine-learning feature extraction.

    IMPORTANT: This function is byte-for-byte identical to the clean_text() function
    used in Cell 18 of the training notebook (notebook.ipynb).

    Text Cleaning Steps:
    --------------------
    1. Lowercase: Normalizes capitalization across all tokens ('URGENT' -> 'urgent').
    2. Punctuation Removal: Strips symbols (!?#$%...) using str.maketrans without regex overhead.
    3. Whitespace Tokenization: Splits normalized string into discrete word tokens.
    4. Stopword Removal & Porter Stemming: Discards non-semantic stopwords ('the', 'is')
       and reduces words to their root stems ('winning' -> 'win').
    5. Token Rejoining: Reconstructs a space-separated string for TF-IDF feature extraction.
    """
    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 3. Tokenize by whitespace
    words = text.split()

    # 4. Remove stopwords and stem remaining words
    stemmed_words = [stemmer.stem(word) for word in words if word not in stop_words]

    # 5. Rejoin stemmed tokens into a clean text string
    return ' '.join(stemmed_words)


# ==============================================================================
# PART 5: MODEL AND VECTORIZER ARTIFACT LOADING
# ==============================================================================

# @st.cache_resource is Streamlit's resource-caching decorator.
# It loads the model and vectorizer into RAM ONCE so they persist across reruns.
@st.cache_resource
def load_artifacts():
    """
    Loads the trained Linear Support Vector Classifier and fitted TF-IDF vectorizer
    from the 'model/' directory using joblib.load().

    Returns:
        tfidf: The fitted TfidfVectorizer instance (vocabulary size: 6,959 terms).
        model: The trained LinearSVC classification model.
    """
    # Relative paths to model artifacts
    tfidf_path = os.path.join("model", "tfidf.joblib")
    model_path = os.path.join("model", "model.joblib")

    # Verify files exist on disk
    if not os.path.exists(tfidf_path) or not os.path.exists(model_path):
        st.error("Model artifacts not found in `model/`. Please execute notebook.ipynb first.")
        st.stop()

    # Deserialize binary joblib files into scikit-learn Python objects
    tfidf = joblib.load(tfidf_path)
    model = joblib.load(model_path)

    return tfidf, model


# Load vectorizer and classifier globally into memory
tfidf, model = load_artifacts()


# ==============================================================================
# PART 6: 20 REALISTIC DEMONSTRATION EXAMPLES (10 SPAM / 10 HAM)
# ==============================================================================
# 20 distinct demonstration messages covering common mobile security scenarios.
# Note: These are demonstration inputs, NOT training data; each is evaluated dynamically.

# 10 Distinct Spam Threat Categories
SPAM_EXAMPLES = [
    {
        "category": "Lottery & Prize Scam",
        "title": "National Lottery Cash Prize",
        "text": "CONGRATULATIONS! You have won £1,000,000 in the National UK Lottery Draw. Claim your cash prize now by calling 09061701461 with ref #WIN992."
    },
    {
        "category": "Fake Bank Warning",
        "title": "Urgent Bank Security Alert",
        "text": "URGENT from Barclays Alert: We detected unauthorized login attempts on your account. To prevent suspension, verify your identity immediately at bit.ly/barclays-auth."
    },
    {
        "category": "Account Verification Scam",
        "title": "PayPal Account Suspension Notice",
        "text": "URGENT Security Notice: Your mobile PayPal account access has been limited. Text CLAIM to 87121 or call customer services now to verify your details."
    },
    {
        "category": "Fake Parcel Delivery Fee",
        "title": "Postal Parcel Shipping Fee",
        "text": "Royal Mail Alert: Your parcel delivery is pending. A payment fee of £1.99 is required. Text POST to 84115 to claim your package or reply STOP to unsubscribe."
    },
    {
        "category": "Urgent Loan Offer",
        "title": "Pre-Approved Personal Loan",
        "text": "Pre-approved Loan Alert! Borrow up to £5,000 today with instant decision and guaranteed credit approval. Text CASH to 88202 to claim funds immediately."
    },
    {
        "category": "Fake Job Opportunity",
        "title": "Work-From-Home Mobile Assistant",
        "text": "Earn £300 to £500 daily working from home on your mobile! No experience needed. Immediate start. Text WORK to 69888 to claim your welcome pack today."
    },
    {
        "category": "Gift Card Scam",
        "title": "Amazon Shopping Gift Card",
        "text": "You have been selected to receive a free £250 Amazon voucher gift card! Claim your reward now before offer expires tonight: text GIFT to 88202."
    },
    {
        "category": "Phishing & Personal Info",
        "title": "HMRC Official Tax Rebate",
        "text": "HMRC Tax Office: You have an unclaimed tax refund of £485. Claim your cash rebate now. Call 09065989182 or text CLAIM to 87007 to verify your refund."
    },
    {
        "category": "Fake Subscription Renewal",
        "title": "Antivirus Mobile Service Renewal",
        "text": "URGENT Notice: Your mobile antivirus service subscription has renewed at £39.99/month. To cancel this charge or stop billing, text STOP to 87066 immediately."
    },
    {
        "category": "Financial Reward Scam",
        "title": "Unclaimed Cash Credit Transfer",
        "text": "Customer Alert: You have £1,500 waiting in unclaimed cash credit rewards on your mobile account. Call 08712400603 now to claim your cash bonus."
    }
]

# 10 Distinct Legitimate (HAM) Communication Categories
HAM_EXAMPLES = [
    {
        "category": "Friend Conversation",
        "title": "Dinner & Football Catch-up",
        "text": "Hey mate, are you free this evening? Was thinking we could grab some burgers and watch the football match at my place later."
    },
    {
        "category": "Family Message",
        "title": "Family Well-being Check-in",
        "text": "Hi sweetheart, just wanted to let you know grandma is doing much better today. Call me whenever you get some free time tonight!"
    },
    {
        "category": "Meeting Reminder",
        "title": "Team Sync Schedule Update",
        "text": "Quick reminder: The product sync meeting has been moved to 3:30 PM in Conference Room B. See you all there shortly."
    },
    {
        "category": "University Notice",
        "title": "Campus Library Revision Hours",
        "text": "Campus Alert: Library hours will be extended until midnight throughout next week for exam preparation. Good luck with your revision!"
    },
    {
        "category": "Work Coordination",
        "title": "Q3 Budget Spreadsheet Review",
        "text": "Hi Sarah, I just emailed you the revised Q3 budget spreadsheet. Please review the marketing section before tomorrow morning's presentation."
    },
    {
        "category": "Genuine Delivery Update",
        "title": "Courier Driver Safe Place Notice",
        "text": "DPD update: Driver Michael will deliver your package today between 14:15 and 15:15. If nobody is home, we will leave it in your designated safe place."
    },
    {
        "category": "Appointment Reminder",
        "title": "Dental Clinic Check-up Notice",
        "text": "Appointment reminder from Oak Dental Practice: You have a dental check-up scheduled for tomorrow, Thursday at 10:30 AM with Dr. Harrison."
    },
    {
        "category": "Routine Payment Confirmation",
        "title": "Shared Household Utility Bill",
        "text": "Hey Mark, just letting you know I transferred my share for the electricity bill and groceries to your bank earlier. Let me know when it arrives!"
    },
    {
        "category": "Travel Coordination",
        "title": "Airport Terminal Pickup",
        "text": "Just landed at Heathrow Terminal 5. Waiting at baggage carousel 4 now, should be outside at the passenger pickup area in about 20 minutes."
    },
    {
        "category": "Social Invitation",
        "title": "Weekend Board Game Night",
        "text": "Hey everyone! We're hosting a small board game night this Saturday around 7 PM. Let us know if you can make it so we know how much food to order!"
    }
]


# ==============================================================================
# PART 7: STREAMLIT SESSION STATE & EVENT CALLBACKS
# ==============================================================================
# st.session_state maintains memory across page reruns.
# Using on_click and on_change callbacks synchronizes the textarea (key='sms_input')
# before the rerun executes, avoiding widget-key conflicts.

# Initialize the workspace text in session state if first load
if "sms_input" not in st.session_state:
    st.session_state["sms_input"] = ""


# Callback function: Loads a chosen string into the workspace
def load_text_into_workspace(text: str):
    st.session_state["sms_input"] = text


# Callback function: Clears the workspace textarea
def clear_workspace():
    st.session_state["sms_input"] = ""


# Callback function: Triggered automatically when the user changes the Spam selectbox
def on_spam_select():
    selected_index = st.session_state.get("spam_selector_idx", 0)
    st.session_state["sms_input"] = SPAM_EXAMPLES[selected_index]["text"]


# Callback function: Triggered automatically when the user changes the Ham selectbox
def on_ham_select():
    selected_index = st.session_state.get("ham_selector_idx", 0)
    st.session_state["sms_input"] = HAM_EXAMPLES[selected_index]["text"]


# ==============================================================================
# PART 8: APPLICATION HEADER & BRANDING UI
# ==============================================================================
# Displays the top navigation bar with brand icon, wordmark, and live status pill.
st.markdown("""
<div class="app-header">
  <div class="brand-group">
    <div class="brand-icon">🛡️</div>
    <div>
      <div class="brand-title">SMS SHIELD</div>
      <div class="brand-sub">Intelligent SMS Threat Detection</div>
    </div>
  </div>
  <div class="status-pill">
    <span class="pulse-dot"></span>
    <span>AI Engine Active</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# PART 9: HERO SECTION
# ==============================================================================
# Editorial cybersecurity headline and value proposition.
st.markdown("""
<div class="hero-box">
  <div class="hero-badge-row">
    <span class="cyber-chip">⚡ Sub-Millisecond Inference</span>
    <span class="cyber-chip">🔒 Zero Retention Policy</span>
    <span class="cyber-chip">🛡️ Linear SVM Hyperplane</span>
  </div>
  <div class="hero-title">
    Your Messages. <span class="hero-title-gradient">Under Protection.</span>
  </div>
  <div class="hero-sub">
    Real-time SMS threat intelligence powered by Natural Language Processing and maximum-margin
    machine learning. Evaluated across 5,169 verified telecommunications.
  </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# PART 10: MODEL EVALUATION METRICS STRIP
# ==============================================================================
# Displays verified offline test-set evaluation metrics from the 80/20 stratified split:
# - Accuracy: 98.26%
# - Precision: 97.48% (Crucial for spam filtering to minimize false positives)
# - Recall: 88.55% (Catches nearly 9 in 10 actual threat messages)
st.markdown("""
<div class="metrics-strip">
  <div class="metric-item">
    <div class="metric-val metric-val-cyan">98.26%</div>
    <div class="metric-lbl">Test Accuracy</div>
  </div>
  <div class="metric-item">
    <div class="metric-val metric-val-violet">97.48%</div>
    <div class="metric-lbl">Threat Precision</div>
  </div>
  <div class="metric-item">
    <div class="metric-val metric-val-emerald">88.55%</div>
    <div class="metric-lbl">Threat Recall</div>
  </div>
</div>
<div class="metric-note">
  Benchmarked on 1,034 held-out test messages (stratified 80/20 split) · Static offline telemetry
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# PART 11: DEMONSTRATION SCENARIO SUITE (20 REAL-WORLD EXAMPLES)
# ==============================================================================
# Interactive categorized scenario library allowing users to select and inspect
# 10 Spam Threat scenarios and 10 Legitimate Communication scenarios.
st.markdown("""
<div class="library-card">
  <div class="library-header">
    <span class="library-title">🧪 Threat Demonstration Suite</span>
    <span class="library-counter">20 Verified Scenarios</span>
  </div>
""", unsafe_allow_html=True)

# st.tabs creates tabbed navigation panels to separate Spam and Legitimate categories cleanly
tab_spam, tab_ham = st.tabs(["🚨 Threat Scenarios (10 Types)", "✅ Legitimate Traffic (10 Types)"])

# ── Tab 1: Spam Threat Scenarios ──
with tab_spam:
    spam_options = [f"{i+1}. {item['category']} — {item['title']}" for i, item in enumerate(SPAM_EXAMPLES)]
    selected_spam_label = st.selectbox(
        "Select a Threat Category:",
        options=range(len(spam_options)),
        format_func=lambda x: spam_options[x],
        key="spam_selector_idx",
        on_change=on_spam_select,
        label_visibility="collapsed",
    )
    current_spam = SPAM_EXAMPLES[selected_spam_label]
    st.markdown(f'<div class="scenario-preview"><strong>Preview ({current_spam["category"]}):</strong><br>"{current_spam["text"]}"</div>', unsafe_allow_html=True)
    col_load_s, _ = st.columns([1, 1])
    col_load_s.button(
        "📥 Load Threat Scenario",
        on_click=load_text_into_workspace,
        args=(current_spam["text"],),
        use_container_width=True,
        key="btn_load_spam_scenario",
    )

# ── Tab 2: Legitimate Communication Scenarios ──
with tab_ham:
    ham_options = [f"{i+1}. {item['category']} — {item['title']}" for i, item in enumerate(HAM_EXAMPLES)]
    selected_ham_label = st.selectbox(
        "Select a Legitimate Category:",
        options=range(len(ham_options)),
        format_func=lambda x: ham_options[x],
        key="ham_selector_idx",
        on_change=on_ham_select,
        label_visibility="collapsed",
    )
    current_ham = HAM_EXAMPLES[selected_ham_label]
    st.markdown(f'<div class="scenario-preview"><strong>Preview ({current_ham["category"]}):</strong><br>"{current_ham["text"]}"</div>', unsafe_allow_html=True)
    col_load_h, _ = st.columns([1, 1])
    col_load_h.button(
        "📥 Load Safe Scenario",
        on_click=load_text_into_workspace,
        args=(current_ham["text"],),
        use_container_width=True,
        key="btn_load_ham_scenario",
    )

st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# PART 12: WORKSPACE CONTROLS & QUICK ACTIONS
# ==============================================================================
# Header bar for the message analysis panel with one-click sample shortcuts and clear.
st.markdown('<div class="workspace-bar"><span class="workspace-label">⚡ Message Analysis Workspace</span></div>', unsafe_allow_html=True)

# st.columns splits layout into 3 equal columns for workspace action buttons
col_q1, col_q2, col_clear = st.columns([1, 1, 1])

# Quick Threat Sample button
col_q1.button(
    "⚠️  Quick Threat",
    on_click=load_text_into_workspace,
    args=(SPAM_EXAMPLES[0]["text"],),
    use_container_width=True,
    help="Load sample National Lottery cash prize scam",
)

# Quick Safe Sample button
col_q2.button(
    "✅  Quick Safe",
    on_click=load_text_into_workspace,
    args=(HAM_EXAMPLES[0]["text"],),
    use_container_width=True,
    help="Load sample conversational message between friends",
)

# Clear button: Resets the textarea
col_clear.button(
    "✕  Clear",
    on_click=clear_workspace,
    use_container_width=True,
    help="Clear message workspace",
)


# ==============================================================================
# PART 13: MESSAGE INPUT TEXTAREA & LIVE CHARACTER COUNTER
# ==============================================================================
# Multiline text input area bound to session state key 'sms_input'.
user_message = st.text_area(
    label="SMS Message Content",
    key="sms_input",
    placeholder="Paste suspicious SMS transmission, or select from the demonstration suite above…",
    height=140,
    label_visibility="collapsed",
)

# Calculate string length dynamically using Python's len()
char_count = len(user_message)

# Display character telemetry alongside the standard 160-character mobile SMS segment benchmark
st.markdown(
    f"""
<div class="counter-row">
  <span>STANDARD SMS SEGMENT: 160 CHARACTERS</span>
  <span>TELEMETRY: <strong>{char_count}</strong> CHARS</span>
</div>
""",
    unsafe_allow_html=True,
)


# ==============================================================================
# PART 14: CLASSIFICATION ACTION BUTTON (CTA)
# ==============================================================================
# Primary CTA button triggers classification inference when clicked.
analyse_clicked = st.button(
    "ANALYZE MESSAGE →",
    type="primary",
    use_container_width=True,
)


# ==============================================================================
# PART 15: PREDICTION WORKFLOW & RESULT DISPLAY
# ==============================================================================
# Runs classification when the user clicks 'ANALYZE MESSAGE →'.
if analyse_clicked:
    # Validate that the user entered non-whitespace text
    if not user_message.strip():
        st.warning("⚠️ Telemetry Error: Please enter or load an SMS message before analyzing.")
    else:
        # Step 1: Preprocess raw input text using clean_text()
        cleaned = clean_text(user_message)

        # Step 2: Extract TF-IDF features using fitted vectorizer
        features = tfidf.transform([cleaned])

        # Step 3: Run classification inference using LinearSVC model
        prediction = model.predict(features)[0]  # 1 = SPAM, 0 = HAM

        # Step 4: Compute raw geometric decision margin (signed distance to hyperplane)
        decision_score = model.decision_function(features)[0]

        # Step 5: Map margin magnitude to visual signal separation strength
        abs_score = abs(decision_score)
        if abs_score >= 1.5:
            band_label = "VERY HIGH SEPARATION"
            bar_pct = 100
        elif abs_score >= 0.8:
            band_label = "HIGH SEPARATION"
            bar_pct = 72
        elif abs_score >= 0.4:
            band_label = "MODERATE SEPARATION"
            bar_pct = 45
        else:
            band_label = "LOW SEPARATION"
            bar_pct = 22

        # Step 6: Render distinct cybersecurity result card
        if prediction == 1:
            # 🚨 SPAM DETECTED STATE (Dark Crimson Surface with Coral Glow)
            st.markdown(f"""
<div class="result-card result-card-spam">
  <div class="result-head">
    <div class="verdict-badge-spam">🚨 THREAT DETECTED · SPAM</div>
    <div class="signal-tag signal-tag-spam">SIGNAL: {band_label}</div>
  </div>
  <div class="result-text result-text-spam">
    This transmission exhibits linguistic syntax and structural anomalies characteristic of fraudulent
    phishing, lottery scamming, or malicious commercial exploitation.
  </div>
  <div class="signal-row">
    <span class="signal-label">Decision Margin</span>
    <div class="signal-track">
      <div class="signal-fill-spam" style="width: {bar_pct}%;"></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
        else:
            # ✅ LEGITIMATE MESSAGE STATE (Dark Emerald Surface with Green Glow)
            st.markdown(f"""
<div class="result-card result-card-ham">
  <div class="result-head">
    <div class="verdict-badge-ham">✅ VERIFIED SAFE · LEGITIMATE</div>
    <div class="signal-tag signal-tag-ham">SIGNAL: {band_label}</div>
  </div>
  <div class="result-text result-text-ham">
    This transmission conforms to standard interpersonal and institutional communication protocols,
    exhibiting zero significant markers of fraudulent intent or spam.
  </div>
  <div class="signal-row">
    <span class="signal-label">Decision Margin</span>
    <div class="signal-track">
      <div class="signal-fill-ham" style="width: {bar_pct}%;"></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        # Step 7: Technical Diagnostics & Feature Inspection Expander
        with st.expander("🔬 Technical Telemetry & NLP Feature Diagnostics"):
            st.markdown('<div class="token-label">Normalized Linguistic Tokens (Stopwords Discarded · Porter Stemmed)</div>', unsafe_allow_html=True)
            token_display = cleaned if cleaned else "(No non-stopword tokens remaining)"
            st.markdown(f'<div class="token-container">{token_display}</div>', unsafe_allow_html=True)
            st.markdown(f"""
<div class="score-box">
  <strong>Linear SVM Decision Score:</strong> <code>{decision_score:+.4f}</code><br>
  <em>Statistical Telemetry Note:</em> This score represents the uncalibrated signed geometric distance from the feature
  sample to the separating hyperplane: <code>f(x) = w^T * x + b</code>. It is <strong>not</strong> a probability or percentage confidence.
  Positive values indicate Threat (Spam); negative values indicate Verified Safe (Ham). The magnitude indicates relative boundary separation.
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# PART 16: 5-STEP MACHINE-LEARNING ARCHITECTURE SEQUENCE
# ==============================================================================
# Elegant visual sequence explaining the 5 technical stages of the pipeline:
# 01 Ingestion -> 02 NLP Preprocessing -> 03 TF-IDF -> 04 Linear SVM -> 05 Verdict
st.markdown("""
<div class="sequence-card">
  <div class="sequence-head">⚙️ Threat Classification Pipeline Architecture</div>
  <div class="pipeline-steps">
    <div class="step-row">
      <div class="step-num">01</div>
      <div>
        <div class="step-title">Message Ingestion & Buffer</div>
        <div class="step-desc">Accepts raw SMS string input through real-time telemetry buffer with zero persistence retention.</div>
      </div>
    </div>
    <div class="step-row">
      <div class="step-num">02</div>
      <div>
        <div class="step-title">NLP Linguistic Normalization</div>
        <div class="step-desc">Applies lowercasing, punctuation stripping, English stopword elimination, and Porter linguistic stemming.</div>
      </div>
    </div>
    <div class="step-row">
      <div class="step-num">03</div>
      <div>
        <div class="step-title">TF-IDF Vector Feature Mapping</div>
        <div class="step-desc">Projects normalized tokens into a fixed 6,959-dimensional sparse feature space trained without leakage.</div>
      </div>
    </div>
    <div class="step-row">
      <div class="step-num">04</div>
      <div>
        <div class="step-title">Linear Support Vector Classifier</div>
        <div class="step-desc">Computes maximum-margin separating hyperplane with squared-hinge loss optimization (C=1.0).</div>
      </div>
    </div>
    <div class="step-row">
      <div class="step-num">05</div>
      <div>
        <div class="step-title">Threat Verdict & Geometric Telemetry</div>
        <div class="step-desc">Generates immediate binary classification accompanied by uncalibrated hyperplane margin telemetry.</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# PART 17: MINIMALIST SAAS FOOTER
# ==============================================================================
st.markdown("""
<div class="app-footer">
  <div class="footer-left">
    <span>🛡️ <strong>SMS SHIELD</strong></span>
    <span>·</span>
    <span>Linear SVM Classifier · 98.26% Accuracy</span>
  </div>
  <div>
    <span>Internship ML Project Showcase · 2026</span>
  </div>
</div>
""", unsafe_allow_html=True)
