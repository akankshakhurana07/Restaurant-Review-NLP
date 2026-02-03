import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk

nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────── CUSTOM CSS ───────────────────────────────
css = """
<style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');

    /* ── Base / Reset ── */
    .stApp {
        background: #0a0e1a;
        color: #e2e8f0;
        font-family: 'Outfit', sans-serif !important;
    }
    * { font-family: 'Outfit', sans-serif !important; }

    /* ── Hide default Streamlit header ── */
    .stApp header { display: none !important; }
    .block-container { padding-top: 1.2rem !important; max-width: 960px !important; margin: 0 auto !important; }

    /* ── Hero Banner ── */
    .hero {
        background: linear-gradient(135deg, #1a1040 0%, #0f1c3a 50%, #0a0e1a 100%);
        border: 1px solid rgba(139,92,246,0.15);
        border-radius: 24px;
        padding: 42px 36px 36px;
        text-align: center;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -60px; left: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(139,92,246,0.18) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero::after {
        content: '';
        position: absolute;
        bottom: -50px; right: -50px;
        width: 200px; height: 200px;
        background: radial-gradient(circle, rgba(16,185,129,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa, #34d399, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        position: relative; z-index: 1;
    }
    .hero-sub {
        color: #64748b;
        font-size: 1rem;
        font-weight: 300;
        position: relative; z-index: 1;
    }

    /* ── Card ── */
    .card {
        background: linear-gradient(145deg, #131828, #0f1420);
        border: 1px solid rgba(139,92,246,0.12);
        border-radius: 18px;
        padding: 26px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    }
    .card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #cbd5e1;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .card-title .icon { font-size: 1.2rem; }

    /* ── Upload Box ── */
    .upload-box {
        border: 2px dashed rgba(139,92,246,0.35);
        border-radius: 14px;
        padding: 32px;
        text-align: center;
        background: rgba(139,92,246,0.04);
        transition: all 0.25s;
    }
    .upload-box:hover {
        border-color: rgba(139,92,246,0.7);
        background: rgba(139,92,246,0.08);
    }
    .upload-box .upload-icon { font-size: 2.2rem; margin-bottom: 8px; }
    .upload-box .upload-text { color: #64748b; font-size: 0.9rem; }
    .upload-box .upload-hint { color: #475569; font-size: 0.78rem; margin-top: 4px; }

    /* ── Streamlit file_uploader override ── */
    .stFileUploader { margin: 0 !important; }
    .stFileUploader label { display: none !important; }
    .stFileUploader [data-testid="fileUploaderContainer"] {
        border: 2px dashed rgba(139,92,246,0.35) !important;
        border-radius: 14px !important;
        background: rgba(139,92,246,0.04) !important;
        padding: 28px !important;
        transition: all 0.25s !important;
    }
    .stFileUploader [data-testid="fileUploaderContainer"]:hover {
        border-color: rgba(139,92,246,0.7) !important;
        background: rgba(139,92,246,0.08) !important;
    }
    .stFileUploader [data-testid="fileUploaderContainer"] p {
        color: #94a3b8 !important;
        font-size: 0.92rem !important;
    }
    .stFileUploader button {
        background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 22px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }

    /* ── Model Buttons ── */
    .model-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
    }
    .model-btn {
        background: #111827;
        border: 1px solid rgba(139,92,246,0.15);
        border-radius: 12px;
        padding: 14px 10px;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s;
        color: #94a3b8;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .model-btn:hover {
        border-color: rgba(139,92,246,0.5);
        background: rgba(139,92,246,0.06);
        color: #c4b5fd;
    }
    .model-btn.active {
        border-color: #7c3aed;
        background: rgba(139,92,246,0.14);
        color: #a78bfa;
        box-shadow: 0 0 12px rgba(139,92,246,0.25);
    }
    .model-btn .model-icon { font-size: 1.3rem; display: block; margin-bottom: 4px; }

    /* ── Train Button ── */
    .stButton button {
        background: linear-gradient(135deg, #7c3aed, #10b981) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        width: 100% !important;
        transition: all 0.2s !important;
        box-shadow: 0 4px 18px rgba(124,58,237,0.35) !important;
    }
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 24px rgba(124,58,237,0.5) !important;
    }
    .stButton button:disabled {
        background: #1e293b !important;
        box-shadow: none !important;
        color: #475569 !important;
        cursor: not-allowed !important;
    }

    /* ── Metric Cards Row ── */
    .metrics-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-bottom: 6px;
    }
    .metric-card {
        background: #111827;
        border-radius: 14px;
        padding: 18px 12px;
        text-align: center;
        border: 1px solid rgba(139,92,246,0.1);
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 800;
        margin-bottom: 2px;
    }
    .metric-label {
        font-size: 0.72rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
    }

    /* ── Progress Bar ── */
    .bar-row { margin-bottom: 12px; }
    .bar-label-row { display: flex; justify-content: space-between; margin-bottom: 5px; font-size: 0.82rem; color: #94a3b8; }
    .bar-label-row span:last-child { font-weight: 700; }
    .bar-bg { height: 9px; background: #1e293b; border-radius: 5px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 5px; transition: width 0.6s ease; }

    /* ── Status Tag ── */
    .status-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 10px;
    }
    .status-tag.success { background: rgba(16,185,129,0.1); color: #34d399; border: 1px solid rgba(16,185,129,0.2); }
    .status-tag.info    { background: rgba(96,165,250,0.1);  color: #60a5fa; border: 1px solid rgba(96,165,250,0.2);  }

    /* ── Plotly chart container ── */
    .stPlotlyChart { margin: 0 !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0a0e1a; }
    ::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 3px; }
</style>
"""

# ─────────────────────── APP ──────────────────────────────────────
st.markdown(css, unsafe_allow_html=True)

# ── Hero ──
st.markdown("""
<div class="hero">
  <div class="hero-title">🍽️ Restaurant Review Classification</div>
  <div class="hero-sub">Upload reviews · Preprocess with TF-IDF · Train & compare 6 ML models</div>
</div>
""", unsafe_allow_html=True)

# ─── SESSION STATE ───────────────────────────────────────────────
if "dataset" not in st.session_state:
    st.session_state["dataset"] = None
if "ready" not in st.session_state:
    st.session_state["ready"] = False

# ─── CARD 1 — UPLOAD ─────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title"><span class="icon">📂</span> Upload Dataset</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=["tsv", "csv", "txt"],
    help="Upload your Restaurant_Reviews.tsv file"
)

if uploaded_file is not None and st.session_state["dataset"] is None:
    try:
        st.session_state["dataset"] = pd.read_csv(uploaded_file, delimiter="\t", quoting=3)
        st.session_state["ready"] = True
    except Exception:
        st.session_state["dataset"] = pd.read_csv(uploaded_file, quoting=3)
        st.session_state["ready"] = True

if st.session_state["ready"]:
    st.markdown(f'<div class="status-tag success">✔ Loaded — {len(st.session_state["dataset"])} rows</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="status-tag info">ℹ Upload a .tsv file to continue</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)   # close card

# ─── CARD 2 — MODEL SELECT + TRAIN ──────────────────────────────
MODEL_INFO = {
    "Logistic Regression":   ("📈", "LR"),
    "KNN (k=5)":             ("🔍", "KNN"),
    "Linear SVM":            ("✂️",  "SVM"),
    "Decision Tree (CART)":  ("🌳", "DT"),
    "Random Forest":         ("🌲", "RF"),
    "Naive Bayes":           ("📊", "NB"),
}

if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = "Logistic Regression"

st.markdown('<div class="card"><div class="card-title"><span class="icon">🤖</span> Choose a Model</div>', unsafe_allow_html=True)

# Build model buttons with HTML + invisible radio behind
cols = st.columns(3)
model_names = list(MODEL_INFO.keys())
for idx, name in enumerate(model_names):
    icon, tag = MODEL_INFO[name]
    active_cls = "active" if st.session_state["selected_model"] == name else ""
    with cols[idx % 3]:
        if st.button(f"{icon}  {name}", key=f"mbtn_{name}", use_container_width=True):
            st.session_state["selected_model"] = name
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)   # close card

# ─── TRAIN BUTTON ────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
train_clicked = st.button(
    "▶  Train & Evaluate Model",
    disabled=not st.session_state["ready"],
    use_container_width=True
)
st.markdown('</div>', unsafe_allow_html=True)

# ─── PREPROCESSING + TRAINING ────────────────────────────────────
if train_clicked and st.session_state["ready"]:
    dataset = st.session_state["dataset"]
    selected_model = st.session_state["selected_model"]

    with st.spinner("Preprocessing & training…"):
        # Preprocess
        ps = PorterStemmer()
        n_rows = min(1000, len(dataset))
        corpus = []
        for i in range(n_rows):
            review = re.sub("[^a-zA-Z]", " ", str(dataset["Review"].iloc[i]))
            review = review.lower().split()
            review = [ps.stem(w) for w in review if w not in set(stopwords.words("english"))]
            corpus.append(" ".join(review))

        # TF-IDF
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus).toarray()
        y = dataset.iloc[:n_rows, 1].values

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Model
        if selected_model == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif selected_model == "KNN (k=5)":
            model = KNeighborsClassifier(n_neighbors=5, metric="cosine")
        elif selected_model == "Linear SVM":
            model = LinearSVC()
        elif selected_model == "Decision Tree (CART)":
            model = DecisionTreeClassifier(random_state=0)
        elif selected_model == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=0)
        elif selected_model == "Naive Bayes":
            model = MultinomialNB()

        model.fit(X_train, y_train)
        y_pred   = model.predict(X_test)
        y_train_pred = model.predict(X_train)

        # Metrics
        acc       = accuracy_score(y_test, y_pred)
        train_acc = accuracy_score(y_train, y_train_pred)
        prec      = precision_score(y_test, y_pred)
        rec       = recall_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred)
        cm        = confusion_matrix(y_test, y_pred)

    # ── RESULTS: Metric Cards ──
    st.markdown("""
    <div class="card">
      <div class="card-title"><span class="icon">📊</span> Model Performance</div>
      <div class="metrics-row">
    """, unsafe_allow_html=True)

    metrics = [
        ("Test Accuracy", acc,        "#a78bfa"),
        ("Train Accuracy", train_acc, "#34d399"),
        ("F1 Score",       f1,        "#60a5fa"),
        ("Precision",      prec,      "#fbbf24"),
        ("Recall",         rec,       "#f472b6"),
        ("Bias-Var Gap",   abs(train_acc - acc), "#fb923c"),
    ]
    for label, val, color in metrics:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="color:{color}">{val*100:.1f}%</div>
          <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)   # close metrics-row + card

    # ── RESULTS: Accuracy Bars ──
    st.markdown('<div class="card"><div class="card-title"><span class="icon">📈</span> Accuracy Bars</div>', unsafe_allow_html=True)
    bars = [
        ("Train Accuracy", train_acc, "#7c3aed"),
        ("Test  Accuracy", acc,       "#10b981"),
        ("F1 Score",        f1,       "#3b82f6"),
    ]
    for label, val, color in bars:
        pct = f"{val*100:.1f}%"
        w   = f"{val*100:.1f}%"
        st.markdown(f"""
        <div class="bar-row">
          <div class="bar-label-row"><span>{label}</span><span style="color:{color}">{pct}</span></div>
          <div class="bar-bg"><div class="bar-fill" style="width:{w}; background:{color};"></div></div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── RESULTS: Confusion Matrix (Plotly heatmap) ──
    st.markdown('<div class="card"><div class="card-title"><span class="icon">🟦</span> Confusion Matrix</div>', unsafe_allow_html=True)

    labels_cm = ["Negative (0)", "Positive (1)"]
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels_cm,
            y=labels_cm,
            colorscale=[[0,"#1e1b4b"],[0.5,"#4c1d95"],[1,"#7c3aed"]],
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 28, "color": "white"},
            colorbar=dict(
                thickness=14,
                tickfont=dict(color="#94a3b8", size=11),
                bgcolor="rgba(0,0,0,0)",
            ),
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Predicted", title_font=dict(color="#94a3b8", size=13), tickfont=dict(color="#cbd5e1", size=12)),
        yaxis=dict(title="Actual", title_font=dict(color="#94a3b8", size=13), tickfont=dict(color="#cbd5e1", size=12), autorange="reversed"),
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
