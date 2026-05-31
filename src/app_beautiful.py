"""
app.py
======
Fake Finders — Beautiful Web UI
All 7 models shown. BERT/RAG shown as static results.
Models cached — only trained ONCE at startup.

Run:
    cd ~/fake-finders-liar
    source venv/bin/activate
    streamlit run src/app.py
"""

import os
import sys
import time
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Fake Finders",
    page_icon  = "🔍",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Beautiful CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
}

.stApp {
    background: #060E18;
    color: #FFFFFF;
}

/* Hide streamlit default header */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main title */
.main-title {
    font-size: 3.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #FFFFFF 0%, #17A589 50%, #2980B9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
    line-height: 1.1;
}

.main-subtitle {
    color: #4A6A80;
    font-size: 14px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 1px;
    margin-bottom: 0;
}

/* Result cards */
.fake-card {
    background: linear-gradient(135deg, rgba(232,85,85,0.15), rgba(192,57,43,0.08));
    border: 1px solid rgba(232,85,85,0.4);
    border-radius: 16px;
    padding: 28px 20px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(232,85,85,0.15);
}

.real-card {
    background: linear-gradient(135deg, rgba(23,165,137,0.15), rgba(20,143,119,0.08));
    border: 1px solid rgba(23,165,137,0.4);
    border-radius: 16px;
    padding: 28px 20px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(23,165,137,0.15);
}

.result-label {
    font-size: 2.5rem;
    font-weight: 700;
    letter-spacing: 4px;
    margin: 8px 0 4px 0;
}

.result-conf {
    font-size: 1.1rem;
    opacity: 0.85;
}

.result-meta {
    font-size: 11px;
    color: #4A6A80;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 10px;
}

/* Evidence box */
.evidence-box {
    background: rgba(41,128,185,0.06);
    border: 1px solid rgba(41,128,185,0.25);
    border-left: 3px solid #2980B9;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 13px;
    color: #A8C4D8;
    line-height: 1.7;
    margin-top: 10px;
}

/* Model result cards */
.model-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 8px;
    transition: all 0.2s;
}

.model-card:hover {
    background: rgba(255,255,255,0.05);
    border-color: rgba(255,255,255,0.15);
}

.model-card-best {
    background: rgba(23,165,137,0.08);
    border: 1px solid rgba(23,165,137,0.3);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 8px;
}

/* Metric boxes */
.metric-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}

.metric-val {
    font-size: 1.8rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}

.metric-lbl {
    font-size: 11px;
    color: #4A6A80;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* Statement display */
.stmt-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 16px 20px;
    font-size: 15px;
    color: #E8F4F8;
    font-style: italic;
    line-height: 1.6;
}

/* BERT info box */
.bert-box {
    background: rgba(168,85,247,0.06);
    border: 1px solid rgba(168,85,247,0.25);
    border-left: 3px solid #A855F7;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 13px;
    color: #C4A8F0;
    line-height: 1.7;
}

/* Tag badges */
.tag {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 1px;
    margin-right: 6px;
}

/* Sidebar */
.sidebar-name {
    font-size: 13px;
    color: #A8C4D8;
    margin: 2px 0;
}

/* Progress bar custom */
.prog-wrap {
    background: rgba(255,255,255,0.06);
    border-radius: 4px;
    height: 6px;
    margin-top: 4px;
    overflow: hidden;
}

.prog-fill {
    height: 100%;
    border-radius: 4px;
}

/* Divider */
.custom-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.07);
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Load models ONCE and cache
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="🔧 Training models... first load takes ~3 minutes")
def load_models():
    import numpy as np
    from sklearn.svm            import SVC
    from sklearn.preprocessing  import MaxAbsScaler
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline       import Pipeline
    from feature_extractor      import FeatureBuilder
    from data_loader            import map_label

    BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TRAIN_PATH = os.path.join(BASE, "data", "train.tsv")

    texts, labels, rows = [], [], []
    with open(TRAIN_PATH, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            lbl = map_label(parts[1].strip())
            if lbl is None:
                continue
            texts.append(parts[2].strip())
            labels.append(1 if lbl == 1 else 0)
            rows.append(parts)

    builder = FeatureBuilder(max_features=8_000, ngram_range=(1, 2))
    builder.fit(texts)
    X    = [builder.transform(t, r) for t, r in zip(texts, rows)]
    X_np = np.array(X,      dtype=np.float32)
    y_np = np.array(labels, dtype=np.int32)

    # SVM
    svm = Pipeline([
        ("scaler", MaxAbsScaler()),
        ("svm",    SVC(C=1.0, kernel="rbf", probability=True,
                       class_weight="balanced", random_state=42)),
    ])
    svm.fit(X_np, y_np)

    # MLP
    mlp_scaler = MaxAbsScaler()
    X_sc       = mlp_scaler.fit_transform(X_np)
    mlp        = MLPClassifier(
        hidden_layer_sizes=(256, 128), activation="relu",
        solver="adam", max_iter=50, early_stopping=True,
        random_state=42, verbose=False,
    )
    mlp.fit(X_sc, y_np)

    return {
        "builder"   : builder,
        "svm"       : svm,
        "mlp"       : mlp,
        "mlp_scaler": mlp_scaler,
    }


def run_predict(statement, model_name, models):
    import numpy as np
    from feature_extractor import retrieve_evidence

    t0 = time.perf_counter()

    # Use dummy row for metadata (neutral — speaker unknown)
    dummy_row          = [""] * 13
    dummy_row[8:13]    = ["0", "0", "0", "0", "0"]
    x_vec = models["builder"].transform(statement, row=dummy_row)
    x_np  = np.array(x_vec, dtype=np.float32).reshape(1, -1)

    if model_name == "SVM":
        label = int(models["svm"].predict(x_np)[0])
        conf  = float(models["svm"].predict_proba(x_np)[0][1])
    else:
        x_sc  = models["mlp_scaler"].transform(x_np)
        label = int(models["mlp"].predict(x_sc)[0])
        conf  = float(models["mlp"].predict_proba(x_sc)[0][1])

    evidence = retrieve_evidence(statement)
    elapsed  = time.perf_counter() - t0

    return {
        "label"     : "FAKE" if label == 1 else "REAL",
        "is_fake"   : label == 1,
        "confidence": conf,
        "evidence"  : evidence,
        "time_ms"   : elapsed * 1000,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:16px 0 8px 0">
        <div style="font-size:2rem">🔍</div>
        <div style="font-size:1.3rem;font-weight:700;color:#FFFFFF">Fake Finders</div>
        <div style="font-size:11px;color:#4A6A80;font-family:'JetBrains Mono',monospace;
                    letter-spacing:1px;margin-top:4px">NLP MASTER PROJECT</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:11px;font-weight:700;color:#4A6A80;
                letter-spacing:2px;text-transform:uppercase;margin-bottom:12px">
        All 7 Models
    </div>
    """, unsafe_allow_html=True)

    models_sidebar = [
        ("Naive Bayes",       "64.46%", "#00C896", False),
        ("Perceptron",        "57.91%", "#E85555", False),
        ("Logistic Reg",      "61.25%", "#00C896", False),
        ("SVM (RBF)",         "72.71%", "#2980B9", False),
        ("MLP (256→128)",     "75.42%", "#17A589", True),
        ("BERT fine-tuned",   "67.24%", "#A855F7", False),
        ("BERT + RAG",        "68.64%", "#F97316", False),
    ]

    for name, f1, color, best in models_sidebar:
        pct = float(f1.replace("%",""))
        if best:
            st.markdown(f"""
            <div class="model-card-best">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <div style="font-size:13px;font-weight:600;color:#17A589">{name} 🏆</div>
                    <div style="font-size:13px;font-weight:700;color:#17A589;
                                font-family:'JetBrains Mono',monospace">{f1}</div>
                </div>
                <div class="prog-wrap">
                    <div class="prog-fill" style="width:{pct}%;background:{color}"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="model-card">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <div style="font-size:12px;color:#A8C4D8">{name}</div>
                    <div style="font-size:12px;font-weight:600;color:{color};
                                font-family:'JetBrains Mono',monospace">{f1}</div>
                </div>
                <div class="prog-wrap">
                    <div class="prog-fill" style="width:{pct}%;background:{color};opacity:0.7"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:11px;font-weight:700;color:#4A6A80;
                letter-spacing:2px;text-transform:uppercase;margin-bottom:10px">Team</div>
    <div class="sidebar-name">👩‍💻 Akhila Pavithran</div>
    <div class="sidebar-name">👩‍💻 Rajana</div>
    <div style="font-size:11px;color:#4A6A80;margin-top:8px">
        Supervisor: Prof. Sabine Weber<br>
        Uni Bamberg · SS 2026
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:11px;color:#4A6A80;font-family:'JetBrains Mono',monospace">
        <a href="https://github.com/akhila-tech12/Fake_Finders_LIAR" 
           style="color:#17A589;text-decoration:none">
            🔗 GitHub Repository
        </a>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main Page
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="padding: 8px 0 24px 0">
    <div class="main-title">Fake Finders</div>
    <div class="main-subtitle">
        AUTOMATED FAKE NEWS DETECTION · LIAR DATASET · UNIVERSITY OF BAMBERG SS 2026
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍  Predict", "📊  All Results", "ℹ️  About"])


# ── Tab 1: Predict ────────────────────────────────────────────────────────────
with tab1:

    st.markdown("#### Enter a Political Statement")

    statement = st.text_area(
        "Statement input",
        placeholder='e.g. "Building a wall on the U.S.-Mexico border will take literally years."',
        height=110,
        label_visibility="collapsed",
    )

    col_model, col_btn = st.columns([3, 1])
    with col_model:
        model_choice = st.selectbox(
            "Model selection",
            ["SVM (F1: 72.71%)", "MLP (F1: 75.42%) ★ Best"],
            label_visibility="collapsed",
        )
    with col_btn:
        st.markdown("<div style='margin-top:4px'>", unsafe_allow_html=True)
        go = st.button("🔍 Analyse", use_container_width=True, type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    # Examples
    st.markdown("""
    <div style="font-size:12px;color:#4A6A80;margin:12px 0 8px 0;
                font-family:'JetBrains Mono',monospace;letter-spacing:1px">
        TRY AN EXAMPLE:
    </div>
    """, unsafe_allow_html=True)

    examples = [
        "Building a wall on the U.S.-Mexico border will take literally years.",
        "Wisconsin is on pace to double the number of layoffs this year.",
        "The federal government paid out 601 million in benefits to deceased employees.",
        "Hillary Clinton agrees with John McCain by voting to give George Bush the benefit of the doubt on Iran.",
        "The crime rate has doubled in the last five years.",
    ]

    ex_cols = st.columns(5)
    for i, (ec, ex) in enumerate(zip(ex_cols, examples)):
        with ec:
            if st.button(f"Example {i+1}", use_container_width=True, key=f"ex{i}"):
                statement = ex
                go        = True

    # BERT info
    st.markdown("""
    <div class="bert-box" style="margin-top:16px">
        <strong style="color:#A855F7">🤖 BERT & BERT+RAG</strong> — run on Google Colab GPU (not available for live prediction)<br>
        BERT fine-tuned: <strong>F1 = 67.24%</strong> &nbsp;·&nbsp; 
        BERT + RAG (Wikipedia evidence): <strong>F1 = 68.64%</strong><br>
        <span style="opacity:0.7;font-size:12px">
        Note: MLP with speaker metadata beats BERT text-only — proving external information is crucial!
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Run prediction
    if go and statement.strip():
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        model_key = "SVM" if "SVM" in model_choice else "MLP"

        with st.spinner(f"Analysing with {model_key}..."):
            try:
                models = load_models()
                result = run_predict(statement, model_key, models)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        r1, r2 = st.columns([1, 2])

        with r1:
            if result["is_fake"]:
                st.markdown(f"""
                <div class="fake-card">
                    <div style="font-size:3rem">🚨</div>
                    <div class="result-label" style="color:#E85555">FAKE</div>
                    <div class="result-conf" style="color:#E85555">
                        {result['confidence']:.0%} confident
                    </div>
                    <div class="result-meta">
                        {model_key} · {result['time_ms']:.0f}ms
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="real-card">
                    <div style="font-size:3rem">✅</div>
                    <div class="result-label" style="color:#17A589">REAL</div>
                    <div class="result-conf" style="color:#17A589">
                        {result['confidence']:.0%} confident
                    </div>
                    <div class="result-meta">
                        {model_key} · {result['time_ms']:.0f}ms
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with r2:
            st.markdown("**Statement analysed:**")
            st.markdown(f"""
            <div class="stmt-box">"{statement}"</div>
            """, unsafe_allow_html=True)

            if result["evidence"]:
                st.markdown("""
                <div style="font-size:12px;color:#4A6A80;margin:12px 0 6px 0;
                            font-family:'JetBrains Mono',monospace;letter-spacing:1px">
                    WIKIPEDIA EVIDENCE FOUND:
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="evidence-box">
                    📖 {result['evidence'][:350]}...
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="color:#4A6A80;font-size:13px;margin-top:12px">
                    ℹ️ No Wikipedia evidence found for this statement.
                </div>
                """, unsafe_allow_html=True)

    elif go:
        st.warning("Please enter a statement first!")


# ── Tab 2: All Results ────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### Final Model Comparison — LIAR Test Set (1,016 samples)")

    # Metric summary row
    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        ("75.42%", "Best F1 (MLP)", "#17A589"),
        ("10,198", "Total Samples", "#2980B9"),
        ("7",      "Models Built",  "#A855F7"),
        ("5.1 min","BERT Training", "#F97316"),
    ]
    for col, (val, lbl, color) in zip([m1,m2,m3,m4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-val" style="color:{color}">{val}</div>
                <div class="metric-lbl">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Full table
    import pandas as pd
    df = pd.DataFrame({
        "Model"     : ["1. Naive Bayes","2. Perceptron","3. Logistic Reg",
                       "4. SVM (RBF)","5. MLP (256→128) ★","6. BERT","7. BERT+RAG"],
        "Accuracy"  : ["62.01%","59.65%","61.02%","71.26%","69.59%","66.24%","67.72%"],
        "Precision" : ["66.04%","67.46%","67.17%","75.68%","67.62%","71.69%","73.27%"],
        "Recall"    : ["62.95%","50.72%","56.29%","69.96%","85.25%","63.31%","64.57%"],
        "F1 Score"  : ["64.46%","57.91%","61.25%","72.71%","75.42%","67.24%","68.64%"],
        "Time"      : ["0.04s","19.7s","97.3s","350s","3.7s","306s","3600s"],
        "Type"      : ["Statistical","Neural 1L","Statistical",
                       "Kernel","Neural 3L","Transformer","T+Retrieval"],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown("#### Key Findings")

    k1, k2 = st.columns(2)
    with k1:
        st.markdown("""
        <div class="model-card-best" style="padding:18px">
            <div style="font-size:13px;font-weight:700;color:#17A589;margin-bottom:8px">
                🏆 MLP beats BERT
            </div>
            <div style="font-size:13px;color:#A8C4D8;line-height:1.7">
                MLP uses <strong>speaker metadata</strong> from LIAR columns 8-12
                (historical credibility scores). BERT uses text only.<br><br>
                External information is more powerful than 
                better language understanding for short political statements!
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="model-card" style="padding:18px;margin-top:8px">
            <div style="font-size:13px;font-weight:700;color:#2980B9;margin-bottom:8px">
                📈 Evidence Retrieval Helps
            </div>
            <div style="font-size:13px;color:#A8C4D8;line-height:1.7">
                BERT alone: F1 = 67.24%<br>
                BERT + RAG (Wikipedia): F1 = 68.64%<br><br>
                +1.4% improvement from Wikipedia evidence retrieval.
                Directly addresses Prof Weber's feedback!
            </div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown("""
        <div class="model-card" style="padding:18px">
            <div style="font-size:13px;font-weight:700;color:#E85555;margin-bottom:8px">
                ❌ Perceptron Never Converged
            </div>
            <div style="font-size:13px;color:#A8C4D8;line-height:1.7">
                1,650 errors at epoch 10 — proves LIAR dataset 
                is NOT linearly separable.<br><br>
                This motivated SVM (max margin) and MLP 
                (non-linear hidden layers) which both significantly 
                outperform linear models.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="model-card" style="padding:18px;margin-top:8px">
            <div style="font-size:13px;font-weight:700;color:#A855F7;margin-bottom:8px">
                🤖 BERT on Colab GPU
            </div>
            <div style="font-size:13px;color:#A8C4D8;line-height:1.7">
                Fine-tuned bert-base-uncased on LIAR in 5.1 minutes 
                on Tesla P100 GPU.<br><br>
                Best val F1: 73.15% at epoch 2 before overfitting.
                Test F1: 67.24%.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Tab 3: About ──────────────────────────────────────────────────────────────
with tab3:
    a1, a2 = st.columns(2)

    with a1:
        st.markdown("#### About the Project")
        st.markdown("""
        **Fake Finders** is an NLP research project that automatically
        detects fake news in political statements using the
        LIAR benchmark dataset.

        **Dataset:** LIAR (Wang, ACL 2017)
        - 10,198 political statements from PolitiFact.com
        - Labels: pants-fire, false, barely-true → **FAKE**
        - Labels: mostly-true, true → **REAL**
        - Train: 8,146 · Valid: 1,036 · Test: 1,016

        **Two problems we solve:**
        1. **Language understanding** → bigrams, TF-IDF, MLP, BERT
        2. **Factual verification** → speaker metadata, Wikipedia RAG

        **Key insight:** Speaker credibility metadata from LIAR
        columns 8-12 is more powerful than BERT's language
        understanding for short political statements (avg 12 words).
        """)

    with a2:
        st.markdown("#### Team & Architecture")
        st.markdown("""
        **Team Fake Finders**
        - 👩‍💻 Akhila Pavithran
        - 👩‍💻 Rajana
        - 👩‍🏫 Supervisor: Prof. Sabine Weber
        - 🏛️ Otto-Friedrich-Universität Bamberg · SS 2026
        """)

        st.markdown("**System Architecture:**")
        st.code("""
Political Statement
        ↓
Tokenize → Bigrams → TF-IDF (8,000 features)
        ↓
+ Metadata: speaker fake rate, party, experience
        ↓
Classical Models:
  NB · Perceptron · LR (from scratch)
  SVM (RBF) · MLP (256→128)
        ↓
BERT fine-tuned (Google Colab GPU)
        ↓
BERT + RAG (Wikipedia evidence retrieval)
        ↓
FAKE or REAL + confidence %
        """, language=None)

        st.markdown(
            "🔗 [GitHub Repository](https://github.com/akhila-tech12/Fake_Finders_LIAR)"
        )
