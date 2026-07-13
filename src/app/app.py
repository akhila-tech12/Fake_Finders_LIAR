"""
app.py -- Fake Finders UI with BERT+FiLM
Run: streamlit run src/app/app.py
"""
import os, sys, time, subprocess, json
import streamlit as st
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add src/ to path

st.set_page_config(page_title="Fake Finders", page_icon="🔍", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif!important}
.stApp{background:#060E18;color:#fff}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding-top:2.5rem;padding-bottom:3rem;max-width:1280px}

/* ── bento tiles: every st.container(key="tile-…") becomes a card ── */
[class*="st-key-tile-"]{
    background:rgba(255,255,255,.03);
    border:1px solid rgba(255,255,255,.08);
    border-radius:20px;
    padding:22px 24px;
    height:100%;
    transition:transform .18s ease,border-color .18s ease,box-shadow .18s ease;
}
[class*="st-key-tile-"]:hover{
    transform:translateY(-2px);
    border-color:rgba(255,255,255,.18);
    box-shadow:0 8px 30px rgba(0,0,0,.35);
}

/* accent variants */
.st-key-tile-best{background:linear-gradient(135deg,rgba(168,85,247,.14),rgba(168,85,247,.04));border-color:rgba(168,85,247,.35)}
.st-key-tile-verdict-fake{background:rgba(232,85,85,.13);border-color:rgba(232,85,85,.45)}
.st-key-tile-verdict-real{background:rgba(23,165,137,.13);border-color:rgba(23,165,137,.45)}
.st-key-tile-evidence{background:rgba(41,128,185,.07);border-color:rgba(41,128,185,.3)}
.st-key-tile-score-5{background:linear-gradient(135deg,rgba(168,85,247,.14),rgba(168,85,247,.04));border-color:rgba(168,85,247,.35)}

/* small mono section labels */
.tile-label{font-size:10px;color:#4A6A80;letter-spacing:2px;font-family:'JetBrains Mono',monospace;margin-bottom:10px}

/* widgets flush with tiles */
.stTextArea textarea{background:rgba(255,255,255,.04)!important;border:1px solid rgba(255,255,255,.1)!important;border-radius:12px;color:#fff!important;caret-color:#fff;font-family:'Space Grotesk',sans-serif}
.stTextArea textarea::placeholder{color:#4A6A80}
.stTextArea [data-baseweb="textarea"]{background:transparent!important;border-color:rgba(255,255,255,.1)!important;border-radius:12px}
.stTextArea textarea:focus{border-color:rgba(168,85,247,.5)!important;box-shadow:none}
.stSelectbox [data-baseweb="select"]>div{background:rgba(255,255,255,.04);border-color:rgba(255,255,255,.1);border-radius:12px}
.stButton button{border-radius:12px}
.stButton button[kind="secondary"]{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.1);color:#A8C4D8;font-size:12px}
.stButton button[kind="secondary"]:hover{border-color:rgba(168,85,247,.5);color:#fff}
.stButton button[kind="primary"]{background:linear-gradient(135deg,#E85555,#A855F7);border:none;font-weight:700}

.ev-box{background:rgba(41,128,185,.06);border-left:3px solid #2980B9;border-radius:8px;padding:14px 18px;font-size:13px;color:#A8C4D8;line-height:1.7}
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
BERT_PATH  = os.path.join(BASE, "models", "bert_film_v3")
RAG_PATH   = os.path.join(BASE, "models", "bert_film_rag")
VENV_BERT  = os.path.join(BASE, "venv_bert", "bin", "python")
BERT_SCRIPT= os.path.join(BASE, "src", "bert", "bert_predict.py")
BERT_OK    = os.path.exists(BERT_PATH) and os.path.exists(VENV_BERT)
RAG_OK     = os.path.exists(RAG_PATH) and os.path.exists(VENV_BERT)


# ── Evidence retrieval ────────────────────────────────────────────────────────
# Prefer the v2 retriever (Wikipedia search + disambiguation filtering +
# lexical reranking — no torch in-process, matching the app's venv). Training
# evidence was reranked semantically (MiniLM); the app reranks lexically —
# an accepted train/serve gap for the demo, search + filtering are identical.
# Falls back to the legacy title-guess retrieval if anything goes wrong.
@st.cache_resource(show_spinner=False)
def _get_retriever():
    try:
        from bert.evidence_retriever import EvidenceRetriever
        return EvidenceRetriever(reranker="lexical")
    except Exception:
        return None


def get_evidence(statement: str) -> str:
    retriever = _get_retriever()
    if retriever is not None:
        try:
            return retriever.retrieve_text(statement)
        except Exception:
            pass
    from feature_extractor import retrieve_evidence
    return retrieve_evidence(statement)


# ── Speaker credibility lookup ────────────────────────────────────────────────
# LIAR rows carry each speaker's credit-history totals (columns 8-12). The
# demo's text box can't know who said a statement, so this map lets the user
# name the speaker and get the same metadata features the evaluation uses.
# Built from train.tsv only — test rows never inform the demo.
@st.cache_resource(show_spinner=False)
def load_speaker_history() -> dict:
    speakers: dict[str, list[str]] = {}
    train_path = os.path.join(BASE, "data", "train.tsv")
    try:
        with open(train_path, encoding="utf-8") as fh:
            for line in fh:
                row = line.rstrip("\n").split("\t")
                if len(row) < 13 or not row[4].strip():
                    continue
                name = row[4].strip().lower()
                try:
                    total = sum(int(row[i] or 0) for i in range(8, 13))
                except ValueError:
                    continue
                best = speakers.get(name)
                if best is None or total > sum(int(best[i] or 0) for i in range(8, 13)):
                    speakers[name] = row
    except Exception:
        pass
    return speakers


def resolve_speaker(name: str):
    """Return (meta_row, meta_str, fake_rate) for a speaker, or neutral defaults."""
    from feature_extractor import metadata_features, speaker_fake_rate
    neutral_row = [""] * 13
    neutral_row[8:13] = ["0", "0", "0", "0", "0"]
    key = name.strip().lower().replace(" ", "-")
    if not key:
        return neutral_row, "0.5,0.5,0.5", None
    row = load_speaker_history().get(key)
    if row is None:
        return neutral_row, "0.5,0.5,0.5", None
    meta = metadata_features(row)
    return row, ",".join(f"{m:.4f}" for m in meta), speaker_fake_rate(row)


# ── Load classical models once ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models... ~3 minutes first time only")
def load_models():
    import numpy as np, math
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import MaxAbsScaler
    from sklearn.pipeline import Pipeline
    from feature_extractor import FeatureBuilder
    from data_loader import map_label
    from classical import naive_bayes as nb_mod
    from classical import perceptron as perc_mod
    from classical import logistic_regression as lr_mod

    TRAIN = os.path.join(BASE, "data", "train.tsv")
    texts, labels, rows = [], [], []
    with open(TRAIN, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3: continue
            lbl = map_label(parts[1].strip())
            if lbl is None: continue
            texts.append(parts[2].strip())
            labels.append(lbl)
            rows.append(parts)

    builder = FeatureBuilder(max_features=8_000, ngram_range=(1,2))
    builder.fit(texts)
    X    = [builder.transform(t, r) for t, r in zip(texts, rows)]
    X_np = np.array(X, dtype=np.float32)
    y_01 = np.array([1 if l==1 else 0 for l in labels], dtype=np.int32)

    # NB
    train_data = list(zip(texts, labels))
    priors, likelihoods, vocab_nb = nb_mod.train(train_data)

    # Perceptron
    vocab_p = perc_mod.build_vocab(train_data)
    w_p, b_p = perc_mod.train(train_data, vocab_p, epochs=10)

    # LR
    vocab_lr = lr_mod.build_vocab(train_data)
    train_01 = lr_mod.convert_labels(train_data)
    w_lr, b_lr = lr_mod.train(train_01, vocab_lr, epochs=20, lr=0.1)

    # SVM
    svm = Pipeline([("sc", MaxAbsScaler()),
                    ("svm", SVC(C=1.0, kernel="rbf", probability=True,
                                class_weight="balanced", random_state=42))])
    svm.fit(X_np, y_01)

    # MLP
    mlp_sc = MaxAbsScaler()
    X_sc   = mlp_sc.fit_transform(X_np)
    mlp    = MLPClassifier(hidden_layer_sizes=(256,128), activation="relu",
                           solver="adam", max_iter=100, early_stopping=True,
                           random_state=42, verbose=False)
    mlp.fit(X_sc, y_01)

    return {
        "builder": builder,
        "nb": (priors, likelihoods, vocab_nb),
        "pc": (vocab_p, w_p, b_p),
        "lr": (vocab_lr, w_lr, b_lr),
        "svm": svm, "mlp": mlp, "mlp_sc": mlp_sc,
    }


def run_predict(statement, model_name, models,
                meta_row=None, meta_str="0.5,0.5,0.5"):
    import numpy as np, math
    from classical import naive_bayes as nb_mod
    from classical import perceptron as perc_mod
    from classical import logistic_regression as lr_mod

    dummy_row = meta_row
    if dummy_row is None:
        dummy_row = [""] * 13
        dummy_row[8:13] = ["0","0","0","0","0"]
    t0 = time.perf_counter()
    evidence = get_evidence(statement)

    if model_name == "Naive Bayes":
        priors, likelihoods, vocab_nb = models["nb"]
        pred_raw = nb_mod.predict(statement, priors, likelihoods, vocab_nb)
        s_f = nb_mod.score(statement,+1,priors,likelihoods,vocab_nb)
        s_r = nb_mod.score(statement,-1,priors,likelihoods,vocab_nb)
        m   = max(s_f, s_r)
        proba = math.exp(s_f-m)/(math.exp(s_f-m)+math.exp(s_r-m))
        pred  = 1 if pred_raw==1 else 0

    elif model_name == "Perceptron":
        vocab_p, w_p, b_p = models["pc"]
        x    = perc_mod.vectorize(statement, vocab_p)
        pred = 1 if perc_mod.predict(x, w_p, b_p)==1 else 0
        raw  = sum(wi*xi for wi,xi in zip(w_p,x)) + b_p
        proba = 1/(1+math.exp(-max(-500,min(500,raw/10))))

    elif model_name == "Logistic Reg":
        vocab_lr, w_lr, b_lr = models["lr"]
        x     = lr_mod.vectorize(statement, vocab_lr)
        proba = lr_mod.predict_proba(x, w_lr, b_lr)
        pred  = lr_mod.predict(x, w_lr, b_lr)

    elif model_name == "SVM":
        x_vec = models["builder"].transform(statement, dummy_row)
        x_np  = np.array(x_vec, dtype=np.float32).reshape(1,-1)
        pred  = int(models["svm"].predict(x_np)[0])
        proba = float(models["svm"].predict_proba(x_np)[0][1])

    elif model_name == "MLP":
        x_vec = models["builder"].transform(statement, dummy_row)
        x_np  = models["mlp_sc"].transform(np.array(x_vec,dtype=np.float32).reshape(1,-1))
        pred  = int(models["mlp"].predict(x_np)[0])
        proba = float(models["mlp"].predict_proba(x_np)[0][1])

    elif model_name == "BERT+FiLM ★" and BERT_OK:
        try:
            result = subprocess.run(
                [VENV_BERT, BERT_SCRIPT, statement,
                 BERT_PATH, meta_str],
                capture_output=True, text=True, timeout=120
            )
            # Get last line of output (JSON)
            lines = [l for l in result.stdout.strip().split('\n') if l.startswith('{')]
            if lines:
                data  = json.loads(lines[-1])
                pred  = 1 if data["is_fake"] else 0
                proba = data["confidence"]
            else:
                # Print error to terminal for debugging
                import sys
                print(f"BERT STDOUT: {result.stdout}", file=sys.stderr)
                print(f"BERT STDERR: {result.stderr[:200]}", file=sys.stderr)
                print(f"BERT RETURNCODE: {result.returncode}", file=sys.stderr)
                pred = 0; proba = 0.5
        except Exception as e:
            pred = 0; proba = 0.5
    elif model_name == "BERT+RAG ★" and BERT_OK:
        try:
            result = subprocess.run(
                [VENV_BERT, BERT_SCRIPT, statement,
                BERT_PATH, meta_str, evidence],
                capture_output=True, text=True, timeout=120
            )
            lines = [l for l in result.stdout.strip().split('\n')
                    if l.startswith('{')]
            if lines:
                data  = json.loads(lines[-1])
                pred  = 1 if data["is_fake"] else 0
                proba = data["confidence"]
            else:
                pred = 0; proba = 0.5
        except Exception as e:
            pred = 0; proba = 0.5
    elif model_name == "BERT+FiLM+RAG ★" and RAG_OK:
        # Model 10: pair-encoded evidence (bert_predict.py switches encoding
        # via the model dir's model_config.json)
        try:
            result = subprocess.run(
                [VENV_BERT, BERT_SCRIPT, statement,
                RAG_PATH, meta_str, evidence],
                capture_output=True, text=True, timeout=120
            )
            lines = [l for l in result.stdout.strip().split('\n')
                    if l.startswith('{')]
            if lines:
                data  = json.loads(lines[-1])
                pred  = 1 if data["is_fake"] else 0
                proba = data["confidence"]
            else:
                pred = 0; proba = 0.5
        except Exception as e:
            pred = 0; proba = 0.5
    else:
        x_vec = models["builder"].transform(statement, dummy_row)
        x_np  = models["mlp_sc"].transform(np.array(x_vec,dtype=np.float32).reshape(1,-1))
        pred  = int(models["mlp"].predict(x_np)[0])
        proba = float(models["mlp"].predict_proba(x_np)[0][1])
        model_name = "MLP (fallback)"

    elapsed = time.perf_counter() - t0
    return {
        "label"     : "FAKE" if pred==1 else "REAL",
        "is_fake"   : pred==1,
        "confidence": proba,
        "evidence"  : evidence,
        "time_ms"   : elapsed*1000,
        "model_used": model_name,
    }


# ── Bento grid ────────────────────────────────────────────────────────────────

def _fill_example(ex):
    st.session_state["statement"] = ex

# Row 1 · hero + best-model highlight
h1, h2 = st.columns([3, 1], gap="medium")
with h1:
    with st.container(key="tile-hero"):
        st.markdown("""
        <div style="font-size:2.6rem;font-weight:700;line-height:1.1;
                    background:linear-gradient(135deg,#fff,#A855F7);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent">
            Fake Finders
        </div>
        <div style="font-size:12px;color:#4A6A80;font-family:'JetBrains Mono',monospace;
                    letter-spacing:1px;margin-top:8px">
            AUTOMATED FAKE NEWS DETECTION · LIAR DATASET · UNIVERSITY OF BAMBERG SS 2026
        </div>
        <div style="font-size:12px;color:#A8C4D8;margin-top:10px">
            NLP Master Project · Supervisor: Prof. Sabine Weber
        </div>""", unsafe_allow_html=True)
with h2:
    with st.container(key="tile-best"):
        st.markdown("""
        <div class="tile-label">BEST MODEL</div>
        <div style="font-size:1.6rem">🏆</div>
        <div style="font-size:1.05rem;font-weight:700;color:#A855F7">BERT+FiLM ★</div>
        <div style="font-size:1.8rem;font-weight:800;font-family:'JetBrains Mono',monospace">78.18%</div>
        <div style="font-size:11px;color:#4A6A80">F1 · LIAR test set</div>""",
        unsafe_allow_html=True)

# Row 2 · statement input + controls
p1, p2 = st.columns([2, 1], gap="medium")
with p1:
    with st.container(key="tile-input"):
        st.markdown('<div class="tile-label">ENTER A POLITICAL STATEMENT</div>',
                    unsafe_allow_html=True)
        statement = st.text_area("stmt", key="statement",
            placeholder='e.g. "Building a wall will take literally years."',
            height=110, label_visibility="collapsed")

        examples = [
            "Building a wall on the U.S.-Mexico border will take literally years.",
            "Wisconsin is on pace to double the number of layoffs this year.",
            "The federal government paid out 601 million in benefits to deceased employees.",
            "Hillary Clinton agrees with John McCain by voting to give George Bush the benefit of the doubt on Iran.",
            "The crime rate has doubled in the last five years.",
        ]
        st.markdown('<div class="tile-label" style="margin-top:6px">EXAMPLES</div>',
                    unsafe_allow_html=True)
        ecols = st.columns(5)
        for i, (ec, ex) in enumerate(zip(ecols, examples)):
            ec.button(f"Ex {i+1}", use_container_width=True, key=f"ex{i}",
                      on_click=_fill_example, args=(ex,))
with p2:
    with st.container(key="tile-controls"):
        st.markdown('<div class="tile-label">MODEL</div>', unsafe_allow_html=True)
        model_options = ["SVM", "MLP", "Naive Bayes",
                         "Perceptron", "Logistic Reg"]
        if BERT_OK:
            model_options.append("BERT+FiLM ★")
            model_options.append("BERT+RAG ★")
        if RAG_OK:
            model_options.append("BERT+FiLM+RAG ★")
        model_choice = st.selectbox("model", model_options,
                                    label_visibility="collapsed")
        st.markdown('<div class="tile-label" style="margin-top:6px">SPEAKER '
                    '(OPTIONAL)</div>', unsafe_allow_html=True)
        speaker_name = st.text_input("speaker", key="speaker",
            placeholder="e.g. facebook-posts, mitt-romney",
            label_visibility="collapsed")
        meta_row, meta_str, known_rate = resolve_speaker(speaker_name)
        if speaker_name.strip():
            if known_rate is not None:
                st.caption(f"✓ known speaker — historical fake rate "
                           f"{known_rate:.0%} (from training data)")
            else:
                st.caption("speaker not in training data — using neutral metadata")
        go = st.button("🔍 Check", use_container_width=True, type="primary")
        if not BERT_OK:
            st.info("ℹ️ BERT+FiLM model not found in models/bert_film_v3/")

# Row 3 · prediction result (verdict / confidence / evidence)
if go and statement.strip():
    spinner_msg = "Analysing with BERT+FiLM... ~5 seconds" \
                  if "BERT" in model_choice else "Analysing..."
    with st.spinner(spinner_msg):
        models_loaded = load_models()
        result = run_predict(statement, model_choice, models_loaded,
                             meta_row=meta_row, meta_str=meta_str)

    r1, r2, r3 = st.columns([1, 1, 2], gap="medium")
    verdict_key   = "tile-verdict-fake" if result["is_fake"] else "tile-verdict-real"
    verdict_color = "#E85555" if result["is_fake"] else "#17A589"
    verdict_icon  = "🚨" if result["is_fake"] else "✅"
    with r1:
        with st.container(key=verdict_key):
            st.markdown(f"""
            <div class="tile-label">VERDICT</div>
            <div style="text-align:center">
                <div style="font-size:2.6rem">{verdict_icon}</div>
                <div style="font-size:2.2rem;font-weight:800;color:{verdict_color}">
                    {result['label']}</div>
            </div>""", unsafe_allow_html=True)
    with r2:
        with st.container(key="tile-confidence"):
            st.markdown(f"""
            <div class="tile-label">CONFIDENCE</div>
            <div style="text-align:center">
                <div style="font-size:2.4rem;font-weight:800;color:{verdict_color};
                            font-family:'JetBrains Mono',monospace">
                    {result['confidence']:.0%}</div>
                <div style="font-size:11px;color:#4A6A80;font-family:'JetBrains Mono',monospace">
                    {result['model_used']} · {result['time_ms']:.0f}ms</div>
            </div>""", unsafe_allow_html=True)
    with r3:
        with st.container(key="tile-evidence"):
            st.markdown('<div class="tile-label">STATEMENT &amp; EVIDENCE</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:13px;color:#A8C4D8;font-style:italic;'
                        f'margin-bottom:10px">"{statement}"</div>',
                        unsafe_allow_html=True)
            if result["evidence"]:
                st.markdown(f'<div class="ev-box">📖 {result["evidence"][:600]}...</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown("ℹ️ No Wikipedia evidence found.")
elif go:
    st.warning("Please enter a statement!")

# Row 4 · model score tiles (formerly the sidebar list)
models_info = [
    ("Naive Bayes",   "64.46%", "#00C896"),
    ("Perceptron",    "57.91%", "#E85555"),
    ("Logistic Reg",  "61.25%", "#00C896"),
    ("SVM",           "72.71%", "#2980B9"),
    ("MLP",           "75.42%", "#17A589"),
    ("BERT+FiLM ★",   "78.18%", "#A855F7"),
]
score_cols = st.columns(6, gap="small")
for i, (sc, (name, f1, color)) in enumerate(zip(score_cols, models_info)):
    with sc:
        with st.container(key=f"tile-score-{i}"):
            st.markdown(f"""
            <div style="font-size:11px;color:#A8C4D8;white-space:nowrap;
                        overflow:hidden;text-overflow:ellipsis">{name}</div>
            <div style="font-size:1.25rem;font-weight:800;color:{color};
                        font-family:'JetBrains Mono',monospace">{f1}</div>""",
            unsafe_allow_html=True)

# Row 5 · comparison table + key findings
t1, t2 = st.columns([2, 1], gap="medium")
with t1:
    with st.container(key="tile-table"):
        st.markdown('<div class="tile-label">FINAL MODEL COMPARISON — LIAR TEST SET</div>',
                    unsafe_allow_html=True)
        import pandas as pd
        df = pd.DataFrame({
            "Model"    : ["1. Naive Bayes","2. Perceptron","3. Logistic Reg",
                          "4. SVM","5. MLP",
                          "6. BERT text-only","7. BERT+RAG",
                          "8. BERT+metadata","9. BERT+FiLM ★"],
            "F1 Score" : ["64.46%","57.91%","61.25%",
                          "72.71%","75.42%",
                          "67.24%","68.64%",
                          "67.43%","78.18%"],
            "Type"     : ["Statistical","Neural 1L","Statistical",
                          "Kernel","Neural 3L",
                          "Transformer","T+Wikipedia",
                          "T+Concat","T+FiLM"],
            "Metadata" : ["❌","❌","❌","✅","✅",
                          "❌","❌","✅","✅"],
            "Live Demo": ["✅","✅","✅","✅","✅",
                          "❌","✅","❌","✅"],
        })
        st.success("🏆 Best: BERT+FiLM differential LR — F1 = 78.18%")
        st.dataframe(df, use_container_width=True, hide_index=True)
with t2:
    with st.container(key="tile-find-1"):
        st.markdown('<div class="tile-label">KEY FINDINGS</div>', unsafe_allow_html=True)
        st.markdown("""
        **🏆 FiLM + Differential LR = Best Result**
        - Uniform LR: F1 = 70.29%
        - Differential LR (FiLM=1e-3, BERT=2e-5): F1 = 78.18%
        - Early stopping found optimal at epoch 3

        **📊 Metadata is crucial**
        - Without metadata: BERT 67.24%
        - With FiLM fusion: 78.18% (+10.94%!)
        """)
    with st.container(key="tile-find-2"):
        st.markdown("""
        **❌ Simple concat barely helped**
        - BERT text-only: 67.24%
        - BERT + concat: 67.43% (+0.19%)
        - FiLM fixed this properly!

        **🔍 Short statements hardest**
        - ≤8 words → 45-47% error rate
        - Motivates RAG evidence retrieval
        """)

# Row 6 · about / team / pipeline footer
f1, f2, f3 = st.columns([2, 1, 1], gap="medium")
with f1:
    with st.container(key="tile-about"):
        st.markdown('<div class="tile-label">ABOUT FAKE FINDERS</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        Automated fake news detection using LIAR benchmark
        dataset (Wang, ACL 2017). 10,198 political statements
        from PolitiFact.com.

        **7 Models:** NB · Perceptron · LR (from scratch)
        SVM · MLP (sklearn) · BERT · BERT+FiLM

        **Key innovation:** FiLM (Feature-wise Linear Modulation)
        with differential learning rates allows metadata to
        meaningfully reshape BERT's 768-dim representations.
        """)
with f2:
    with st.container(key="tile-team"):
        st.markdown('<div class="tile-label">TEAM</div>', unsafe_allow_html=True)
        st.markdown("""
        👩‍💻 **Akhila Pavithran**
        👩‍💻 **Rajana**
        👩‍🏫 Supervisor: Prof. Sabine Weber
        🏛️ Uni Bamberg · SS 2026

        [🔗 GitHub](https://github.com/akhila-tech12/Fake_Finders_LIAR)
        """)
with f3:
    with st.container(key="tile-pipeline"):
        st.markdown('<div class="tile-label">PIPELINE</div>', unsafe_allow_html=True)
        st.code("""
Input → Tokenize → TF-IDF + Metadata
     → NB/Perc/LR/SVM/MLP
     → BERT fine-tuned
     → BERT + FiLM (gamma*cls + beta)
     → FAKE or REAL + confidence
        """, language=None)
