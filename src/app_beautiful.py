"""
app.py -- Fake Finders Final UI with BERT+FiLM
Run: streamlit run src/app.py
"""
import os, sys, time, subprocess, json
import streamlit as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Fake Finders", page_icon="🔍", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif!important}
.stApp{background:#060E18;color:#fff}
#MainMenu,footer,header{visibility:hidden}
.fake-card{background:rgba(232,85,85,.15);border:1px solid rgba(232,85,85,.4);border-radius:16px;padding:28px 20px;text-align:center}
.real-card{background:rgba(23,165,137,.15);border:1px solid rgba(23,165,137,.4);border-radius:16px;padding:28px 20px;text-align:center}
.ev-box{background:rgba(41,128,185,.06);border:1px solid rgba(41,128,185,.25);border-left:3px solid #2980B9;border-radius:8px;padding:14px 18px;font-size:13px;color:#A8C4D8;line-height:1.7}
.model-card{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);border-radius:10px;padding:14px;margin-bottom:6px}
.best-card{background:rgba(168,85,247,.08);border:1px solid rgba(168,85,247,.3);border-radius:10px;padding:14px;margin-bottom:6px}
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = os.path.expanduser("~/fake-finders-liar")
BERT_PATH  = os.path.join(BASE, "models", "bert_film_v3")
VENV_BERT  = os.path.join(BASE, "venv_bert", "bin", "python")
BERT_SCRIPT= os.path.join(BASE, "src", "bert_predict.py")
BERT_OK    = os.path.exists(BERT_PATH) and os.path.exists(VENV_BERT)

# Debug - remove after fixing
import sys
print(f"DEBUG BASE={BASE}", file=sys.stderr)
print(f"DEBUG BERT_OK={BERT_OK}", file=sys.stderr)
print(f"DEBUG BERT_PATH exists={os.path.exists(BERT_PATH)}", file=sys.stderr)
print(f"DEBUG VENV_BERT exists={os.path.exists(VENV_BERT)}", file=sys.stderr)

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
    import naive_bayes as nb_mod
    import perceptron as perc_mod
    import logistic_regression as lr_mod

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


def run_predict(statement, model_name, models):
    import numpy as np, math
    from feature_extractor import retrieve_evidence
    import naive_bayes as nb_mod
    import perceptron as perc_mod
    import logistic_regression as lr_mod

    dummy_row = [""] * 13
    dummy_row[8:13] = ["0","0","0","0","0"]
    t0 = time.perf_counter()
    evidence = retrieve_evidence(statement)

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
                 BERT_PATH, "0.5,0.5,0.5"],
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
                BERT_PATH, "0.5,0.5,0.5", evidence],
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


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:12px 0">
        <div style="font-size:2rem">🔍</div>
        <div style="font-size:1.2rem;font-weight:700">Fake Finders</div>
        <div style="font-size:10px;color:#4A6A80;font-family:monospace;letter-spacing:1px">
            NLP MASTER PROJECT
        </div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    models_info = [
        ("Naive Bayes",   "64.46%", "#00C896", False),
        ("Perceptron",    "57.91%", "#E85555", False),
        ("Logistic Reg",  "61.25%", "#00C896", False),
        ("SVM",           "72.71%", "#2980B9", False),
        ("MLP",           "75.42%", "#17A589", False),
        ("BERT+FiLM ★",   "78.18%", "#A855F7", True),
    ]
    st.markdown("<div style='font-size:10px;color:#4A6A80;letter-spacing:2px;font-family:monospace;margin-bottom:8px'>ALL MODELS</div>", unsafe_allow_html=True)
    for name, f1, color, best in models_info:
        css = "best-card" if best else "model-card"
        st.markdown(f"""
        <div class="{css}">
            <div style="display:flex;justify-content:space-between">
                <div style="font-size:12px;color:#A8C4D8">{name}</div>
                <div style="font-size:12px;font-weight:700;color:{color};
                            font-family:monospace">{f1}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="font-size:12px;color:#A8C4D8">👩‍💻 Akhila Pavithran</div>
    <div style="font-size:12px;color:#A8C4D8">👩‍💻 Rajana</div>
    <div style="font-size:11px;color:#4A6A80;margin-top:6px">
        Supervisor: Prof. Sabine Weber<br>Uni Bamberg · SS 2026
    </div>""", unsafe_allow_html=True)
    st.markdown("[🔗 GitHub](https://github.com/akhila-tech12/Fake_Finders_LIAR)")


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:8px 0 24px 0">
    <div style="font-size:3rem;font-weight:700;background:linear-gradient(135deg,#fff,#A855F7);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent">
        Fake Finders
    </div>
    <div style="font-size:13px;color:#4A6A80;font-family:monospace;letter-spacing:1px">
        AUTOMATED FAKE NEWS DETECTION · LIAR DATASET · UNIVERSITY OF BAMBERG SS 2026
    </div>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍  Predict", "📊  Results", "ℹ️  About"])

with tab1:
    st.markdown("#### Enter a Political Statement")
    statement = st.text_area("stmt",
        placeholder='e.g. "Building a wall will take literally years."',
        height=100, label_visibility="collapsed")

    model_options = ["SVM", "MLP", "Naive Bayes",
                     "Perceptron", "Logistic Reg"]
    if BERT_OK:
        model_options.append("BERT+FiLM ★")
        model_options.append("BERT+RAG ★")

    c1, c2 = st.columns([3,1])
    with c1:
        model_choice = st.selectbox("model", model_options,
                                    label_visibility="collapsed")
    with c2:
        go = st.button("🔍 Check", use_container_width=True, type="primary")

    st.markdown("**Examples:**")
    examples = [
        "Building a wall on the U.S.-Mexico border will take literally years.",
        "Wisconsin is on pace to double the number of layoffs this year.",
        "The federal government paid out 601 million in benefits to deceased employees.",
        "Hillary Clinton agrees with John McCain by voting to give George Bush the benefit of the doubt on Iran.",
        "The crime rate has doubled in the last five years.",
    ]
    ecols = st.columns(5)
    for i,(ec,ex) in enumerate(zip(ecols,examples)):
        with ec:
            if st.button(f"Ex {i+1}", use_container_width=True, key=f"ex{i}"):
                statement = ex; go = True

    if not BERT_OK:
        st.info("ℹ️ BERT+FiLM model not found in models/bert_film_v3/")

    if go and statement.strip():
        st.divider()
        spinner_msg = "Analysing with BERT+FiLM... ~5 seconds" \
                      if "BERT" in model_choice else "Analysing..."
        with st.spinner(spinner_msg):
            models_loaded = load_models()
            result = run_predict(statement, model_choice, models_loaded)

        r1, r2 = st.columns([1,2])
        with r1:
            if result["is_fake"]:
                st.markdown(f"""<div class="fake-card">
                    <div style="font-size:3rem">🚨</div>
                    <div style="font-size:2.5rem;font-weight:800;color:#E85555">FAKE</div>
                    <div style="color:#E85555">{result['confidence']:.0%} confident</div>
                    <div style="font-size:11px;color:#4A6A80;margin-top:8px;font-family:monospace">
                        {result['model_used']} · {result['time_ms']:.0f}ms</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="real-card">
                    <div style="font-size:3rem">✅</div>
                    <div style="font-size:2.5rem;font-weight:800;color:#17A589">REAL</div>
                    <div style="color:#17A589">{result['confidence']:.0%} confident</div>
                    <div style="font-size:11px;color:#4A6A80;margin-top:8px;font-family:monospace">
                        {result['model_used']} · {result['time_ms']:.0f}ms</div>
                </div>""", unsafe_allow_html=True)
        with r2:
            st.info(f'"{statement}"')
            if result["evidence"]:
                st.markdown("**Wikipedia evidence:**")
                st.markdown(f'<div class="ev-box">📖 {result["evidence"][:600]}...</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown("ℹ️ No Wikipedia evidence found.")

    elif go:
        st.warning("Please enter a statement!")

with tab2:
    st.markdown("### Final Model Comparison — LIAR Test Set")
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
    st.success("🏆 Best: BERT+FiLM with differential LR — F1 = 78.18%")

    st.markdown("### Key Findings")
    k1, k2 = st.columns(2)
    with k1:
        st.markdown("""
        **🏆 FiLM + Differential LR = Best Result**
        - Uniform LR: F1 = 70.29%
        - Differential LR (FiLM=1e-3, BERT=2e-5): F1 = 78.18%
        - Early stopping found optimal at epoch 3

        **📊 Metadata is crucial**
        - Without metadata: BERT 67.24%
        - With FiLM fusion: 78.18% (+10.94%!)
        """)
    with k2:
        st.markdown("""
        **❌ Simple concat barely helped**
        - BERT text-only: 67.24%
        - BERT + concat: 67.43% (+0.19%)
        - FiLM fixed this properly!

        **🔍 Short statements hardest**
        - ≤8 words → 45-47% error rate
        - Motivates RAG evidence retrieval
        """)

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        ### About Fake Finders
        Automated fake news detection using LIAR benchmark
        dataset (Wang, ACL 2017). 10,198 political statements
        from PolitiFact.com.

        **7 Models:** NB · Perceptron · LR (from scratch)
        SVM · MLP (sklearn) · BERT · BERT+FiLM

        **Key innovation:** FiLM (Feature-wise Linear Modulation)
        with differential learning rates allows metadata to
        meaningfully reshape BERT's 768-dim representations.
        """)
    with c2:
        st.markdown("""
        ### Team
        👩‍💻 **Akhila Pavithran**
        👩‍💻 **Rajana**
        👩‍🏫 Supervisor: Prof. Sabine Weber
        🏛️ Uni Bamberg · SS 2026

        [🔗 GitHub](https://github.com/akhila-tech12/Fake_Finders_LIAR)
        """)
        st.code("""
Input → Tokenize → TF-IDF + Metadata
     → NB/Perc/LR/SVM/MLP
     → BERT fine-tuned
     → BERT + FiLM (gamma*cls + beta)
     → FAKE or REAL + confidence
        """, language=None)