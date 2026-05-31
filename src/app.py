"""
app.py
======
Fake Finders — Simple Web UI
Does NOT retrain models on every load.
Models are cached — only trained ONCE at startup.

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

st.set_page_config(
    page_title="Fake Finders",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
.fake-result {
    background: rgba(232,85,85,0.15);
    border: 2px solid #E85555;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.real-result {
    background: rgba(0,200,150,0.15);
    border: 2px solid #00C896;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models... takes ~2 minutes on first load only!")
def load_models():
    """Train SVM and MLP once — cached forever after first load."""
    import numpy as np
    from sklearn.svm            import SVC
    from sklearn.preprocessing  import MaxAbsScaler
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline       import Pipeline
    from feature_extractor      import FeatureBuilder
    from data_loader            import map_label

    BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TRAIN_PATH = os.path.join(BASE, "data", "train.tsv")

    # Load training data
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

    # Build features
    builder = FeatureBuilder(max_features=8_000, ngram_range=(1, 2))
    builder.fit(texts)
    X    = [builder.transform(t, r) for t, r in zip(texts, rows)]
    X_np = np.array(X,      dtype=np.float32)
    y_np = np.array(labels, dtype=np.int32)

    # Train SVM
    svm = Pipeline([
        ("scaler", MaxAbsScaler()),
        ("svm",    SVC(C=1.0, kernel="rbf", probability=True,
                       class_weight="balanced", random_state=42)),
    ])
    svm.fit(X_np, y_np)

    # Train MLP
    mlp_scaler = MaxAbsScaler()
    X_sc       = mlp_scaler.fit_transform(X_np)
    mlp        = MLPClassifier(
        hidden_layer_sizes=(256, 128), activation="relu",
        solver="adam", max_iter=50, early_stopping=True,
        random_state=42, verbose=False,
    )
    mlp.fit(X_sc, y_np)

    return {"builder": builder, "svm": svm,
            "mlp": mlp, "mlp_scaler": mlp_scaler}


def run_predict(statement, model_name, models):
    import numpy as np
    from feature_extractor import retrieve_evidence

    t0    = time.perf_counter()
    dummy_row = [""] * 13
    dummy_row[8:13] = ["0","0","0","0","0"]
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


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Fake Finders")
    st.markdown("NLP Master Project · Bamberg SS 2026")
    st.divider()
    st.markdown("### 📊 Model Results")
    model_results = [
        ("Naive Bayes",      "64.46%", False),
        ("Perceptron",       "57.91%", False),
        ("Logistic Reg",     "61.25%", False),
        ("SVM (RBF)",        "72.71%", False),
        ("MLP (256→128)",    "75.42%", True),
        ("BERT",             "67.24%", False),
        ("BERT + RAG",       "68.64%", False),
    ]
    for name, f1, best in model_results:
        if best:
            st.markdown(f"**{name} → {f1} 🏆**")
        else:
            st.markdown(f"{name} → {f1}")
    st.divider()
    st.markdown("Akhila Pavithran · Rajana")


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🔍 Fake Finders")
st.caption("Fake News Detection · University of Bamberg · SS 2026")

tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📊 All Results", "ℹ️ About"])

with tab1:
    st.markdown("### Check a Statement")

    statement = st.text_area(
        "",
        placeholder='e.g. "Building a wall on the border will take years."',
        height=100,
        label_visibility="collapsed",
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        model_choice = st.selectbox("", ["SVM", "MLP"],
                                    label_visibility="collapsed")
    with c2:
        go = st.button("🔍 Check", use_container_width=True, type="primary")

    st.markdown("**Examples:**")
    examples = [
        "Building a wall on the U.S.-Mexico border will take literally years.",
        "Wisconsin is on pace to double the number of layoffs this year.",
        "The federal government paid out 601 million in benefits to deceased employees.",
        "Hillary Clinton agrees with John McCain on Iran.",
        "The crime rate has doubled in the last five years.",
    ]
    ecols = st.columns(5)
    for i, (ec, ex) in enumerate(zip(ecols, examples)):
        with ec:
            if st.button(f"Ex {i+1}", use_container_width=True):
                statement = ex
                go        = True

    if go and statement.strip():
        st.divider()
        with st.spinner("Analysing..."):
            models = load_models()
            result = run_predict(statement, model_choice, models)

        r1, r2 = st.columns([1, 2])
        with r1:
            if result["is_fake"]:
                st.markdown(f"""
                <div class="fake-result">
                    <div style="font-size:48px">🚨</div>
                    <div style="font-size:30px;font-weight:800;color:#E85555">FAKE</div>
                    <div style="color:#E85555">{result['confidence']:.0%} confident</div>
                    <div style="font-size:11px;color:#888;margin-top:6px">
                        {model_choice} · {result['time_ms']:.0f}ms
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="real-result">
                    <div style="font-size:48px">✅</div>
                    <div style="font-size:30px;font-weight:800;color:#00C896">REAL</div>
                    <div style="color:#00C896">{result['confidence']:.0%} confident</div>
                    <div style="font-size:11px;color:#888;margin-top:6px">
                        {model_choice} · {result['time_ms']:.0f}ms
                    </div>
                </div>""", unsafe_allow_html=True)

        with r2:
            st.info(f'"{statement}"')
            if result["evidence"]:
                st.markdown("**Wikipedia evidence:**")
                st.markdown(f"> {result['evidence'][:300]}...")
            else:
                st.markdown("ℹ️ No Wikipedia evidence found.")

    elif go:
        st.warning("Please enter a statement!")


with tab2:
    st.markdown("### Final Comparison — LIAR Test Set (1,016 samples)")
    import pandas as pd
    df = pd.DataFrame({
        "Model"    : ["1. Naive Bayes","2. Perceptron","3. Logistic Reg",
                      "4. SVM","5. MLP ★","6. BERT","7. BERT+RAG"],
        "Accuracy" : ["62.01%","59.65%","61.02%","71.26%","69.59%","66.24%","67.72%"],
        "F1 Score" : ["64.46%","57.91%","61.25%","72.71%","75.42%","67.24%","68.64%"],
        "Time"     : ["0.04s","19.7s","97.3s","350s","3.7s","306s","3600s"],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.success("🏆 Best: MLP (256→128) — F1 = 75.42%")
    st.info("💡 MLP beats BERT because it uses speaker metadata (credibility scores). "
            "This directly confirms Prof Weber's feedback: external information beyond text is crucial!")


with tab3:
    st.markdown("### About Fake Finders")
    st.markdown("""
    **Dataset:** LIAR Benchmark (Wang, ACL 2017)
    - 10,198 political statements from PolitiFact.com
    - Binary: FAKE vs REAL

    **7 Models implemented:**
    - Naive Bayes, Perceptron, Logistic Regression (from scratch)
    - SVM (RBF kernel), MLP (256→128)
    - BERT fine-tuned (HuggingFace)
    - BERT + RAG (Wikipedia evidence retrieval)

    **Key finding:** MLP + metadata beats BERT text-only.
    External speaker credibility information is more valuable
    than better language understanding for short political statements.

    **Team:** Akhila Pavithran · Rajana
    **Supervisor:** Prof. Sabine Weber
    **University:** Otto-Friedrich-Universität Bamberg · SS 2026
    """)
