import streamlit as st
from transformers import AutoTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf

# Page config
st.set_page_config(
    page_title="💬 Analyse de Sentiment",
    page_icon="🎭",
    layout="centered",
)

# load local CSS
def load_css(path: str):
    with open(path, 'r') as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# inject CSS
load_css("style.css")

# ——— MODEL LOADING ———
@st.cache_resource
def load_model_and_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model     = TFDistilBertForSequenceClassification.from_pretrained(path)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer(
    "C:/Users\DELL/OneDrive/Bureau/Master/Semestre 2/Fouilles_de_données/sentiment_analysis/model_distilBERT"
)

# ——— PREDICTION FUNCTION ———
def predict_sentiment(text):
    inputs  = tokenizer(text, return_tensors="tf", padding=True, truncation=True)
    logits  = model(**inputs).logits
    probs   = tf.nn.softmax(logits, axis=-1).numpy().flatten()  # [neg, neu, pos]
    return probs[2], probs[1], probs[0]                         # (pos, neu, neg)

# ——— UI LAYOUT ———
st.markdown('<div class="title">✨ Analyse de Sentiment ✨</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Tapez un commentaire en anglais pour savoir s’il est positif, neutre ou négatif.</div>', unsafe_allow_html=True)

with st.expander("🔤 Entrer le texte à analyser", expanded=True):
    st.markdown('<div class="input-area">', unsafe_allow_html=True)
    input_text = st.text_area("Votre texte :", placeholder="Tapez votre texte ici...", height=150)
    st.markdown('</div>', unsafe_allow_html=True)

if st.button("🔍 Prédire le sentiment"):
    if not input_text.strip():
        st.warning("⚠️ Veuillez entrer un texte avant de prédire.")
    else:
        pos, neu, neg = predict_sentiment(input_text)
        # bars
        st.markdown("### Résultats :")
        for label, score, cls in zip(["Positif","Neutre","Négatif"], [pos,neu,neg], ["positive","neutral","negative"]):
            pct = f"{score*100:.1f}%"
            bar = f'<div class="bar-container"><div class="bar {cls}" style="width:{score*100:.1f}%">{pct}</div></div>'
            st.markdown(bar, unsafe_allow_html=True)

        # final sentiment
        best = max([(pos,"positif","😊"), (neu,"neutre","😐"), (neg,"négatif","😔")], key=lambda x: x[0])
        label, emoji = best[1], best[2]
        if label=="positif":
            st.success(f"Le sentiment est **positif** {emoji}")
        elif label=="neutre":
            st.info   (f"Le sentiment est **neutre** {emoji}")
        else:
            st.error  (f"Le sentiment est **négatif** {emoji}")

st.markdown('<div class="footer">🚀 Développé avec ❤️ par Ndeye Khary Souhaibou Touré 🧕🏿🕋 & Sokhna Maimounatou Kabyr NDIAYE🌟</div>', unsafe_allow_html=True)
