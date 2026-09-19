import os
import string
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS styling for modern and sleek presentation
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        padding: 1.25rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 1.5rem;
        animation: fadeIn 0.4s ease-in-out;
    }
    .spam-badge {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 2px solid #F87171;
    }
    .ham-badge {
        background-color: #DCFCE7;
        color: #166534;
        border: 2px solid #4ADE80;
    }
    .cleaned-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.8rem;
        font-family: monospace;
        font-size: 0.9rem;
        color: #334155;
        margin-top: 0.5rem;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords', quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Text cleaning function (identical to the notebook implementation)
def clean_text(text: str) -> str:
    # 1. Convert to lowercase
    text = text.lower()
    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 3. Tokenize by whitespace
    words = text.split()
    # 4. Remove stopwords and stem remaining words
    stemmed_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(stemmed_words)

# Load model artifacts with Streamlit caching
@st.cache_resource
def load_artifacts():
    tfidf_path = os.path.join("model", "tfidf.joblib")
    model_path = os.path.join("model", "model.joblib")
    
    if not os.path.exists(tfidf_path) or not os.path.exists(model_path):
        st.error("Model artifacts not found in `model/`. Please execute notebook.ipynb first.")
        st.stop()
        
    tfidf = joblib.load(tfidf_path)
    model = joblib.load(model_path)
    return tfidf, model

tfidf, model = load_artifacts()

# App Header
st.markdown("<div class='main-title'>🛡️ SMS Spam Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Detect spam messages using a Linear Support Vector Machine trained on TF-IDF features</div>", unsafe_allow_html=True)

# Sample message quick loader
sample_spam = "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
sample_ham = "Hey, are we still meeting up for coffee around 4pm today? Let me know!"

st.write("##### Quick Samples:")
col_s1, col_s2, col_clear = st.columns([1, 1, 1])

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

if col_s1.button("📋 Load Spam Sample"):
    st.session_state.input_text = sample_spam

if col_s2.button("📋 Load Ham Sample"):
    st.session_state.input_text = sample_ham

if col_clear.button("🗑️ Clear Input"):
    st.session_state.input_text = ""

# Message input area
user_message = st.text_area(
    "Enter SMS message to check:",
    value=st.session_state.input_text,
    placeholder="Paste or type your SMS message here...",
    height=130,
    key="message_input"
)

# Check button
if st.button("🔍 Check", type="primary", use_container_width=True):
    if not user_message.strip():
        st.warning("⚠️ Please enter a message before clicking Check.")
    else:
        # Preprocess text
        cleaned = clean_text(user_message)
        
        # Transform with fitted TF-IDF
        features = tfidf.transform([cleaned])
        
        # Predict using Linear SVM
        prediction = model.predict(features)[0]
        
        # Display Result
        if prediction == 1:
            st.markdown(
                "<div class='result-card spam-badge'>🚨 SPAM</div>",
                unsafe_allow_html=True
            )
            st.error("This message has been identified as spam or an unsolicited promotional offer.")
        else:
            st.markdown(
                "<div class='result-card ham-badge'>✅ NOT SPAM</div>",
                unsafe_allow_html=True
            )
            st.success("This message has been identified as legitimate (ham).")
            
        with st.expander("Details: Cleaned Text & Diagnostics"):
            st.write("**Preprocessed message tokens:**")
            st.markdown(f"<div class='cleaned-box'>{cleaned if cleaned else '(empty after stopwords removal)'}</div>", unsafe_allow_html=True)
            if hasattr(model, "decision_function"):
                decision_score = model.decision_function(features)[0]
                st.write(f"**SVM Decision Margin:** `{decision_score:.4f}` (positive = spam, negative = ham)")
