import json
import string
import nltk
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- Download NLTK Data ---
nltk.download('punkt')
nltk.download('stopwords')

# --- Load FAQs ---
with open("faq_data.json", "r") as file:
    faqs = json.load(file)

faq_questions = list(faqs.keys())
faq_answers = list(faqs.values())

# --- Preprocessing Function ---
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in stopwords.words('english') and t not in string.punctuation]
    return " ".join(tokens)

# --- Create TF-IDF Vectors ---
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform([preprocess(q) for q in faq_questions])

# --- Find Best Answer ---
def get_answer(user_input):
    user_vector = vectorizer.transform([preprocess(user_input)])
    similarity = cosine_similarity(user_vector, faq_vectors)
    index = similarity.argmax()
    score = similarity[0][index]
    if score < 0.3:
        return "😕 Sorry, I couldn't find an answer for that."
    return faq_answers[index]

# --- Streamlit Page Config ---
st.set_page_config(page_title="Cafe Delight Chatbot", page_icon="☕", layout="centered")

# --- Custom CSS (Updated to remove top bar) ---
st.markdown("""
    <style>
    
    /* Hide Streamlit default top header & toolbar 
    header[data-testid="stHeader"] {
        display: none !important;
    }
    div[data-testid="stToolbar"] {
        display: none !important;
    }*/

    /*Remove top padding & margin */
    .block-container {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    .stApp {
        background-color: #f8f9fa;
        font-family: 'Poppins', sans-serif;
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    .stTextInput, .stTextInput > div {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    label, .stTextInput label {
        display: none !important;
    }

    .chat-container {
        max-width: 650px;
        margin: auto;
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .chat-header {
        text-align: center;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f1f1f1;
    }

    .chat-header h1 {
        color: #4b3832;
        font-size: 2rem;
        margin-bottom: 0.2rem;
    }

    .chat-header h3 {
        color: #7b6f67;
        font-weight: 400;
    }

    .message {
        padding: 12px 18px;
        border-radius: 12px;
        margin: 10px 0;
        line-height: 1.5;
        max-width: 85%;
    }

    .user-msg {
        background-color: #d9edf7;
        align-self: flex-end;
        border-top-right-radius: 0;
        margin-left: auto;
    }

    .bot-msg {
        background-color: #f7f4ea;
        border-left: 5px solid #d4a373;
        color: #333;
        border-top-left-radius: 0;
    }

    .footer {
        text-align: center;
        font-size: 0.85rem;
        color: #aaa;
        margin-top: 2rem;
    }

    input[type="text"] {
        flex: 1;
        border-radius: 10px !important;
        border: 1px solid #ccc !important;
        padding: 12px !important;
        background-color: #fff !important;
        box-shadow: none !important;
    }

    button[kind="primary"] {
        background-color: #d4a373 !important;
        color: white !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
    }

    </style>
""", unsafe_allow_html=True)

# --- Chat Layout ---
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

st.markdown("""
    <div class='chat-header'>
        <h1>☕ Cafe Delight Chatbot</h1>
        <h3>Welcome! Ask me anything about our cafe services 🍰</h3>
    </div>
""", unsafe_allow_html=True)

# --- Input at the bottom only ---
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("", placeholder="Type your question here...", label_visibility="collapsed")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    answer = get_answer(user_input)
    st.markdown(f"<div class='message user-msg'><b>You:</b> {user_input}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='message bot-msg'><b>Bot:</b> {answer}</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>Powered by Cafe Delight © 2025</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
