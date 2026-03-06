import streamlit as st
import torch
import joblib
import requests
from streamlit_lottie import st_lottie
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Movie AI Studio",
    page_icon="🎬",
    layout="centered"
)

# ---------------- HIDE DEFAULT UI ----------------
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stApp {
    background: radial-gradient(circle at center, #0f0f0f, #000000);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- CINEMATIC INTRO ----------------
if "intro_done" not in st.session_state:
    st.session_state.intro_done = False

if not st.session_state.intro_done:

    intro = requests.get(
        "https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json"
    ).json()

    st_lottie(intro, height=400)

    st.markdown(
        "<h2 style='text-align:center;'>🎬 Welcome to Movie AI</h2>",
        unsafe_allow_html=True
    )

    if st.button("Start Prediction 🚀"):
        st.session_state.intro_done = True
        st.rerun()

    st.stop()

# ---------------- LOAD MODEL ----------------
model_path = "movie_genre_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

mlb = joblib.load("label_binarizer.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ---------------- MAIN UI ----------------
st.markdown("""
<h1 style='text-align:center; font-size:42px;'>
🎬 Movie Genre Predictor
</h1>
<p style='text-align:center; color:gray;'>
AI-powered cinematic classification system
</p>
""", unsafe_allow_html=True)

text = st.text_area("Enter Movie Overview", height=150)

if st.button("Analyze with AI 🚀"):

    if text.strip() == "":
        st.warning("Please enter movie description.")
    else:

        with st.spinner("Analyzing cinematic data..."):

            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            inputs.pop("token_type_ids", None)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            probs = torch.sigmoid(outputs.logits)
            probs = probs.cpu().numpy()[0]

            threshold = 0.3
            predicted_indices = [i for i, p in enumerate(probs) if p > threshold]

        if predicted_indices:

            st.success("🎥 Predicted Genres")

            for i in predicted_indices:
                st.markdown(f"""
                <div style="
                    background:#111;
                    padding:12px;
                    border-radius:10px;
                    margin-top:8px;
                    border-left:4px solid red;
                    animation: fadeIn 0.8s ease-in-out;">
                    🔥 {mlb.classes_[i]}
                </div>

                <style>
                @keyframes fadeIn {{
                    from {{opacity:0; transform:translateY(15px);}}
                    to {{opacity:1; transform:translateY(0);}}
                }}
                </style>
                """, unsafe_allow_html=True)

        else:
            st.info("No strong genre detected.")