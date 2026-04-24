import streamlit as st
import joblib
import time
import matplotlib.pyplot as plt
from preprocess import clean_text
from wordcloud import WordCloud

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Spam Detector", page_icon="📧", layout="wide")

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Settings")
dark_mode = st.sidebar.toggle("🌙 Dark Mode")
show_wordcloud = st.sidebar.checkbox("Show WordCloud", value=True)
show_chart = st.sidebar.checkbox("Show Probability Chart", value=True)

# ---------------- CUSTOM CSS ----------------
if dark_mode:
    bg_color = "#0e1117"
    text_color = "white"
else:
    bg_color = "white"
    text_color = "black"

st.markdown(f"""
<style>
body {{
    background-color: {bg_color};
    color: {text_color};
}}
.title {{
    font-size:42px;
    font-weight:800;
    text-align:center;
    color:#ff4b2b;
}}
.card {{
    padding:20px;
    border-radius:12px;
    box-shadow:0px 4px 10px rgba(0,0,0,0.2);
    text-align:center;
}}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">📧 Advanced Spam Detection System</div>', unsafe_allow_html=True)
st.write("")

# ---------------- LOAD MODEL ----------------
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except:
    st.error("⚠️ Run train.py first")
    st.stop()

# ---------------- INPUT OPTIONS ----------------
option = st.radio("Choose Input Method:", ["✍️ Manual Input", "📁 Upload File"])

if option == "✍️ Manual Input":
    user_input = st.text_area("Enter Email Text", height=150)
else:
    uploaded_file = st.file_uploader("Upload .txt file", type=["txt"])
    if uploaded_file:
        user_input = uploaded_file.read().decode("utf-8")
        st.text_area("File Content", user_input, height=150)
    else:
        user_input = ""

# ---------------- ANALYZE ----------------
if st.button("🚀 Analyze Email"):

    if not user_input.strip():
        st.warning("⚠️ Please provide input")
    else:
        with st.spinner("Analyzing with AI... 🤖"):
            time.sleep(1)

            cleaned = clean_text(user_input)
            vectorized = vectorizer.transform([cleaned])

            prediction = model.predict(vectorized)[0]
            prob = model.predict_proba(vectorized)[0]

        st.write("---")

        col1, col2, col3 = st.columns(3)

        # ---------------- RESULT CARDS ----------------
        if prediction == 1:
            col1.error("🚨 SPAM")
            confidence = prob[1]
        else:
            col1.success("✅ NOT SPAM")
            confidence = prob[0]

        col2.metric("Confidence", f"{confidence*100:.2f}%")
        col3.metric("Spam Score", f"{prob[1]*100:.2f}%")

        # ---------------- PROGRESS ----------------
        st.progress(float(confidence))

        # ---------------- CHART ----------------
        if show_chart:
            st.subheader("📊 Probability Distribution")
            labels = ['Not Spam', 'Spam']
            values = [prob[0], prob[1]]

            fig, ax = plt.subplots()
            ax.bar(labels, values)
            st.pyplot(fig)

        # ---------------- WORD CLOUD ----------------
        if show_wordcloud:
            st.subheader("☁️ WordCloud Analysis")
            wc = WordCloud(width=600, height=300, background_color='white').generate(cleaned)
            st.image(wc.to_array())

        # ---------------- EXPLANATION ----------------
        st.subheader("🧠 Why this prediction?")
        spam_keywords = ["free", "win", "click", "offer", "urgent", "money"]

        found = [word for word in spam_keywords if word in cleaned]

        if found:
            st.warning(f"Detected suspicious keywords: {', '.join(found)}")
        else:
            st.info("No strong spam keywords detected.")

        # ---------------- TIPS ----------------
        if prediction == 1:
            st.error("⚠️ Avoid clicking unknown links!")
        else:
            st.success("👍 Safe communication detected")

# ---------------- FOOTER ----------------
st.write("---")
st.caption("🚀 Built with ML + Streamlit | Advanced UI Version")