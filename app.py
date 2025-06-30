import streamlit as st
import joblib
from PIL import Image

# Load saved components
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# ----- Banner Image -----
image = Image.open("image.png")
st.image(image, use_container_width=True)


# ----- Header -----
st.markdown(
    "<h1 style='text-align: center; color: #38bdf8;'>📰 Fake News Detection App</h1>",
    unsafe_allow_html=True
)


# ----- Info Box -----
st.markdown(
    """
    <div style="background-color: #1e293b; padding: 1rem; border-left: 6px solid #38bdf8;
    border-radius: 10px; margin-top: 1rem; margin-bottom: 2rem;">
        <span style="color: #f1f5f9; font-size: 15px;">
        <b>ℹ️ Note:</b> This model is trained primarily on full article content.
        For best results, please enter a complete news article or a meaningful excerpt.
        Headlines may give less accurate results due to limited context.
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

# ----- Input Area -----
st.markdown("### 🖊️ Paste the news content below:")
user_input = st.text_area("", height=220, placeholder="Enter complete news article...")

# ----- Predict Button -----
if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        predicted_label = label_encoder.inverse_transform([prediction])[0]

        if predicted_label == "FAKE":
            st.error("❌ This news appears to be *Fake*.")
        else:
            st.success("✅ This news appears to be *Real*.")

# ----- Footer -----
st.markdown(
    "<hr style='margin-top: 3rem;'>"
    "<div style='text-align: center; color: gray;'>"
    "Built with ❤️ by Sneha Pandey | Powered by Streamlit & Scikit-learn"
    "</div>",
    unsafe_allow_html=True
)



