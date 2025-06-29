import streamlit as st
import joblib

# Load model, vectorizer, and label encoder
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Set page configuration
st.set_page_config(
    page_title="Fake News Detection App",
    page_icon="📰",
    layout="centered"
)

st.markdown(
    """
    <h2 style='text-align:center; color:#2e7d32;'>📰 Fake News Detection App</h2>
    
    """,
    unsafe_allow_html=True
)



st.markdown("""
<style>
@media (prefers-color-scheme: dark) {
  .description {
    color: #e0e0e0;
  }
}
@media (prefers-color-scheme: light) {
  .description {
    color: #333333;
  }
}
</style>

<p class='description' style='text-align: center; font-size: 17px; line-height: 1.6;'>
Welcome to the <strong>Fake News Detection App</strong> — a lightweight, AI-powered tool trained on real-world news articles.<br><br>
Enter any news paragraph or headline below to check whether it is <strong>Real</strong> or <strong>Fake</strong>, along with a confidence score.
</p>
""", unsafe_allow_html=True)




st.markdown("---")

# Text Input
user_input = st.text_area("📝 Paste News Content Here:", height=200, help="Use full article or paragraph for better accuracy.")

# Prediction
if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some news content to analyze.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        label = label_encoder.inverse_transform([prediction])[0]

        if label == "FAKE":
            st.error("❌ This news appears to be **Fake**.")
        else:
            st.success("✅ This news appears to be **Real**.")

        # Confidence Score
        confidence = model.decision_function(input_vector)[0]
        st.caption(f"🧠 Confidence Score: {round(abs(confidence), 2)}")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 13px; color: gray;'>Built by Sneha Pandey | Powered by Streamlit & Scikit-Learn</p>",
    unsafe_allow_html=True
)


