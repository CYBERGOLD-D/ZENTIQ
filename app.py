import streamlit as st
import joblib
import json
from pathlib import Path

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Nigerian Retail Support",
    page_icon="🛍️",
    layout="centered"
)

# --------------------------------------------------
# 2. PROJECT AND MODEL PATH CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"


# --------------------------------------------------
# 3. LOAD SAVED MODEL COMPONENTS
# --------------------------------------------------

@st.cache_resource
def load_components():

    vectorizer_path = MODEL_DIR / "zentIQ_tfidf_vectorizer.pkl"
    model_path = MODEL_DIR / "zentIQ_logistic_regression_model.pkl"
    mapping_path = MODEL_DIR / "zentIQ_intent_response_mapping.json"

    # Verify that all required model files exist
    required_files = [
        vectorizer_path,
        model_path,
        mapping_path
    ]

    for file_path in required_files:
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Required model file not found: {file_path}"
            )

    # Load TF-IDF vectorizer
    vectorizer = joblib.load(vectorizer_path)

    # Load Logistic Regression model
    model = joblib.load(model_path)

    # Load intent-response mapping
    with open(
        mapping_path,
        "r",
        encoding="utf-8"
    ) as file:
        response_mapping = json.load(file)

    return vectorizer, model, response_mapping


# Load chatbot components
try:
    vectorizer, model, response_mapping = load_components()

except Exception as error:
    st.error(f"Unable to load chatbot components: {error}")
    st.stop()


# --------------------------------------------------
# 4. PREDICTION FUNCTION
# --------------------------------------------------

def predict_customer_query(question):

    question_tfidf = vectorizer.transform([question])

    predicted_intent = model.predict(question_tfidf)[0]

    probabilities = model.predict_proba(question_tfidf)[0]
    confidence = max(probabilities) * 100

    response = response_mapping.get(
        predicted_intent,
        "Sorry, I could not find a suitable response."
    )

    return predicted_intent, response, confidence


# --------------------------------------------------
# 5. APPLICATION HEADER
# --------------------------------------------------

st.title("🛍️ Nigerian Retail Support")
st.subheader("Your Intelligent Retail Support Assistant")

st.write(
    "Welcome! Ask questions about your retail business, "
    "sales, payments, inventory, deliveries and more. "
    "I'm here to help."
)

st.divider()


# --------------------------------------------------
# 6. SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("About the Assistant")

    st.write(
        "Nigerian Retail Support is a machine-learning "
        "powered chatbot designed to assist Nigerian "
        "retail business owners."
    )

    st.subheader("Available Support Areas")

    st.markdown("""
    - Business Expenses
    - Customer Acquisition
    - Daily Sales
    - Damaged Products
    - Delivery Costs and Delays
    - Inventory Reordering
    - Order Tracking
    - Payment Confirmation
    - Payment Methods
    - POS Issues
    - Profit Calculation
    - Record Keeping
    - Refund Requests
    - Selling Prices
    - Social Media Marketing
    - Stock Availability
    - Supplier Onboarding
    - Supplier Price Changes
    """)

    st.divider()

    st.caption("Powered by TF-IDF and Logistic Regression.")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------
# 7. CHAT HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":
            if "intent" in message:

                st.caption(
                    f"Predicted Intent: {message['intent']}"
                )

                st.caption(
                    f"Prediction Confidence: "
                    f"{message['confidence']:.2f}%"
                )


# --------------------------------------------------
# 8. CUSTOMER INPUT AND CHATBOT RESPONSE
# --------------------------------------------------

customer_question = st.chat_input(
    "Ask your retail support question..."
)

if customer_question:

    st.session_state.messages.append({
        "role": "user",
        "content": customer_question
    })

    with st.chat_message("user"):
        st.markdown(customer_question)

    with st.chat_message("assistant"):

        with st.spinner("Analysing your question..."):

            predicted_intent, response, confidence = (
                predict_customer_query(customer_question)
            )

        st.markdown(response)

        st.caption(
            f"Predicted Intent: {predicted_intent}"
        )

        st.caption(
            f"Prediction Confidence: {confidence:.2f}%"
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "intent": predicted_intent,
        "confidence": confidence
    })


# --------------------------------------------------
# 9. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Nigerian Retail Support | Machine Learning Project"
)

st.caption(
    "Designed to support better retail business operations."
)