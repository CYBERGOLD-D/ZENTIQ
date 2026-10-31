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
# 2. LOAD SAVED MODEL COMPONENTS
# --------------------------------------------------

dataset_dir = Path(
    r"C:\Users\everm\Documents_Local\ZENTIQ\datasets"
)

processed_dir = dataset_dir / "processed"

@st.cache_resource
def load_components():

    vectorizer = joblib.load(
        processed_dir / "zentIQ_tfidf_vectorizer.pkl"
    )

    model = joblib.load(
        processed_dir / "zentIQ_logistic_regression_model.pkl"
    )

    with open(
        processed_dir / "zentIQ_intent_response_mapping.json",
        "r",
        encoding="utf-8"
    ) as file:
        response_mapping = json.load(file)

    return vectorizer, model, response_mapping


try:
    vectorizer, model, response_mapping = load_components()
except Exception as error:
    st.error(f"Unable to load chatbot components: {error}")
    st.stop()


# --------------------------------------------------
# 3. PREDICTION FUNCTION
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
# 4. APPLICATION HEADER
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
# 5. SIDEBAR
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
# 6. CHAT HISTORY
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
# 7. CUSTOMER INPUT AND CHATBOT RESPONSE
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

        st.caption(f"Predicted Intent: {predicted_intent}")

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
# 8. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Nigerian Retail Support | Machine Learning Project"
)

st.caption(
    "Designed to support better retail business operations."
)