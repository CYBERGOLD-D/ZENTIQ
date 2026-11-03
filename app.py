
"""
ZENTIQ - Intelligent Business Assistant
Hybrid AI Streamlit Application

Mode 1: Machine Learning Retail Support
Mode 2: Hybrid AI Assistant (Groq + Ollama)
"""

import sys
import json
import joblib
import streamlit as st

from pathlib import Path


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ZENTIQ - Intelligent Business Assistant",
    page_icon="🤖",
    layout="centered"
)


# --------------------------------------------------
# 2. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


# --------------------------------------------------
# 3. LOAD MACHINE LEARNING COMPONENTS
# --------------------------------------------------

@st.cache_resource
def load_components():

    vectorizer_path = MODEL_DIR / "zentIQ_tfidf_vectorizer.pkl"
    model_path = MODEL_DIR / "zentIQ_logistic_regression_model.pkl"
    mapping_path = MODEL_DIR / "zentIQ_intent_response_mapping.json"

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

    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)

    with open(mapping_path, "r", encoding="utf-8") as file:
        response_mapping = json.load(file)

    return vectorizer, model, response_mapping


try:
    vectorizer, model, response_mapping = load_components()

except Exception as error:
    st.error(f"Unable to load ML chatbot components: {error}")
    st.stop()


# --------------------------------------------------
# 4. MACHINE LEARNING PREDICTION FUNCTION
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
# 5. LOAD HYBRID AI SERVICE
# --------------------------------------------------

@st.cache_resource
def load_llm_service():

    from ai_engine.llm_service import LLMService

    return LLMService()


# --------------------------------------------------
# 6. APPLICATION HEADER
# --------------------------------------------------

st.title("🤖 ZENTIQ")

st.subheader("Smarter Business. Better Decisions.")

st.write(
    "Welcome to ZENTIQ, your intelligent business assistant. "
    "Get business support, explore ideas, and find practical "
    "solutions to everyday business challenges."
)

st.divider()


# --------------------------------------------------
# 7. SIDEBAR AND ASSISTANT SELECTION
# --------------------------------------------------

with st.sidebar:

    st.title("ZENTIQ Assistant")

    assistant_mode = st.radio(
        "Choose your assistant:",
        [
            "ZENTIQ AI Assistant",
            "Machine Learning Retail Support"
        ]
    )

    st.divider()

    if assistant_mode == "ZENTIQ AI Assistant":

        st.subheader("Hybrid AI Business Assistance")

        st.write(
            "Powered by Groq Cloud as the primary AI provider "
            "and Ollama as the secondary provider."
        )

        prompt_options = {
            "General Business": "general",
            "Customer Support": "customer_support",
            "Financial Assistance": "financial",
            "Sales Assistance": "sales"
        }

        selected_prompt = st.selectbox(
            "Choose assistance area:",
            list(prompt_options.keys())
        )

        template_name = prompt_options[selected_prompt]

        history_key = "zentiq_ai_messages"

    else:

        st.subheader("Retail Support")

        st.write(
            "Powered by TF-IDF and Logistic Regression."
        )

        st.markdown("""
        **Available Support Areas**

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

        history_key = "ml_retail_messages"

    st.divider()

    if st.button("Clear Current Chat", use_container_width=True):

        st.session_state[history_key] = []
        st.rerun()

    st.caption("ZENTIQ | Intelligent Business Platform")


# --------------------------------------------------
# 8. INITIALISE SEPARATE CHAT HISTORIES
# --------------------------------------------------

if "zentiq_ai_messages" not in st.session_state:
    st.session_state.zentiq_ai_messages = []

if "ml_retail_messages" not in st.session_state:
    st.session_state.ml_retail_messages = []


messages = st.session_state[history_key]


# --------------------------------------------------
# 9. DISPLAY CHAT HISTORY
# --------------------------------------------------

for message in messages:

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

            if "provider" in message:

                st.caption(
                    f"AI Provider: {message['provider']}"
                )


# --------------------------------------------------
# 10. USER INPUT
# --------------------------------------------------

if assistant_mode == "ZENTIQ AI Assistant":

    placeholder = "Ask ZENTIQ about your business..."

else:

    placeholder = "Ask your retail support question..."


user_question = st.chat_input(placeholder)


# --------------------------------------------------
# 11. PROCESS USER QUESTION
# --------------------------------------------------

if user_question:

    messages.append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):

        if assistant_mode == "ZENTIQ AI Assistant":

            try:

                with st.spinner("ZENTIQ is thinking..."):

                    llm = load_llm_service()

                    response = llm.generate_response(
                        prompt=user_question,
                        task="general",
                        template_name=template_name,
                        temperature=0.3
                    )

                    provider = (
                        llm.model_manager.active_provider
                        or "Hybrid AI"
                    )

                st.markdown(response)

                st.caption(
                    f"AI Provider: {provider.upper()}"
                )

                messages.append({
                    "role": "assistant",
                    "content": response,
                    "provider": provider
                })

            except Exception as error:

                st.error(
                    "ZENTIQ AI could not generate a response. "
                    "Please check your AI provider configuration "
                    "and network connectivity."
                )

                st.caption(f"Technical details: {error}")

        else:

            try:

                with st.spinner("Analysing your question..."):

                    predicted_intent, response, confidence = (
                        predict_customer_query(user_question)
                    )

                st.markdown(response)

                st.caption(
                    f"Predicted Intent: {predicted_intent}"
                )

                st.caption(
                    f"Prediction Confidence: {confidence:.2f}%"
                )

                messages.append({
                    "role": "assistant",
                    "content": response,
                    "intent": predicted_intent,
                    "confidence": confidence
                })

            except Exception as error:

                st.error(
                    f"Unable to process your question: {error}"
                )


# --------------------------------------------------
# 12. APPLICATION FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "ZENTIQ | Smarter Business. Better Decisions. "
    "Unlimited Possibilities."
)
