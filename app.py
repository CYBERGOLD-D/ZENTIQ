""" ZENTIQ - Intelligent Business Assistant Hybrid AI Streamlit Application PRODUCTION AI: Primary: Cloudflare Workers AI - Gemma 4 Fallback: Groq Cloud LOCAL DEVELOPMENT AI: Primary: Ollama - Llama 3.2:1B Fallback: Groq Cloud SAFETY: Production: Cloudflare Llama Guard 3 8B Local development: Ollama Llama Guard 3:1B MODES: 1. Machine Learning Retail Support 2. Hybrid AI Assistant FEATURES: - Supabase email authentication - User-specific persistent conversations - Conversation recovery and retry saving - Incoming and outgoing safety screening in both modes - PDF conversation export Developed by EverMark Creative Solutions. """

import sys
import json
import io

import joblib
import streamlit as st

from pathlib import Path
from datetime import datetime, timezone
from xml.sax.saxutils import escape


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="ZENTIQ - Intelligent Business Assistant",
    page_icon="ðŸ¤–",
    layout="centered"
)


# ==================================================
# 2. PROJECT CONFIGURATION
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


# ==================================================
# 3. DATABASE AND AUTHENTICATION IMPORTS
# ==================================================

from database.auth import (
    register_user,
    login_user
)

from database.session_manager import (
    get_authenticated_client,
    save_authenticated_session,
    logout_authenticated_user,
    USER_ID_KEY,
    USER_EMAIL_KEY
)

from database.repositories import (
    create_conversation,
    get_conversations,
    get_conversation,
    update_conversation
)


# ==================================================
# 4. LOAD MACHINE LEARNING COMPONENTS
# ==================================================

@st.cache_resource
def load_components():
    """Load the trained TF-IDF model and response mapping."""

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

    if not isinstance(response_mapping, dict):
        raise ValueError(
            "The intent-response mapping must be a JSON object."
        )

    return vectorizer, model, response_mapping


try:
    vectorizer, model, response_mapping = load_components()

except Exception as error:
    st.error(
        "Unable to load Machine Learning components. "
        "Check that all three model files exist and are valid."
    )
    st.exception(error)
    st.stop()


# ==================================================
# 5. MACHINE LEARNING PREDICTION
# ==================================================

def predict_customer_query(question):
    """Predict a retail-support intent and return its response."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("The customer question cannot be empty.")

    question_tfidf = vectorizer.transform([question])
    predicted_intent = model.predict(question_tfidf)[0]
    probabilities = model.predict_proba(question_tfidf)[0]
    confidence = float(max(probabilities) * 100)

    response = response_mapping.get(
        predicted_intent,
        "Sorry, I could not find a suitable response."
    )

    if not isinstance(response, str) or not response.strip():
        raise ValueError(
            "The Machine Learning model returned an invalid response."
        )

    return predicted_intent, response.strip(), confidence


# ==================================================
# 6. LOAD HYBRID AI SERVICE
# ==================================================

def load_llm_service():
    """ Create an LLM service for the current request. This is intentionally not globally cached because the service records the active provider after generation. A globally shared, mutable provider field could cause concurrent sessions to display another request's provider. """

    from ai_engine.llm_service import LLMService

    return LLMService()


# ==================================================
# 7. AUTHENTICATION INTERFACE
# ==================================================

def show_authentication_page(client):
    """Display the sign-in and account-registration interface."""

    st.title("ðŸ¤– ZENTIQ")
    st.subheader(
        "Welcome to your Intelligent Business Assistant"
    )

    st.write(
        "Sign in to access your business assistant "
        "and securely manage your conversation history."
    )

    login_tab, register_tab = st.tabs(
        ["Sign In", "Create Account"]
    )

    # ----------------------------------------------
    # SIGN IN
    # ----------------------------------------------

    with login_tab:
        with st.form("zentiq_login_form"):
            email = st.text_input(
                "Email address",
                placeholder="Enter your email"
            )

            password = st.text_input(
                "Password",
                type="password"
            )

            login_button = st.form_submit_button(
                "Sign In",
                use_container_width=True
            )

        if login_button:
            if not email.strip() or not password:
                st.error(
                    "Please enter your email address and password."
                )
            else:
                try:
                    user, session = login_user(
                        client,
                        email.strip(),
                        password
                    )

                    if session is None:
                        st.error(
                            "No active session was returned. "
                            "Please confirm your email address "
                            "before signing in."
                        )

                    elif user is None:
                        st.error(
                            "Unable to identify your account."
                        )

                    else:
                        save_authenticated_session(session)
                        st.session_state[USER_ID_KEY] = user.id
                        st.session_state[USER_EMAIL_KEY] = (
                            user.email or email.strip()
                        )

                        st.success("Login successful.")
                        st.rerun()

                except Exception:
                    st.error(
                        "Unable to sign in. Check your email, "
                        "password and account verification."
                    )

    # ----------------------------------------------
    # CREATE ACCOUNT
    # ----------------------------------------------

    with register_tab:
        with st.form("zentiq_register_form"):
            new_email = st.text_input(
                "Email address",
                placeholder="Enter your email"
            )

            new_password = st.text_input(
                "Password",
                type="password",
                help="Use at least 8 characters."
            )

            confirm_password = st.text_input(
                "Confirm password",
                type="password"
            )

            register_button = st.form_submit_button(
                "Create Account",
                use_container_width=True
            )

        if register_button:
            if not new_email.strip():
                st.error("Please enter your email address.")

            elif len(new_password) < 8:
                st.error(
                    "Your password must contain at least 8 characters."
                )

            elif new_password != confirm_password:
                st.error("Your passwords do not match.")

            else:
                try:
                    user, session = register_user(
                        client,
                        new_email.strip(),
                        new_password
                    )

                    if session is None:
                        st.success(
                            "Your account has been created. "
                            "Please check your email and click "
                            "the confirmation link before signing in."
                        )

                    elif user is not None:
                        save_authenticated_session(session)
                        st.session_state[USER_ID_KEY] = user.id
                        st.session_state[USER_EMAIL_KEY] = (
                            user.email or new_email.strip()
                        )

                        st.success("Your account is ready.")
                        st.rerun()

                    else:
                        st.error(
                            "Account creation did not return a valid user."
                        )

                except Exception:
                    st.error(
                        "Unable to create your account. "
                        "Please check your details and try again."
                    )


# ==================================================
# 8. RESTORE AUTHENTICATED USER SESSION
# ==================================================

try:
    supabase, authenticated_user, session_error = (
        get_authenticated_client()
    )

except Exception:
    st.error(
        "Unable to connect to the authentication service. "
        "Please check your Supabase configuration."
    )
    st.stop()


if session_error:
    st.warning(session_error)

    if st.button(
        "Retry Connection",
        use_container_width=True
    ):
        st.rerun()

    st.stop()


if authenticated_user is None:
    show_authentication_page(supabase)
    st.stop()


user_id = authenticated_user.id
user_email = (
    authenticated_user.email or "Authenticated user"
)


# ==================================================
# 9. INITIALISE SESSION STATE
# ==================================================

SESSION_DEFAULTS = {
    "zentiq_ai_messages": [],
    "ml_retail_messages": [],
    "zentiq_active_conversations": {},
    "zentiq_unsaved_modes": set()
}

for key, default_value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = (
            default_value.copy()
            if isinstance(default_value, (dict, list, set))
            else default_value
        )


# ==================================================
# 10. CONVERSATION PERSISTENCE
# ==================================================

def persist_conversation( client, authenticated_user_id, mode, messages ):
    """ Create or update a conversation in Supabase. The active conversation ID prevents duplicate records. The database repository must enforce user_id ownership through Supabase Row Level Security. """

    if not messages:
        return None

    active_conversations = st.session_state[
        "zentiq_active_conversations"
    ]

    conversation_id = active_conversations.get(mode)

    first_user_message = next(
        (
            message.get("content", "")
            for message in messages
            if message.get("role") == "user"
        ),
        "ZENTIQ Conversation"
    )

    title = str(first_user_message).strip()[:80]
    if not title:
        title = "ZENTIQ Conversation"

    if conversation_id:
        result = update_conversation(
            client=client,
            user_id=authenticated_user_id,
            conversation_id=conversation_id,
            messages=messages,
            title=title
        )

        if result is None:
            raise RuntimeError(
                "The conversation could not be updated."
            )

    else:
        result = create_conversation(
            client=client,
            user_id=authenticated_user_id,
            title=title,
            mode=mode,
            messages=messages
        )

        if not result or not result.get("id"):
            raise RuntimeError(
                "Supabase did not return a conversation ID."
            )

        active_conversations[mode] = result["id"]

    st.session_state["zentiq_unsaved_modes"].discard(mode)
    return result


# ==================================================
# 11. PDF EXPORT
# ==================================================

def build_conversation_pdf(messages, mode):
    """Build a downloadable PDF containing the conversation."""

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import (
        getSampleStyleSheet,
        ParagraphStyle
    )
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer
    )
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import reportlab

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch
    )

    styles = getSampleStyleSheet()
    font_name = "Helvetica"
    bold_font = "Helvetica-Bold"

    font_directory = Path(reportlab.__file__).parent / "fonts"

    try:
        regular_font = font_directory / "Vera.ttf"
        bold_font_path = font_directory / "VeraBd.ttf"

        if regular_font.exists() and bold_font_path.exists():
            registered_fonts = pdfmetrics.getRegisteredFontNames()

            if "ZENTIQVera" not in registered_fonts:
                pdfmetrics.registerFont(
                    TTFont("ZENTIQVera", str(regular_font))
                )
                pdfmetrics.registerFont(
                    TTFont("ZENTIQVera-Bold", str(bold_font_path))
                )

            font_name = "ZENTIQVera"
            bold_font = "ZENTIQVera-Bold"

    except Exception:
        pass

    title_style = ParagraphStyle(
        "ZENTIQTitle",
        parent=styles["Title"],
        fontName=bold_font,
        fontSize=19,
        textColor=colors.HexColor("#17365D"),
        spaceAfter=12
    )

    heading_style = ParagraphStyle(
        "ZENTIQHeading",
        parent=styles["Heading2"],
        fontName=bold_font,
        fontSize=11,
        spaceBefore=10,
        spaceAfter=5
    )

    body_style = ParagraphStyle(
        "ZENTIQBody",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=9,
        leading=14,
        spaceAfter=8,
        wordWrap="CJK"
    )

    story = [
        Paragraph("ZENTIQ", title_style),
        Paragraph(
            "Intelligent Business Assistant",
            heading_style
        ),
        Paragraph(
            "Assistant: " + escape(str(mode)),
            body_style
        ),
        Paragraph(
            "Generated: "
            + datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M UTC"
            ),
            body_style
        ),
        Spacer(1, 15)
    ]

    for message in messages:
        role = str(
            message.get("role", "unknown")
        ).title()

        content = str(message.get("content", ""))
        safe_content = escape(content).replace(
            "\n",
            "<br/>"
        )

        # Do not wrap an entire potentially long message in KeepTogether.
        # ReportLab can then split long paragraphs across PDF pages.
        story.append(
            Paragraph(escape(role), heading_style)
        )
        story.append(
            Paragraph(safe_content, body_style)
        )

        if message.get("provider"):
            story.append(
                Paragraph(
                    "AI Provider: "
                    + escape(str(message["provider"])),
                    body_style
                )
            )

        if message.get("engine"):
            story.append(
                Paragraph(
                    "Response Engine: "
                    + escape(str(message["engine"])),
                    body_style
                )
            )

        if message.get("intent"):
            story.append(
                Paragraph(
                    "Predicted Intent: "
                    + escape(str(message["intent"])),
                    body_style
                )
            )

        if message.get("confidence") is not None:
            story.append(
                Paragraph(
                    "Confidence: "
                    + escape(
                        f"{float(message['confidence']):.2f}%"
                    ),
                    body_style
                )
            )

        story.append(Spacer(1, 6))

    document.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ==================================================
# 12. APPLICATION HEADER
# ==================================================

st.title("ðŸ¤– ZENTIQ")
st.subheader("Smarter Business. Better Decisions.")

st.write(
    "Welcome to ZENTIQ, your intelligent business assistant. "
    "Get business support, explore ideas, and find practical "
    "solutions to everyday business challenges."
)

st.caption(f"Signed in as: {user_email}")
st.divider()


# ==================================================
# 13. SIDEBAR AND ASSISTANT SELECTION
# ==================================================

with st.sidebar:
    st.title("ZENTIQ Assistant")

    # ----------------------------------------------
    # SIGN OUT
    # ----------------------------------------------

    if st.button(
        "Sign Out",
        use_container_width=True
    ):
        success, message = logout_authenticated_user(supabase)

        for key in (
            "zentiq_active_conversations",
            "zentiq_unsaved_modes",
            "zentiq_ai_messages",
            "ml_retail_messages",
            USER_ID_KEY,
            USER_EMAIL_KEY
        ):
            st.session_state.pop(key, None)

        st.success(message) if success else st.warning(message)
        st.rerun()

    st.divider()

    # ----------------------------------------------
    # ASSISTANT MODE
    # ----------------------------------------------

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
            "Production: Cloudflare Gemma 4 is the primary AI "
            "provider, with Groq Cloud as fallback. "
            "Local development: Ollama is the primary provider."
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
            "Powered by TF-IDF and Logistic Regression, "
            "with safety screening."
        )

        st.markdown(""" **Available Support Areas** - Business Expenses - Customer Acquisition - Daily Sales - Damaged Products - Delivery Costs and Delays - Inventory Reordering - Order Tracking - Payment Confirmation - Payment Methods - POS Issues - Profit Calculation - Record Keeping - Refund Requests - Selling Prices - Social Media Marketing - Stock Availability - Supplier Onboarding - Supplier Price Changes """)

        history_key = "ml_retail_messages"

    st.divider()

    # ----------------------------------------------
    # ACTIVE CONVERSATION
    # ----------------------------------------------

    active_conversations = st.session_state[
        "zentiq_active_conversations"
    ]

    # ----------------------------------------------
    # NEW CONVERSATION
    # ----------------------------------------------

    if st.button(
        "New Conversation",
        use_container_width=True
    ):
        active_conversations[assistant_mode] = None
        st.session_state[history_key] = []

        st.session_state["zentiq_unsaved_modes"].discard(
            assistant_mode
        )

        st.rerun()

    # ----------------------------------------------
    # RECENT CONVERSATIONS
    # ----------------------------------------------

    st.subheader("Recent Conversations")

    try:
        saved_conversations = get_conversations(
            supabase,
            user_id
        )

        mode_conversations = [
            conversation
            for conversation in saved_conversations
            if conversation.get("mode") == assistant_mode
        ]

        for conversation in mode_conversations[:10]:
            conversation_id = conversation.get("id")
            conversation_title = conversation.get(
                "title",
                "ZENTIQ Conversation"
            )

            if st.button(
                conversation_title[:45],
                key=f"conversation_{assistant_mode}_{conversation_id}",
                use_container_width=True
            ):
                selected_conversation = get_conversation(
                    supabase,
                    user_id,
                    conversation_id
                )

                if selected_conversation:
                    active_conversations[assistant_mode] = (
                        conversation_id
                    )

                    restored_messages = selected_conversation.get(
                        "messages",
                        []
                    )

                    st.session_state[history_key] = (
                        restored_messages
                        if isinstance(restored_messages, list)
                        else []
                    )

                    st.session_state["zentiq_unsaved_modes"].discard(
                        assistant_mode
                    )

                    st.rerun()

    except Exception:
        st.warning(
            "Unable to load recent conversations. "
            "Please try again later."
        )

    st.divider()

    # ----------------------------------------------
    # RETRY UNSAVED CONVERSATIONS
    # ----------------------------------------------

    if assistant_mode in st.session_state["zentiq_unsaved_modes"]:
        if st.button(
            "Retry Saving Conversation",
            use_container_width=True
        ):
            try:
                persist_conversation(
                    supabase,
                    user_id,
                    assistant_mode,
                    st.session_state[history_key]
                )

                st.success("Conversation saved successfully.")
                st.rerun()

            except Exception:
                st.error(
                    "Conversation could not be saved. "
                    "Please try again."
                )

    # ----------------------------------------------
    # CLEAR CURRENT CHAT
    # ----------------------------------------------

    if st.button(
        "Clear Current Chat",
        use_container_width=True
    ):
        try:
            conversation_id = active_conversations.get(
                assistant_mode
            )

            if conversation_id:
                update_conversation(
                    client=supabase,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    messages=[],
                    title="ZENTIQ Conversation"
                )

            st.session_state[history_key] = []

            st.session_state["zentiq_unsaved_modes"].discard(
                assistant_mode
            )

            st.success("Current conversation cleared.")
            st.rerun()

        except Exception:
            st.error("Unable to clear this conversation.")


# ==================================================
# 14. DISPLAY CONVERSATION HISTORY
# ==================================================

messages = st.session_state[history_key]

for message in messages:
    role = message.get("role", "assistant")
    content = message.get("content", "")

    with st.chat_message(role):
        st.markdown(content)

        if message.get("provider"):
            st.caption(
                f"AI Provider: {message['provider']}"
            )

        if message.get("engine"):
            st.caption(
                f"Response Engine: {message['engine']}"
            )

        if message.get("intent"):
            st.caption(
                f"Predicted Intent: {message['intent']}"
            )

        if message.get("confidence") is not None:
            st.caption(
                f"Confidence: {float(message['confidence']):.2f}%"
            )


# ==================================================
# 15. CHAT INPUT AND RESPONSE GENERATION
# ==================================================

user_question = st.chat_input(
    "Ask ZENTIQ a business question..."
)

if user_question:
    user_message = {
        "role": "user",
        "content": user_question
    }

    messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(user_question)

    try:
        with st.chat_message("assistant"):
            with st.spinner("ZENTIQ is preparing your response..."):

                # ------------------------------------------
                # MODE 1: MACHINE LEARNING RETAIL SUPPORT
                # ------------------------------------------

                if assistant_mode == "Machine Learning Retail Support":
                    llm = load_llm_service()

                    # Screen the incoming customer message.
                    input_safe = llm.check_input_safety(
                        user_question
                    )

                    if not input_safe:
                        response = (
                            "I'm sorry, but I cannot process this "
                            "request because it did not pass "
                            "ZENTIQ's safety checks."
                        )

                        assistant_message = {
                            "role": "assistant",
                            "content": response,
                            "engine": "Safety screening"
                        }

                        st.markdown(response)

                    else:
                        # Generate the ML retail-support response.
                        intent, response, confidence = (
                            predict_customer_query(user_question)
                        )

                        # Screen the generated ML response.
                        output_safe = llm.check_output_safety(
                            response,
                            user_message=user_question
                        )

                        if not output_safe:
                            response = (
                                "ZENTIQ generated a response that "
                                "did not pass the safety checks. "
                                "The response has been withheld."
                            )

                            assistant_message = {
                                "role": "assistant",
                                "content": response,
                                "engine": "Safety screening"
                            }

                            st.markdown(response)

                        else:
                            assistant_message = {
                                "role": "assistant",
                                "content": response,
                                "engine": "TF-IDF + Logistic Regression",
                                "intent": intent,
                                "confidence": confidence
                            }

                            st.markdown(response)
                            st.caption(
                                f"Predicted Intent: {intent}"
                            )
                            st.caption(
                                f"Confidence: {confidence:.2f}%"
                            )

                # ------------------------------------------
                # MODE 2: HYBRID AI ASSISTANT
                # ------------------------------------------

                else:
                    llm = load_llm_service()

                    # Prevent a previous request's provider from
                    # being shown if this request is refused early.
                    llm.model_manager.active_provider = None

                    response = llm.generate_response(
                        prompt=user_question,
                        task="general",
                        template_name=template_name,
                        temperature=0.3
                    )

                    provider = (
                        llm.model_manager.active_provider
                        or "Not used / response withheld"
                    )

                    assistant_message = {
                        "role": "assistant",
                        "content": response,
                        "provider": str(provider)
                    }

                    st.markdown(response)

                    st.caption(
                        f"AI Provider: {provider}"
                    )

        # Append only after a response has been generated and
        # any required safety screening has completed.
        messages.append(assistant_message)

        # Save the complete conversation.
        try:
            persist_conversation(
                supabase,
                user_id,
                assistant_mode,
                messages
            )

        except Exception:
            st.session_state["zentiq_unsaved_modes"].add(
                assistant_mode
            )

            st.warning(
                "Your response was generated, but the conversation "
                "could not be saved. Use Retry Saving Conversation "
                "in the sidebar."
            )

        st.rerun()

    except Exception:
        st.error(
            "ZENTIQ could not complete your request. "
            "The response was not released if safety screening "
            "could not be completed. Please try again. If the "
            "problem continues, check the AI provider and safety "
            "service configuration."
        )

        # Remove the unanswered user message from the current
        # in-memory conversation. It can be submitted again.
        if messages and messages[-1] == user_message:
            messages.pop()


# ==================================================
# 16. PDF CONVERSATION EXPORT
# ==================================================

if messages:
    st.divider()
    st.subheader("Export Your Conversation")

    try:
        pdf_data = build_conversation_pdf(
            messages,
            assistant_mode
        )

        st.download_button(
            label="Download Conversation as PDF",
            data=pdf_data,
            file_name=(
                "ZENTIQ_Conversation_"
                + datetime.now().strftime("%Y%m%d_%H%M%S")
                + ".pdf"
            ),
            mime="application/pdf",
            use_container_width=True
        )

    except Exception:
        st.error(
            "Unable to generate the PDF. Please try again."
        )


# ==================================================
# 17. APPLICATION FOOTER
# ==================================================

st.divider()

st.caption(
    "ZENTIQ â€” Intelligent Business Assistant"
)

st.caption(
    "Developed by EverMark Creative Solutions."
)

st.caption(
    "Intelligence that empowers business."
)
