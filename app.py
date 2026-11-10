"""
ZENTIQ - Intelligent Business Assistant
Redesigned Hybrid AI Streamlit Application

PRODUCTION AI:
    Primary: Cloudflare Workers AI - Gemma 4
    Fallback: Groq Cloud

LOCAL DEVELOPMENT AI:
    Primary: Ollama - Llama 3.2:1B
    Fallback: Groq Cloud

SAFETY:
    Production: Cloudflare Llama Guard 3 8B
    Local: Ollama Llama Guard 3:1B

FEATURES:
    - Supabase email authentication
    - User-specific persistent conversations
    - Conversation recovery and retry saving
    - Input and output safety screening
    - Direct response display
    - Response regeneration
    - Answer feedback
    - Copy-friendly response view
    - Individual answer download
    - PDF conversation export
    - Responsive ZENTIQ interface

Developed by EverMark Creative Solutions.
"""

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
    page_icon="🤖",
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
# 3. ZENTIQ VISUAL IDENTITY
# ==================================================

PRIMARY_BLUE = "#087BFF"
DEEP_NAVY = "#0B1F4A"
VIOLET = "#742BFF"
SOFT_BLUE = "#EAF2FF"
MUTED_TEXT = "#52658F"


def apply_zentiq_design():
    """Apply the ZENTIQ visual identity to the application."""

    st.markdown(
        """
        <style>
        :root {
            --z-primary: #087BFF;
            --z-navy: #0B1F4A;
            --z-violet: #742BFF;
            --z-soft: #EAF2FF;
            --z-muted: #52658F;
        }

        .stApp {
            background:
                radial-gradient(
                    circle at 5% 0%,
                    rgba(8,123,255,0.13),
                    transparent 34%
                ),
                radial-gradient(
                    circle at 100% 100%,
                    rgba(116,43,255,0.10),
                    transparent 32%
                ),
                #F6F9FF;
            color: var(--z-navy);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                #EAF2FF 0%,
                #F7F9FF 100%
            );
            border-right: 1px solid #DCE7FA;
        }

        h1, h2, h3 {
            color: var(--z-navy);
            letter-spacing: -0.5px;
        }

        .zentiq-brand {
            text-align: center;
            padding: 10px 0 4px 0;
        }

        .zentiq-brand-mark {
            font-size: 54px;
            line-height: 1.1;
            font-weight: 900;
            letter-spacing: -3px;
            color: var(--z-navy);
        }

        .zentiq-brand-mark span {
            background: linear-gradient(
                110deg,
                #087BFF,
                #742BFF
            );
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }

        .zentiq-tagline {
            color: var(--z-primary);
            font-size: 17px;
            font-weight: 600;
            text-align: center;
            margin-top: 2px;
        }

        .zentiq-subtitle {
            color: var(--z-muted);
            text-align: center;
            font-size: 14px;
            margin-bottom: 20px;
        }

        .zentiq-card {
            background: rgba(255,255,255,0.94);
            border: 1px solid #DCE6F8;
            border-radius: 20px;
            padding: 20px;
            box-shadow: 0 12px 35px rgba(11,31,74,0.07);
            margin: 12px 0;
        }

        .zentiq-footer {
            text-align: center;
            color: var(--z-muted);
            font-size: 13px;
            padding: 18px 0 8px 0;
        }

        .zentiq-footer strong {
            color: var(--z-primary);
        }

        .stButton > button {
            border-radius: 12px;
            border: 1px solid #D5E2F7;
            font-weight: 600;
            transition: all 0.2s ease;
        }

        .stButton > button[kind="primary"],
        .stFormSubmitButton > button {
            background: linear-gradient(
                105deg,
                #087BFF,
                #742BFF
            );
            color: white;
            border: none;
        }

        .stButton > button:hover {
            border-color: var(--z-primary);
            color: var(--z-primary);
        }

        .stButton > button[kind="primary"]:hover,
        .stFormSubmitButton > button:hover {
            color: white;
            filter: brightness(1.05);
        }

        [data-testid="stChatMessage"] {
            background: rgba(255,255,255,0.92);
            border: 1px solid #E0E9F8;
            border-radius: 16px;
            padding: 14px;
            margin-bottom: 12px;
            box-shadow: 0 5px 18px rgba(11,31,74,0.035);
        }

        [data-testid="stChatInput"] {
            border-radius: 16px;
        }

        [data-testid="stChatInput"] textarea {
            border-radius: 14px;
        }

        [data-testid="stMetric"] {
            background: white;
            border: 1px solid #E0E9F8;
            border-radius: 14px;
            padding: 12px;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
        }

        .stTabs [data-baseweb="tab"] {
            color: var(--z-muted);
            font-weight: 600;
        }

        .stTabs [aria-selected="true"] {
            color: var(--z-primary) !important;
        }

        div[data-testid="stExpander"] {
            border: 1px solid #DCE6F8;
            border-radius: 12px;
            background: rgba(255,255,255,0.8);
        }

        @media (max-width: 640px) {
            .zentiq-brand-mark {
                font-size: 43px;
            }

            .zentiq-card {
                padding: 14px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )


apply_zentiq_design()


# ==================================================
# 4. DATABASE AND AUTHENTICATION IMPORTS
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
# 5. LOAD MACHINE LEARNING COMPONENTS
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
        "Unable to load the Machine Learning components. "
        "Check that all three model files exist and are valid."
    )
    st.exception(error)
    st.stop()


# ==================================================
# 6. MACHINE LEARNING PREDICTION
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
# 7. LOAD HYBRID AI SERVICE
# ==================================================

def load_llm_service():
    """Create a fresh service for the current request."""

    from ai_engine.llm_service import LLMService

    return LLMService()


# ==================================================
# 8. AUTHENTICATION INTERFACE
# ==================================================

def show_authentication_page(client):
    """Display the redesigned ZENTIQ sign-in interface."""

    st.markdown(
        """
        <div class="zentiq-brand">
            <div class="zentiq-brand-mark">
                <span>Z</span>ENTIQ
            </div>
            <div class="zentiq-tagline">
                Smarter Business. Together.
            </div>
            <div class="zentiq-subtitle">
                Welcome to your Intelligent Business Assistant
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="zentiq-card">
            <h3 style="text-align:center;color:#0B1F4A;">
                Your business intelligence starts here
            </h3>
            <p style="text-align:center;color:#52658F;">
                Sign in to access your assistant and securely
                manage your conversation history.
            </p>
        </div>
        """,
        unsafe_allow_html=True
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
                type="password",
                placeholder="Enter your password"
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

                        st.success("Welcome back to ZENTIQ.")
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
                placeholder="Create a password",
                help="Use at least 8 characters."
            )

            confirm_password = st.text_input(
                "Confirm password",
                type="password",
                placeholder="Confirm your password"
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

                        st.success("Your ZENTIQ account is ready.")
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

    st.markdown(
        """
        <div class="zentiq-footer">
            <strong>Intelligence that empowers business.</strong><br>
            Developed by EverMark Creative Solutions.
        </div>
        """,
        unsafe_allow_html=True
    )


# ==================================================
# 9. RESTORE AUTHENTICATED USER SESSION
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
# 10. INITIALISE SESSION STATE
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
# 11. CONVERSATION PERSISTENCE
# ==================================================

def persist_conversation(
    client,
    authenticated_user_id,
    mode,
    messages
):
    """Create or update a user-owned Supabase conversation."""

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
# 12. PDF CONVERSATION EXPORT
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
        fontSize=20,
        textColor=colors.HexColor("#0B1F4A"),
        spaceAfter=12
    )

    heading_style = ParagraphStyle(
        "ZENTIQHeading",
        parent=styles["Heading2"],
        fontName=bold_font,
        fontSize=11,
        textColor=colors.HexColor("#087BFF"),
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

        if message.get("feedback"):
            story.append(
                Paragraph(
                    "User feedback: "
                    + escape(str(message["feedback"])),
                    body_style
                )
            )

        story.append(Spacer(1, 6))

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# ==================================================
# 13. RESPONSE GENERATION
# ==================================================

def generate_answer(question, assistant_mode, template_name):
    """
    Generate one answer.

    Both modes pass through the appropriate input and output
    safety screening before an answer is released.
    """

    if assistant_mode == "Machine Learning Retail Support":

        llm = load_llm_service()

        input_safe = llm.check_input_safety(question)

        if not input_safe:
            return {
                "role": "assistant",
                "content": (
                    "I'm sorry, but I cannot process this request "
                    "because it did not pass ZENTIQ's safety checks."
                ),
                "engine": "Safety screening"
            }

        intent, response, confidence = predict_customer_query(
            question
        )

        output_safe = llm.check_output_safety(
            response,
            user_message=question
        )

        if not output_safe:
            return {
                "role": "assistant",
                "content": (
                    "ZENTIQ generated a response that did not pass "
                    "the safety checks. The response has been withheld."
                ),
                "engine": "Safety screening"
            }

        return {
            "role": "assistant",
            "content": response,
            "engine": "TF-IDF + Logistic Regression",
            "intent": intent,
            "confidence": confidence
        }

    llm = load_llm_service()

    llm.model_manager.active_provider = None

    response = llm.generate_response(
        prompt=question,
        task="general",
        template_name=template_name,
        temperature=0.3
    )

    provider = (
        llm.model_manager.active_provider
        or getattr(llm, "last_provider", None)
        or "Not reported"
    )

    return {
        "role": "assistant",
        "content": response,
        "provider": str(provider)
    }


# ==================================================
# 14. APPLICATION HEADER
# ==================================================

st.markdown(
    """
    <div class="zentiq-brand">
        <div class="zentiq-brand-mark">
            <span>Z</span>ENTIQ
        </div>
        <div class="zentiq-tagline">
            Smarter Business. Together.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="zentiq-subtitle">
        Intelligence that empowers business.
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(f"Signed in as: {user_email}")

st.divider()


# ==================================================
# 15. SIDEBAR AND ASSISTANT SELECTION
# ==================================================

with st.sidebar:

    st.markdown("## ZENTIQ Assistant")

    if st.button(
        "Sign Out",
        use_container_width=True
    ):
        success, message = logout_authenticated_user(
            supabase
        )

        for key in (
            "zentiq_active_conversations",
            "zentiq_unsaved_modes",
            "zentiq_ai_messages",
            "ml_retail_messages",
            USER_ID_KEY,
            USER_EMAIL_KEY
        ):
            st.session_state.pop(key, None)

        if success:
            st.success(message)
        else:
            st.warning(message)

        st.rerun()

    st.divider()

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
            "Production: Cloudflare Gemma 4 is the primary "
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

        st.markdown(
            """
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
            """
        )

        template_name = "general"

        history_key = "ml_retail_messages"

    st.divider()

    active_conversations = st.session_state[
        "zentiq_active_conversations"
    ]

    # ----------------------------------------------
    # NEW CONVERSATION
    # ----------------------------------------------

    if st.button(
        "＋ New Conversation",
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

        if not mode_conversations:
            st.caption("No saved conversations yet.")

        for conversation in mode_conversations[:10]:

            conversation_id = conversation.get("id")

            conversation_title = conversation.get(
                "title",
                "ZENTIQ Conversation"
            )

            if st.button(
                conversation_title[:45],
                key=(
                    f"conversation_{assistant_mode}_"
                    f"{conversation_id}"
                ),
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

                    restored_messages = (
                        selected_conversation.get(
                            "messages",
                            []
                        )
                    )

                    st.session_state[history_key] = (
                        restored_messages
                        if isinstance(restored_messages, list)
                        else []
                    )

                    st.session_state[
                        "zentiq_unsaved_modes"
                    ].discard(assistant_mode)

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

    if assistant_mode in st.session_state[
        "zentiq_unsaved_modes"
    ]:

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

            st.session_state[
                "zentiq_unsaved_modes"
            ].discard(assistant_mode)

            st.success("Current conversation cleared.")
            st.rerun()

        except Exception:
            st.error("Unable to clear this conversation.")


# ==================================================
# 16. DISPLAY CONVERSATION HISTORY
# ==================================================

messages = st.session_state[history_key]

for index, message in enumerate(messages):

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
                f"Confidence: "
                f"{float(message['confidence']):.2f}%"
            )

        # ------------------------------------------
        # ANSWER ACTIONS
        # ------------------------------------------

        if role == "assistant" and content:

            with st.expander(
                "Copy or share this answer",
                expanded=False
            ):

                st.caption(
                    "Use the copy icon on the text block, "
                    "or download the answer as a text file."
                )

                st.code(
                    content,
                    language=None
                )

                st.download_button(
                    "Download This Answer",
                    data=content.encode("utf-8"),
                    file_name="ZENTIQ_Answer.txt",
                    mime="text/plain",
                    key=f"answer_download_{index}",
                    use_container_width=True
                )

            # --------------------------------------
            # FEEDBACK
            # --------------------------------------

            feedback_col1, feedback_col2 = st.columns(2)

            with feedback_col1:

                if st.button(
                    "👍 Helpful",
                    key=f"feedback_up_{index}",
                    use_container_width=True
                ):

                    message["feedback"] = "Helpful"

                    try:
                        persist_conversation(
                            supabase,
                            user_id,
                            assistant_mode,
                            messages
                        )

                        st.toast("Thank you for your feedback.")

                    except Exception:
                        st.session_state[
                            "zentiq_unsaved_modes"
                        ].add(assistant_mode)

                        st.warning(
                            "Feedback recorded locally but "
                            "could not be saved yet."
                        )

                    st.rerun()

            with feedback_col2:

                if st.button(
                    "👎 Not Helpful",
                    key=f"feedback_down_{index}",
                    use_container_width=True
                ):

                    message["feedback"] = "Not Helpful"

                    try:
                        persist_conversation(
                            supabase,
                            user_id,
                            assistant_mode,
                            messages
                        )

                        st.toast("Thank you. We value your feedback.")

                    except Exception:
                        st.session_state[
                            "zentiq_unsaved_modes"
                        ].add(assistant_mode)

                        st.warning(
                            "Feedback recorded locally but "
                            "could not be saved yet."
                        )

                    st.rerun()

            if message.get("feedback"):
                st.caption(
                    f"Your feedback: {message['feedback']}"
                )

            with st.expander(
                "Add written feedback",
                expanded=False
            ):

                feedback_text = st.text_area(
                    "Tell us how this answer could improve:",
                    key=f"feedback_text_{index}",
                    height=90
                )

                if st.button(
                    "Submit Feedback",
                    key=f"feedback_submit_{index}"
                ):

                    if feedback_text.strip():

                        message["feedback_comment"] = (
                            feedback_text.strip()
                        )

                        try:

                            persist_conversation(
                                supabase,
                                user_id,
                                assistant_mode,
                                messages
                            )

                            st.success(
                                "Thank you. Your feedback has been saved."
                            )

                        except Exception:

                            st.session_state[
                                "zentiq_unsaved_modes"
                            ].add(assistant_mode)

                            st.warning(
                                "Feedback could not be saved yet."
                            )

                        st.rerun()

                    else:
                        st.warning(
                            "Please enter your feedback first."
                        )


# ==================================================
# 17. REGENERATE LAST RESPONSE
# ==================================================

if messages:

    last_assistant_index = None

    for index in range(len(messages) - 1, -1, -1):

        if messages[index].get("role") == "assistant":
            last_assistant_index = index
            break

    if last_assistant_index is not None:

        last_assistant = messages[last_assistant_index]

        last_user_message = None

        for index in range(
            last_assistant_index - 1,
            -1,
            -1
        ):

            if messages[index].get("role") == "user":
                last_user_message = messages[index]
                break

        if last_user_message:

            if st.button(
                "↻ Regenerate Last Response",
                use_container_width=True,
                type="secondary"
            ):

                try:

                    with st.spinner(
                        "ZENTIQ is generating another answer..."
                    ):

                        regenerated = generate_answer(
                            last_user_message.get("content", ""),
                            assistant_mode,
                            template_name
                        )

                    messages[last_assistant_index] = regenerated

                    try:

                        persist_conversation(
                            supabase,
                            user_id,
                            assistant_mode,
                            messages
                        )

                    except Exception:

                        st.session_state[
                            "zentiq_unsaved_modes"
                        ].add(assistant_mode)

                        st.warning(
                            "The new response was generated, "
                            "but saving failed. Retry saving "
                            "from the sidebar."
                        )

                    st.rerun()

                except Exception:

                    st.error(
                        "ZENTIQ could not regenerate the answer. "
                        "Please try again."
                    )


# ==================================================
# 18. CHAT INPUT AND RESPONSE GENERATION
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

            with st.spinner(
                "ZENTIQ is preparing your response..."
            ):

                assistant_message = generate_answer(
                    user_question,
                    assistant_mode,
                    template_name
                )

            st.markdown(
                assistant_message["content"]
            )

            if assistant_message.get("provider"):
                st.caption(
                    "AI Provider: "
                    + assistant_message["provider"]
                )

            if assistant_message.get("engine"):
                st.caption(
                    "Response Engine: "
                    + assistant_message["engine"]
                )

            if assistant_message.get("intent"):
                st.caption(
                    "Predicted Intent: "
                    + assistant_message["intent"]
                )

            if assistant_message.get("confidence") is not None:
                st.caption(
                    "Confidence: "
                    f"{assistant_message['confidence']:.2f}%"
                )

        messages.append(assistant_message)

        # ------------------------------------------
        # SAVE THE COMPLETE CONVERSATION
        # ------------------------------------------

        try:

            persist_conversation(
                supabase,
                user_id,
                assistant_mode,
                messages
            )

        except Exception:

            st.session_state[
                "zentiq_unsaved_modes"
            ].add(assistant_mode)

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
            "could not be completed. Please try again. "
            "If the problem continues, check the AI provider "
            "and safety service configuration."
        )

        if messages and messages[-1] == user_message:
            messages.pop()


# ==================================================
# 19. PDF CONVERSATION EXPORT
# ==================================================

if messages:

    st.divider()

    st.subheader("Your Conversation")

    st.write(
        "Download a copy of your conversation whenever you need it."
    )

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
# 20. APPLICATION FOOTER
# ==================================================

st.divider()

st.markdown(
    """
    <div class="zentiq-footer">
        <strong>Smarter Business. Together.</strong><br><br>
        ZENTIQ — Intelligent Business Assistant<br>
        Developed by EverMark Creative Solutions.<br>
        Intelligence that empowers business.
    </div>
    """,
    unsafe_allow_html=True
)
