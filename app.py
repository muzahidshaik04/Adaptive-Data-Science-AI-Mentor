import streamlit as st
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Adaptive Data Science AI Mentor",
    page_icon="🤖",
    layout="centered"
)

# ===================== ENV =====================
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("gemini")

if not os.getenv("GOOGLE_API_KEY"):
    st.error("Google API key not found. Please add it your .env file")
    st.stop()

# ===================== DARK MODE TOGGLE =====================
with st.sidebar:
    dark_mode = st.toggle("🌙 Dark Mode")


# ===================== MODERN STYLES =====================
if dark_mode:
    background = "#0f172a"
    text_color = "#f1f5f9"
    card_bg = "#1e293b"
else:
    background = "#f8fafc"
    text_color = "#0f172a"
    card_bg = "#ffffff"

st.markdown(f"""
<style>
[data-testid="stApp"] {{
    background: {background};
    color: {text_color};
}}

html, body, [class*="css"] {{
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI",
                 Roboto, Helvetica, Arial, sans-serif;
}}

.hero {{
    text-align: center;
    padding: 40px 20px;
}}

.hero h1 {{
    font-size: 42px;
    font-weight: 700;
}}

.hero p {{
    font-size: 18px;
    opacity: 0.7;
}}

.card {{
    background: {card_bg};
    padding: 30px;
    border-radius: 16px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    margin-top: 30px;
}}

.gradient-header {{
    background: linear-gradient(90deg,#2563eb,#4f46e5);
    padding: 14px;
    border-radius: 12px;
    color: white;
    font-weight: 600;
    margin-bottom: 20px;
}}

.stChatMessage {{
    border-radius: 14px;
    padding: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.04);
    transition: all 0.3s ease-in-out;
}}
</style>
""", unsafe_allow_html=True)


# ===================== HERO SECTION =====================
st.markdown("""
<div class="hero">
    <h1>🚀 Adaptive Data Science AI Mentor</h1>
    <p>Personalized AI guidance across Data Science domains</p>
</div>
""", unsafe_allow_html=True)


# ===================== SUBJECT MAPPING =====================
SUBJECT_MAP = {

    "Foundations": [
        "Python Programming",
        "Python for Data Science",
        "Data Structures & Algorithms (Basics)",
        "Math for Data Science",
        "Statistics Fundamentals",
        "Probability",
        "Linear Algebra (Essentials)"
    ],

    "Data Analysis": [
        "NumPy",
        "Pandas",
        "Data Cleaning",
        "Exploratory Data Analysis (EDA)",
        "Feature Engineering",
        "Data Visualization",
        "Handling Missing Values",
        "Outlier Detection"
    ],

    "Databases & SQL": [
        "SQL Fundamentals",
        "Advanced SQL",
        "Joins",
        "Subqueries",
        "Window Functions",
        "SQL for Analytics",
        "Query Optimization"
    ],

    "Statistics & Experiments": [
        "Descriptive Statistics",
        "Inferential Statistics",
        "Hypothesis Testing",
        "A/B Testing",
        "Confidence Intervals",
        "Probability Distributions",
        "Bayesian Statistics (Intro)"
    ],

    "Machine Learning": [
        "Introduction to Machine Learning",
        "Supervised Learning",
        "Unsupervised Learning",
        "Linear Regression",
        "Logistic Regression",
        "Decision Trees",
        "KNN",
        "Naive Bayes",
        "Support Vector Machines (SVM)",
        "Model Evaluation Metrics",
        "Cross Validation",
        "Bias vs Variance"
    ],

    "Advanced Machine Learning": [
        "Ensemble Methods",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost",
        "LightGBM",
        "Feature Selection",
        "Hyperparameter Tuning",
        "Imbalanced Data Handling",
        "Time Series Forecasting",
        "Recommendation Systems"
    ],

    "Deep Learning": [
        "Neural Networks Basics",
        "Backpropagation",
        "Activation Functions",
        "TensorFlow",
        "PyTorch",
        "Convolutional Neural Networks (CNN)",
        "Recurrent Neural Networks (RNN)",
        "LSTM",
        "GRU",
        "Transformers Basics"
    ],

    "NLP & Generative AI": [
        "Text Preprocessing",
        "Tokenization",
        "TF-IDF",
        "Word2Vec",
        "GloVe",
        "BERT",
        "Large Language Models (LLMs)",
        "Prompt Engineering",
        "RAG (Retrieval Augmented Generation)",
        "Vector Databases",
        "FAISS",
        "ChromaDB"
    ],

    "MLOps & Deployment": [
        "Model Deployment Basics",
        "FastAPI",
        "Streamlit",
        "Docker",
        "MLflow",
        "Model Versioning",
        "CI/CD for Machine Learning",
        "Monitoring & Data Drift",
        "Model Retraining Strategies"
    ],

    "Cloud & Big Data": [
        "AWS for Data Science",
        "S3",
        "EC2",
        "Lambda",
        "BigQuery",
        "Apache Spark Basics",
        "ETL Pipelines",
        "Batch vs Streaming Data"
    ],

    "Analytics & BI": [
        "Power BI",
        "DAX Fundamentals",
        "Power BI Data Modeling",
        "Tableau",
        "Excel for Analytics",
        "Dashboard Design",
        "Business Metrics",
        "Storytelling with Data"
    ],

    "Projects & Case Studies": [
        "End-to-End Data Science Project",
        "EDA Case Studies",
        "Machine Learning Project Lifecycle",
        "NLP Project",
        "Time Series Project",
        "Business Problem Solving",
        "Capstone Project"
    ],

    "Interview Preparation": [
        "Python Interview Questions",
        "SQL Interview Questions",
        "Statistics Interview Questions",
        "Machine Learning Interview Questions",
        "Deep Learning Interview Questions",
        "Data Science Case Studies",
        "Behavioral Interview Questions",
        "Resume Building",
        "ATS Resume Optimization",
        "Mock Interviews"
    ]
}

# ===================== PROMPT =====================
def get_mentor_prompt(subject, experience):
    system = SystemMessagePromptTemplate.from_template(
        """
You are a professional AI mentor specialized ONLY in {subject}.

User experience level: {experience} years.

Rules:
- Greet politely and professionally.
- Answer only questions related to {subject}.
- Adjust depth based on experience.
- If the question is unrelated, reply exactly:
"I can only answer questions related to {subject}. Please ask a relevant question."
"""
    )

    human = HumanMessagePromptTemplate.from_template("{question}")

    return ChatPromptTemplate.from_messages([system, human])

# ===================== SIDEBAR SETUP =====================
if "mentor_subject" not in st.session_state:

    with st.sidebar:
        st.header("🧑‍🏫 Data Science AI Mentor Setup")

        category = st.selectbox(
            "Choose domain",
            list(SUBJECT_MAP.keys())
        )

        subject = st.selectbox(
            "Choose skill",
            SUBJECT_MAP[category]
        )

        experience = st.slider(
            "Your experience (years)",
            0, 20, 0
        )

        st.markdown("---")

        if st.button("Start Mentor Session"):
            st.session_state.mentor_subject = subject
            st.session_state.experience = experience
            st.session_state.chat = []
            st.session_state.memory = []

            prompt = get_mentor_prompt(subject, experience)
            system_msg = prompt.format_messages(
                subject=subject,
                experience=experience,
                question=""
            )[0]

            st.session_state.memory.append(system_msg)
            st.rerun()

        st.markdown(f"""
    <div class="card">
        <h3>Start Your AI Mentor Session</h3>
        <p>Select a domain and skill from the sidebar to begin learning.</p>
    </div>
    """, unsafe_allow_html=True)

# ===================== CHAT PAGE =====================
else:
    # ===== SIDEBAR: SESSION CONTROLS =====
    with st.sidebar:
        st.header("⚙️ Session Controls")

        st.markdown("### 📌 Current Mentor")
        st.write("**Subject:**", st.session_state.mentor_subject)
        st.write("**Experience:**", f"{st.session_state.experience} yrs")

        if st.button("🔄 Change Mentor"):
            for key in [
                "mentor_subject",
                "experience",
                "chat",
                "memory"
            ]:
                if key in st.session_state:
                    del st.session_state[key]

            st.rerun()

    st.markdown(f"""
    <div class="gradient-header">
        Active Mentor: {st.session_state.mentor_subject}
    </div>
    """, unsafe_allow_html=True)

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask your question...")

    if user_input:
        user_msg = HumanMessage(content=user_input)
        st.session_state.memory.append(user_msg)
        st.session_state.chat.append(
            {"role": "user", "content": user_input}
        )

        try:
            response = model.invoke(
                [st.session_state.memory[0]] +
                st.session_state.memory[-6:]
            )
        except Exception:
            st.error("Something went wrong. Please check API key or model.")
            st.stop()

        ai_msg = AIMessage(content=response.content)
        st.session_state.memory.append(ai_msg)
        st.session_state.chat.append(
            {"role": "assistant", "content": response.content}
        )

        st.rerun()

    # Download Chat
    st.markdown("---")

    chat_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in st.session_state.chat
    )

    st.download_button(
        "📥 Download Chat",
        chat_text,
        file_name=f"{st.session_state.mentor_subject}_mentor_chat.txt"
    )
