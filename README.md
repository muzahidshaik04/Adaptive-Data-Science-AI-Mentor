# 🚀 Adaptive Data Science AI Mentor

A structured, domain-restricted AI mentoring system built using Streamlit, LangChain, and Google Gemini.

Designed to simulate real-world technical mentoring with experience-aware explanations and controlled LLM behavior.

---

## 📌 Overview

Adaptive Data Science AI Mentor simulates a focused mentoring system instead of a generic chatbot.

Users can:
- Select a Data Science domain
- Choose a specific skill
- Set their experience level
- Ask structured, domain-specific questions

The AI:
- Adjusts explanation depth based on experience
- Refuses off-topic questions
- Maintains contextual memory
- Allows session download for revision

---

## 🎯 Key Features

- Domain-specific AI mentoring
- Experience-aware response adaptation
- Prompt-engineered behavior control
- Context-limited memory window
- Modern Streamlit UI with dark mode
- Session reset and mentor switching
- Downloadable chat history

---

## 🧠 Supported Domains

- Foundations (Python, Math, Statistics)
- Data Analysis
- Databases & SQL
- Statistics & Experiments
- Machine Learning
- Advanced Machine Learning
- Deep Learning
- NLP & Generative AI
- MLOps & Deployment
- Cloud & Big Data
- Analytics & BI
- Projects & Case Studies
- Interview Preparation

---

## 🏗️ Tech Stack

Frontend:
- Streamlit

LLM Orchestration:
- LangChain
- Prompt Templates (System + Human messages)
- Context window memory control

Model:
- Google Gemini (gemini-2.5-flash-lite)

Security:
- python-dotenv for environment-based API key handling

---

## 📂 Project Structure

Adaptive-Data-Science-AI-Mentor/ │ ├── app.py ├── requirements.txt ├── .env └── README.md 

---

## 🔄 System Workflow

### Step 1: Domain Selection
User selects a Data Science domain.

### Step 2: Skill Selection
User chooses a specific skill within the selected domain.

### Step 3: Experience Configuration
User sets experience level (0–20 years).

### Step 4: Query Submission
User asks structured, domain-focused questions.

### Step 5: Validation Layer
System verifies whether the question belongs to the selected domain.

### Step 6: LLM Response Generation
Gemini generates an experience-aware response using controlled prompt templates.

### Step 7: Context Management
Conversation memory window is maintained for relevant continuity.

### Step 8: Session Export
User can download the full mentoring session.

---

## 🧠 Core Engineering Concepts Demonstrated

- Large Language Model (LLM) integration
- Prompt engineering for response control
- Domain-restricted AI behavior enforcement
- Context window memory management
- Experience-aware explanation adaptation
- Clean UI development using Streamlit
- Secure environment-based API management

This project showcases structured AI system design — not a generic chatbot implementation.

---

## ⚙️ Installation

1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/Adaptive-Data-Science-AI-Mentor.git
cd Adaptive-Data-Science-AI-Mentor
```

2️⃣ Create Virtual Environment

```Bash
python -m venv venv
```

Activate:

Windows:
```
venv\Scripts\activate
```

Mac/Linux:
```
source venv/bin/activate
```

3️⃣ Install Dependencies

```Bash
pip install -r requirements.txt
```

4️⃣ Add Gemini API Key

Create a .env file in the root folder:

gemini=YOUR_GOOGLE_API_KEY

Get your API key from Google AI Studio.

---

5️⃣ Run the Application

```Bash
streamlit run app.py
```

## 👨‍💻 Author

Shaik Muzahid  
Focused on developing production-oriented Data Science and AI systems with practical LLM integration.
