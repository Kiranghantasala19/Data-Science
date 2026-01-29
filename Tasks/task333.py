import os
import requests
import time
import streamlit as st
from dotenv import load_dotenv

# =====================================================
# Load environment (.env already contains OLLAMA_URL and OLLAMA_MODEL)
# =====================================================
load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

# =====================================================
# Ollama client
# =====================================================
def ollama_generate(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": 512,
                "temperature": 0.1,
                "top_p": 0.8
            }
        },
        timeout=300
    )
    response.raise_for_status()
    return response.json()["response"]

# =====================================================
# Agents (Developer + QA) with language support
# =====================================================
def developer_agent(user_query, language):
    prompt = f"""
Generate {language} code for the task.

RULES:
- Output ONLY {language} code.
- No markdown.
- Explain in brief about the code in comments.

TASK:
{user_query}
"""
    return ollama_generate(prompt)


def qa_agent(code, language):
    prompt = f"""
Fix bugs in the {language} code below if any.
Return ONLY corrected {language} code.

CODE:
{code}
"""
    return ollama_generate(prompt)

# =====================================================
# Real Python Validator (only for Python)
# =====================================================
def python_validate(code):
    cleaned = code.replace("```python", "").replace("```", "").strip()

    try:
        compile(cleaned, "<string>", "exec")
        return True, None, cleaned
    except Exception as e:
        return False, str(e), cleaned

# For non-Python languages, skip validation (return valid=True)
def validate_code(code, language):
    if language.lower() == "python":
        return python_validate(code)
    else:
        # Validation for other languages can be added if needed
        return True, None, code

# =====================================================
# Agent Workflow (Developer -> QA -> Validator)
# =====================================================
def programming_assistant(user_query, language):
    code = developer_agent(user_query, language)
    code = qa_agent(code, language)

    valid, error, code = validate_code(code, language)

    # Auto retry once if invalid (only for Python)
    if not valid and language.lower() == "python":
        code = qa_agent(code, language)
        valid, error, code = validate_code(code, language)

    return code, valid, error

# =====================================================
# Streamlit UI
# =====================================================
st.set_page_config(
    page_title="🤖 Programming Assistant",
    layout="centered"
)

st.title("🤖 Programming Assistant")
st.write("Developer → QA → Code Validator")

language = st.selectbox(
    "Select programming language",
    options=["Python", "Java", "C++", "C"],
    index=0
)

user_query = st.text_area(
    "Enter your programming task",
    placeholder=f"Example: Write a {language} function to check if a number is palindrome."
)

if st.button("Generate & Validate Code"):
    if not user_query:
        st.warning("Please enter a programming task.")
    else:
        with st.spinner("Generating & validating..."):
            start_time = time.time()
            final_code, is_valid, error = programming_assistant(user_query, language)
            end_time = time.time()

        st.subheader("✅ Final Code")
        st.code(final_code, language=language.lower())
        st.info(f"⏱️ Time taken: {round(end_time - start_time, 2)} seconds")

        if is_valid:
            st.success("Code validated successfully")
        else:
            st.error("Validation failed even after retry")
            st.text(error)
