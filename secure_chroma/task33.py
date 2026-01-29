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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")  # you already updated this

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
                # keep this small for speed
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
# Agents (Developer + QA)
# =====================================================
def developer_agent(user_query):
    prompt = f"""
Generate Python code for the task.

RULES:
- Output ONLY Python code.
- No markdown.
- Explain in brief about the code in comments.

TASK:
{user_query}
"""
    return ollama_generate(prompt)


def qa_agent(code):
    prompt = f"""
Fix bugs in the Python code below if any.
Return ONLY corrected Python code.

CODE:
{code}
"""
    return ollama_generate(prompt)

# =====================================================
# Real Python Validator
# =====================================================
def python_validate(code):
    # Remove markdown if model adds it
    cleaned = code.replace("```python", "").replace("```", "").strip()

    try:
        compile(cleaned, "<string>", "exec")
        return True, None, cleaned
    except Exception as e:
        return False, str(e), cleaned

# =====================================================
# Agent Workflow (Developer -> QA -> Python Validator)
# =====================================================
def programming_assistant(user_query):
    code = developer_agent(user_query)
    code = qa_agent(code)

    valid, error, code = python_validate(code)

    # Auto retry once if invalid
    if not valid:
        code = qa_agent(code)
        valid, error, code = python_validate(code)

    return code, valid, error

# =====================================================
# Streamlit UI
# =====================================================
st.set_page_config(
    page_title="🤖 Programming Assistant",
    layout="centered"
)

st.title("🤖 Programming Assistant")
st.write("Developer → QA → Python Validator ")

user_query = st.text_area(
    "Enter your programming task",
    placeholder="Example: Write a Python function to check if a number is palindrome."
)

if st.button("Generate & Validate Code"):
    if not user_query:
        st.warning("Please enter a programming task.")
    else:
        with st.spinner("Generating & validating..."):
            start_time = time.time()
            final_code, is_valid, error = programming_assistant(user_query)
            end_time = time.time()

        st.subheader("✅ Final Code")
        st.code(final_code, language="python")
        st.info(f"⏱️ Time taken: {round(end_time - start_time, 2)} seconds")


        if is_valid:
            st.success("Code validated successfully")
        else:
            st.error("Validation failed even after retry")
            st.text(error)