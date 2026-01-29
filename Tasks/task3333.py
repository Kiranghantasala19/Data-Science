import os
import time
import requests
import streamlit as st
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM

# =====================================================
# Load environment variables
# =====================================================
load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

# =====================================================
# Ollama LLM wrapper for CrewAI
# =====================================================
ollama_llm = LLM(
    model=f"ollama/{OLLAMA_MODEL}",
    base_url=OLLAMA_URL,
    temperature=0.1,
    top_p=0.8,
    num_ctx=512
)

# =====================================================
# Python Validator
# =====================================================
def python_validate(code):
    cleaned = code.replace("```python", "").replace("```", "").strip()
    try:
        compile(cleaned, "<string>", "exec")
        return True, None, cleaned
    except Exception as e:
        return False, str(e), cleaned


def validate_code(code, language):
    if language.lower() == "python":
        return python_validate(code)
    return True, None, code


# =====================================================
# CrewAI Agents
# =====================================================
developer_agent = Agent(
    role="Software Developer",
    goal="Generate clean, correct, and efficient code",
    backstory=(
        "You are a senior software engineer who writes production-ready code "
        "with brief inline comments for explanation."
    ),
    llm=ollama_llm,
    verbose=False
)

qa_agent = Agent(
    role="Code Quality Analyst",
    goal="Identify and fix bugs or logical issues in code",
    backstory=(
        "You are a strict QA engineer who reviews code for correctness, syntax, "
        "and best practices. You return only corrected code."
    ),
    llm=ollama_llm,
    verbose=False
)

# =====================================================
# CrewAI Workflow
# =====================================================
def programming_assistant(user_query, language):
    dev_task = Task(
        description=f"""
Generate {language} code for the following task.

RULES:
- Output ONLY {language} code
- No markdown
- Include brief explanation as comments

TASK:
{user_query}
""",
        expected_output=f"Valid {language} source code",
        agent=developer_agent
    )

    qa_task = Task(
        description=f"""
Review and fix bugs in the {language} code produced by the developer.
Return ONLY corrected {language} code.
""",
        expected_output=f"Corrected {language} code",
        agent=qa_agent
    )

    crew = Crew(
        agents=[developer_agent, qa_agent],
        tasks=[dev_task, qa_task],
        process=Process.sequential
    )

    result = crew.kickoff()
    code = result.raw

    valid, error, cleaned_code = validate_code(code, language)

    # Retry once if Python validation fails
    if not valid and language.lower() == "python":
        retry_task = Task(
            description=f"""
The following Python code has errors. Fix them.

ERROR:
{error}

CODE:
{cleaned_code}

Return ONLY corrected Python code.
""",
            expected_output="Corrected Python code",
            agent=qa_agent
        )

        retry_crew = Crew(
            agents=[qa_agent],
            tasks=[retry_task],
            process=Process.sequential
        )

        retry_result = retry_crew.kickoff()
        cleaned_code = retry_result.raw
        valid, error, cleaned_code = validate_code(cleaned_code, language)

    return cleaned_code, valid, error


# =====================================================
# Streamlit UI
# =====================================================
st.set_page_config(
    page_title="Programming Assistant (CrewAI)",
    layout="centered"
)

st.title("🤖 Programming Assistant")
st.write("Developer Agent → QA Agent → Validator")

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
        with st.spinner("Agents are working..."):
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
