import os
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

import time
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
# Ollama LLM wrapper
# =====================================================
ollama_llm = LLM(
    model=f"ollama/{OLLAMA_MODEL}",
    base_url=OLLAMA_URL,
    temperature=0.1,
    top_p=0.8,
    num_ctx=1024
)

# =====================================================
# Agents
# =====================================================
usecase_agent = Agent(
    role="Use Case Analyst",
    goal="Produce detailed functional and edge-case analysis",
    backstory="Expert requirements engineer producing formal reports.",
    llm=ollama_llm
)

developer_agent = Agent(
    role="Software Developer",
    goal="Generate clean, correct Python code",
    backstory="Senior Python engineer.",
    llm=ollama_llm
)

qa_agent = Agent(
    role="Code Quality Analyst",
    goal="Fix logical, syntactic, and edge-case issues",
    backstory="Strict QA reviewer.",
    llm=ollama_llm
)

validator_agent = Agent(
    role="Validation Agent",
    goal="Validate Python code correctness and executability",
    backstory="Mentally executes Python code.",
    llm=ollama_llm
)

usecase_test_agent = Agent(
    role="Use Case Test Agent",
    goal="Exhaustively test Python code against all use cases",
    backstory="Test engineer generating formal test reports.",
    llm=ollama_llm
)

# =====================================================
# Workflow
# =====================================================
def programming_assistant(user_query):
    reports = {}

    usecase_task = Task(
        description=f"""
Analyze the task below and produce a DETAILED USE CASE REPORT.

TASK:
{user_query}

MANDATORY SECTIONS:
1. Functional Requirements (min 5)
2. Non-Functional Requirements
3. Edge Cases (min 5)
4. Assumptions
5. Constraints
6. Input / Output Specification
""",
        expected_output="Detailed structured use case report",
        agent=usecase_agent
    )

    dev_task = Task(
        description="""
Using the use case report, write Python code.
Handle all edge cases.
Output ONLY Python code.
""",
        expected_output="Python source code",
        agent=developer_agent
    )

    qa_task = Task(
        description="""
Review the Python code for bugs and bad practices.
Fix issues.
Return ONLY corrected Python code.
""",
        expected_output="QA-corrected Python code",
        agent=qa_agent
    )

    validation_task = Task(
        description="""
Validate Python code by simulating execution.
Fix runtime or syntax errors.
Return ONLY corrected Python code.
""",
        expected_output="Validated Python code",
        agent=validator_agent
    )

    usecase_test_task = Task(
        description="""
Create a FORMAL USE CASE TEST REPORT.

MANDATORY:
1. Test Strategy
2. Test Case Table (Use Case | Input | Expected | Actual | Status)
3. Edge Case Validation
4. Failure Analysis
5. Final Verdict
""",
        expected_output="Formal use case test report",
        agent=usecase_test_agent
    )

    final_code_task = Task(
        description="""
Return FINAL Python code after all fixes.
Output ONLY Python code.
""",
        expected_output="Final executable Python code",
        agent=developer_agent
    )

    crew = Crew(
        agents=[
            developer_agent,
            qa_agent,
            validator_agent,
            usecase_agent,      # 4th position (as requested)
            usecase_test_agent
        ],
        tasks=[
            usecase_task,
            dev_task,
            qa_task,
            validation_task,
            usecase_test_task,
            final_code_task
        ],
        process=Process.sequential
    )

    crew.kickoff()

    reports["Use Case Report"] = usecase_task.output
    reports["Use Case Test Report"] = usecase_test_task.output
    reports["Final Python Code"] = final_code_task.output

    return reports

# =====================================================
# Streamlit UI
# =====================================================
st.set_page_config(page_title="Agentic Python Programming Assistant")

st.title("🤖 Agentic Python Programming Assistant")

user_query = st.text_area(
    "Enter Python programming task",
    placeholder="Example: Write a Python function to validate an email address."
)

if st.button("Generate"):
    if not user_query:
        st.warning("Please enter a task.")
    else:
        with st.spinner("Agents working..."):
            start = time.time()
            reports = programming_assistant(user_query)
            end = time.time()

        st.subheader("📘 Use Case Report")
        st.text(reports["Use Case Report"])

        st.subheader("🧪 Use Case Test Report")
        st.text(reports["Use Case Test Report"])

        st.subheader("✅ Final Python Code")
        st.code(reports["Final Python Code"], language="python")

        st.info(f"⏱️ Time taken: {round(end - start, 2)} seconds")
