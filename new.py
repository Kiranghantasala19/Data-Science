import inspect
import re
import streamlit as st

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

from phi.agent import Agent
from phi.model.ollama import Ollama


def extract_code(text: str) -> str:
    """
    Extract only Python code from an LLM response.
    The app relies on this to keep code clean and executable.
    """
    if not text:
        return ""

    python_block = re.search(r"```python\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if python_block:
        return python_block.group(1).strip()

    generic_block = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if generic_block:
        return generic_block.group(1).strip()

    return text.replace("```", "").strip()

# ------------------------------------------------------------
# Helper: safely run an AI agent
# ------------------------------------------------------------

def run_agent(agent: Agent, prompt: str) -> str:
    """
    Runs an agent and safely returns text output.
    Prevents the app from crashing if the model fails.
    """
    try:
        response = agent.run(prompt)
        return response.content.strip() if response and response.content else ""
    except Exception as e:
        return f"[ERROR] {agent.name}: {e}"

llm = Ollama(
    id="tinyllama",
    temperature=0.3,
    top_p=0.9
)

#agents
explainer = Agent(
    name="Explainer",
    model=llm,
    instructions=[
        "Explain the program logic in simple bullet points.",
        "No code."
    ],
    markdown=True
)

developer = Agent(
    name="Developer",
    model=llm,
    instructions=[
        "Write clean, correct Python code.",
        "Return ONLY code.",
        "Wrap code in ```python```."
    ],
    markdown=True
)

debugger = Agent(
    name="Debugger",
    model=llm,
    instructions=[
        "Fix bugs and improve clarity.",
        "Return the FULL corrected code in ```python```."
    ],
    markdown=True
)

reviewer = Agent(
    name="Reviewer",
    model=llm,
    instructions=[
        "Review the code for correctness.",
        "If correct, reply with: STATUS: APPROVED"
    ],
    markdown=True
)

use_case_gen = Agent(
    name="UseCaseGenerator",
    model=llm,
    instructions=[
        "Generate exactly 3 real-world use cases.",
        "Format clearly as: Input → Expected Output"
    ],
    markdown=True
)

# streamlit ui 

st.set_page_config(page_title="Use-Case Driven Code Generator", layout="wide")

st.title("🧩 Use-Case Driven Python Code Generator")
st.write(
    "This app focuses on **real use cases** so you understand "
    "how the generated code should actually be used."
)

task = st.text_area(
    "Describe the Python problem you want to solve:",
    placeholder="Example: Validate an email address"
)

if st.button("🚀 Generate Solution"):

    if not task.strip():
        st.error("Please enter a problem description.")
        st.stop()

    # --------------------------------------------------------
    # Step 1: Understand the problem
    # --------------------------------------------------------
    with st.spinner("Understanding the problem..."):
        explanation = run_agent(explainer, task)

    st.subheader("1️⃣ Problem Understanding")
    st.markdown(explanation)

    # --------------------------------------------------------
    # Step 2: Write initial code
    # --------------------------------------------------------
    with st.spinner("Writing initial code..."):
        raw_code = run_agent(developer, f"Create a Python program for: {task}")
        code = extract_code(raw_code)

    if not code:
        st.error("Code generation failed.")
        st.stop()

    st.subheader("2️⃣ Initial Implementation")
    st.code(code, language="python")

    # --------------------------------------------------------
    # Step 3: Fix and improve the code
    # --------------------------------------------------------
    with st.spinner("Improving code quality..."):
        improved = run_agent(debugger, f"Review and fix this code:\n\n{code}")
        code = extract_code(improved)

    st.subheader("3️⃣ Improved Code")
    st.code(code, language="python")

    # --------------------------------------------------------
    # Step 4: Review correctness
    # --------------------------------------------------------
    with st.spinner("Reviewing correctness..."):
        review = run_agent(reviewer, code)

    st.subheader("4️⃣ Code Review Result")
    if review.startswith("STATUS: APPROVED"):
        st.success(review)
    else:
        st.warning(review)

    # --------------------------------------------------------
    # Step 5: Generate USE CASES (most important part)
    # --------------------------------------------------------
    with st.spinner("Generating real-world use cases..."):
        use_cases = run_agent(use_case_gen, code)

    st.subheader("5️⃣ How to Use This Code (Use Cases)")
    st.markdown(
        "These examples show **how the code should behave** "
        "when used in real situations."
    )
    st.markdown(use_cases)

    # --------------------------------------------------------
    # Step 6: Final verified code
    # --------------------------------------------------------
    st.subheader("✅ Final Verified Python Code")
    st.code(code, language="python")