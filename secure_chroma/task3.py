import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM

# --------------------------------------------------
# Load environment variables (.env contains GROQ_API_KEY)
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# Configure Groq LLM (ACTIVE MODEL ONLY)
# --------------------------------------------------
groq_llm = LLM(
    model="groq/llama-3.1-8b-instant",
    temperature=0.2
)

# --------------------------------------------------
# Agent 1: Problem Analyzer
# --------------------------------------------------
analyzer_agent = Agent(
    role="Problem Analyzer",
    goal="Analyze the programming problem and define the correct algorithm",
    backstory=(
        "You are an expert software architect who specializes in "
        "breaking down programming problems into clear algorithmic steps."
    ),
    llm=groq_llm,
    verbose=True
)

# --------------------------------------------------
# Agent 2: Python Developer
# --------------------------------------------------
developer_agent = Agent(
    role="Python Developer",
    goal="Write clean, correct, and executable Python code",
    backstory=(
        "You are a senior Python engineer known for writing efficient, "
        "readable, and production-ready code."
    ),
    llm=groq_llm,
    verbose=True
)

# --------------------------------------------------
# Agent 3: Code Reviewer
# --------------------------------------------------
reviewer_agent = Agent(
    role="Code Reviewer",
    goal="Validate logic, correctness, and best practices",
    backstory=(
        "You are a strict code reviewer who ensures correctness, "
        "efficiency, and adherence to Python best practices."
    ),
    llm=groq_llm,
    verbose=True
)

# --------------------------------------------------
# Task 1: Problem Analysis (Fibonacci)
# --------------------------------------------------
analysis_task = Task(
    description=(
        "Analyze the following problem:\n"
        "Write a Python function that generates the Fibonacci series "
        "up to n terms.\n"
        "Explain the correct algorithm and logic."
    ),
    expected_output="A clear explanation of the Fibonacci series algorithm.",
    agent=analyzer_agent
)

# --------------------------------------------------
# Task 2: Code Generation (Fibonacci)
# --------------------------------------------------
development_task = Task(
    description=(
        "Based on the analysis, write executable Python code for a function "
        "named fibonacci(n) that returns the Fibonacci series up to n terms.\n"
        "Return ONLY valid Python code."
    ),
    expected_output="Executable Python function fibonacci(n).",
    agent=developer_agent
)

# --------------------------------------------------
# Task 3: Code Review & Validation
# --------------------------------------------------
review_task = Task(
    description=(
        "Review the generated Fibonacci Python code for correctness, "
        "edge cases, and efficiency.\n"
        "Return FINAL executable Python code only."
    ),
    expected_output="Final, corrected Fibonacci Python code.",
    agent=reviewer_agent
)

# --------------------------------------------------
# Create Crew (Sequential SDLC Simulation)
# --------------------------------------------------
crew = Crew(
    agents=[
        analyzer_agent,
        developer_agent,
        reviewer_agent
    ],
    tasks=[
        analysis_task,
        development_task,
        review_task
    ],
    process=Process.sequential,
    verbose=True
)

# --------------------------------------------------
# Run Crew
# IMPORTANT: Do NOT print the result manually
# CrewAI handles all output internally
# --------------------------------------------------
if __name__ == "__main__":
    crew.kickoff()
