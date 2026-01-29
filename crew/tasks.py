from agents import translator_agent,summarizer_agent
from crewai import Task
# Task 1: Translation
translation_task = Task(
    description="Translate the following text to {target_language}: {user_input}",
    expected_output="The translated text in {target_language}.",
    agent=translator_agent
)

# Task 2: Summarization (uses output from Task 1 automatically)
summarization_task = Task(
    description="Summarize the translated text from the previous task.in the ueser needed langauge which is {target_language}",
    expected_output="A brief summary of the translated content.",
    agent=summarizer_agent
)

