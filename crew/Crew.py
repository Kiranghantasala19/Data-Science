from agents import translator_agent,summarizer_agent
from tasks import translation_task,summarization_task
from crewai import Crew

# Create and run the crew
crew = Crew(
    agents=[translator_agent, summarizer_agent],
    tasks=[translation_task, summarization_task],
    verbose=True
)


# Run with user inputs
result = crew.kickoff(inputs={
    "user_input": "Artificial intelligence is transforming how we work and live.",
    "target_language": "Telugu"
})

