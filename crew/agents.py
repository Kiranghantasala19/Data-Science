import os

os.environ["GOOGLE_API_KEY"] = "AIzaSyAeHb96PvJlX8-Nf8lAVSXf92XuumBWV3U"

from crewai import Agent,LLM
from langchain_google_genai import ChatGoogleGenerativeAI


#model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)


# Agent 1: Translator
translator_agent = Agent(
    role="Language Translator",
    goal="Translate user input into the requested language accurately",
    backstory="You are an expert multilingual translator fluent in all major languages.",
    llm=llm,
    verbose=True
)


# Agent 2: Summarizer
summarizer_agent = Agent(
    role="Content Summarizer",
    goal="Summarize the translated text concisely",
    backstory="You are skilled at creating clear, concise summaries of any content.",
    llm=llm,
    verbose=True
)
