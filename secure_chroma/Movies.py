import json
import httpx
import streamlit as st
from phi.agent import Agent
from phi.model.ollama import Ollama
from fpdf import FPDF

# Tool: Fetch movie data from OMDb API
def get_movie_data(title: str) -> str:
    """Fetch movie data from OMDb API."""
    api_key = "51e24d4c-2067-429e-b716-fc214e869c97" 
    url = f"http://www.omdbapi.com/?t={title}&apikey={api_key}&plot=full"
    response = httpx.get(url)
    return json.dumps(response.json(), indent=2)

def get_movie_ratings(imdb_id: str) -> str:
    """Fetch detailed ratings for a movie."""
    api_key = "51e24d4c-2067-429e-b716-fc214e869c97" 
    url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={api_key}"
    response = httpx.get(url)
    return json.dumps(response.json(), indent=2)

def search_movies(title: str) -> str:
    """Search for movies by title."""
    api_key = "51e24d4c-2067-429e-b716-fc214e869c97" 
    url = f"http://www.omdbapi.com/?s={title}&apikey={api_key}"
    response = httpx.get(url)
    return json.dumps(response.json(), indent=2)

# Agent 1: Primary Analysis Agent - fetches and analyzes movie data
analysis_agent = Agent(
    name="Movie Analysis Agent",
    model=Ollama(id="llama3.2"),
    tools=[get_movie_data, get_movie_ratings, search_movies],
    instructions=[
        "Fetch comprehensive movie data using the tools provided.",
        "Answer user questions based strictly on the retrieved movie data.",
        "Generate a well-structured paragraph summary combining all movie details."
    ],
    description="Retrieves and analyzes movie metadata from OMDb API.",
    show_tool_calls=True,
    markdown=True,
)

# Agent 2: Document Agent - generates PDF reports
def save_to_pdf(content: str, filename: str = "movie_report.pdf") -> bytes:
    """Save content to a PDF file and return bytes."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, content)
    pdf_bytes = pdf.output(dest="S").encode("latin1")
    return pdf_bytes

pdf_agent = Agent(
    name="Document Agent",
    model=Ollama(id="llama3.2"),
    tools=[save_to_pdf],
    instructions=["Transform summarized content into a professionally formatted PDF."],
)

# Streamlit UI
st.title("🎬 Movie Intelligence System")

movie_title = st.text_input("Enter a movie title:")
user_question = st.text_area("Ask a question about the movie (optional):")

if st.button("Generate Movie Report"):
    if movie_title:
        # Step 1: Fetch and analyze movie data
        query = f"Get comprehensive data for the movie '{movie_title}'"
        if user_question:
            query += f" and answer: {user_question}"
        
        analysis_response = analysis_agent.run(query, stream=False)
        analysis_content = analysis_response.content

        # Step 2: Generate summary
        summary_prompt = f"Create a detailed, well-structured summary of this movie data: {analysis_content}"
        summary_response = analysis_agent.run(summary_prompt, stream=False)
        summary_content = summary_response.content

        # Step 3: Generate PDF
        pdf_bytes = save_to_pdf(summary_content)

        # Display results
        st.subheader("Movie Analysis")
        st.write(summary_content)

        st.download_button(
            label="Download Movie Report PDF",
            data=pdf_bytes,
            file_name=f"{movie_title}_report.pdf",
            mime="application/pdf"
        )
    else:
        st.warning("Please enter a movie title.")