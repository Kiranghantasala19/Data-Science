import os
import streamlit as st
import requests
import json
from fpdf import FPDF
from phi.agent import Agent
from phi.model.ollama import Ollama

os.environ["OPENAI_API_KEY"] = ""

st.set_page_config(
    page_title="SkyFetch - AI Weather System",
    page_icon="🌤️",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stApp {background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);}
.hero-title {
    font-family: 'Inter', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(90deg, #00d4ff, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

weather_key = "c99afaea5f9fe2355d4ad13aec77d0d7"

ollama_model = Ollama(
    id="tinyllama",
    options={
        "num_ctx": 256,
        "num_predict": 200
    }
)

def get_weather(city: str) -> dict:
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": weather_key, "units": "metric"}
    response = requests.get(url, params=params)
    data = response.json()
    if response.status_code != 200:
        raise Exception(data.get("message", "Weather fetch failed"))
    temp_c = data["main"]["temp"]
    temp_f = round((temp_c * 9 / 5) + 32, 1)
    return {
        "City": data["name"],
        "Temperature": f"{temp_c}°C / {temp_f}°F",
        "Description": data["weather"][0]["description"],
        "Humidity": f"{data['main']['humidity']}%",
        "Wind Speed": f"{data['wind']['speed']} m/s"
    }

def create_weather_pdf(text: str) -> str:
    filename = "weather_report.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "SkyFetch Weather Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    safe_text = text.encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(0, 10, safe_text)
    pdf.output(filename)
    return filename

content_agent = Agent(
    name="WeatherWriter",
    model=ollama_model,
    instructions=[
        "You are a friendly weather reporter.",
        "Summarize the weather clearly.",
        "Give simple clothing advice.",
        "Answer the user's question naturally.",
        "Do not mention JSON or raw data."
    ]
)

st.markdown("<h1 class='hero-title'>🌤️ SkyFetch</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtitle'>AI Weather Intelligence System</p>", unsafe_allow_html=True)

city = st.text_input("🏙️ Enter City Name", "Hyderabad")
question = st.text_input("💬 Optional Question", "Is it good for a picnic?")

if st.button("🚀 Generate Weather Report"):
    try:
        st.info("Fetching weather data...")
        weather_data = get_weather(city)

        prompt = f"""
Weather Data:
{json.dumps(weather_data, indent=2)}

User Question:
{question}
"""

        st.info("Writing report...")
        report = content_agent.run(prompt).content

        st.info("Generating PDF...")
        create_weather_pdf(report)

        st.success("Report ready!")

        st.markdown(
            f"""
            <div style="background:rgba(255,255,255,0.05);padding:15px;border-radius:12px;">
                <h3>Weather Report for {city}</h3>
                <p>{report}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        with open("weather_report.pdf", "rb") as f:
            st.download_button(
                "📥 Download PDF Report",
                f,
                file_name=f"SkyFetch_{city}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    except Exception as e:
        st.error(str(e))
        st.info("Make sure Ollama is running and tinyllama is pulled")