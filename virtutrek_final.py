import streamlit as st
import os

import nltk
nltk.download('punkt')

nltk_data_path = "/tmp/nltk_data"

# Ensure the directory exists
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# Tell NLTK to use the custom directory
nltk.data.path.append(nltk_data_path)

# Download the punkt tokenizer if it's not already available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

os.environ["STREAMLIT_FILE_WATCHER_TYPE"] = "none"

st.legacy_caching.clear_cache()

def main_page():
    st.set_page_config(page_title="VirtuTrek-Personalized AI Travel Assistant ",page_icon="🌍", layout="wide")
    # Title of the page with custom color
    st.markdown("<h1 style='color: #00FFFF;'>VirtuTek: Experience History Through Technology</h1>", unsafe_allow_html=True)

    # Vision of the Project with custom color
    st.markdown("<h2 style='color: #00FFFF;'>Project Vision</h2>", unsafe_allow_html=True)
    st.write("""
    **Heritage Horizons** aims to bridge the gap between modern technology and ancient history. Our vision is to create an immersive and educational virtual experience that allows users to explore heritage sites, monuments, and cultural artifacts in an intuitive and interactive way.

    Whether you're a student, traveler, history enthusiast, or researcher, our platform makes exploring the past accessible and engaging.
    """)

    # Services provided by the project with custom color
    st.markdown("<h2 style='color: #00FFFF;'>Services We Provide</h2>", unsafe_allow_html=True)
    st.write("""
    Explore the rich cultural heritage of the world using our suite of intelligent services:

    - **Home Dashboard**: Your central hub to navigate all the features of the platform and access general information.
    - **AI Chatbot**: Ask questions about historical monuments, cultures, or ancient artifacts — our AI-powered assistant responds like a virtual tour guide.
    - **Image Analysis**: Upload photos of monuments or artifacts and get intelligent descriptions and historical insights instantly.
    - **Map Viewer**: Visually explore heritage sites on an interactive map to learn about their significance and location-based context.
    - **Tour Planner**: Plan your perfect journey from photo to destination — explore landmarks, routes, and live weather, all in one smart tour planner.
    """)

    # Impact of the Project with custom color
    st.markdown("<h2 style='color: #00FFFF;'>Why Heritage Horizons Matters</h2>", unsafe_allow_html=True)
    st.write("""
    Our platform redefines how people learn and interact with cultural heritage:

    - **Immersive Education**: Engage with history in an interactive format, far beyond traditional textbooks or static articles.
    - **Accessible Heritage**: Bring global monuments to your screen, breaking geographical barriers to learning.
    - **Empowering Tourism**: Equip travelers with insightful context before or during visits to historic places.
    - **Smart Exploration**: Leverage AI to understand cultural relevance, symbolism, and historical timelines from images or text queries.
    """)
    
    st.markdown("<h2 style='color: #00FFFF;'>About Us</h2>", unsafe_allow_html=True)
    st.markdown("""
**VirtuTrek** is built by:
- **Anand** – AI Developer 
- **Atishay** – AI Developer 
- **Kelvin** – AI Developer 
- **Vineetha** – AI Developer 


        
""")
    st.markdown("We're a passionate team of engineers dedicated to making heritage accessible through tech!")
    
# =============================================SIDEBAR_PANNEL=====================================================


    # Sidebar navigation
    
    st.sidebar.markdown("<h2 style='color: #00FFFF;'>VirtuTrek-Personalized AI Travel Assistant </h2>", unsafe_allow_html=True)
    st.sidebar.markdown("<h2 style='color: #00FFFF;'>Explore Our Services</h2>", unsafe_allow_html=True)
    st.sidebar.write("Select a section to begin your virtual tour:")

    # Sidebar buttons for navigation
    
    if st.sidebar.button("🗺️ Tour Planner"):
        st.session_state.page = "tour_plan"
        st.rerun()

    if st.sidebar.button("🖼️ Image Analysis"):
        st.session_state.page = "image_analysis"
        st.rerun()
    
    if st.sidebar.button("💬 Chatbot"):
        st.session_state.page = "chatbot"
        st.rerun()


    # Footer Section with custom color
    st.write("VirtuTrek | All rights reserved © 2025")
    
    
    
# ===============================================CHATBOT================================================================  
    
    
    
def chatbot_page():
    
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.agents import Agent
    from langchain.tools import Tool
    from langchain.agents import AgentExecutor,ZeroShotAgent
    from langchain_community.utilities import SerpAPIWrapper
    from langchain.prompts import PromptTemplate
    from langchain.agents import initialize_agent, AgentType
    from langchain.memory import ConversationBufferMemory
    import requests
    from dotenv import load_dotenv
    import os
    import re
    import json
    import streamlit as st


    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    serp_api_key = os.getenv("SERPAPI_API_KEY")
    weather = os.getenv("WEATHER_API_KEY")

    def search_google(query: str) -> str:
        """Search Google using SerpAPI."""
        serp = SerpAPIWrapper()
        results = serp.run(query)
        return results


    def get_weather(city:str):
        params = {
            "q": city,
            "key": weather,
            "aqi": "yes"
        }
        response = requests.get("https://api.weatherapi.com/v1/current.json", params=params).json()

        #print(json.dumps(response, indent=2))  # Debug the full response

        if "current" in response:
            return f"""Weather in {city}:
                    - Condition: {response["current"]["condition"]["text"]}
                    - Temperature: {response["current"]["temp_c"]}°C
                    - Feels Like: {response["current"]["feelslike_c"]}°C
                    - Humidity: {response["current"]["humidity"]}%
                    - Wind Speed: {response["current"]["wind_kph"]} kph
                    - UV Index: {response["current"].get("uv", "N/A")}
                    """
        return "Weather data not available."
            
            
    web_search = Tool(
        name = "Web Search",
        func=search_google,
        description="A tool to search the web using Google. Input should be a string like 'What is the capital of France?'.",
    )

    getweather = Tool(
        name = "Weather Search",
        func=get_weather,
        description="A tool to get teh weather o fteh heritage site or any city. Input should be a string like 'how is the weather in Paris?'.",
    )

    tools = [web_search, getweather]


    class CategorizerAgent:
        def __init__(self):
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
            self.tools = tools
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                memory=st.session_state.memory,
                handle_parsing_errors=True
            )
        
        def categorize_topic(self, text):
            prompt = f"""
            You are the Categorizer Agent, a specialized AI component responsible for analyzing the user's input {text} and classifying it into one of the predefined categories based strictly on the content and intent of the message.

            Your job is to accurately detect the user’s intent and return only the matching category label from the list below. You should not perform any processing or delegation yourself — your role is limited to categorization only.

            📋 Available Categories
            Choose only one of the following:

            "Historical Information" – if the user is asking about the history of a place, monument, or heritage site.

            "Architectural Details" – if the user is interested in design, structure, style, or architecture of a place.

            "Travel or Logistics" – if the user is asking about how to reach a place, travel duration, entry fee, timings, routes, etc.

            "Accommodation and Dining" – if the user is seeking places to stay, eat, nearby attractions, or leisure activities.

            "General Conversation" – if the user is making casual remarks, greetings, jokes, or off-topic chit-chat.
            
            "Weather Information"  if the user is asking about the weather of a place.

            "Unrecognized" – if the input does not fit into any of the above categories.

            ✅ Output Format
            Return only a single line of text with just the category label, nothing else.
            Do not explain or comment on your decision. Do not repeat the input.
            """
            response = self.llm.invoke(prompt).content.strip()
            
            # Clean markdown-style formatting (if any)
            response = re.sub(r"^```json|```$", "", response).strip()

            try:
                return response
            except json.JSONDecodeError:
                return {"error": "Invalid JSON format returned", "raw_response": response}
            

    class History_Expert_Subagent:
        def __init__(self):
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
            self.tools = tools
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                memory=st.session_state.memory,
                handle_parsing_errors=True
            )
        
        def history_info(self, text):
            prompt = f"""
                You are the History Expert Subagent, responsible for providing detailed, accurate, and insightful historical context for any heritage site, landmark, or culturally significant location the user inquires about.

                Your responsibilities:

                Provide verified historical facts (e.g., date of construction, founder, purpose, major events).

                Highlight the cultural and political significance of the location.

                Mention any legends, folklore, or myths associated with the place, if applicable.

                Be clear, concise, and structured — no fluff or filler.
                
                user input: {text}

                Return your output in this format:

                Name of Site:
                Historical Overview:
                Timeline of Key Events:
                Interesting Facts or Stories:
                Source Reliability: High / Medium / Low
            """
            response = self.llm.invoke(prompt).content.strip()
            
            # Clean markdown-style formatting (if any)
            response = re.sub(r"^```json|```$", "", response).strip()

            try:
                return response
            except json.JSONDecodeError:
                return {"error": "Invalid JSON format returned", "raw_response": response}

    class Weather_forecaster_Subagent:
        def __init__(self):
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
            self.tools = tools
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                handle_parsing_errors=True
            )
        
        def weather_info(self, text):
            prompt = f"""
                        You are a Weather Location Extractor Subagent. Your job is to extract the name of the city or place the user is asking about in their query.

                        Only return the name of the city or place — nothing else.

                        The place name should be suitable to pass into a weather API.

                        User input: {text}

                        Extracted location:
                    """

            response = self.llm.invoke(prompt).content.strip()
            
            # Clean markdown-style formatting (if any)
            response = re.sub(r"^```json|```$", "", response).strip()

            try:
                result = get_weather(response)
                prompt = f"you are a weather forecaster, you are ordered to provide weatehr details ina systematic and easy way fro teh user to understand by using the providee weather details: {result}"
                response = self.llm.invoke(prompt).content.strip()
                return response
            except json.JSONDecodeError:
                return {"error": "Invalid JSON format returned", "raw_response": response}        

    class Architectural_Expert_Subagent:
        def __init__(self):
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
            self.tools = tools
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                memory=st.session_state.memory,
                handle_parsing_errors=True
            )
        
        def architecture_info(self, text):
            prompt = f"""
            You are the Architectural Expert Subagent, responsible for analyzing and explaining the structural, stylistic, and artistic aspects of a monument, temple, fort, or other built heritage.

            Your responsibilities:

            Identify the architectural style (e.g., Mughal, Dravidian, Gothic, Colonial).

            Mention the materials, structural techniques, and artistic features used.

            Point out any symbolic design elements or layout significance.

            Provide comparisons with similar structures if relevant.
            
            user input: {text}

            Return your output in this format:

            Name of Site:
            Architectural Style:
            Materials Used:
            Structural Highlights:
            Artistic Elements (e.g., carvings, frescoes, motifs):
            Special Features / Innovations:
            """
            response = self.llm.invoke(prompt).content.strip()
            
            # Clean markdown-style formatting (if any)
            response = re.sub(r"^```json|```$", "", response).strip()

            try:
                return response
            except json.JSONDecodeError:
                return {"error": "Invalid JSON format returned", "raw_response": response}
        

    class Travel_Logistics_Subagent:
        def __init__(self):
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
            self.tools = tools
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                memory=st.session_state.memory,
                handle_parsing_errors=True
            )
        
        def travel_info(self, text):
            prompt = f"""
            You are the Travel Logistics Subagent, tasked with providing up-to-date and practical information on how to reach a specific tourist location.

            Your responsibilities:

            Provide recommended modes of transport (train, bus, flight, taxi).

            Mention nearest transport hubs (airport, railway station).

            Approximate travel time and cost from common points (e.g., major cities).

            Include entry fees, opening/closing times, and best visiting seasons.

            Mention accessibility (elderly-friendly, wheelchair access, etc.) if applicable.
            
            user input: {text}

            Return your output in this format:

            Location:
            Nearest Airport/Station:
            Travel Options:
            Travel Duration & Cost Estimates:
            Entry Fee & Timings:
            Best Time to Visit:
            Accessibility Notes:

            """
            response = self.llm.invoke(prompt).content.strip()
            
            # Clean markdown-style formatting (if any)
            response = re.sub(r"^```json|```$", "", response).strip()

            try:
                return response
            except json.JSONDecodeError:
                return {"error": "Invalid JSON format returned", "raw_response": response}
            

    class Accommodation_and_Dining_Expert_Subagent:
        def __init__(self):
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
            self.tools = tools
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                memory=st.session_state.memory,
                handle_parsing_errors=True
            )
        
        def accommodation_info(self, text):
            prompt = f"""
            You are the Accommodation and Dining Expert Subagent, responsible for suggesting where to stay, what to eat, and what else to explore nearby.

            Your responsibilities:

            Recommend accommodations across budget ranges (luxury, mid-range, budget).

            Suggest authentic or popular local dining spots.

            Recommend nearby attractions and activities for tourists.

            Mention safety tips and local etiquette if relevant.
            
            user input: {text}

            Return your output in this format:

            Location:
            Top Accommodation Picks:
            - Luxury: 
            - Mid-range: 
            - Budget:
            Recommended Eateries:
            - Local Cuisine:
            - Vegetarian/Vegan Options:
            Nearby Attractions/Activities:
            Safety Tips & Local Etiquette:
            """
            response = self.llm.invoke(prompt).content.strip()
            
            # Clean markdown-style formatting (if any)
            response = re.sub(r"^```json|```$", "", response).strip()

            try:
                return response
            except json.JSONDecodeError:
                return {"error": "Invalid JSON format returned", "raw_response": response}
            
            
    class Conversational_tour_guide_subagent:
        def __init__(self):
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
            self.tools = tools
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                memory=st.session_state.memory,
                handle_parsing_errors=True
            )
        
        def convo_info(self, text):
            prompt = f"""
            You are a Conversational tour guide subagent. Your role is to engage in friendly, casual conversation with users as if you’re accompanying them on a relaxed tour — through a city, a museum, a campus, or even a digital platform.

            Your tone is warm, conversational, and attentive. You don’t wait for commands or tasks; instead, you respond naturally to whatever the user says — like a human guide would during small talk.

            If the user shares a feeling, observation, or random thought, respond with something related, thoughtful, or playful. You are not here to perform tasks, give definitions, or answer deep factual queries — just vibe and keep the energy light, interesting, and human.

            Think of yourself as a mix of a knowledgeable host and a good listener — someone who knows the place well, but never dominates the conversation.

            Keep responses short and context-aware. Ask casual follow-up questions if it feels right. Never prompt the user to “ask a question.” Just go with the flow.
            
            user input: {text}
            """
            response = self.llm.invoke(prompt).content.strip()
            
            # Clean markdown-style formatting (if any)
            response = re.sub(r"^```json|```$", "", response).strip()

            try:
                return response
            except json.JSONDecodeError:
                return {"error": "Invalid JSON format returned", "raw_response": response}
            
    
    # Initialize memory if not already
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True)        
            
    def get_agent_response(agent, category, topic,memory=st.session_state.memory):
        
        agent.memory = memory
        
        architectural_keywords = ["architectural", "building", "monument", "fort", "temple"]
        if any(keyword in topic.lower() for keyword in architectural_keywords):
            category = "Architectural Details"
        
        if category == "Historical Information":
            return agent.history_info(topic)
        elif category == "Architectural Details":
            return agent.architecture_info(topic)
        elif category == "Travel or Logistics":
            return agent.travel_info(topic)
        elif category == "Weather Information":
            return agent.weather_info(topic)
        elif category == "Accommodation and Dining":
            return agent.accommodation_info(topic)
        elif category == "General Conversation" or category == "Unrecognized":
            return agent.convo_info(topic)
        else:
            return "Sorry, I couldn't categorize your query. Can you try rephrasing?"
        

    # Streamlit UI
    st.set_page_config(page_title="VirtuTrek-Personalized AI Travel Assistant ",page_icon="🌍", layout="centered")
    st.markdown("<h1 style='color: #00FFFF;'>🧑‍💼 Your Virtual Tour Guide</h1>", unsafe_allow_html=True)
    



    # Initialize chat history if not already
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Initialize agents for each category (using memory)
    if "agents" not in st.session_state:
        st.session_state.agents = {
            "Historical Information": History_Expert_Subagent(),
            "Architectural Details": Architectural_Expert_Subagent(),
            "Travel or Logistics": Travel_Logistics_Subagent(),
            "Weather Information": Weather_forecaster_Subagent(),
            "Accommodation and Dining": Accommodation_and_Dining_Expert_Subagent(),
            "General Conversation": Conversational_tour_guide_subagent(),
            "Unrecognized": Conversational_tour_guide_subagent()  # Default for unrecognized
        }

    user_input = st.chat_input("Ask me about a heritage site...")

    if user_input:
        # Categorize the user input
        categorizer = CategorizerAgent()
        category = categorizer.categorize_topic(user_input)

        # Use the appropriate agent based on the category
        agent = st.session_state.agents.get(category, st.session_state.agents["Unrecognized"])
        response = get_agent_response(agent, category, user_input)  # Passing agent directly
        
        # Add chat history
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))
        
        st.session_state.memory.chat_memory.add_user_message(user_input)
        st.session_state.memory.chat_memory.add_ai_message(response)


    # Display conversation
    for speaker, message in st.session_state.chat_history:
        with st.chat_message("user" if speaker == "You" else "assistant"):
            st.markdown(message)

# =============================================SIDEBAR_PANNEL=====================================================


    # Sidebar navigation
    
    st.sidebar.markdown("<h2 style='color: #00FFFF;'>VirtuTrek-Personalized AI Travel Assistant </h2>", unsafe_allow_html=True)
    st.sidebar.markdown("<h2 style='color: #00FFFF;'>Explore Our Services</h2>", unsafe_allow_html=True)
    st.sidebar.write("Select a section to begin your virtual tour:")

    # Sidebar buttons for navigation
    
    if st.sidebar.button("🗺️ Tour Planner"):
        st.session_state.page = "tour_plan"
        st.rerun()

    if st.sidebar.button("🖼️ Image Analysis"):
        st.session_state.page = "image_analysis"
        st.rerun()
    
    if st.sidebar.button("💬 Chatbot"):
        st.session_state.page = "chatbot"
        st.rerun()


    



    if st.sidebar.button("Return to Main Page"):
        st.session_state["page"] = "main"
        st.rerun()
        
        
# ===============================================IMAGE_ANALYSIS================================================================  



def image_page():
    
    import streamlit as st
    from PIL import Image
    import googlemaps
    import io
    import json
    import google.generativeai as genai
    from geopy.geocoders import Nominatim
    from datetime import datetime
    import pydeck as pdk
    from gtts import gTTS
    import tempfile
    import base64
    import folium
    from streamlit_folium import st_folium
    import requests
    import openrouteservice
    from openrouteservice import convert
    from io import BytesIO
    import streamlit.components.v1 as components
    from dotenv import load_dotenv
    import os


    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    serp_api_key = os.getenv("SERPAPI_API_KEY")
    openroutservice_api_key = os.getenv("OPENROUTESERVICE_API_KEY")
    openweather_api_key = os.getenv("OPENWEATHERMAP_API_KEY")

    # === UTILS ===
    def text_to_speech(text):
        tts = gTTS(text)
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return mp3_fp




    def get_real_route_info(origin_coords, destination_coords):
        try:
            gmaps = googlemaps.Client(key=google_api_key)
            directions_result = gmaps.directions(origin_coords, destination_coords, mode="driving")

            if directions_result:
                duration = directions_result[0]["legs"][0]["duration"]["value"] // 60  # in minutes
                summary = directions_result[0]["legs"][0]["duration"]["text"]
                return {
                    "summary": f"Estimated travel time by car is {summary}.",
                    "duration_minutes": duration
                }
            else:
                return {
                    "summary": "Could not fetch route information.",
                    "duration_minutes": 0
                }
        except Exception as e:
            return {
                "summary": f"Error fetching directions: {e}",
                "duration_minutes": 0
            }

    def format_duration(minutes):
        days, minutes = divmod(minutes, 1440)
        hours, minutes = divmod(minutes, 60)
        parts = []
        if days: parts.append(f"{days} days")
        if hours: parts.append(f"{hours} hours")
        if minutes: parts.append(f"{minutes} minutes")
        return ", ".join(parts) if parts else "0 minutes"

    def describe_image_with_gemini(image_bytes):
        genai.configure(api_key=google_api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-001")
        image = Image.open(io.BytesIO(image_bytes))

        prompt = """
        You are a landmark detection assistant.
        Given an image, identify the landmark, its full name, city, and country.
        Return your answer in the following JSON format:

        {
            "landmark": "Eiffel Tower",
            "city": "Paris",
            "country": "France",
            "description": "A wrought iron lattice tower built on the Champ de Mars in Paris.",
            "coordinates": [48.8584, 2.2945]
        }
        for the "description" give teh details in this way: give a comprehensive overview covering its historical background, cultural importance, geographical features, famous attractions, and why it’s worth visiting in max 3 paragraph. provide description in  markdown.
        If you're unsure, write "Unknown" for fields.
        """

        try:
            response = model.generate_content([prompt, image])
            import re, json
            match = re.search(r'\{.*?\}', response.text, re.DOTALL)
            if match:
                output = json.loads(match.group())
            else:
                raise ValueError("Gemini response was not valid JSON.")

            coords = output.get("coordinates", None)
            if coords and isinstance(coords, list) and len(coords) == 2:
                coords = tuple(coords)
            else:
                coords = None

            return {
                "landmark": output.get("landmark", "Unknown"),
                "city": output.get("city", "Unknown"),
                "country": output.get("country", "Unknown"),
                "description": output.get("description", "No description provided."),
                "coordinates": coords
            }

        except Exception as e:
            st.error(f"Gemini failed to analyze image: {e}")
            return {
                "landmark": "Unknown",
                "city": "Unknown",
                "country": "Unknown",
                "description": "Analysis failed.",
                "coordinates": None
            }

    def get_route_and_duration(origin_coords, destination_coords):
        try:
            client = openrouteservice.Client(key=openroutservice_api_key)
            coords = [origin_coords[::-1], destination_coords[::-1]]
            routes = client.directions(coords)
            geometry = routes['routes'][0]['geometry']
            decoded_geometry = convert.decode_polyline(geometry)
            distance = routes['routes'][0]['summary']['distance'] / 1000
            duration = routes['routes'][0]['summary']['duration'] / 60

            return {
                "geometry": geometry,
                "distance_km": round(distance, 2),
                "duration_minutes": round(duration, 2)
            }

        except Exception as e:
            st.error(f"Route calculation failed: {e}")
            return None

    def get_weather(lat, lon):
        try:
            url = (
                f"https://api.openweathermap.org/data/2.5/weather?"
                f"lat={lat}&lon={lon}&appid={openweather_api_key}&units=metric"
            )
            response = requests.get(url).json()
            if response.get("main"):
                temp = response["main"]["temp"]
                desc = response["weather"][0]["description"]
                humidity = response["main"]["humidity"]
                wind_speed = response["wind"]["speed"]
                return {
                    "summary": f"{desc.capitalize()} with temperature of {temp}°C",
                    "details": {
                        "temperature": temp,
                        "description": desc,
                        "humidity": humidity,
                        "wind_speed": wind_speed
                    }
                }
            return {"summary": "Weather data not available.", "details": None}
        except Exception as e:
            st.error(f"Weather API error: {str(e)}")
            return {"summary": "Weather data not available.", "details": None}



    # === MAIN INTERFACE ===
    st.set_page_config(page_title="VirtuTrek-Personalized AI Travel Assistant ",page_icon="🌍", layout="wide")
    st.markdown("<h1 style='color: #00FFFF;'>SightSage Your Snap 📸 </h1>", unsafe_allow_html=True)


    # === LOCATION SETTING ===
    if 'user_location' not in st.session_state:
        st.markdown("#### Set Your Location")
        manual_location = st.text_input("Enter your city and country:", placeholder="e.g. Harare, Zimbabwe")
        if manual_location:
            geolocator = Nominatim(user_agent="tour_agent")
            location = geolocator.geocode(manual_location)
            if location:
                st.session_state['user_location'] = [location.latitude, location.longitude]
                st.success(f"Location set to {manual_location}")
            else:
                st.error("Could not find the location you entered.")

    st.markdown("#### Upload an Image:")
    uploaded_image = st.file_uploader("Upload an image of a landmark", type=["jpg", "jpeg", "png"])
    if uploaded_image and 'user_location' in st.session_state:
        if 'image_analysis' not in st.session_state:
            with st.spinner("Analyzing your image..."):
                bytes_data = uploaded_image.getvalue()
                image_analysis = describe_image_with_gemini(bytes_data)
                st.session_state['image_analysis'] = image_analysis
                st.session_state['audio'] = text_to_speech(image_analysis['description'])
                st.session_state['destination_coords'] = image_analysis.get("coordinates")

        image_analysis = st.session_state['image_analysis']
        destination_coords = st.session_state.get('destination_coords')
        audio_fp = st.session_state.get('audio')

        st.image(uploaded_image)
        st.write("Image Analysis Results")

        landmark = image_analysis["landmark"]
        city = image_analysis["city"]
        country = image_analysis["country"]
        description = image_analysis["description"]

        if landmark != "Unknown":
            st.markdown(f"### {landmark}, {city}, {country}")
        else:
            st.markdown("### 🏛️ Landmark not recognized")

        st.write("#### Description")
        st.write(description)
        st.audio(audio_fp, format="audio/mp3")
                
        x = "completed"

        if x == "completed" and destination_coords:
            import pandas as pd

            st.subheader("📍 Landmark Location")
            if destination_coords:
                st.map(pd.DataFrame([{
                    "lat": destination_coords[0],
                    "lon": destination_coords[1]
                }]))

            
            st.subheader("🌤️ Live Weather at Destination")
            weather_info = get_weather(*destination_coords)
            st.success(weather_info.get("summary", "Weather info not available."))

            if weather_info.get("details"):
                weather_col1, weather_col2 = st.columns(2)
                with weather_col1:
                    st.metric("Temperature", f"{weather_info['details']['temperature']}°C")
                    st.metric("Humidity", f"{weather_info['details']['humidity']}%")
                with weather_col2:
                    st.metric("Conditions", weather_info['details']['description'].capitalize())
                    st.metric("Wind Speed", f"{weather_info['details']['wind_speed']} m/s")
            else:
                st.warning("Weather details not available.")

            st.subheader("🛣️ Travel Route & Map")
            # Add your routing logic here (continued in your next script block)
            try:
                origin_coords = st.session_state['user_location']
                destination_coords = image_analysis.get("coordinates", None)

                if destination_coords:
                    # Create map centered between user and landmark
                    midpoint_lat = (origin_coords[0] + destination_coords[0]) / 2
                    midpoint_lon = (origin_coords[1] + destination_coords[1]) / 2
                    map_obj = folium.Map(location=[midpoint_lat, midpoint_lon], zoom_start=5)

                    # Add marker for user
                    folium.Marker(
                        location=origin_coords,
                        tooltip="Your Location",
                        icon=folium.Icon(color='blue', icon='user')
                    ).add_to(map_obj)

                    # Add marker for landmark
                    folium.Marker(
                        location=destination_coords,
                        tooltip=f"{landmark}",
                        icon=folium.Icon(color='red', icon='flag')
                    ).add_to(map_obj)

                    # Draw a line between them
                    folium.PolyLine(locations=[origin_coords, destination_coords], color="green", weight=3).add_to(map_obj)

                    # Display the map
                    st_folium(map_obj, width=700, height=500)
                else:
                    st.warning("Destination coordinates not found. Please ensure the image was analyzed successfully.")

            except Exception as e:
                st.error(f"Could not load map: {e}")


# =============================================SIDEBAR_PANNEL=====================================================


    # Sidebar navigation
    
    st.sidebar.markdown("<h2 style='color: #00FFFF;'>VirtuTrek-Personalized AI Travel Assistant </h2>", unsafe_allow_html=True)
    st.sidebar.markdown("<h2 style='color: #00FFFF;'>Explore Our Services</h2>", unsafe_allow_html=True)
    st.sidebar.write("Select a section to begin your virtual tour:")

    # Sidebar buttons for navigation
    
    if st.sidebar.button("🗺️ Tour Planner"):
        st.session_state.page = "tour_plan"
        st.rerun()

    if st.sidebar.button("🖼️ Image Analysis"):
        st.session_state.page = "image_analysis"
        st.rerun()
    
    if st.sidebar.button("💬 Chatbot"):
        st.session_state.page = "chatbot"
        st.rerun()


   

        
    if st.sidebar.button("Return to Main Page"):
        st.session_state["page"] = "main"
        st.rerun()
        
# ===============================================TOUR_PLANNER================================================================    
    
def tour_page():
    
    
    # streamlit_app.py
    import streamlit as st
    import requests
    import os
    import json
    import faiss
    import numpy as np
    import nltk
    
    from sentence_transformers import SentenceTransformer
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain_community.utilities import SerpAPIWrapper
    from langchain.tools import Tool
    from langchain.agents import initialize_agent, AgentType
    from langchain.prompts import PromptTemplate
    from nltk.tokenize import sent_tokenize
    


    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    os.environ["GOOGLE_API_KEY"] = "AIzaSyCoB8kXfj4IPVxqYy57EW5RDOLWsI0BpXQ"
    llm = ChatGoogleGenerativeAI(model ='gemini-2.0-flash')
    # API keys (replace with your actual keys)
    WEATHER_API_KEY = "f11eb8bd11a24873a20203749250704"
    SERPAPI_KEY = "41e810ea71ef73c3b8aa116de475d65203024759f54e4eae23853a624a01ec90"

    # Page config
    st.set_page_config(page_title="VirtuTrek-Personalized AI Travel Assistant ",page_icon="🌍", layout="wide")
    st.markdown("<h1 style='color: #00FFFF;'>🌍 Personalized Tour Planner </h1>", unsafe_allow_html=True)
    


    def language(text):
        prompt_template = f"""You are proficient in detecting the type of language based on the input given by the user and to reply with the language type in one word.
                                user input: {text}
        """
        resp = llm.invoke(prompt_template)
        lang = resp.content
        
        prompt_template2 = f""" 
                if {lang} is not english, then translate {text} into english else let it in english only. you just have to act as a translator and reply with just the translated text.
        """
        resp1 = llm.invoke(prompt_template2)
        translated = resp1.content
        return lang,translated

    def convert(lang,text):
        prompt_template = f"""if the given text: {text} is not in {lang}, then translate it completely into the {lang} and print teh translated text only, else print the {text} as it is with nothing else.
        """
        resp = llm.invoke(prompt_template)
        final = resp.content
        
        return final


    # User inputs
    user_city = st.text_input("Which city are you exploring today?")
    user_preferences = st.text_input("Your interests (e.g., museums, food, history, nature):")
    user_mood = st.selectbox("How are you feeling today?", ["adventurous", "relaxed", "curious", "romantic", "energetic"])

    if st.button("Plan"):
        if user_city and user_preferences:
            lang,translated_city_input = language(user_city)
            
            
            lang,translated_user_mood = language(user_mood)
            
            lang,translated_preferences_input = language(user_preferences)
            
            # ================== Wikipedia Fetch ==================
            def fetch_wiki_data(city):
                title = city.replace(" ", "_")
                URL = "https://en.wikipedia.org/w/api.php"
                PARAMS = {
                    "action": "query",
                    "format": "json",
                    "prop": "extracts",
                    "titles": title,
                    "explaintext": True,
                    "exlimit": 1
                }
                response = requests.get(URL, params=PARAMS)
                pages = response.json()["query"]["pages"]
                page = next(iter(pages.values()))
                return page.get("extract", "")

            wiki_text = fetch_wiki_data(translated_city_input)

            # ================ Weather Fetch ==================
            def get_weather(city):
                params = {
                    "q": city,
                    "key": WEATHER_API_KEY,
                    "aqi": "yes"
                }
                response = requests.get("https://api.weatherapi.com/v1/current.json", params=params).json()
                if "current" in response:
                    return {
                        "description": response["current"]["condition"]["text"],
                        "temperature": response["current"]["temp_c"],
                        "humidity": response["current"]["humidity"],
                        "wind": response["current"]["wind_kph"]
                    }
                return {"description": "Data unavailable"}

            weather_info = get_weather(translated_city_input)

            # ================ Food Recommendation ==================
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
            search = SerpAPIWrapper(serpapi_api_key=SERPAPI_KEY)
            search_tool = Tool(name="Search", func=search.run, description="Web search for local food")
            agent = initialize_agent([search_tool], llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False, handle_parsing_errors=True)
            
            
            
            def get_food_recs(city):
                query = f"Top vegetarian and non-vegetarian food in {city}"
                return agent.invoke({"input": query})
            
            

            food_recs = get_food_recs(translated_city_input)

            # ================ Text Chunking and Embedding ==================
            def chunk_text(text, chunk_size=5):
                sentences = sent_tokenize(text)
                return [" ".join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]

            chunks = chunk_text(wiki_text)
            embedder = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            embeddings = embedder.embed_documents(chunks)
            
            
            
            # FAISS Vector Index
            dimension = len(embeddings[0])
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(embeddings).astype("float32"))

            def get_custom_tour(query):
                query_vec = np.array([embedder.embed_query(query)]).astype("float32")
                D, I = index.search(query_vec, k=5)
                
                return "\n\n".join([chunks[i] for i in I[0]])

            user_query = f"Suggest a {translated_user_mood} tour in {translated_city_input} that includes {translated_preferences_input}"
            tour_info = get_custom_tour(user_query)

            
            
            # ================ Final Tour Plan Generation ==================
            prompt = PromptTemplate(
                input_variables=["city", "mood", "preferences", "weather", "tour", "food"],
                template="""
                You are a friendly and knowledgeable AI travel guide helping users plan a personalized tour.

                Using the following details:
                - City: {city}
                - Mood: {mood}
                - Preferences: {preferences}
                - Weather: {weather}
                - Tour Info: {tour}
                - Food: {food}

                Generate a detailed and engaging travel plan. 

                Make sure to:
                1. Suggest **activities that match the mood and preferences**.
                2. **Adapt the plan based on the weather** (e.g., indoor if rainy).
                3. Include **must-see sights or hidden gems** from the tour info.
                4. Recommend **local dishes or food experiences** tailored to the user.
                5. Use a friendly and conversational tone throughout.
                6. End with an inviting summary of the experience.

                Keep the tone warm, positive, and helpful—like a local friend planning a fun day!
                """
            )

            chain = prompt | llm
            response = chain.invoke({
                "city": translated_city_input,
                "mood": translated_user_mood,
                "preferences": translated_preferences_input,
                "weather": str(weather_info),
                "tour": tour_info,
                "food": str(food_recs)
            })

            
            
            st.subheader("🏞️ Your Personalized Tour Plan")
            final_response = convert(lang,response.content)
            st.markdown(final_response)

# =============================================SIDEBAR_PANNEL=====================================================


    # Sidebar navigation
    
    st.sidebar.markdown("<h2 style='color: #00FFFF;'>VirtuTrek-Personalized AI Travel Assistant </h2>", unsafe_allow_html=True)
    st.sidebar.markdown("<h2 style='color: #00FFFF;'>Explore Our Services</h2>", unsafe_allow_html=True)
    st.sidebar.write("Select a section to begin your virtual tour:")

    # Sidebar buttons for navigation
    
    if st.sidebar.button("🗺️ Tour Planner"):
        st.session_state.page = "tour_plan"
        st.rerun()

    if st.sidebar.button("🖼️ Image Analysis"):
        st.session_state.page = "image_analysis"
        st.rerun()
    
    if st.sidebar.button("💬 Chatbot"):
        st.session_state.page = "chatbot"
        st.rerun()


    


   
    if st.sidebar.button("Return to Main Page"):
        st.session_state["page"] = "main"
        st.rerun()
        
            
    
if 'page' not in st.session_state:
    st.session_state.page = "main"

if st.session_state.page == "main":
    main_page()
elif st.session_state.page == "chatbot":
    chatbot_page()
elif st.session_state.page == "image_analysis":
    image_page()
elif st.session_state.page == "tour_plan":
    tour_page()
