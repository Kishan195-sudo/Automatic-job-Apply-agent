import streamlit as st
import PyPDF2
from io import BytesIO
import json
import re
import requests
from urllib.parse import quote

# Try to import the Gemini SDK, but don't fail the app if it's not installed
try:
    import google.generativeai as genai  # Optional dependency
except Exception:
    genai = None

# --- Configuration and Setup ---

st.set_page_config(
    page_title="AI Job Search Assistant",
    page_icon="🤖",
    layout="wide",
)

# --- Google Gemini API Configuration ---

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (KeyError, AttributeError):
    st.error("🚨 Google API Key not found! Please add it to your Streamlit secrets.")
    st.stop()

# Configure SDK only if available
if genai is not None:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        st.warning(f"Could not configure Gemini SDK, will use REST fallback. Details: {e}")

# Default model name (works for both SDK and REST). You can change via secrets if you want.
MODEL_NAME = st.secrets.get("GEMINI_MODEL", "gemini-1.5-flash")


# --- Core Functions ---

def _clean_llm_text(text: str) -> str:
    """
    Clean LLM response text by stripping code fences like ```json ... ```
    and trimming whitespace.
    """
    if not text:
        return ""
    # Remove ```json and ``` fences
    cleaned = re.sub(r"```json\s*|\s*```", "", text.strip(), flags=re.IGNORECASE)
    return cleaned.strip()


def _gemini_rest_generate_content(prompt: str, model: str, api_key: str) -> str:
    """
    REST fallback for Gemini content generation.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    params = {"key": api_key}
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }
    resp = requests.post(url, params=params, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # Extract text from the first candidate
    try:
        text = data["candidates"][0]["content"]["parts"][0].get("text", "")
    except (KeyError, IndexError, TypeError):
        text = ""
    return text


def get_gemini_response(prompt, retries=3):
    """
    Sends a prompt to Gemini and returns the response text.
    - Uses the Gemini SDK if available.
    - Falls back to the REST API if the SDK is missing or fails.
    Includes retry logic for robustness.
    """
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            if genai is not None:
                # Try SDK path first
                model = genai.GenerativeModel(MODEL_NAME)
                response = model.generate_content(prompt)
                cleaned_text = _clean_llm_text(getattr(response, "text", "") or "")
                if cleaned_text:
                    return cleaned_text
                # If SDK returns nothing, try REST as a backup in same attempt
            # REST fallback
            text = _gemini_rest_generate_content(prompt, MODEL_NAME, GOOGLE_API_KEY)
            cleaned_text = _clean_llm_text(text)
            if cleaned_text:
                return cleaned_text
        except Exception as e:
            last_error = e
            st.warning(f"AI call failed on attempt {attempt}: {e}. Retrying...")
    st.error(f"Failed to get response from Gemini after {retries} attempts. Last error: {last_error}")
    return None


def extract_text_from_pdf(file_bytes):
    """Extracts text content from a PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None


def _parse_queries_from_text(response_text: str):
    """
    Try to parse the "queries" list from the LLM response text.
    Handles plain JSON, fenced JSON, and cases where JSON is embedded in prose.
    """
    if not response_text:
        return []

    # First try direct JSON
    try:
        obj = json.loads(response_text)
        if isinstance(obj, dict) and "queries" in obj and isinstance(obj["queries"], list):
            return obj["queries"]
    except json.JSONDecodeError:
        pass

    # Try to extract the first {...} JSON object
    m = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "queries" in obj and isinstance(obj["queries"], list):
                return obj["queries"]
        except json.JSONDecodeError:
            pass

    return []


def generate_search_queries_from_resume(resume_text):
    """
    Uses the LLM to analyze resume text and generate relevant job search queries.
    """
    prompt = f"""
    Analyze the following resume text and generate exactly 5 diverse, effective job search queries.
    These queries should be used to find job postings on the internet.
    Focus on job titles, key skills, and technologies mentioned.
    Include a mix of general and specific queries. For example: "remote software engineer jobs", "python data analyst jobs in New York".

    Resume Text:
    ---
    {resume_text}
    ---

    Return the result as a JSON object with a single key "queries" which is a list of strings.
    Example format: {{"queries": ["query 1", "query 2", "query 3", "query 4", "query 5"]}}
    Only return JSON.
    """
    response_text = get_gemini_response(prompt)
    if response_text:
        queries = _parse_queries_from_text(response_text)
        if queries:
            # Ensure only strings and strip whitespace
            queries = [str(q).strip() for q in queries if str(q).strip()]
            # Keep max 5
            return queries[:5]
        st.error("Could not parse the search queries from the AI response. Please try again.")
        return []
    return []


def display_job_search_links(queries):
    """
    Displays clickable links for Google job searches based on the generated queries.
    """
    st.subheader("✅ Here are your personalized job search links:")
    st.markdown("Click the links below to see job results on Google. The tool has done the searching for you!")

    for query in queries:
        search_url = f"https://www.google.com/search?q={quote(query)}&ibp=htl;jobs"
        st.markdown(f"- **[{query}]({search_url})**")


# --- Streamlit User Interface ---

st.title("🤖 AI Job Search Assistant")
st.markdown("Upload your resume, and this tool will generate personalized job search links based on your skills and experience.")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("📝 Your Details")
    st.markdown("Please upload your resume in PDF format.")
    uploaded_file = st.file_uploader("Upload Your Resume (PDF only)", type=["pdf"])
    start_button = st.button("Generate Job Searches", type="primary", use_container_width=True)

# --- Main Content Area ---
if start_button and uploaded_file is not None:
    with st.spinner("Analyzing your resume and finding jobs..."):
        # 1. Read and parse the resume
        st.info("Step 1: Reading your resume...")
        resume_bytes = uploaded_file.getvalue()
        resume_text = extract_text_from_pdf(resume_bytes)

        if not resume_text:
            st.error("Could not extract text from your resume. Please ensure it's not an image-based PDF and try again.")
            st.stop()

        # 2. Generate search queries using the LLM
        st.info("Step 2: Asking the AI to identify key skills and job titles...")
        search_queries = generate_search_queries_from_resume(resume_text)

        if not search_queries:
            st.error("The AI could not generate search queries from your resume. The document might be too short or in an unsupported format.")
            st.stop()

        # 3. Display the results
        st.success("Analysis complete!")
        display_job_search_links(search_queries)

elif start_button and uploaded_file is None:
    st.warning("Please upload your resume before starting.")

# --- Welcome Message and Instructions ---
if not start_button:
    st.info("Get started by uploading your resume in the sidebar and clicking the 'Generate Job Searches' button.")
