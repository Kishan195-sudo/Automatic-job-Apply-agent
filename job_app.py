import streamlit as st
import google.generativeai as genai
import PyPDF2
from io import BytesIO
import json
import re

# --- Configuration and Setup ---

# Set the page configuration for the Streamlit app
st.set_page_config(
    page_title="AI Job Search Assistant",
    page_icon="🤖",
    layout="wide",
)

# --- Google Gemini API Configuration ---

# Fetch the API key from Streamlit's secrets management
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except (KeyError, AttributeError):
    st.error("🚨 Google API Key not found! Please add it to your Streamlit secrets.")
    st.stop()

# --- Core Functions ---

def get_gemini_response(prompt, retries=3):
    """
    Sends a prompt to the Google Gemini Pro model and returns the response.
    Includes retry logic for robustness.
    """
    model = genai.GenerativeModel('gemini-pro')
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            # Clean the response text from markdown and other artifacts
            cleaned_text = re.sub(r'```json\s*|\s*```', '', response.text.strip())
            return cleaned_text
        except Exception as e:
            st.warning(f"API call failed on attempt {attempt + 1}: {e}. Retrying...")
            if attempt + 1 == retries:
                st.error("Failed to get response from Gemini API after several retries.")
                return None
    return None

def extract_text_from_pdf(file_bytes):
    """Extracts text content from a PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

def generate_search_queries_from_resume(resume_text):
    """
    Uses the LLM to analyze resume text and generate relevant job search queries.
    """
    prompt = f"""
    Analyze the following resume text and generate 5 diverse, effective job search queries.
    These queries should be used to find job postings on the internet.
    Focus on job titles, key skills, and technologies mentioned.
    Include a mix of general and specific queries. For example: "remote software engineer jobs", "python data analyst jobs in New York".

    Resume Text:
    ---
    {resume_text}
    ---

    Return the result as a JSON object with a single key "queries" which is a list of strings.
    Example format: {{"queries": ["query 1", "query 2", "query 3", "query 4", "query 5"]}}
    """
    response_text = get_gemini_response(prompt)
    if response_text:
        try:
            # Parse the JSON response
            response_json = json.loads(response_text)
            return response_json.get("queries", [])
        except json.JSONDecodeError:
            st.error("Could not parse the search queries from the LLM response. Please try again.")
            return []
    return []

def display_job_search_links(queries):
    """
    Displays clickable links for Google job searches based on the generated queries.
    """
    st.subheader("✅ Here are your personalized job search links:")
    st.markdown("Click the links below to see job results on Google. The tool has done the searching for you!")

    for query in queries:
        # URL encode the query to make it safe for a URL
        from urllib.parse import quote
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
