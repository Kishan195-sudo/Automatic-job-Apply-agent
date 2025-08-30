import streamlit as st
from io import BytesIO
import json
import re
import requests
from urllib.parse import quote

# Optional imports: don't crash if not installed
try:
    import google.generativeai as genai  # SDK fallback
except Exception:
    genai = None

try:
    import PyPDF2  # PDF parsing
except Exception:
    PyPDF2 = None

# --- Streamlit App Setup ---
st.set_page_config(
    page_title="AI Job Search Assistant",
    page_icon="🤖",
    layout="wide",
)

# --- Secrets / Config ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (KeyError, AttributeError):
    st.error("🚨 Google API Key not found! Add GOOGLE_API_KEY to your Streamlit secrets.")
    st.stop()

MODEL_NAME = st.secrets.get("GEMINI_MODEL", "gemini-1.5-flash")

# Configure SDK if present
if genai is not None:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        st.warning(f"Could not configure Gemini SDK; will use REST fallback. Details: {e}")


# --- Helper Functions ---

def _clean_llm_text(text: str) -> str:
    """Strip code fences like ```json ... ``` and trim whitespace."""
    if not text:
        return ""
    cleaned = re.sub(r"```json\s*|\s*```", "", text.strip(), flags=re.IGNORECASE)
    return cleaned.strip()


def _gemini_rest_generate_content(prompt: str, model: str, api_key: str) -> str:
    """REST call to Gemini generateContent."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    params = {"key": api_key}
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    resp = requests.post(url, params=params, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0].get("text", "")
    except (KeyError, IndexError, TypeError):
        return ""


def get_gemini_response(prompt: str, retries: int = 3):
    """Generate text using SDK if available; fallback to REST. Retries on failure."""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            # SDK path
            if genai is not None:
                model = genai.GenerativeModel(MODEL_NAME)
                response = model.generate_content(prompt)
                cleaned = _clean_llm_text(getattr(response, "text", "") or "")
                if cleaned:
                    return cleaned
            # REST fallback
            text = _gemini_rest_generate_content(prompt, MODEL_NAME, GOOGLE_API_KEY)
            cleaned = _clean_llm_text(text)
            if cleaned:
                return cleaned
        except Exception as e:
            last_error = e
            st.warning(f"AI call failed on attempt {attempt}: {e}. Retrying...")
    st.error(f"Failed to get response from Gemini after {retries} attempts. Last error: {last_error}")
    return None


def extract_text_from_pdf(file_bytes: bytes):
    """Extract text from PDF if PyPDF2 is installed; else return None."""
    if PyPDF2 is None:
        st.info("PDF text extraction library (PyPDF2) is not installed. You can paste your resume text instead.")
        return None
    try:
        reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        text_chunks = []
        for page in reader.pages:
            pg = page.extract_text() or ""
            if pg:
                text_chunks.append(pg)
        text = "\n".join(text_chunks).strip()
        return text if text else None
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None


def _parse_queries_from_text(response_text: str):
    """Parse {"queries": [...]} from possibly messy LLM output."""
    if not response_text:
        return []
    # Direct JSON attempt
    try:
        obj = json.loads(response_text)
        if isinstance(obj, dict) and isinstance(obj.get("queries"), list):
            return obj["queries"]
    except json.JSONDecodeError:
        pass
    # Try extracting a JSON object substring
    m = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and isinstance(obj.get("queries"), list):
                return obj["queries"]
        except json.JSONDecodeError:
            pass
    return []


def generate_search_queries_from_resume(resume_text: str):
    """Ask the LLM for exactly 5 search queries based on resume text."""
    prompt = f"""
Analyze the following resume text and generate exactly 5 diverse, effective job search queries.
These queries should be used to find job postings on the internet.
Focus on job titles, key skills, and technologies mentioned.
Include a mix of general and specific queries. For example: "remote software engineer jobs", "python data analyst jobs in New York".

Resume Text:
---
{resume_text}
---

Return ONLY a JSON object with a single key "queries" whose value is a list of 5 strings.
Example: {{"queries": ["query 1", "query 2", "query 3", "query 4", "query 5"]}}
"""
    response_text = get_gemini_response(prompt)
    if response_text:
        queries = _parse_queries_from_text(response_text)
        if queries:
            # sanitize and cap at 5
            queries = [str(q).strip() for q in queries if str(q).strip()]
            return queries[:5]
        st.error("Could not parse the search queries from the AI response. Please try again.")
        return []
    return []


def display_job_search_links(queries):
    """Show Google Jobs links for each query."""
    st.subheader("✅ Here are your personalized job search links:")
    st.markdown("Click the links below to see job results on Google. The tool has done the searching for you!")
    for query in queries:
        search_url = f"https://www.google.com/search?q={quote(query)}&ibp=htl;jobs"
        st.markdown(f"- **[{query}]({search_url})**")


# --- UI ---

st.title("🤖 AI Job Search Assistant")
st.markdown(
    "Upload your resume as a PDF or paste the text below. "
    "This tool will generate personalized job search links based on your skills and experience."
)

with st.sidebar:
    st.header("📝 Your Details")
    uploaded_file = st.file_uploader("Upload Your Resume (PDF only)", type=["pdf"])
    st.markdown("Or paste your resume text:")
    pasted_text = st.text_area("Resume text (optional if uploading a PDF)", height=200, placeholder="Paste your resume text here...")
    start_button = st.button("Generate Job Searches", type="primary", use_container_width=True)

if start_button:
    with st.spinner("Analyzing your resume and finding jobs..."):
        resume_text = None

        # Prefer pasted text if provided
        if pasted_text and pasted_text.strip():
            resume_text = pasted_text.strip()
            st.info("Using pasted resume text.")
        elif uploaded_file is not None:
            st.info("Reading your uploaded PDF...")
            resume_bytes = uploaded_file.getvalue()
            resume_text = extract_text_from_pdf(resume_bytes)
        else:
            st.warning("Please upload a PDF or paste your resume text, then click Generate.")

        if not resume_text:
            if uploaded_file is not None and PyPDF2 is None:
                st.error("PDF could not be processed because PyPDF2 is not installed in this environment. Please paste your resume text into the textbox and try again.")
            else:
                st.error("Could not obtain resume text. Please ensure the PDF contains selectable text or paste your resume into the textbox.")
            st.stop()

        st.info("Identifying key skills and job titles with AI...")
        queries = generate_search_queries_from_resume(resume_text)

        if not queries:
            st.error("The AI could not generate search queries from your resume. Try adding more detail or using a different resume format.")
            st.stop()

        st.success("Analysis complete!")
        display_job_search_links(queries)
else:
    st.info("Get started by uploading your resume or pasting it in the sidebar, then click 'Generate Job Searches'.")
