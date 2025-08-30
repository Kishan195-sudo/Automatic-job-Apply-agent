import streamlit as st
import requests
import json
import re
from io import BytesIO
from urllib.parse import quote, urlencode
import sys
import subprocess
import base64
import zipfile
from datetime import datetime

# ---------------- Page config & background GIF ----------------

st.set_page_config(page_title="AI Job Hunter (No Email creds)", page_icon="🤖", layout="wide")

# Default background GIF - you can change it in the UI
DEFAULT_GIF = "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZ2JmN3dyeG5laGZtYWMyY2dnOGpoZndjM3ZxYjZxOTF5cWF1ZzFwdiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/gw3IWyGkC0rsazTi3U/giphy.gif"

def set_background_gif(gif_url: str):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url('{gif_url}') no-repeat center center fixed;
            background-size: cover;
        }}
        .app-content {{
            background: rgba(255,255,255,0.86);
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        }}
        [data-testid="stSidebar"] > div:first-child {{
            background: rgba(255,255,255,0.92);
            backdrop-filter: blur(4px);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------------- Secrets / Config ----------------

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    st.error("Missing secret: GOOGLE_API_KEY. Add it in Streamlit secrets and reload.")
    st.stop()

MODEL_NAME = st.secrets.get("GEMINI_MODEL", "gemini-1.5-flash")

# ---------------- PDF reader bootstrap (pypdf fallback & runtime install) ----------------

PdfReader = None

def ensure_pdf_reader():
    global PdfReader
    if PdfReader:
        return
    try:
        from pypdf import PdfReader as _PdfReader
        PdfReader = _PdfReader
        return
    except Exception:
        pass
    try:
        # If pypdf not available, try installing it at runtime
        with st.spinner("Installing PDF library (pypdf)..."):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf", "-q"])
        from pypdf import PdfReader as _PdfReader
        PdfReader = _PdfReader
    except Exception as e:
        st.error(f"Could not install/import pypdf: {e}. Add 'pypdf' to your environment.")
        st.stop()

def extract_text_from_pdf(file_bytes: bytes) -> str | None:
    ensure_pdf_reader()
    try:
        reader = PdfReader(BytesIO(file_bytes))
        pages = getattr(reader, "pages", [])
        chunks = []
        for p in pages:
            text = p.extract_text() or ""
            if text.strip():
                chunks.append(text)
        text = "\n\n".join(chunks).strip()
        return text if text else None
    except Exception as e:
        st.error(f"PDF extraction error: {e}")
        return None

# ---------------- Gemini REST helpers ----------------

def _clean_llm_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"```json\s*|\s*```", "", text.strip(), flags=re.IGNORECASE).strip()

def gemini_generate(prompt: str, model: str = MODEL_NAME, retries: int = 3) -> str | None:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    params = {"key": GOOGLE_API_KEY}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, params=params, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            try:
                raw = data["candidates"][0]["content"]["parts"][0].get("text", "")
            except Exception:
                raw = ""
            cleaned = _clean_llm_text(raw)
            if cleaned:
                return cleaned
            else:
                last_err = "empty response"
        except Exception as e:
            last_err = e
    st.error(f"LLM request failed after {retries} attempts. Last error: {last_err}")
    return None

# ---------------- Query generation ----------------

def generate_search_queries(resume_text: str, count: int = 6) -> list[str]:
    prompt = f"""
Analyze the resume below and generate exactly {count} diverse, effective job search queries to find job postings on the web.
Focus on job titles, seniority, skills, and technologies found. Provide a mix of general and specific queries (e.g., "remote python backend engineer", "senior machine learning engineer new york", "data analyst entry level remote").

Resume:
---
{resume_text}
---

Return ONLY a JSON object: {{ "queries": ["q1","q2", ...] }}
"""
    resp = gemini_generate(prompt)
    if not resp:
        return []
    # parse JSON or try to extract JSON substring
    try:
        obj = json.loads(resp)
        queries = obj.get("queries", [])
        return [q.strip() for q in queries if isinstance(q, str) and q.strip()][:count]
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", resp, flags=re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                queries = obj.get("queries", [])
                return [q.strip() for q in queries if isinstance(q, str) and q.strip()][:count]
            except Exception:
                pass
    # fallback: naive extraction of lines
    lines = [ln.strip("-• \t") for ln in resp.splitlines() if ln.strip()]
    return lines[:count]

# ---------------- Job board searchers ----------------

def search_remotive(query: str, limit: int = 25) -> list[dict]:
    try:
        url = "https://remotive.com/api/remote-jobs"
        r = requests.get(url, params={"search": query}, timeout=30)
        r.raise_for_status()
        data = r.json()
        jobs = data.get("jobs", [])[:limit]
        out = []
        for j in jobs:
            out.append({
                "source": "Remotive",
                "title": j.get("title"),
                "company": j.get("company_name"),
                "location": j.get("candidate_required_location"),
                "tags": j.get("tags") or [],
                "url": j.get("url"),
                "apply_email": None,
                "description": re.sub(r"<[^>]+>", "", j.get("description", "") or "")[:2000],
            })
        return out
    except Exception:
        return []

def search_remoteok(query: str, limit: int = 25) -> list[dict]:
    try:
        url = "https://remoteok.com/api"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        r.raise_for_status()
        data = r.json()
        out = []
        q_tokens = [t for t in re.split(r"[,\s]+", query.lower()) if t]
        for item in data:
            if not isinstance(item, dict):
                continue
            title = (item.get("position") or item.get("title") or "").strip()
            description = (item.get("description") or "").lower()
            tags = item.get("tags") or []
            company = item.get("company") or item.get("company_name")
            url_ = item.get("url") or item.get("apply_url") or item.get("slug")
            location = item.get("location") or item.get("region") or "Remote"
            apply_email = item.get("apply_email")
            text_blob = " ".join([title.lower(), description, " ".join([t.lower() for t in tags])])
            if any(tok in text_blob for tok in q_tokens):
                out.append({
                    "source": "RemoteOK",
                    "title": title,
                    "company": company,
                    "location": location,
                    "tags": tags,
                    "url": url_,
                    "apply_email": apply_email,
                    "description": (item.get("description") or "")[:2000],
                })
            if len(out) >= limit:
                break
        return out
    except Exception:
        return []

def aggregate_jobs(queries: list[str], per_query: int = 25, max_total: int = 80) -> list[dict]:
    seen = set()
    results = []
    for q in queries:
        r_jobs = search_remotive(q, limit=per_query)
        o_jobs = search_remoteok(q, limit=per_query)
        for job in (r_jobs + o_jobs):
            url = job.get("url") or ""
            if url and url not in seen:
                seen.add(url)
                results.append(job)
            if len(results) >= max_total:
                break
        if len(results) >= max_total:
            break
    return results

# ---------------- Cover letter generation (Gemini) ----------------

def generate_cover_letter(resume_text: str, job_title: str, company: str) -> str | None:
    prompt = f"""
Write a concise, professional cover email (120-180 words) tailored to the following job.
Tone: confident, outcome-focused. Include 2 short bullet points of relevant achievements. No markdown.

Candidate resume:
---
{resume_text}
---

Job:
- Title: {job_title}
- Company: {company}
"""
    return gemini_generate(prompt)

# ---------------- Utilities: mailto and zip generation ----------------

def make_mailto_link(to_email: str, subject: str, body: str) -> str:
    # mailto: supports subject and body; URL-encode
    params = {"subject": subject, "body": body}
    return f"mailto:{to_email}?{urlencode(params)}"

def create_zip(resume_bytes: bytes, resume_filename: str, cover_letters: dict, jobs: list[dict]) -> bytes:
    bio = BytesIO()
    with zipfile.ZipFile(bio, mode="w") as zf:
        # add resume
        zf.writestr(resume_filename, resume_bytes)
        # add cover letters
        for fname, txt in cover_letters.items():
            zf.writestr(f"cover_letters/{fname}", txt)
        # add jobs JSON
        zf.writestr("jobs.json", json.dumps(jobs, indent=2, ensure_ascii=False))
    return bio.getvalue()

# ---------------- User Interface ----------------

set_background_gif(DEFAULT_GIF)

st.markdown("<div class='app-content'>", unsafe_allow_html=True)
st.title("AI Job Hunter — Prepare Applications (no creds required)")
st.write("Upload your PDF resume. The agent will analyze it, search job boards, and prepare cover letters and mailto links. No email keys required; nothing is sent automatically.")

with st.sidebar:
    st.header("Upload & options")
    uploaded = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
    gif_url = st.text_input("Background GIF URL (optional)", value=DEFAULT_GIF)
    if gif_url and gif_url != DEFAULT_GIF:
        set_background_gif(gif_url)
    candidate_name = st.text_input("Your full name", value="")
    candidate_contact_email = st.text_input("Your contact email (used in cover letter)", value="")
    max_jobs = st.slider("Max jobs to collect", 10, 120, 40, 10)
    generate_button = st.button("Analyze & Search Jobs")

if generate_button:
    if not uploaded:
        st.error("Please upload a PDF resume to continue.")
        st.stop()

    resume_bytes = uploaded.getvalue()
    st.info("Extracting text from PDF...")
    resume_text = extract_text_from_pdf(resume_bytes)
    if not resume_text:
        st.error("Could not extract text from the PDF. Ensure it has selectable text (not just images).")
        st.stop()

    st.success("Resume text extracted.")
    with st.expander("Preview extracted text (first 1200 chars)"):
        st.write(resume_text[:1200] + ("..." if len(resume_text) > 1200 else ""))

    st.info("Generating job search queries with Gemini...")
    queries = generate_search_queries(resume_text, count=6)
    if not queries:
        st.error("Could not generate search queries. Try a different resume or retry.")
        st.stop()

    st.success("Queries generated.")
    st.markdown("**Queries:**")
    for q in queries:
        st.markdown(f"- {q}")

    st.info("Searching job boards (Remotive & RemoteOK)...")
    jobs = aggregate_jobs(queries, per_query=25, max_total=max_jobs)
    if not jobs:
        st.warning("No jobs found for the generated queries.")
    else:
        st.success(f"Found {len(jobs)} job listings.")
        st.markdown("Top matches (click links to open):")
        for i, j in enumerate(jobs[:50], start=1):
            title = j.get("title") or "Untitled"
            company = j.get("company") or "Unknown"
            url = j.get("url") or ""
            apply_email = j.get("apply_email")
            source = j.get("source", "")
            line = f"{i}. {title} — {company} ({source})"
            if url:
                st.markdown(f"- [{line}]({url})")
            else:
                st.markdown(f"- {line}")
            if apply_email:
                st.markdown(f"  - Apply email: `{apply_email}`  — [Open mail client](mailto:{apply_email})")
        st.markdown("---")

    # Generate cover letters for top N jobs (limit to reasonable number to save API usage)
    cover_letters = {}
    n_cover = st.number_input("Generate cover letters for top N jobs (Gemini API calls)", min_value=0, max_value=min(30, len(jobs)), value=5, step=1)
    if n_cover > 0 and jobs:
        st.info(f"Generating {n_cover} tailored cover letters (one Gemini call each)...")
        progress = st.progress(0)
        for idx in range(int(n_cover)):
            job = jobs[idx]
            title = job.get("title") or ""
            company = job.get("company") or ""
            key = f"{idx+1:02d}_{(title[:40] or 'role').replace('/', '_').replace('\\n',' ')}_{(company[:30] or 'company').replace('/', '_')}.txt"
            # Create some context: include candidate name and contact if available
            cover = generate_cover_letter(resume_text + ("\n\nContact: " + candidate_contact_email if candidate_contact_email else ""), title, company) or ""
            # Add header and contact
            header = f"{title} — {company}\nPrepared for: {candidate_name or ''}\n\n"
            if candidate_contact_email:
                header += f"Contact: {candidate_contact_email}\n\n"
            full = header + cover
            cover_letters[key] = full
            progress.progress((idx + 1) / int(n_cover))
        st.success("Cover letters generated.")

        st.markdown("Preview cover letters:")
        for fname, txt in list(cover_letters.items())[:3]:
            st.subheader(fname)
            st.code(txt[:1200] + ("..." if len(txt) > 1200 else ""), language=None)

    # Prepare mailto links for jobs with apply_email
    st.markdown("---")
    st.subheader("Prepared mailto links (for jobs that expose an apply email)")
    mailto_count = 0
    for idx, j in enumerate(jobs):
        apply_email = j.get("apply_email")
        title = j.get("title") or ""
        company = j.get("company") or ""
        if apply_email:
            body = cover_letters.get(list(cover_letters.keys())[0], "") if cover_letters else f"Hello,\n\nI am interested in the {title} role at {company}.\n\nRegards,\n{candidate_name or ''}"
            subject = f"Application: {title} at {company}"
            # Append contact details to body if provided
            if candidate_contact_email and candidate_contact_email not in body:
                body = body + f"\n\nContact: {candidate_contact_email}"
            mailto = make_mailto_link(apply_email, subject, body)
            st.markdown(f"- {title} — {company} — [Open mail client]({mailto})  — Email: `{apply_email}`")
            mailto_count += 1
    if mailto_count == 0:
        st.info("No apply-email addresses found in discovered jobs.")

    # Offer downloadable ZIP with resume, cover letters, and jobs.json
    if jobs:
        st.markdown("---")
        st.subheader("Download prepared files")
        zip_bytes = create_zip(resume_bytes, uploaded.name, cover_letters, jobs)
        b64 = base64.b64encode(zip_bytes).decode()
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        filename = f"ai_job_hunter_package_{ts}.zip"
        href = f'<a href="data:application/zip;base64,{b64}" download="{filename}">Download ZIP with resume, cover letters, and jobs.json</a>'
        st.markdown(href, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <small>Notes:</small>
    <ul>
      <li>This tool requires only your GOOGLE_API_KEY in Streamlit secrets to call Gemini. No email credentials are used.</li>
      <li>Auto-sending emails is intentionally disabled. Use the provided mailto links or the generated cover letters to apply manually via your email or ATS.</li>
      <li>Job board APIs (Remotive, RemoteOK) are used. Results depend on their current data and availability.</li>
    </ul>
    """,
    unsafe_allow_html=True,
)
