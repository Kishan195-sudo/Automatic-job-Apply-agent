import streamlit as st
import requests
import json
import re
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from io import BytesIO
from urllib.parse import quote
import PyPDF2  # required (not optional)


# ---------------------- Page & Theming ----------------------

st.set_page_config(
    page_title="AI Job Hunter & Auto-Apply",
    page_icon="🤖",
    layout="wide",
)

# Background GIF (replace with your own if desired)
BACKGROUND_GIF_URL = "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZ2JmN3dyeG5laGZtYWMyY2dnOGpoZndjM3ZxYjZxOTF5cWF1ZzFwdiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/gw3IWyGkC0rsazTi3U/giphy.gif"

def add_bg_gif(gif_url: str):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url('{gif_url}') no-repeat center center fixed;
            background-size: cover;
        }}
        /* Glass effect for main content */
        .main > div {{
            background: rgba(255, 255, 255, 0.80);
            backdrop-filter: blur(6px);
            border-radius: 12px;
            padding: 1rem 1.2rem;
        }}
        /* Sidebar styling */
        [data-testid="stSidebar"] > div:first-child {{
            background: rgba(255, 255, 255, 0.88);
            backdrop-filter: blur(6px);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

add_bg_gif(BACKGROUND_GIF_URL)


# ---------------------- Secrets / Config ----------------------

# Required secrets
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    GMAIL_USERNAME = st.secrets["GMAIL_USERNAME"]
    GMAIL_APP_PASSWORD = st.secrets["GMAIL_APP_PASSWORD"]
except (KeyError, AttributeError):
    st.error("Missing required secrets. Please set GOOGLE_API_KEY, GMAIL_USERNAME, and GMAIL_APP_PASSWORD in Streamlit secrets.")
    st.stop()

# Model (Gemini REST)
MODEL_NAME = st.secrets.get("GEMINI_MODEL", "gemini-1.5-flash")


# ---------------------- LLM Helpers (Gemini via REST) ----------------------

def _clean_llm_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"```json\s*|\s*```", "", text.strip(), flags=re.IGNORECASE).strip()

def _gemini_rest_generate_content(prompt: str, model: str, api_key: str, timeout: int = 60) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    params = {"key": api_key}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    resp = requests.post(url, params=params, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0].get("text", "")
    except (KeyError, IndexError, TypeError):
        return ""

def get_gemini_response(prompt: str, retries: int = 3) -> str | None:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            text = _gemini_rest_generate_content(prompt, MODEL_NAME, GOOGLE_API_KEY)
            cleaned = _clean_llm_text(text)
            if cleaned:
                return cleaned
        except Exception as e:
            last_error = e
            st.warning(f"LLM call failed on attempt {attempt}: {e}. Retrying...")
    st.error(f"Failed to get response from Gemini after {retries} attempts. Last error: {last_error}")
    return None


# ---------------------- PDF Parsing ----------------------

def extract_text_from_pdf(file_bytes: bytes) -> str | None:
    try:
        reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        text_chunks = []
        for page in reader.pages:
            pg = page.extract_text() or ""
            if pg.strip():
                text_chunks.append(pg)
        text = "\n".join(text_chunks).strip()
        return text if text else None
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None


# ---------------------- Query Generation ----------------------

def generate_search_queries_from_resume(resume_text: str) -> list[str]:
    prompt = f"""
Analyze the following resume and generate exactly 6 diverse job search queries.
Mix general and specific searches (role, tech, seniority, location, remote/hybrid).
Return ONLY JSON in the form: {{"queries": ["...", "...", "...", "...", "...", "..."]}}

Resume:
---
{resume_text}
---
"""
    response = get_gemini_response(prompt)
    if not response:
        return []
    # Parse JSON
    try:
        obj = json.loads(response)
        queries = obj.get("queries", [])
        queries = [str(q).strip() for q in queries if str(q).strip()]
        return queries[:6]
    except json.JSONDecodeError:
        # fallback: try to find JSON object inside
        m = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                queries = obj.get("queries", [])
                queries = [str(q).strip() for q in queries if str(q).strip()]
                return queries[:6]
            except json.JSONDecodeError:
                pass
    return []


# ---------------------- Job Search (Remotive + RemoteOK) ----------------------

def search_remotive(query: str, limit: int = 20) -> list[dict]:
    url = "https://remotive.com/api/remote-jobs"
    params = {"search": query}
    r = requests.get(url, params=params, timeout=30)
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
            "apply_email": None,  # Remotive usually links out to ATS
            "description": j.get("description") or "",
        })
    return out

def search_remoteok(query: str, limit: int = 20) -> list[dict]:
    url = "https://remoteok.com/api"
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    data = r.json()
    jobs = []
    # First element is metadata; skip non-job dicts
    for item in data:
        if not isinstance(item, dict):
            continue
        title = (item.get("position") or item.get("title") or "").strip()
        description = (item.get("description") or "").lower()
        tags = item.get("tags") or []
        company = item.get("company") or item.get("company_name")
        url_ = item.get("url") or item.get("apply_url") or item.get("slug")
        location = item.get("location") or item.get("region") or "Remote"
        apply_email = item.get("apply_email")  # sometimes present
        # Filter by query terms (simple contains any token)
        q_tokens = [t for t in re.split(r"[,\s]+", query.lower()) if t]
        text_blob = " ".join([title.lower(), description, " ".join([t.lower() for t in tags])])
        if any(tok in text_blob for tok in q_tokens):
            jobs.append({
                "source": "RemoteOK",
                "title": title or item.get("position") or "",
                "company": company,
                "location": location,
                "tags": tags,
                "url": url_,
                "apply_email": apply_email,
                "description": item.get("description") or "",
            })
        if len(jobs) >= limit:
            break
    return jobs

def aggregate_jobs(queries: list[str], per_query_limit: int = 20, max_total: int = 60) -> list[dict]:
    seen_urls = set()
    results = []
    for q in queries:
        try:
            remotive_jobs = search_remotive(q, limit=per_query_limit)
        except Exception:
            remotive_jobs = []
        try:
            remoteok_jobs = search_remoteok(q, limit=per_query_limit)
        except Exception:
            remoteok_jobs = []
        for j in remotive_jobs + remoteok_jobs:
            url = j.get("url") or ""
            if url and url not in seen_urls:
                seen_urls.add(url)
                results.append(j)
        if len(results) >= max_total:
            break
    return results[:max_total]


# ---------------------- Cover Letter & Auto-Apply (Email) ----------------------

def generate_cover_letter(resume_text: str, job_title: str, company: str) -> str:
    prompt = f"""
Write a concise, professional cover email (120-180 words) tailored to this job.
Tone: confident, straightforward, outcome-focused. Use bullet points for 2-3 achievements.
No markdown formatting.

Candidate resume:
---
{resume_text}
---

Job:
- Title: {job_title}
- Company: {company}
"""
    text = get_gemini_response(prompt) or ""
    # Basic cleanup if model adds extra spacing
    return text.strip()

def send_email(
    to_email: str,
    subject: str,
    body_text: str,
    attachment_name: str | None = None,
    attachment_bytes: bytes | None = None,
    cc: list[str] | None = None,
) -> tuple[bool, str]:
    try:
        msg = MIMEMultipart()
        msg["From"] = GMAIL_USERNAME
        msg["To"] = to_email
        if cc:
            msg["Cc"] = ", ".join(cc)
        msg["Subject"] = subject

        msg.attach(MIMEText(body_text, "plain"))

        if attachment_bytes and attachment_name:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment_bytes)
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f'attachment; filename="{attachment_name}"')
            msg.attach(part)

        recipients = [to_email] + (cc or [])
        context = ssl.create_default_context()
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls(context=context)
            server.login(GMAIL_USERNAME, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_USERNAME, recipients, msg.as_string())
        return True, "sent"
    except Exception as e:
        return False, str(e)


def attempt_auto_apply_by_email(job: dict, resume_bytes: bytes, resume_filename: str, resume_text: str, candidate_email: str, candidate_name: str) -> dict:
    to_addr = job.get("apply_email")
    if not to_addr:
        return {"applied": False, "reason": "No apply email available", "details": ""}

    company = job.get("company") or "Hiring Team"
    title = job.get("title") or "Role"
    cover = generate_cover_letter(resume_text, title, company)

    # Add candidate contact footer
    cover += f"\n\nBest regards,\n{candidate_name}\nEmail: {candidate_email}"

    subject = f"Application: {title} at {company} — {candidate_name}"
    ok, msg = send_email(
        to_email=to_addr,
        subject=subject,
        body_text=cover,
        attachment_name=resume_filename,
        attachment_bytes=resume_bytes,
        cc=[candidate_email],  # CC you on each application email
    )
    return {
        "applied": ok,
        "reason": "" if ok else "Email send failed",
        "details": msg,
    }


# ---------------------- Report Email to You ----------------------

def send_summary_report(
    recipient_email: str,
    candidate_name: str,
    queries: list[str],
    jobs: list[dict],
    application_results: list[dict],
):
    lines = []
    lines.append(f"Hi {candidate_name or 'there'},")
    lines.append("")
    lines.append("Here is your AI Job Hunter summary:")
    lines.append("")
    lines.append("Search queries:")
    for q in queries:
        lines.append(f"- {q}")
    lines.append("")

    applied_count = 0
    for j, res in zip(jobs, application_results):
        title = j.get("title") or "Untitled"
        company = j.get("company") or "N/A"
        url = j.get("url") or ""
        src = j.get("source") or "Unknown"
        applied = res.get("applied", False)
        status = "APPLIED (email)" if applied else f"NOT APPLIED ({res.get('reason', 'n/a')})"
        if applied:
            applied_count += 1
        lines.append(f"- [{src}] {title} — {company} — {status}")
        if url:
            lines.append(f"  Link: {url}")
    lines.append("")
    lines.append(f"Total applied (via email): {applied_count} of {len(jobs)}")
    lines.append("")
    lines.append("Note: Only postings with a public apply email can be auto-applied. Others require submission on their ATS page.")

    body = "\n".join(lines)
    subject = "Your AI Job Hunter Report"
    ok, msg = send_email(to_email=recipient_email, subject=subject, body_text=body)
    return ok, msg


# ---------------------- UI ----------------------

st.title("🤖 AI Job Hunter & Auto‑Apply")
st.markdown("Let the assistant analyze your resume, find matching jobs, attempt email-based applications where possible, and send you a report.")

with st.sidebar:
    st.header("Upload & Preferences")
    uploaded_pdf = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
    candidate_name = st.text_input("Your full name", value="Kishan Pratap")
    candidate_email = st.text_input("Your email", value="kishanpratap3@gmail.com")
    recipient_report_email = st.text_input("Send summary report to", value="kishanpratap3@gmail.com")
    max_jobs = st.slider("Max jobs to evaluate", min_value=10, max_value=100, value=40, step=10)
    auto_apply_toggle = st.checkbox("Auto-apply by email when available", value=True)
    start_button = st.button("Start Job Hunt", type="primary", use_container_width=True)


# ---------------------- Main Logic ----------------------

if start_button:
    if not uploaded_pdf:
        st.error("Please upload your resume PDF first.")
        st.stop()

    resume_bytes = uploaded_pdf.getvalue()
    st.info("Extracting text from your resume...")
    resume_text = extract_text_from_pdf(resume_bytes)
    if not resume_text:
        st.error("Could not extract text from the PDF. Ensure it has selectable text (not scanned images).")
        st.stop()

    st.info("Generating targeted search queries...")
    queries = generate_search_queries_from_resume(resume_text)
    if not queries:
        st.error("The AI could not generate search queries. Please try again or use a different resume.")
        st.stop()

    st.success("Search queries ready.")
    with st.expander("View generated queries"):
        for q in queries:
            st.markdown(f"- {q}")

    st.info("Searching job boards for matching roles...")
    jobs = aggregate_jobs(queries, per_query_limit=20, max_total=max_jobs)
    if not jobs:
        st.warning("No jobs found for your queries right now. Try adjusting your resume or try again later.")
        st.stop()

    st.success(f"Found {len(jobs)} jobs across sources.")
    st.markdown("Preview of top matches:")
    for j in jobs[:15]:
        st.markdown(f"- [{j.get('title','Untitled')}]({j.get('url','')}) — {j.get('company','N/A')} ({j.get('source','')})")

    application_results = []
    if auto_apply_toggle:
        st.info("Attempting email-based applications where possible...")
        progress = st.progress(0)
        for idx, job in enumerate(jobs):
            res = attempt_auto_apply_by_email(
                job=job,
                resume_bytes=resume_bytes,
                resume_filename=uploaded_pdf.name,
                resume_text=resume_text,
                candidate_email=candidate_email,
                candidate_name=candidate_name,
            )
            application_results.append(res)
            progress.progress((idx + 1) / len(jobs))
        st.success("Auto-apply (email) step complete.")
    else:
        application_results = [{"applied": False, "reason": "Auto-apply disabled", "details": ""} for _ in jobs]

    st.info("Emailing your consolidated report...")
    ok, msg = send_summary_report(
        recipient_email=recipient_report_email,
        candidate_name=candidate_name,
        queries=queries,
        jobs=jobs,
        application_results=application_results,
    )
    if ok:
        st.success(f"Report sent to {recipient_report_email}.")
    else:
        st.error(f"Failed to send report: {msg}")

else:
    st.info("Upload your resume PDF, adjust preferences in the sidebar, and click Start Job Hunt.")


# ---------------------- Footer Note ----------------------
st.markdown(
    "<small>Note: Auto-apply is possible only for postings that expose an application email. "
    "For other roles, please use the provided links to apply via their ATS or portal.</small>",
    unsafe_allow_html=True,
)
