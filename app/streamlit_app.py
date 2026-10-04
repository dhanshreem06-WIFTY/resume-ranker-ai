from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.extraction import SUPPORTED_EXTENSIONS, extract_resume  # noqa: E402
from core.ranking import analyze_candidates  # noqa: E402
from core.web_ingest import fetch_job_description  # noqa: E402

st.set_page_config(
    page_title="ResumeRanker AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Styling: intentionally close to the supplied soft lavender / cream reference.
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

:root {
  --bg: #f4ece6;
  --panel: #fbf6f0;
  --lav: #e8dce7;
  --lav2: #d8c6d8;
  --plum: #8d6b8e;
  --plum-dark: #5b485b;
  --ink: #4d424b;
  --muted: #877982;
  --line: rgba(112, 87, 110, .16);
  --pink: #ef718d;
  --cyan: #79c5c7;
  --gold: #f2bd52;
}

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: var(--bg); color: var(--ink); }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem; max-width: 1440px; }

section[data-testid="stSidebar"] {
  background: #e9dfe2;
  border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] > div { padding-top: 1.2rem; }

.brand { font-family: 'Playfair Display', serif; font-size: 1.55rem; font-weight: 700; color: var(--plum-dark); margin-bottom: .25rem; }
.brand-sub { color: var(--muted); font-size: .78rem; margin-bottom: 1.2rem; }

.hero { display:flex; justify-content:space-between; align-items:center; gap:1rem; margin-bottom:1.1rem; }
.hero h1 { font-family:'Playfair Display',serif; font-size:2.2rem; margin:0; color:var(--plum-dark); letter-spacing:-.03em; }
.hero p { margin:.25rem 0 0; color:var(--muted); }

.card { background:var(--panel); border:1px solid var(--line); border-radius:24px; padding:1rem 1.1rem; box-shadow:0 8px 26px rgba(89,63,84,.05); }
.card h3 { margin:.05rem 0 .55rem; font-size:1rem; color:var(--plum-dark); }
.card-title { font-weight:700; color:var(--plum-dark); }

.metric-card { background:var(--lav); border-radius:20px; padding:1rem; min-height:120px; border:1px solid rgba(120,91,117,.08); }
.metric-label { color:var(--muted); font-size:.82rem; }
.metric-value { color:var(--plum-dark); font-size:1.8rem; font-weight:700; margin:.3rem 0; }
.metric-note { font-size:.72rem; color:var(--muted); }

.pill { display:inline-block; padding:.27rem .55rem; margin:.12rem; border-radius:999px; background:#eee3ec; color:#6f576e; font-size:.72rem; }
.pill.missing { background:#f3dfe5; color:#9a5369; }
.pill.good { background:#dceeea; color:#4f7775; }

.rank-row { background:#fbf7f3; border:1px solid var(--line); border-radius:18px; padding:.75rem .9rem; margin:.45rem 0; }
.rank-number { font-size:1.2rem; font-weight:700; color:var(--plum); }
.score { font-size:1.15rem; font-weight:700; color:var(--plum-dark); }

div.stButton > button { border-radius:14px; border:1px solid var(--line); background:#8d6b8e; color:white; font-weight:600; padding:.55rem 1rem; }
div.stButton > button:hover { border-color:#8d6b8e; background:#765876; color:white; }

.stTextArea textarea, .stTextInput input { border-radius:15px !important; border:1px solid var(--line) !important; background:#fffaf5 !important; }
[data-testid="stFileUploader"] { background:#fffaf5; border-radius:18px; border:1px dashed #b99fb5; padding:.3rem; }

.small { font-size:.78rem; color:var(--muted); }
.warning { padding:.75rem 1rem; border-radius:14px; background:#fff3d9; color:#735d33; }
.footer-note { color:var(--muted); font-size:.72rem; text-align:center; margin-top:1.5rem; }
</style>
""",
    unsafe_allow_html=True,
)


def pill_list(items, missing=False, limit=12):
    if not items:
        return '<span class="small">None detected</span>'
    cls = "pill missing" if missing else "pill good"
    return "".join(f'<span class="{cls}">{x}</span>' for x in items[:limit])


def metric(label, value, note):
    return f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>'


def reset_results():
    st.session_state.pop("analysis", None)
    st.session_state.pop("selected_candidate", None)

# Sidebar
with st.sidebar:
    st.markdown('<div class="brand">ResumeRanker AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Open-source computer vision / NLP-style candidate intelligence</div>', unsafe_allow_html=True)
    page = st.radio("Navigate", ["Dashboard", "Candidate details", "Settings"], index=0)
    st.markdown("---")
    st.markdown("**Workflow**")
    st.caption("1. Add job description\n2. Upload resumes\n3. Set threshold\n4. Analyze\n5. Review evidence")
    st.markdown("---")
    st.markdown("**Privacy**")
    st.caption("Files are processed in this local Streamlit session. No resume is sent to a paid AI API.")

if page == "Settings":
    st.markdown('<div class="hero"><div><h1>Settings</h1><p>Transparent scoring controls and project notes.</p></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><h3>Model</h3><p><b>sentence-transformers/all-MiniLM-L6-v2</b></p><p class="small">The app uses local sentence embeddings for semantic matching. If the model cannot be downloaded, it falls back to TF-IDF.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><h3>Scoring weights</h3><p>Semantic similarity: <b>65%</b> · Skill coverage: <b>25%</b> · Keyword overlap: <b>10%</b></p><p class="small">These are transparent demo weights and can be changed in <code>core/ranking.py</code>.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><h3>Responsible-use note</h3><p>Do not use this score as an automatic hiring decision. Review actual resume evidence and job requirements. The application does not intentionally score protected characteristics.</p></div>', unsafe_allow_html=True)
    st.stop()

# Dashboard input area
st.markdown('<div class="hero"><div><h1>HELLO, RECRUITER ✦</h1><p>Rank resumes, surface strengths, and expose skill gaps.</p></div></div>', unsafe_allow_html=True)

left, right = st.columns([1.65, 1], gap="large")
with left:
    st.markdown('<div class="card"><h3>Job description</h3>', unsafe_allow_html=True)
    jd_url = st.text_input("Public job URL (optional)", placeholder="https://example.com/jobs/...", label_visibility="collapsed")
    if st.button("Fetch job description from URL", use_container_width=False):
        if not jd_url.strip():
            st.warning("Paste a public job URL first.")
        else:
            try:
                st.session_state["job_text"] = fetch_job_description(jd_url)
                st.success("Job description fetched.")
            except Exception as exc:
                st.error(f"Could not fetch the page: {exc}")
    job_text = st.text_area(
        "Paste the job description",
        value=st.session_state.get("job_text", (ROOT / "data" / "sample_job_description.txt").read_text(encoding="utf-8")),
        height=255,
        label_visibility="collapsed",
        help="Paste the complete role description, including requirements and nice-to-have skills.",
    )
    st.session_state["job_text"] = job_text
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card"><h3>Candidate resumes</h3>', unsafe_allow_html=True)
    uploads = st.file_uploader(
        "Upload resumes",
        type=[x.lstrip(".") for x in sorted(SUPPORTED_EXTENSIONS)],
        accept_multiple_files=True,
        help="Supported: PDF, DOC/DOCX, TXT, MD, CSV, RTF, ODT, PNG, JPG/JPEG, WEBP, BMP and TIFF. Images and scanned PDFs are read with local OCR.",
    )
    if uploads:
        st.caption(f"{len(uploads)} file(s) selected")
        for up in uploads:
            st.write(f"• {up.name} · {up.size / 1024:.0f} KB")
    threshold = st.slider("Shortlist threshold", 0, 100, 40, 5, help="Candidates at or above this score get detailed strengths/gap explanations.")
    use_semantic = st.checkbox("Use AI semantic model", value=False, help="Off = near-instant TF-IDF matching. On = downloads/loads the local MiniLM model and gives stronger semantic matching. The model is cached after the first run.")
    st.markdown(f'<div class="small">Detailed analysis: candidates ≥ <b>{threshold}%</b> · {"AI semantic matching" if use_semantic else "Fast matching"}</div>', unsafe_allow_html=True)
    analyze = st.button("Analyze candidates →", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if analyze:
    if not job_text.strip():
        st.error("Add a job description first.")
    elif not uploads:
        st.error("Upload at least one resume.")
    else:
        candidates = []
        errors = []
        for up in uploads:
            try:
                raw = up.getvalue()
                text = extract_resume(up.name, raw)
                if len(text) < 50:
                    errors.append(f"{up.name}: not enough readable text")
                    continue
                stem = Path(up.name).stem.replace("_", " ").replace("-", " ").strip()
                candidates.append({"name": stem.title(), "filename": up.name, "text": text})
            except Exception as exc:
                errors.append(f"{up.name}: {exc}")
        if errors:
            for e in errors:
                st.warning(e)
        if candidates:
            with st.spinner("Embedding resumes and calculating transparent scores..."):
                st.session_state["analysis"] = analyze_candidates(job_text, candidates, threshold, use_semantic=use_semantic)
                st.session_state["selected_candidate"] = 0
            st.success(f"Analyzed {len(candidates)} candidate(s).")

            with st.expander("Preview extracted resume text", expanded=False):
                for candidate in candidates:
                    st.markdown(f"**{candidate['filename']}**")
                    st.code(candidate["text"][:3500], language="text")

analysis = st.session_state.get("analysis")
if not analysis:
    st.markdown('<div class="card" style="margin-top:1rem"><h3>Ready when you are</h3><p>Use the included sample data or upload your own resumes. The first model run may take longer while the open-source embedding model is cached.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="footer-note">ResumeRanker AI · local-first demo · transparent scoring · human review required</div>', unsafe_allow_html=True)
    st.stop()

results = analysis["results"]
qualified = [r for r in results if r["above_threshold"]]

if page == "Candidate details":
    st.markdown('<div class="hero"><div><h1>Candidate details</h1><p>Evidence-backed strengths and skill gaps.</p></div></div>', unsafe_allow_html=True)
    names = [f"#{r['rank']} · {r['name']} · {r['score']:.1f}%" for r in results]
    idx = st.selectbox("Candidate", range(len(results)), format_func=lambda i: names[i])
    r = results[idx]
    st.markdown(f'<div class="card"><h3>{r["name"]}</h3><p class="small">Rank #{r["rank"]} · {r["filename"]} · threshold {analysis["threshold"]}%</p></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown(metric("Overall fit", f"{r['score']:.1f}%", "hybrid score"), unsafe_allow_html=True)
    c2.markdown(metric("Semantic", f"{r['semantic_score']:.1f}%", "meaning-level match"), unsafe_allow_html=True)
    c3.markdown(metric("Skill coverage", f"{r['skill_score']:.1f}%", "detected job skills"), unsafe_allow_html=True)
    st.markdown("### Strengths")
    st.markdown(pill_list(r["strengths"]), unsafe_allow_html=True)
    st.markdown("### Skill gaps")
    st.markdown(pill_list(r["missing_skills"], missing=True), unsafe_allow_html=True)
    st.markdown("### Evidence")
    for skill, evidence in r["evidence"].items():
        st.markdown(f"**{skill}** — {evidence}")
    if not r["above_threshold"]:
        st.info("This candidate is below the configured threshold, so the dashboard does not label the result as a shortlist recommendation.")
    st.stop()

# Dashboard results
st.markdown("### Candidate overview")
c1, c2, c3, c4 = st.columns(4)
avg = sum(r["score"] for r in results) / len(results)
c1.markdown(metric("Candidates", len(results), "resumes analyzed"), unsafe_allow_html=True)
c2.markdown(metric("Threshold", f"{analysis['threshold']}%", "detailed review line"), unsafe_allow_html=True)
c3.markdown(metric("Above threshold", len(qualified), "candidate profiles"), unsafe_allow_html=True)
c4.markdown(metric("Average fit", f"{avg:.1f}%", "across all resumes"), unsafe_allow_html=True)

st.markdown("### Ranking")
for r in results:
    status = "Above threshold" if r["above_threshold"] else "Below threshold"
    st.markdown(
        f'''<div class="rank-row"><div style="display:flex;align-items:center;justify-content:space-between;gap:1rem;">
        <div><span class="rank-number">#{r['rank']}</span>&nbsp;&nbsp;<b>{r['name']}</b><div class="small">{status} · {r['filename']}</div></div>
        <div style="text-align:right"><div class="score">{r['score']:.1f}%</div><div class="small">semantic {r['semantic_score']:.0f}% · skills {r['skill_score']:.0f}%</div></div>
        </div></div>''', unsafe_allow_html=True
    )

if qualified:
    st.markdown("### Strengths & weaknesses beyond threshold")
    for r in qualified:
        with st.expander(f"{r['name']} · {r['score']:.1f}%", expanded=(r["rank"] == 1)):
            a, b = st.columns(2)
            with a:
                st.markdown("**Strengths**")
                st.markdown(pill_list(r["strengths"]), unsafe_allow_html=True)
                st.markdown("**Evidence**")
                for skill, evidence in list(r["evidence"].items())[:5]:
                    st.caption(f"{skill}: {evidence}")
            with b:
                st.markdown("**Skill gaps / weaknesses to review**")
                st.markdown(pill_list(r["missing_skills"], missing=True), unsafe_allow_html=True)
                st.caption("A gap means the skill was detected in the job description but not detected in this resume. It does not prove the candidate lacks the skill.")
else:
    st.info("No candidate is at or above the configured threshold. Lower the threshold if you want to inspect more profiles.")

st.markdown("### Job skills detected")
st.markdown(pill_list(analysis["job_skills"]), unsafe_allow_html=True)

# Export
export_rows = []
for r in results:
    export_rows.append({
        "rank": r["rank"], "candidate": r["name"], "filename": r["filename"],
        "overall_score": r["score"], "semantic_score": r["semantic_score"],
        "skill_score": r["skill_score"], "keyword_score": r["keyword_score"],
        "above_threshold": r["above_threshold"],
        "matched_skills": ", ".join(r["matched_skills"]),
        "skill_gaps": ", ".join(r["missing_skills"]),
    })
export_df = pd.DataFrame(export_rows)
json_payload = json.dumps({"job_skills": analysis["job_skills"], "threshold": analysis["threshold"], "results": export_rows}, indent=2)

st.markdown("### Export")
d1, d2 = st.columns(2)
d1.download_button("Download ranking CSV", export_df.to_csv(index=False).encode("utf-8"), "resume_ranking.csv", "text/csv", use_container_width=True)
d2.download_button("Download analysis JSON", json_payload.encode("utf-8"), "resume_analysis.json", "application/json", use_container_width=True)

st.markdown(f'<div class="footer-note">Engine: {analysis["engine"]} · Scores are decision-support signals, not hiring decisions.</div>', unsafe_allow_html=True)
