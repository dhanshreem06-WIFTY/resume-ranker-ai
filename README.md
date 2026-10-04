# ResumeRanker AI

A local, open-source-model resume ranking application built with Streamlit.

It compares multiple resumes against a job description, ranks candidates by a transparent score, and explains **strengths and skill gaps** for candidates above a configurable threshold.

## Features

- Upload multiple resumes at once: PDF, DOC/DOCX, TXT, MD, CSV, RTF, ODT, PNG, JPG/JPEG, WEBP, BMP and TIFF.
- Paste a job description or fetch a public job-description webpage URL.
- Semantic matching with `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0).
- Transparent hybrid score:
  - 65% semantic similarity
  - 25% skill coverage
  - 10% explicit keyword coverage
- Configurable shortlist threshold.
- Strengths and weaknesses/skill gaps with evidence snippets.
- Candidate detail view.
- CSV and JSON export.
- Pastel dashboard UI inspired by the supplied reference image: rounded cards, left navigation, soft lavender/cream palette, compact analytics cards.
- Runs locally on Windows, macOS and Linux and opens in a browser.
- No API key and no paid AI service required.
- Optional public URL ingestion for job descriptions.
- Local OCR for image resumes and scanned PDFs using RapidOCR.
- Legacy `.doc` extraction through LibreOffice when available.
- Extracted-text preview so you can verify what the parser/OCR saw.
- TF-IDF fallback if the embedding model cannot be downloaded.

## Project structure

```text
ResumeRankerAI/
├── app.py
├── app/
│   └── streamlit_app.py
├── core/
│   ├── __init__.py
│   ├── extraction.py
│   ├── ocr.py
│   ├── ranking.py
│   └── web_ingest.py
├── data/
│   ├── skills_taxonomy.json
│   ├── sample_job_description.txt
│   └── sample_resumes/
│       ├── arjun_resume.txt
│       ├── meera_resume.txt
│       └── rohan_resume.txt
├── .streamlit/
│   └── config.toml
├── .gitignore
├── LICENSE
├── requirements.txt
├── run.bat
├── run.sh
├── Dockerfile
├── render.yaml
└── README.md
```

## 1. Requirements

Install Python 3.10–3.12. For the easiest local OCR setup, Python 3.11 is recommended.

The first run downloads the open-source embedding model and the OCR runtime/model. After that, they are cached locally.

## 2. Run in VS Code

### Windows

Open the project folder in VS Code, open the terminal, then:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Or double-click `run.bat` after Python is installed.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Or:

```bash
chmod +x run.sh
./run.sh
```

The browser normally opens at `http://localhost:8501`.

## 3. Try the included demo

1. Start the app.
2. Use the sample job description from `data/sample_job_description.txt`.
3. Upload the three files inside `data/sample_resumes/`.
4. Set the threshold to 40%.
5. Click **Analyze candidates**.

## Scoring logic

The score is deliberately transparent and is not a black-box hiring decision:

```text
Semantic score   = cosine similarity between job and resume embeddings
Skill score      = matched required/important skills / detected job skills
Keyword score    = explicit keyword overlap

Final score = 0.65 × semantic + 0.25 × skill + 0.10 × keyword
```

The app also identifies skill gaps by comparing the job's detected skills with the candidate's detected skills.

## File types and OCR

- **PDF:** normal text extraction first; scanned PDFs automatically fall back to OCR.
- **DOCX:** paragraphs and tables are extracted.
- **DOC:** converted to text through LibreOffice when available. The included Docker/Render setup installs LibreOffice automatically.
- **TXT/MD/CSV/RTF/ODT:** parsed locally.
- **Images:** OCR is applied to PNG/JPG/JPEG/WEBP/BMP/TIFF resumes.

For local Windows use, `.doc` files may require LibreOffice installed on the machine. If you do not need old Word files, `.docx` is preferred.

## URL ingestion

You can paste a publicly accessible job-description URL. The app extracts page text locally with `requests` + `BeautifulSoup`. Some sites block automated requests or render content only in JavaScript; in that case, paste the job description manually instead.

## Render deployment

The included `Dockerfile` and `render.yaml` are ready for a Docker-based Render web service. The Docker image installs LibreOffice so legacy `.doc` resumes work in the deployed app.

## Important limitation

This is a decision-support/demo system, not an autonomous hiring system. It should not be used as the sole basis for employment decisions. It intentionally does not score candidates using age, gender, race, religion, photo, address, or other protected/personal attributes. Semantic similarity can still reproduce bias present in training data, so human review is required.

## Open-source references

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` — Apache-2.0. The model card documents local use through Sentence Transformers.
- Streamlit provides the file uploader and download-button components used by this project.
- O*NET is a useful public occupational-information resource for extending the skill taxonomy.

See `data/skills_taxonomy.json` for the starter taxonomy used by the demo.


### Performance
The dashboard defaults to Fast matching (TF-IDF) so it responds quickly. Turn on AI semantic matching only when you want the local MiniLM model; it is cached after the first load. OCR is also cached so multiple image resumes do not recreate the OCR engine.
