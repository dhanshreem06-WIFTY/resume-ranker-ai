from __future__ import annotations

import json
import re
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_taxonomy(path: str | Path | None = None) -> Dict[str, List[str]]:
    path = Path(path or Path(__file__).resolve().parents[1] / "data" / "skills_taxonomy.json")
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_skills(taxonomy: Dict[str, List[str]]) -> List[str]:
    return [s.lower() for values in taxonomy.values() for s in values]


def detect_skills(text: str, taxonomy: Dict[str, List[str]]) -> List[str]:
    low = text.lower()
    found = []
    for skill in flatten_skills(taxonomy):
        # allow punctuation variants such as node.js / nodejs
        pattern = re.escape(skill).replace(r"\.", r"[. ]?")
        if re.search(r"(?<![a-z0-9])" + pattern + r"(?![a-z0-9])", low):
            found.append(skill)
    return sorted(set(found))


def extract_keywords(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]{2,}", text.lower())
    stop = {
        "the", "and", "for", "with", "from", "that", "this", "are", "will", "you",
        "your", "our", "have", "has", "into", "about", "using", "work", "working",
        "team", "role", "job", "candidate", "experience", "knowledge", "skills", "years",
        "required", "requirements", "responsibilities", "good", "strong", "ability", "looking",
    }
    return sorted(set(w for w in words if w not in stop and len(w) > 3))


def _clip(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def semantic_scores(job: str, resumes: List[str], use_semantic: bool = True) -> Tuple[List[float], str]:
    if not use_semantic:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        matrix = vectorizer.fit_transform([job] + resumes)
        sims = cosine_similarity(matrix[1:], matrix[0:1]).ravel()
        return [_clip(float(x)) for x in sims], "TF-IDF (fast mode)"
    try:
        from sentence_transformers import SentenceTransformer

        @lru_cache(maxsize=1)
        def get_model():
            return SentenceTransformer(MODEL_NAME)

        model = get_model()
        emb = model.encode([job] + resumes, normalize_embeddings=True, show_progress_bar=False)
        sims = np.dot(emb[1:], emb[0])
        # cosine similarity is typically 0..1 for this use; clamp for a UI score.
        scores = [_clip(float(x)) for x in sims]
        return scores, "Sentence Transformers / all-MiniLM-L6-v2"
    except Exception:
        # Lightweight local fallback. This keeps the demo usable if the model
        # cannot be downloaded (e.g. no internet on first launch).
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        matrix = vectorizer.fit_transform([job] + resumes)
        sims = cosine_similarity(matrix[1:], matrix[0:1]).ravel()
        return [_clip(float(x)) for x in sims], "TF-IDF fallback (no model download)"


def keyword_score(job: str, resume: str) -> float:
    job_kw = set(extract_keywords(job))
    resume_kw = set(extract_keywords(resume))
    if not job_kw:
        return 0.0
    return _clip(len(job_kw & resume_kw) / len(job_kw))


def skill_score(job_skills: List[str], resume_skills: List[str]) -> float:
    if not job_skills:
        return 0.0
    return _clip(len(set(job_skills) & set(resume_skills)) / len(set(job_skills)))


def find_evidence(skill: str, resume: str) -> str:
    lines = [line.strip() for line in resume.splitlines() if line.strip()]
    low = skill.lower()
    for line in lines:
        if low in line.lower():
            return line[:220]
    # Fallback: return a useful short resume excerpt.
    return " ".join(lines)[:220]


def analyze_candidates(job: str, candidates: List[dict], threshold: int = 60, use_semantic: bool = True) -> dict:
    taxonomy = load_taxonomy()
    job_skills = detect_skills(job, taxonomy)
    semantic, engine = semantic_scores(job, [c["text"] for c in candidates], use_semantic=use_semantic)

    results = []
    for candidate, sem in zip(candidates, semantic):
        resume_skills = detect_skills(candidate["text"], taxonomy)
        matched = sorted(set(job_skills) & set(resume_skills))
        missing = sorted(set(job_skills) - set(resume_skills))
        sk = skill_score(job_skills, resume_skills)
        kw = keyword_score(job, candidate["text"])
        final = (0.65 * sem) + (0.25 * sk) + (0.10 * kw)
        result = {
            "name": candidate["name"],
            "filename": candidate.get("filename", candidate["name"]),
            "score": round(final * 100, 1),
            "semantic_score": round(sem * 100, 1),
            "skill_score": round(sk * 100, 1),
            "keyword_score": round(kw * 100, 1),
            "matched_skills": matched,
            "missing_skills": missing,
            "strengths": matched[:10],
            "evidence": {s: find_evidence(s, candidate["text"]) for s in matched[:8]},
            "above_threshold": final * 100 >= threshold,
            "resume_text": candidate["text"],
        }
        results.append(result)

    results.sort(key=lambda x: x["score"], reverse=True)
    for idx, r in enumerate(results, start=1):
        r["rank"] = idx

    return {
        "engine": engine,
        "job_skills": job_skills,
        "threshold": threshold,
        "results": results,
    }
