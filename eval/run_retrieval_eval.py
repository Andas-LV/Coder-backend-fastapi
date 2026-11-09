"""Retrieval quality evaluation: does the search find the chunk that holds the answer?

Dataset: eval/dataset.jsonl, one JSON object per line:
    {"question": "...", "document": "<filename in the DB>", "evidence": "<text that must
     appear in the right chunk>"}

For every question the script runs the same search the chat uses
(DocumentService.search: embed the query -> pgvector cosine search), restricted to the
question's document, and looks for the evidence in the returned chunks.

Run (from the project root):
    uv run python -m eval.run_retrieval_eval --label baseline
    uv run python -m eval.run_retrieval_eval --label chunk300 --dataset eval/my.jsonl

Metrics:
    Recall@k  share of questions whose evidence chunk is in the top k
    MRR@10    mean of 1/rank of the first correct chunk (0 if not in the top 10)
    score     mean similarity of the first correct chunk / of the best wrong chunk;
              the gap between them shows how separable right and wrong answers are
              (and where RAG_MIN_SCORE can sit)
"""

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy import select

from app.core.config import settings
from app.db.session import SessionLocal, engine
from app.models import Document
from app.models.document import STATUS_READY
from app.repositories.document import SearchRow
from app.services.document_service import DocumentService
from app.services.embeddings import get_embedding_provider

EVAL_DIR = Path(__file__).parent
DEFAULT_DATASET = EVAL_DIR / "dataset.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
TOP_K = 10
KS = (1, 3, 5, 10)
PREVIEW_CHARS = 100


@dataclass
class Case:
    line: int
    question: str
    document: str
    evidence: str


@dataclass
class DocRef:
    id: uuid.UUID
    user_id: uuid.UUID


def normalize(text: str) -> str:
    """lower, ё -> е, all whitespace (spaces, newlines, tabs) collapsed to one space.

    NFKC first: PDFs and uploaded filenames often carry decomposed letters (Й as
    И + U+0306, ё as е + U+0308) or ligatures (ﬁ) that look identical but don't match.
    """
    text = unicodedata.normalize("NFKC", text).lower().replace("ё", "е")
    return re.sub(r"\s+", " ", text).strip()


def load_dataset(path: Path) -> list[Case]:
    if not path.exists():
        sys.exit(
            f"Dataset not found: {path}\n"
            f"Create it (see {EVAL_DIR / 'dataset.example.jsonl'} for the format)."
        )
    cases: list[Case] = []
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
            case = Case(n, item["question"], item["document"], item["evidence"])
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            sys.exit(
                f"{path}:{n}: bad line ({e!r}); expected question/document/evidence"
            )
        if not all(
            isinstance(v, str) and v.strip()
            for v in (case.question, case.document, case.evidence)
        ):
            sys.exit(f"{path}:{n}: question, document and evidence must be non-empty")
        cases.append(case)
    if not cases:
        sys.exit(f"{path} has no questions")
    return cases


async def find_documents(session, filenames: set[str]) -> dict[str, DocRef]:
    """Dataset filename -> the most recent ready document with that name.

    Names are compared normalized (see normalize), so a filename typed by hand matches
    the one stored from the upload even if their Unicode forms differ.
    """
    wanted = {normalize(f): f for f in filenames}
    rows = await session.execute(
        select(Document.filename, Document.id, Document.user_id)
        .where(Document.status == STATUS_READY)
        .order_by(Document.created_at)
    )
    found: dict[str, DocRef] = {}
    counts: dict[str, int] = {}
    for stored, doc_id, user_id in rows.all():
        name = wanted.get(normalize(stored))
        if name is not None:
            found[name] = DocRef(doc_id, user_id)  # later rows win -> newest
            counts[name] = counts.get(name, 0) + 1
    for name, n in counts.items():
        if n > 1:
            print(f"! {name}: {n} ready copies in the DB, using the newest")
    return found


def preview(text: str) -> str:
    flat = re.sub(r"\s+", " ", text).strip()
    return flat[:PREVIEW_CHARS] + ("…" if len(flat) > PREVIEW_CHARS else "")


def evaluate_case(case: Case, rows: list[SearchRow], seconds: float) -> dict:
    evidence = normalize(case.evidence)
    hits = [evidence in normalize(r.content) for r in rows]
    rank = hits.index(True) + 1 if True in hits else None
    wrong = [r.score for r, hit in zip(rows, hits, strict=True) if not hit]
    return {
        "line": case.line,
        "question": case.question,
        "document": case.document,
        "evidence": case.evidence,
        "rank": rank,
        "correct_score": rows[rank - 1].score if rank else None,
        "best_wrong_score": max(wrong) if wrong else None,
        "search_seconds": round(seconds, 3),
        "top": [
            {
                "rank": i,
                "hit": hit,
                "score": round(r.score, 4),
                "page": r.page,
                "position": r.position,
                "preview": preview(r.content),
            }
            for i, (r, hit) in enumerate(zip(rows, hits, strict=True), start=1)
        ],
    }


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def compute_metrics(results: list[dict]) -> dict:
    n = len(results)
    ranks = [r["rank"] for r in results]
    metrics = {
        f"recall@{k}": round(sum(1 for x in ranks if x and x <= k) / n, 4) for k in KS
    }
    metrics["mrr@10"] = round(sum(1 / x for x in ranks if x) / n, 4)
    metrics["mean_correct_score"] = _mean(
        [r["correct_score"] for r in results if r["correct_score"] is not None]
    )
    metrics["mean_best_wrong_score"] = _mean(
        [r["best_wrong_score"] for r in results if r["best_wrong_score"] is not None]
    )
    metrics["questions"] = n
    metrics["found_in_top10"] = sum(1 for x in ranks if x)
    return metrics


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def print_report(metrics: dict, results: list[dict], skipped: list[dict]) -> None:
    n = metrics["questions"]
    print()
    print(f"Questions evaluated: {n}   found in top-10: {metrics['found_in_top10']}")
    if skipped:
        print(f"Skipped (document not found / not ready): {len(skipped)}")
    print()
    print(f"{'metric':<24}{'value':>8}")
    print("-" * 32)
    for k in KS:
        print(f"{f'Recall@{k}':<24}{_fmt(metrics[f'recall@{k}']):>8}")
    print(f"{'MRR@10':<24}{_fmt(metrics['mrr@10']):>8}")
    print(f"{'mean score (correct)':<24}{_fmt(metrics['mean_correct_score']):>8}")
    print(f"{'mean score (best wrong)':<24}{_fmt(metrics['mean_best_wrong_score']):>8}")
    print(f"{'RAG_MIN_SCORE (current)':<24}{settings.rag_min_score:>8.3f}")

    misses = [r for r in results if r["rank"] is None or r["rank"] > 1]
    if not misses:
        print("\nNo misses: every evidence chunk was ranked first.")
        return
    print(f"\nMisses (correct chunk not at rank 1): {len(misses)}")
    for r in misses:
        where = "not in top-10" if r["rank"] is None else f"rank {r['rank']}"
        print(f"\n[{where}] line {r['line']} · {r['document']}")
        print(f"  Q: {r['question']}")
        print(f"  evidence: {preview(r['evidence'])}")
        for t in r["top"][:3]:
            mark = "✓" if t["hit"] else " "
            print(
                f"  {mark} #{t['rank']} {t['score']:.3f} p.{t['page']}  {t['preview']}"
            )
    for s in skipped:
        print(f"\n[skipped] line {s['line']}: {s['document']!r} {s['reason']}")


def save(label: str, dataset: Path, metrics: dict, results, skipped) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_label = re.sub(r"[^\w.-]+", "_", label).strip("_") or "run"
    now = datetime.now()
    path = RESULTS_DIR / f"{now:%Y-%m-%d_%H%M%S}_{safe_label}.json"
    payload = {
        "label": label,
        "created_at": now.isoformat(timespec="seconds"),
        "dataset": str(dataset),
        "config": {
            "embed_model": settings.embed_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": TOP_K,
            "rag_min_score": settings.rag_min_score,
        },
        "metrics": metrics,
        "results": results,
        "skipped": skipped,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


async def run(label: str, dataset: Path) -> None:
    cases = load_dataset(dataset)
    embeddings = get_embedding_provider()
    results: list[dict] = []
    skipped: list[dict] = []

    async with SessionLocal() as session:
        docs = await find_documents(session, {c.document for c in cases})
        search = DocumentService(session, embeddings).search
        for i, case in enumerate(cases, start=1):
            doc = docs.get(case.document)
            if doc is None:
                skipped.append(
                    {
                        "line": case.line,
                        "document": case.document,
                        "reason": "is not in the DB or not ready",
                    }
                )
                continue
            started = time.perf_counter()
            rows = await search(doc.user_id, case.question, TOP_K, [doc.id])
            results.append(evaluate_case(case, rows, time.perf_counter() - started))
            print(f"\r{i}/{len(cases)}", end="", flush=True)
    await engine.dispose()

    if not results:
        sys.exit("\nNothing evaluated: none of the dataset documents is in the DB.")
    metrics = compute_metrics(results)
    print_report(metrics, results, skipped)
    print(f"\nSaved: {save(label, dataset, metrics, results, skipped)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate document retrieval quality.")
    parser.add_argument(
        "--label", required=True, help="name of this run, e.g. baseline"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"JSONL dataset (default: {DEFAULT_DATASET})",
    )
    args = parser.parse_args()
    # Windows consoles default to cp1251/cp866; Russian text and ✓ need UTF-8
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(run(args.label, args.dataset))


if __name__ == "__main__":
    main()
