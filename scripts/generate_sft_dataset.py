#!/usr/bin/env python3
"""
Generate NL2BM25 SFT dataset.

Converts natural language queries from BEIR datasets into Tantivy boolean
search queries using multiple models via NVIDIA NIM, cycling round-robin to
spread load across quotas. Each example includes an explicit chain-of-thought
reasoning trace, making it suitable for both SFT and as a warm-start for GRPO.

Resumes automatically — every completed query is checkpointed to JSONL
before the next request is made.

Usage:
    uv run scripts/generate_sft_dataset.py
    uv run scripts/generate_sft_dataset.py --max-per-dataset 20 --dry-run
    uv run scripts/generate_sft_dataset.py --push-to-hub your-org/nl2bm25-sft
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import Dataset, load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm

from searchlm.services.search import SearchEngine

# ── Paths ─────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("data/sft")
CHECKPOINT_PATH = OUTPUT_DIR / "progress.jsonl"
DATASET_PATH = OUTPUT_DIR / "dataset"

# ── API config ────────────────────────────────────────────────────────────────

NVIDIA_API_KEY = os.environ.get(
    "NVIDIA_API_KEY",
    "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",
)

# Cycled round-robin to spread load across per-model quotas.
# (name, rpm) — keep rpm conservative so bursts from parallel workers don't trip 429s.
MODELS: list[tuple[str, int]] = [
    ("mistralai/mistral-medium-3.5-128b", 20),
    ("meta/llama-3.3-70b-instruct", 20),
    ("mistralai/mistral-large-2-123b", 15),
    ("nvidia/llama-3.3-nemotron-super-49b-v1", 20),
]

# ── Dataset registry ──────────────────────────────────────────────────────────
# has_index: True → full retrieval + NDCG validation using the local Tantivy index
# max: cap queries sampled from this dataset (None = use all)

# Target ~5K total: nfcorpus(110) + scifact(809) + fiqa(1500) + arguana(1000) + hotpotqa(1000) + nq(800)
DATASET_CONFIGS = [
    {
        "id": "nfcorpus",
        "hf": "mteb/nfcorpus",
        "qrels_split": "train",
        "max": None,
    },  # ~110
    {
        "id": "scifact",
        "hf": "mteb/scifact",
        "qrels_split": "train",
        "max": None,
    },  # ~809
    {"id": "fiqa", "hf": "mteb/fiqa", "qrels_split": "train", "max": 1500},
    {"id": "arguana", "hf": "mteb/arguana", "qrels_split": "test", "max": 1000},
    {"id": "hotpotqa", "hf": "mteb/hotpotqa", "qrels_split": "train", "max": 1000},
    {"id": "nq", "hf": "mteb/nq", "qrels_split": "test", "max": 800},
]

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert information retrieval specialist. Your job is to convert \
natural language queries into high-quality Tantivy boolean search queries that \
maximise retrieval of relevant documents.

## Tantivy Boolean Syntax Reference

| Construct | Syntax | Example |
|-----------|--------|---------|
| Single term | `word` | cancer |
| Exact phrase | `"two or more words"` | "bone density" |
| Both required | `A AND B` | vitamin AND calcium |
| Either matches | `A OR B` | cancer OR tumor OR malignancy |
| Exclude term | `NOT A` | NOT review |
| Grouping | `(A OR B) AND C` | (cat OR feline) AND behavior |
| Field scope | `field:term` | title:"machine learning" |
| Boost term | `term^2` | cancer^2 OR tumor |

**Does NOT work:** wildcards (*), regex, fuzzy (~), numeric ranges, `site:`, \
`filetype:`, `age:`, `genome:` or any other non-indexed field names.

## Construction Strategy

1. Identify 2–4 core concepts in the query (ignore stop words)
2. For each concept, collect domain synonyms and related terms
3. Connect concepts with AND; connect synonyms with OR
4. Quote multi-word concepts as phrases
5. Scope high-precision terms to `title:` when they are central to the topic
6. Keep queries readable — avoid OR chains with 8+ terms

## Examples

**NL:** effects of caffeine on sleep quality
**Boolean:** (caffeine OR coffee OR "coffee consumption" OR theophylline) AND \
(sleep OR insomnia OR "sleep quality" OR "sleep duration" OR "sleep disturbance")

**NL:** machine learning for early cancer detection
**Boolean:** ("machine learning" OR "deep learning" OR "neural network" OR \
"artificial intelligence" OR AI) AND (cancer OR tumor OR malignancy OR \
carcinoma OR neoplasm) AND (detection OR diagnosis OR screening OR \
"early detection" OR classification)

**NL:** What causes antibiotic resistance in hospital settings?
**Boolean:** ("antibiotic resistance" OR "antimicrobial resistance" OR AMR OR \
MRSA OR "drug resistance") AND (hospital OR nosocomial OR "healthcare-associated" \
OR clinical) AND (cause OR mechanism OR emergence OR factor OR acquisition)

**NL:** economic consequences of deforestation in the Amazon
**Boolean:** (deforestation OR "forest loss" OR "forest clearance" OR logging) \
AND (Amazon OR Amazonia OR "tropical forest" OR "rainforest") AND \
(economic OR economy OR "economic impact" OR cost OR livelihood OR poverty)

**NL:** Who won the 2018 FIFA World Cup and what was the final score?
**Boolean:** ("FIFA World Cup" OR "World Cup 2018" OR "2018 World Cup") AND \
(winner OR champion OR final OR result OR score)

## Output Format

You MUST respond in exactly this format — no other text:

<reasoning>
Identify the key concepts in the query. For each concept list candidate synonyms \
and decide which to include. Explain your AND/OR structure choices and any \
field-scoping decisions.
</reasoning>
<query>your single-line boolean query here</query>"""

USER_TEMPLATE = "Convert to a Tantivy boolean search query:\n\n{nl_query}"


# ── Rate limiter ──────────────────────────────────────────────────────────────


class RateLimiter:
    """Sliding-window rate limiter: at most `rpm` calls per 60-second window."""

    def __init__(self, rpm: int):
        self.rpm = rpm
        self._window: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            # Drop timestamps older than 60 s
            while self._window and now - self._window[0] > 60.0:
                self._window.popleft()

            if len(self._window) >= self.rpm:
                sleep_for = 60.0 - (now - self._window[0]) + 0.05
                await asyncio.sleep(sleep_for)
                now = time.monotonic()
                while self._window and now - self._window[0] > 60.0:
                    self._window.popleft()

            self._window.append(time.monotonic())


# ── Dataset loading ───────────────────────────────────────────────────────────


def _load_query_ids_from_qrels(cfg: dict) -> set[str]:
    """Return query IDs that have relevance judgments in the target split."""
    hf_name = cfg["hf"]
    split = cfg["qrels_split"]
    # MTEB datasets store qrels in the default config, split by train/dev/test
    try:
        qrels_ds = load_dataset(hf_name, "default", split=split)
        return {row["query-id"] for row in qrels_ds}
    except Exception:
        pass
    # Some datasets use a separate -qrels repo
    try:
        qrels_ds = load_dataset(f"{hf_name}-qrels", split=split)
        return {row["query-id"] for row in qrels_ds}
    except Exception:
        pass
    return set()  # fallback: caller will use all queries


def load_queries_for_dataset(
    cfg: dict, max_override: Optional[int] = None
) -> list[dict]:
    """Load (id, nl_query) pairs for one BEIR/MTEB dataset."""
    dataset_id = cfg["id"]
    print(f"  [{dataset_id}] loading queries...")

    try:
        raw = load_dataset(cfg["hf"], "queries", split="queries")
    except Exception as e:
        print(f"  [{dataset_id}] WARNING: could not load queries — {e}")
        return []

    id_to_text: dict[str, str] = {}
    for row in raw:
        qid = row.get("_id") or row.get("id", "")
        text = row.get("text") or row.get("query", "")
        if qid and text:
            id_to_text[qid] = text

    # Filter to queries that actually have relevance judgments
    valid_ids = _load_query_ids_from_qrels(cfg)
    if valid_ids:
        id_to_text = {k: v for k, v in id_to_text.items() if k in valid_ids}

    queries = [
        {
            "id": f"{dataset_id}__{qid}",
            "original_query_id": qid,
            "dataset": dataset_id,
            "nl_query": text,
        }
        for qid, text in id_to_text.items()
    ]

    cap = max_override if max_override is not None else cfg.get("max")
    if cap and len(queries) > cap:
        random.seed(42)
        queries = random.sample(queries, cap)

    print(f"  [{dataset_id}] {len(queries)} queries")
    return queries


def load_all_queries(
    configs: list[dict], max_per_dataset: Optional[int] = None
) -> list[dict]:
    print("\nLoading queries from all datasets...")
    all_queries: list[dict] = []
    for cfg in configs:
        all_queries.extend(load_queries_for_dataset(cfg, max_override=max_per_dataset))
    random.seed(0)
    random.shuffle(all_queries)
    print(f"Total: {len(all_queries)} queries\n")
    return all_queries


# ── Checkpoint helpers ────────────────────────────────────────────────────────


def load_checkpoint() -> tuple[set[str], list[dict]]:
    """Return (done_ids, completed_records) from the checkpoint file."""
    if not CHECKPOINT_PATH.exists():
        return set(), []
    done_ids: set[str] = set()
    records: list[dict] = []
    with open(CHECKPOINT_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done_ids.add(rec["id"])
                records.append(rec)
            except json.JSONDecodeError:
                pass
    return done_ids, records


def append_checkpoint(record: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Generation + parsing ──────────────────────────────────────────────────────


async def call_api(
    client: AsyncOpenAI, model: str, nl_query: str, max_retries: int = 8
) -> str:
    """Call the given model with fixed 20s retry on 429s."""
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": USER_TEMPLATE.format(nl_query=nl_query),
                    },
                ],
                temperature=0.7,
                top_p=0.95,
                max_tokens=2048,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            is_rate_limit = "429" in str(e) or "Too Many Requests" in str(e)
            if is_rate_limit and attempt < max_retries - 1:
                print(
                    f"\n  [429] {model} rate limited — waiting 20s (attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(20)
            else:
                raise


def parse_response(raw: str) -> tuple[Optional[str], Optional[str]]:
    """Extract (reasoning, boolean_query) from the model response."""
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", raw, re.DOTALL)
    query_match = re.search(r"<query>(.*?)</query>", raw, re.DOTALL)

    reasoning = reasoning_match.group(1).strip() if reasoning_match else None
    boolean_query = query_match.group(1).strip() if query_match else None

    # Collapse accidental multi-line queries to a single line
    if boolean_query:
        boolean_query = " ".join(boolean_query.split())

    return reasoning, boolean_query


# ── Retrieval validation (sync — run in executor) ─────────────────────────────


def validate_query(
    boolean_query: str,
    query_record: dict,
    engine: Optional[SearchEngine],
    preloaded_qrels: dict[str, dict],  # dataset_id -> {query_id -> {doc_id -> score}}
) -> dict:
    """
    Syntax-check and measure retrieval quality for a generated boolean query.

    Every dataset is validated the same way:
      1. Parse + execute the query (catches bad syntax).
      2. Check num_retrieved > 0.
      3. If qrels are available for this dataset, compute NDCG@10.
    """
    if engine is None:
        return {
            "syntax_valid": None,
            "retrieval_valid": None,
            "ndcg_at_10": None,
            "num_retrieved": None,
        }

    dataset_id = query_record["dataset"]
    original_qid = query_record["original_query_id"]

    # ── Step 1 & 2: syntax + retrieval ──────────────────────────────────────
    try:
        results = engine.search(boolean_query, limit=100, dataset_filter=dataset_id)
        syntax_valid = True
        num_retrieved = len(results)
        retrieval_valid = num_retrieved > 0
    except Exception:
        return {
            "syntax_valid": False,
            "retrieval_valid": False,
            "ndcg_at_10": None,
            "num_retrieved": 0,
        }

    # ── Step 3: NDCG (whenever qrels are available) ──────────────────────────
    ndcg_at_10 = None
    query_qrels = preloaded_qrels.get(dataset_id, {}).get(original_qid, {})
    if query_qrels:
        try:
            from searchlm.services.metrics import calculate_ndcg

            relevance_scores = [query_qrels.get(r["doc_id"], 0.0) for r in results]
            ndcg_at_10 = calculate_ndcg(relevance_scores, k=10)
        except Exception:
            pass

    return {
        "syntax_valid": syntax_valid,
        "retrieval_valid": retrieval_valid,
        "ndcg_at_10": ndcg_at_10,
        "num_retrieved": num_retrieved,
    }


# ── Core per-query worker ─────────────────────────────────────────────────────


async def process_query(
    query_record: dict,
    client: AsyncOpenAI,
    model: str,
    limiter: RateLimiter,
    semaphore: asyncio.Semaphore,
    engine: Optional[SearchEngine],
    preloaded_qrels: dict,
    dry_run: bool = False,
) -> Optional[dict]:
    async with semaphore:
        await limiter.acquire()

        if dry_run:
            reasoning = "This is a dry-run placeholder reasoning trace."
            boolean_query = f'("{query_record["nl_query"][:30]}")'
            raw_response = (
                f"<reasoning>{reasoning}</reasoning><query>{boolean_query}</query>"
            )
        else:
            try:
                raw_response = await call_api(client, model, query_record["nl_query"])
            except Exception as e:
                print(f"\n  [ERROR] API call failed for {query_record['id']}: {e}")
                return None

        reasoning, boolean_query = parse_response(raw_response)

        if not boolean_query:
            # Model failed to follow the format; skip
            return None

        # Validation runs in a thread so it doesn't block the event loop
        loop = asyncio.get_running_loop()
        val = await loop.run_in_executor(
            None, validate_query, boolean_query, query_record, engine, preloaded_qrels
        )

        # Build the SFT-formatted conversation
        assistant_content = (
            f"<reasoning>\n{reasoning}\n</reasoning>\n<query>{boolean_query}</query>"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(nl_query=query_record["nl_query"]),
            },
            {"role": "assistant", "content": assistant_content},
        ]

        record = {
            # Identity
            "id": query_record["id"],
            "original_query_id": query_record["original_query_id"],
            "dataset": query_record["dataset"],
            # Core fields
            "nl_query": query_record["nl_query"],
            "boolean_query": boolean_query,
            "reasoning": reasoning,
            # SFT-ready conversation
            "messages": messages,
            # Quality signals
            "syntax_valid": val["syntax_valid"],
            "retrieval_valid": val["retrieval_valid"],
            "ndcg_at_10": val["ndcg_at_10"],
            "num_retrieved": val["num_retrieved"],
            # Provenance
            "generator_model": model,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        append_checkpoint(record)
        return record


# ── Dataset assembly ──────────────────────────────────────────────────────────


def assemble_and_save(records: list[dict], push_to_hub: Optional[str]) -> None:
    print(f"\nAssembling dataset from {len(records)} records...")

    syntax_ok = sum(1 for r in records if r.get("syntax_valid") is not False)
    retrieval_ok = sum(1 for r in records if r.get("retrieval_valid") is True)
    with_ndcg = [r for r in records if r.get("ndcg_at_10") is not None]
    avg_ndcg = (
        sum(r["ndcg_at_10"] for r in with_ndcg) / len(with_ndcg) if with_ndcg else 0.0
    )

    per_dataset: dict[str, int] = {}
    for r in records:
        per_dataset[r["dataset"]] = per_dataset.get(r["dataset"], 0) + 1

    print(f"  Syntax valid (or unverified):   {syntax_ok}/{len(records)}")
    print(f"  Retrieval non-empty:            {retrieval_ok}/{len(records)}")
    print(f"  Mean NDCG@10 (indexed subsets): {avg_ndcg:.4f}")
    print(f"  Per-dataset counts: {per_dataset}")

    ds = Dataset.from_list(records)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(DATASET_PATH))
    print(f"  Saved to {DATASET_PATH}")

    if push_to_hub:
        print(f"  Pushing to HuggingFace Hub: {push_to_hub}")
        ds.push_to_hub(push_to_hub)
        print("  Done.")


# ── Pre-load qrels for indexed datasets ───────────────────────────────────────


def preload_qrels(configs: list[dict]) -> dict:
    """Load only relevance judgments for all datasets — skips loading the corpus."""
    from searchlm.data.loaders.factory import create_loader

    qrels: dict[str, dict] = {}
    for cfg in configs:
        try:
            print(f"  Pre-loading qrels for {cfg['id']} ({cfg['qrels_split']})...")
            loader = create_loader(cfg["id"])
            qrels[cfg["id"]] = loader.load_qrels(split=cfg["qrels_split"])
            print(f"  → {len(qrels[cfg['id']])} queries with relevance judgments")
        except Exception as e:
            print(f"  WARNING: could not load qrels for {cfg['id']}: {e}")
    return qrels


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NL2BM25 SFT dataset")
    parser.add_argument(
        "--max-per-dataset",
        type=int,
        default=None,
        help="Cap queries per dataset (for testing)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Skip API calls; use placeholder outputs"
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="HuggingFace repo to push the final dataset to",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Max concurrent in-flight API requests (default: 8)",
    )
    parser.add_argument(
        "--index-path", type=str, default="./search_index", help="Path to Tantivy index"
    )
    args = parser.parse_args()

    # ── Load checkpoint ──────────────────────────────────────────────────────
    done_ids, completed = load_checkpoint()
    if done_ids:
        print(f"Resuming: {len(done_ids)} queries already completed.")

    # ── Load queries ─────────────────────────────────────────────────────────
    all_queries = load_all_queries(
        DATASET_CONFIGS, max_per_dataset=args.max_per_dataset
    )
    pending = [q for q in all_queries if q["id"] not in done_ids]
    print(f"Pending: {len(pending)} queries  |  Done: {len(done_ids)}")

    if not pending:
        print("Nothing to do. Assembling existing checkpoint into dataset...")
        assemble_and_save(completed, args.push_to_hub)
        return

    # ── Setup search engine (optional) ───────────────────────────────────────
    engine: Optional[SearchEngine] = None
    preloaded_qrels: dict = {}

    try:
        engine = SearchEngine(index_path=args.index_path)
        print("Search index loaded — full retrieval validation enabled.")
        print("Pre-loading qrels (queries only, no corpus)...")
        preloaded_qrels = preload_qrels(DATASET_CONFIGS)
    except ValueError:
        print(
            f"No index found at {args.index_path} — syntax-only validation will be skipped."
        )

    # ── API client & per-model rate limiters ─────────────────────────────────
    client = AsyncOpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY,
    )
    # One RateLimiter per model; queries cycle round-robin across models
    model_limiters: list[tuple[str, RateLimiter]] = [
        (name, RateLimiter(rpm=rpm)) for name, rpm in MODELS
    ]
    total_rpm = sum(rpm for _, rpm in MODELS)
    semaphore = asyncio.Semaphore(args.concurrency)

    # ── Generate ──────────────────────────────────────────────────────────────
    eta_minutes = len(pending) / total_rpm
    model_names = ", ".join(m for m, _ in MODELS)
    print(
        f"\nGenerating {len(pending)} queries at ≤{total_rpm} RPM combined (~{eta_minutes:.0f} min ETA)"
    )
    print(f"Models ({len(MODELS)}): {model_names}")
    print(f"Concurrency: {args.concurrency}  |  Dry-run: {args.dry_run}\n")

    tasks = [
        process_query(
            q,
            client,
            model_limiters[i % len(model_limiters)][0],  # round-robin model name
            model_limiters[i % len(model_limiters)][1],  # its limiter
            semaphore,
            engine,
            preloaded_qrels,
            args.dry_run,
        )
        for i, q in enumerate(pending)
    ]

    new_records: list[dict] = []
    for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating"):
        result = await coro
        if result is not None:
            new_records.append(result)

    all_records = completed + new_records
    print(f"\nGenerated {len(new_records)} new records. Total: {len(all_records)}")

    # ── Assemble final dataset ────────────────────────────────────────────────
    assemble_and_save(all_records, args.push_to_hub)


if __name__ == "__main__":
    asyncio.run(main())
