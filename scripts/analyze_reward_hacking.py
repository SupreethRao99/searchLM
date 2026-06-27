#!/usr/bin/env python3
"""
Comprehensive reward hacking analysis: benchmark evaluation + behavioral fingerprinting.

Evaluates three model checkpoints (base / SFT / GRPO) on NFCorpus and SciFact test
splits, collecting per-query metrics and completion statistics to characterise what
the GRPO model actually learned.

Outputs:
  <outputs_dir>/reward_hacking/
    per_query_results.jsonl        -- one line per (model, dataset, query)
    aggregate_metrics.json         -- benchmark table
    behavioral_stats.json          -- completion/query statistics per model
    qualitative_examples.json      -- same 20 queries shown for all 3 models
    report.md                      -- human-readable analysis report
"""

import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from searchlm.config import get_config, get_data_path
from searchlm.data import create_loader
from searchlm.inference import VllmEngine
from searchlm.prompts import create_chat_prompt, extract_query_from_output
from searchlm.services.evaluator import SearchEvaluator

# ── Configuration ─────────────────────────────────────────────────────────────

DATASETS = ["nfcorpus", "scifact"]
SPLIT = "test"

# ── Helpers ───────────────────────────────────────────────────────────────────


def count_tokens_approx(text: str) -> int:
    """Rough whitespace-based token count (good enough for relative comparisons)."""
    return len(text.split())


def classify_query(query: str) -> dict:
    """Extract structural features from a boolean query string."""
    q = query.upper()
    has_and = " AND " in q
    has_or = " OR " in q
    has_not = " NOT " in q or q.startswith("NOT ")
    has_phrases = '"' in query
    has_field_scope = bool(re.search(r"\b(title|text):", query, re.IGNORECASE))
    has_boost = "^" in query

    # Count distinct AND-connected clauses (very rough)
    num_and_clauses = query.upper().count(" AND ") + 1 if has_and else 1
    num_or_terms = query.upper().count(" OR ")

    return {
        "has_and": has_and,
        "has_or": has_or,
        "has_not": has_not,
        "has_phrases": has_phrases,
        "has_field_scope": has_field_scope,
        "has_boost": has_boost,
        "num_and_clauses": num_and_clauses,
        "num_or_terms": num_or_terms,
        "complexity_score": int(has_and)
        + int(has_or)
        + int(has_not)
        + int(has_phrases)
        + int(has_field_scope),
    }


def extract_reasoning(text: str) -> str:
    """Extract content of <reasoning>...</reasoning> block."""
    match = re.search(r"<reasoning>\s*(.*?)\s*</reasoning>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def analyse_completion(completion: str) -> dict:
    """Return structural breakdown of a single model completion."""
    has_reasoning_tag = bool(re.search(r"<reasoning>", completion))
    has_query_tag = bool(re.search(r"<query>", completion))
    reasoning_text = extract_reasoning(completion)
    query_text = extract_query_from_output(completion)

    # If extract_query_from_output fell back to the whole completion it means
    # there were no <query> tags — treat as empty query.
    if not has_query_tag:
        query_text = ""

    return {
        "completion_tokens": count_tokens_approx(completion),
        "has_reasoning_tag": has_reasoning_tag,
        "has_query_tag": has_query_tag,
        "reasoning_tokens": count_tokens_approx(reasoning_text),
        "query_tokens": count_tokens_approx(query_text),
        "query_text": query_text,
        **classify_query(query_text),
    }


# ── Per-model evaluation ───────────────────────────────────────────────────────


def evaluate_model(
    model_label: str,
    model_path: str,
    evaluator: SearchEvaluator,
    all_queries: dict,  # dataset_name -> {qid: Query}
    all_qrels: dict,  # dataset_name -> {qid: {doc_id: float}}
    qualitative_qids: dict,  # dataset_name -> [qid, ...]
    config,
) -> tuple[list[dict], dict]:
    """
    Generate completions for every test query and evaluate.

    Returns:
        per_query_rows  -- list of result dicts (one per query)
        qualitative     -- {dataset: {qid: completion_text}}
    """
    print(f"\n{'=' * 70}")
    print(f"Evaluating: {model_label}  ({model_path})")
    print(f"{'=' * 70}")

    engine = VllmEngine(model_name=model_path)

    per_query_rows = []
    qualitative = defaultdict(dict)

    for dataset_name in DATASETS:
        queries = all_queries[dataset_name]
        qrels_all = all_qrels[dataset_name]

        qids = list(queries.keys())
        prompts = [create_chat_prompt(queries[qid].text) for qid in qids]

        print(f"\n  {dataset_name}: generating {len(prompts)} completions …")
        t0 = time.time()
        completions = engine.generate(
            prompts,
            temperature=config.evaluation.temperature,
            max_tokens=config.evaluation.max_tokens,
        )
        elapsed = time.time() - t0
        print(f"  done in {elapsed:.0f}s  ({elapsed / len(prompts):.2f}s/query)")

        # Evaluate each
        print(f"  evaluating {len(qids)} queries …")
        for qid, completion in zip(qids, completions):
            comp_info = analyse_completion(completion)
            query_text = comp_info["query_text"]
            qrels = qrels_all.get(qid, {})

            ndcg10 = mrr = reward = 0.0
            num_retrieved = 0
            eval_error: Optional[str] = None

            if query_text and qrels:
                metrics, error = evaluator.evaluate_query(
                    query_text=query_text,
                    qrels=qrels,
                    k=100,
                    dataset_filter=dataset_name,
                )
                if error:
                    eval_error = error
                else:
                    ndcg10 = metrics["ndcg@10"]
                    mrr = metrics["mrr"]
                    num_retrieved = metrics.get("retrieved", 0)
                    reward = (
                        config.reward.ndcg_weight * ndcg10
                        + config.reward.mrr_weight * mrr
                    )
            elif not qrels:
                eval_error = "no_qrels"
            elif not query_text:
                eval_error = "empty_query"

            row = {
                "model": model_label,
                "dataset": dataset_name,
                "query_id": qid,
                "nl_query": queries[qid].text,
                "completion": completion,
                "ndcg_at_10": ndcg10,
                "mrr": mrr,
                "reward": reward,
                "num_retrieved": num_retrieved,
                "eval_error": eval_error,
                **comp_info,
            }
            per_query_rows.append(row)

            # Stash for qualitative comparison
            if qid in qualitative_qids.get(dataset_name, []):
                qualitative[dataset_name][qid] = {
                    "nl_query": queries[qid].text,
                    "completion": completion,
                    "query_text": query_text,
                    "ndcg_at_10": ndcg10,
                    "mrr": mrr,
                    "reward": reward,
                }

    # Free GPU memory before next model
    del engine
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()

    return per_query_rows, dict(qualitative)


# ── Aggregate statistics ───────────────────────────────────────────────────────


def compute_aggregate(rows: list[dict]) -> dict:
    """Compute mean IR metrics and behavioral stats for a set of per-query rows."""
    if not rows:
        return {}

    def safe_mean(vals):
        return mean(vals) if vals else 0.0

    def safe_std(vals):
        return stdev(vals) if len(vals) > 1 else 0.0

    # IR metrics (skip rows with eval errors)
    valid = [r for r in rows if r["eval_error"] is None]
    errored = [r for r in rows if r["eval_error"] is not None]
    zero_reward = [r for r in valid if r["reward"] == 0.0]

    ndcgs = [r["ndcg_at_10"] for r in valid]
    mrrs = [r["mrr"] for r in valid]
    rewards = [r["reward"] for r in valid]
    comp_tokens = [r["completion_tokens"] for r in rows]
    query_tokens = [r["query_tokens"] for r in rows]
    reason_tokens = [r["reasoning_tokens"] for r in rows]
    retrieved = [r["num_retrieved"] for r in valid]

    return {
        # IR metrics
        "ndcg_at_10": {
            "mean": safe_mean(ndcgs),
            "std": safe_std(ndcgs),
            "median": median(ndcgs) if ndcgs else 0.0,
        },
        "mrr": {
            "mean": safe_mean(mrrs),
            "std": safe_std(mrrs),
            "median": median(mrrs) if mrrs else 0.0,
        },
        "reward": {
            "mean": safe_mean(rewards),
            "std": safe_std(rewards),
            "median": median(rewards) if rewards else 0.0,
        },
        # Retrieval behavior
        "num_retrieved": {
            "mean": safe_mean(retrieved),
            "median": median(retrieved) if retrieved else 0.0,
        },
        "frac_zero_reward": len(zero_reward) / len(valid) if valid else 1.0,
        # Format adherence
        "frac_has_reasoning_tag": safe_mean(
            [int(r["has_reasoning_tag"]) for r in rows]
        ),
        "frac_has_query_tag": safe_mean([int(r["has_query_tag"]) for r in rows]),
        # Length distributions
        "completion_tokens": {
            "mean": safe_mean(comp_tokens),
            "std": safe_std(comp_tokens),
            "median": median(comp_tokens) if comp_tokens else 0.0,
            "p10": sorted(comp_tokens)[len(comp_tokens) // 10] if comp_tokens else 0,
            "p90": sorted(comp_tokens)[int(len(comp_tokens) * 0.9)]
            if comp_tokens
            else 0,
        },
        "reasoning_tokens": {
            "mean": safe_mean(reason_tokens),
            "median": median(reason_tokens) if reason_tokens else 0.0,
        },
        "query_tokens": {
            "mean": safe_mean(query_tokens),
            "median": median(query_tokens) if query_tokens else 0.0,
            "p10": sorted(query_tokens)[len(query_tokens) // 10] if query_tokens else 0,
            "p90": sorted(query_tokens)[int(len(query_tokens) * 0.9)]
            if query_tokens
            else 0,
        },
        # Query complexity
        "frac_has_and": safe_mean([int(r["has_and"]) for r in rows]),
        "frac_has_or": safe_mean([int(r["has_or"]) for r in rows]),
        "frac_has_not": safe_mean([int(r["has_not"]) for r in rows]),
        "frac_has_phrases": safe_mean([int(r["has_phrases"]) for r in rows]),
        "frac_has_field_scope": safe_mean([int(r["has_field_scope"]) for r in rows]),
        "mean_complexity_score": safe_mean([r["complexity_score"] for r in rows]),
        "mean_num_and_clauses": safe_mean([r["num_and_clauses"] for r in rows]),
        "mean_num_or_terms": safe_mean([r["num_or_terms"] for r in rows]),
        # Coverage
        "num_valid": len(valid),
        "num_errored": len(errored),
        "num_zero_reward": len(zero_reward),
    }


# ── Report generation ─────────────────────────────────────────────────────────


def fmt(val, fmt_str=".4f"):
    return f"{val:{fmt_str}}" if val is not None else "—"


def generate_report(aggregate: dict, qualitative: dict, output_dir: Path) -> str:
    lines = []
    a = lines.append

    a("# Reward Hacking Analysis Report")
    a(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    a("\n---\n")

    # ── Benchmark table ────────────────────────────────────────────────────────
    a("## 1. Benchmark Scores (test split)\n")
    a("### NDCG@10\n")
    a(f"{'Model':<18} {'NFCorpus':>12} {'SciFact':>12}")
    a("-" * 44)
    for model in ["base", "sft", "grpo"]:
        row = []
        for ds in DATASETS:
            key = (model, ds)
            v = aggregate.get(key, {}).get("ndcg_at_10", {}).get("mean")
            row.append(f"{v:.4f}" if v is not None else "—")
        a(f"{model:<18} {row[0]:>12} {row[1]:>12}")

    a("\n### MRR\n")
    a(f"{'Model':<18} {'NFCorpus':>12} {'SciFact':>12}")
    a("-" * 44)
    for model in ["base", "sft", "grpo"]:
        row = []
        for ds in DATASETS:
            key = (model, ds)
            v = aggregate.get(key, {}).get("mrr", {}).get("mean")
            row.append(f"{v:.4f}" if v is not None else "—")
        a(f"{model:<18} {row[0]:>12} {row[1]:>12}")

    a("\n### Mean Reward (0.6×NDCG@10 + 0.4×MRR)\n")
    a(f"{'Model':<18} {'NFCorpus':>12} {'SciFact':>12}")
    a("-" * 44)
    for model in ["base", "sft", "grpo"]:
        row = []
        for ds in DATASETS:
            key = (model, ds)
            v = aggregate.get(key, {}).get("reward", {}).get("mean")
            row.append(f"{v:.4f}" if v is not None else "—")
        a(f"{model:<18} {row[0]:>12} {row[1]:>12}")

    a("\n---\n")

    # ── Completion length ──────────────────────────────────────────────────────
    a("## 2. Completion Length Statistics\n")
    a(
        f"{'Model':<8} {'Dataset':<12} {'Mean tokens':>12} {'Median':>10} {'P10':>8} {'P90':>8}"
    )
    a("-" * 62)
    for model in ["base", "sft", "grpo"]:
        for ds in DATASETS:
            key = (model, ds)
            ct = aggregate.get(key, {}).get("completion_tokens", {})
            a(
                f"{model:<8} {ds:<12} {ct.get('mean', 0):>12.1f} {ct.get('median', 0):>10.1f} "
                f"{ct.get('p10', 0):>8} {ct.get('p90', 0):>8}"
            )

    a("\n---\n")

    # ── Query complexity ───────────────────────────────────────────────────────
    a("## 3. Query Complexity & Format Adherence\n")
    a(
        f"{'Model':<8} {'Dataset':<12} "
        f"{'%AND':>8} {'%OR':>8} {'%phrase':>9} {'%field':>8} "
        f"{'AND_clauses':>12} {'OR_terms':>10} {'complexity':>11}"
    )
    a("-" * 100)
    for model in ["base", "sft", "grpo"]:
        for ds in DATASETS:
            key = (model, ds)
            ag = aggregate.get(key, {})
            a(
                f"{model:<8} {ds:<12} "
                f"{ag.get('frac_has_and', 0) * 100:>8.1f} "
                f"{ag.get('frac_has_or', 0) * 100:>8.1f} "
                f"{ag.get('frac_has_phrases', 0) * 100:>9.1f} "
                f"{ag.get('frac_has_field_scope', 0) * 100:>8.1f} "
                f"{ag.get('mean_num_and_clauses', 0):>12.2f} "
                f"{ag.get('mean_num_or_terms', 0):>10.2f} "
                f"{ag.get('mean_complexity_score', 0):>11.2f}"
            )

    a("\n---\n")

    # ── Format adherence ───────────────────────────────────────────────────────
    a("## 4. Format Adherence\n")
    a(
        f"{'Model':<8} {'Dataset':<12} {'%has_reasoning':>16} {'%has_query':>12} {'%zero_reward':>14}"
    )
    a("-" * 65)
    for model in ["base", "sft", "grpo"]:
        for ds in DATASETS:
            key = (model, ds)
            ag = aggregate.get(key, {})
            a(
                f"{model:<8} {ds:<12} "
                f"{ag.get('frac_has_reasoning_tag', 0) * 100:>16.1f} "
                f"{ag.get('frac_has_query_tag', 0) * 100:>12.1f} "
                f"{ag.get('frac_zero_reward', 0) * 100:>14.1f}"
            )

    a("\n---\n")

    # ── Query token length ─────────────────────────────────────────────────────
    a("## 5. Query Token Length (the boolean query only)\n")
    a(f"{'Model':<8} {'Dataset':<12} {'Mean':>8} {'Median':>8} {'P10':>6} {'P90':>6}")
    a("-" * 56)
    for model in ["base", "sft", "grpo"]:
        for ds in DATASETS:
            key = (model, ds)
            qt = aggregate.get(key, {}).get("query_tokens", {})
            a(
                f"{model:<8} {ds:<12} {qt.get('mean', 0):>8.1f} {qt.get('median', 0):>8.1f} "
                f"{qt.get('p10', 0):>6} {qt.get('p90', 0):>6}"
            )

    a("\n---\n")

    # ── Qualitative examples ───────────────────────────────────────────────────
    # qualitative is {ds: [{"qid": ..., "base": {...}, "sft": {...}, "grpo": {...}}, ...]}
    a("## 6. Qualitative Examples (same queries, all 3 models)\n")
    for ds, entries in qualitative.items():
        a(f"### {ds.upper()}\n")
        for entry in entries[:10]:
            qid = entry.get("qid", "?")
            nl = entry.get("base", {}).get("nl_query", "?")
            a(f"**NL Query** (id={qid}): *{nl}*\n")
            for model in ["base", "sft", "grpo"]:
                info = entry.get(model, {})
                qt = info.get("query_text", "—")
                r = info.get("reward", 0.0)
                n = info.get("ndcg_at_10", 0.0)
                a(f"**{model.upper()}** (reward={r:.3f}, NDCG@10={n:.3f})")
                a("```")
                a(qt[:400] + ("…" if len(qt) > 400 else ""))
                a("```\n")
            a("---\n")

    a("\n## 7. Interpretation\n")
    a(
        """
*(Fill in after reviewing numbers above)*

Key questions to answer:
- Does GRPO improve over SFT, or degrade?
- Are GRPO queries shorter than SFT? How much?
- Does GRPO still use AND/OR structure, or collapse to single terms?
- What fraction of GRPO completions are missing the <reasoning> block?
- Is zero-reward fraction higher for GRPO than SFT?
- Does the SFT model generalise better (higher NDCG) despite lower training reward?
""".strip()
    )

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────


def run_analysis(version: str = "v1"):
    """
    Args:
        version: "v1" — compare base / sft / grpo (original run).
                 "v2" — compare base / sft_v2 / grpo_v2 (improved run).
                 "compare" — compare sft / grpo / sft_v2 / grpo_v2 side-by-side.
    """
    config = get_config()
    models_dir = get_data_path("models")
    indices_dir = get_data_path("indices")
    outputs_dir = get_data_path("outputs") / f"reward_hacking_{version}"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Resolve model paths for this version
    if version == "v2":
        MODELS = [
            ("base", config.model.name),
            ("sft_v2", str(models_dir / "sft_v2" / "final")),
            ("grpo_v2", str(models_dir / "grpo_v2" / "final")),
        ]
    elif version == "compare":
        MODELS = [
            ("base", config.model.name),
            ("sft", str(models_dir / "sft" / "final")),
            ("grpo", str(models_dir / "final")),
            ("sft_v2", str(models_dir / "sft_v2" / "final")),
            ("grpo_v2", str(models_dir / "grpo_v2" / "final")),
        ]
    else:
        # v1 default
        MODELS = [
            ("base", config.model.name),
            ("sft", str(models_dir / "sft" / "final")),
            ("grpo", str(models_dir / "final")),
        ]

    # Pre-load all test data (shared across models)
    print("Loading test queries and qrels …")
    all_queries = {}
    all_qrels = {}
    for ds in DATASETS:
        loader = create_loader(ds)
        split = loader.load_split(split=SPLIT)
        all_queries[ds] = split.queries
        all_qrels[ds] = split.qrels
        print(f"  {ds}: {len(split.queries)} queries, {len(split.qrels)} qrel entries")

    # Pick 20 qualitative example queries per dataset (fixed across models)
    qualitative_qids: dict[str, list[str]] = {}
    for ds in DATASETS:
        qids_with_qrels = [q for q in all_queries[ds] if q in all_qrels[ds]]
        # Take first 20 that have qrels
        qualitative_qids[ds] = qids_with_qrels[:20]

    evaluator = SearchEvaluator(index_path=str(indices_dir))

    # ── Evaluate each model ───────────────────────────────────────────────────
    all_rows: list[dict] = []
    # {dataset: {qid: {model: {nl_query, completion, query_text, ...}}}}
    qualitative_by_ds: dict[str, dict[str, dict]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for model_label, model_path in MODELS:
        rows, qual = evaluate_model(
            model_label,
            model_path,
            evaluator,
            all_queries,
            all_qrels,
            qualitative_qids,
            config,
        )
        all_rows.extend(rows)

        for ds, by_qid in qual.items():
            for qid, info in by_qid.items():
                qualitative_by_ds[ds][qid][model_label] = info

    # ── Compute aggregate metrics ─────────────────────────────────────────────
    aggregate: dict[tuple, dict] = {}
    for model_label, _ in MODELS:
        for ds in DATASETS:
            rows_subset = [
                r for r in all_rows if r["model"] == model_label and r["dataset"] == ds
            ]
            aggregate[(model_label, ds)] = compute_aggregate(rows_subset)

    # ── Serialise aggregate with string keys ──────────────────────────────────
    aggregate_serialisable = {f"{m}|{d}": v for (m, d), v in aggregate.items()}

    # ── Save outputs ──────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    jsonl_path = outputs_dir / f"per_query_results_{ts}.jsonl"
    with open(jsonl_path, "w") as f:
        for row in all_rows:
            # Don't dump full completion in JSONL — save separately
            compact = {k: v for k, v in row.items() if k != "completion"}
            f.write(json.dumps(compact) + "\n")
    print(f"Saved per-query results → {jsonl_path}")

    agg_path = outputs_dir / f"aggregate_metrics_{ts}.json"
    with open(agg_path, "w") as f:
        json.dump(aggregate_serialisable, f, indent=2)
    print(f"Saved aggregate metrics → {agg_path}")

    qual_path = outputs_dir / f"qualitative_examples_{ts}.json"
    # Reformat for JSON: {ds: [{qid, nl_query, base: {...}, sft: {...}, grpo: {...}}]}
    qual_out = {}
    for ds, by_qid in qualitative_by_ds.items():
        entries = []
        for qid, by_model in by_qid.items():
            entries.append({"qid": qid, **by_model})
        qual_out[ds] = entries
    with open(qual_path, "w") as f:
        json.dump(qual_out, f, indent=2)
    print(f"Saved qualitative examples → {qual_path}")

    report_text = generate_report(aggregate, qual_out, outputs_dir)
    report_path = outputs_dir / f"report_{ts}.md"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"Saved report → {report_path}")

    # Print quick summary to stdout
    print("\n" + "=" * 70)
    print("QUICK SUMMARY — NDCG@10")
    print("=" * 70)
    for model_label, _ in MODELS:
        for ds in DATASETS:
            v = (
                aggregate.get((model_label, ds), {})
                .get("ndcg_at_10", {})
                .get("mean", 0)
            )
            print(f"  {model_label:<6} {ds:<12}  {v:.4f}")

    print("\nCOMPLETION TOKENS (mean)")
    for model_label, _ in MODELS:
        for ds in DATASETS:
            v = (
                aggregate.get((model_label, ds), {})
                .get("completion_tokens", {})
                .get("mean", 0)
            )
            print(f"  {model_label:<6} {ds:<12}  {v:.1f}")

    print("\nFRAC ZERO REWARD")
    for model_label, _ in MODELS:
        for ds in DATASETS:
            v = aggregate.get((model_label, ds), {}).get("frac_zero_reward", 0) * 100
            print(f"  {model_label:<6} {ds:<12}  {v:.1f}%")

    print(f"\nAll outputs saved to: {outputs_dir}")


if __name__ == "__main__":
    ver = sys.argv[1] if len(sys.argv) > 1 else "v1"
    run_analysis(version=ver)
