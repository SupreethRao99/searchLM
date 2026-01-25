import re

import pandas as pd
from datasets import load_dataset

from searchlm import SearchEvaluator

df = pd.read_csv("./data/generated_queries.tsv", sep="\t")
print("Total queries: ", len(df))

df["cleaned_query"] = df["query"].str.extract(
    r"<query>\s*(.*?)\s*</query>", flags=re.DOTALL
)
df = df.dropna(subset=["cleaned_query"])
df["cleaned_query"] = df["cleaned_query"].str.strip()
df = df.reset_index(drop=True)
print("Total queries after cleaning: ", len(df))

dataset = load_dataset("mteb/nfcorpus", name="default", split="test")
df = df[df["id"].isin(dataset["query-id"])]


evaluator = SearchEvaluator()
results = evaluator.evaluate_batch(
    queries=zip(df["cleaned_query"], df["id"]),
    k=100,
    dataset_name="nfcorpus",
    split="test",
    dataset_filter=None,
)
print(results)
