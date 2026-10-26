import re
import pandas as pd


def parse_pages(value):
    """'40; 41' -> {40, 41}"""
    return {int(x) for x in re.findall(r"\d+", str(value))}


def evaluate_retrieval(retriever, eval_df, k=5, mode="hybrid"):
    eval_df = eval_df.dropna(subset=["question", "expected_pages"])
    if eval_df.empty:
        raise ValueError("No labelled questions - fill expected_pages and re-run the CSV cell")
    rows = []
    for _, row in eval_df.iterrows():
        wanted = parse_pages(row["expected_pages"])
        results = retriever.search(row["question"], k=k, mode=mode)
        rank = next((r for r, p in enumerate(results, start=1) if p["page"] in wanted), None)
        rows.append({"question": row["question"], "hit": rank is not None, "rank": rank})
    df = pd.DataFrame(rows)
    df["rank"] = pd.to_numeric(df["rank"])
    summary = {
        "mode": mode, "k": k, "n": len(df),
        "hit_rate": round(float(df["hit"].mean()), 3),
        "mrr": round(float((1 / df["rank"]).fillna(0).mean()), 3),
    }
    return df, summary
