import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def parse_labels(x):
    """Gold/pred labels -> list of label strings."""
    if isinstance(x, str):
        x = json.loads(x) if x else []
    if isinstance(x, dict):  # model_dump'd {"topics": [...]}
        x = x.get("topics", [])
    return x or []


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def main():
    rows = []
    path = sys.argv[1] if len(sys.argv) > 1 else "results/topic.jsonl"
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))

    per_model = defaultdict(lambda: {
        "n": 0,
        "errors": 0,
        "primary_correct": 0,
        "exact": 0,
        "tp": 0, "fp": 0, "fn": 0,          # micro counts
        "label_tp": Counter(), "label_fp": Counter(), "label_fn": Counter(),
        "jaccard": 0.0,
        "flat": [],
    })

    for r in rows:
        m = per_model[r["model"]]
        if r.get("error"):
            m["errors"] += 1
            continue

        gold = parse_labels(r["gold"])
        pred = r["topics"] if isinstance(r["topics"], list) else parse_labels(r["topics"])
        if not gold:
            continue

        m["n"] += 1
        gold_set, pred_set = set(gold), set(pred)

        if pred and gold and pred[0] == gold[0]:
            m["primary_correct"] += 1
        if pred_set == gold_set:
            m["exact"] += 1

        tp = len(gold_set & pred_set)
        m["tp"] += tp
        m["fp"] += len(pred_set - gold_set)
        m["fn"] += len(gold_set - pred_set)
        m["jaccard"] += tp / len(gold_set | pred_set) if gold_set | pred_set else 0.0

        for lab in gold_set & pred_set:
            m["label_tp"][lab] += 1
        for lab in pred_set - gold_set:
            m["label_fp"][lab] += 1
        for lab in gold_set - pred_set:
            m["label_fn"][lab] += 1

        m["flat"].append({
            "id": r["id"], "model": r["model"],
            "gold": "|".join(gold), "pred": "|".join(pred),
            "primary_gold": gold[0], "primary_pred": pred[0] if pred else None,
        })

    for model, m in per_model.items():
        n = max(m["n"], 1)
        mic_p, mic_r, mic_f1 = prf(m["tp"], m["fp"], m["fn"])
        f1s = [prf(m["label_tp"][l], m["label_fp"][l], m["label_fn"][l])[2]
               for l in set(m["label_tp"]) | set(m["label_fn"]) | set(m["label_fp"])]
        macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

        print(f"\n=== {model} ===")
        print(f"articles={m['n']}  api_errors={m['errors']}")
        print(f"primary-label accuracy: {m['primary_correct'] / n:.3f}")
        print(f"exact-match (set):      {m['exact'] / n:.3f}")
        print(f"mean Jaccard:           {m['jaccard'] / n:.3f}")
        print(f"micro P/R/F1:           {mic_p:.3f} {mic_r:.3f} {mic_f1:.3f}")
        print(f"macro-F1 (per label):   {macro_f1:.3f}")

        print(f"\n  {'label':45s} {'f1':>6s} {'sup':>5s}")
        labels = sorted(
            set(m["label_tp"]) | set(m["label_fn"]),
            key=lambda l: -(m["label_tp"][l] + m["label_fn"][l]),
        )
        for lab in labels:
            _, _, f1 = prf(m["label_tp"][lab], m["label_fp"][lab], m["label_fn"][lab])
            sup = m["label_tp"][lab] + m["label_fn"][lab]
            print(f"  {lab:45s} {f1:6.3f} {sup:5d}")

    df = pd.DataFrame([e for m in per_model.values() for e in m["flat"]])
    out = f"results/{Path(path).stem}_articles.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
