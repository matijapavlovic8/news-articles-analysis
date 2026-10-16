import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

SENTIMENTS = {0: "neutral", 1: "negative", 2: "positive"}
CLASSES = ["neutral", "negative", "positive"]


def gold_label(g):
    if isinstance(g, dict):
        return g.get("sentiment_label") or SENTIMENTS.get(g.get("sentiment"))
    return SENTIMENTS.get(g)


def prf(conf):
    """Per-class precision/recall/F1 + macro-F1 from a gold x pred counter."""
    per_class = {}
    for c in CLASSES:
        tp = conf[c][c]
        fp = sum(conf[g][c] for g in CLASSES) - tp
        fn = sum(conf[c][p] for p in CLASSES) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = (prec, rec, f1)
    macro = sum(v[2] for v in per_class.values()) / len(CLASSES)
    return per_class, macro


def main():
    rows = {}
    path = sys.argv[1] if len(sys.argv) > 1 else "results/tsa.jsonl"
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            k = (r["id"], r["model"])
            if "error" in r:
                rows.setdefault(k, r)
            else:
                rows[k] = r
    rows = list(rows.values())

    per_model = defaultdict(lambda: {
        "conf": defaultdict(Counter),   # conf[gold][pred]
        "n_entities": 0,
        "missing": 0,
        "errors": 0,
        "flat": [],
    })

    for r in rows:
        m = per_model[r["model"]]
        if r.get("error"):
            m["errors"] += 1
            continue

        gold = json.loads(r["gold"]) if isinstance(r["gold"], str) else r["gold"]
        pred = {e["i"]: e["sentiment"] for e in r["tsa"]["entities"]}

        for i, g in enumerate(gold):
            gl = gold_label(g)
            pl = pred.get(i)
            m["n_entities"] += 1
            if pl is None:
                m["missing"] += 1
                pl = "<missing>"
            m["conf"][gl][pl] += 1
            m["flat"].append({
                "id": r["id"],
                "model": r["model"],
                "entity": g.get("entity") if isinstance(g, dict) else None,
                "gold": gl,
                "pred": pl,
            })

    for model, m in per_model.items():
        per_class, macro = prf(m["conf"])
        correct = sum(m["conf"][c][c] for c in CLASSES)
        acc = correct / m["n_entities"] if m["n_entities"] else 0.0

        print(f"\n=== {model} ===")
        print(f"entities={m['n_entities']}  accuracy={acc:.3f}  macro-F1={macro:.3f}  "
              f"missing={m['missing']}  api_errors={m['errors']}")
        print(f"{'class':10s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'support':>8s}")
        for c in CLASSES:
            p, r_, f1 = per_class[c]
            support = sum(m["conf"][c].values())
            print(f"{c:10s} {p:6.3f} {r_:6.3f} {f1:6.3f} {support:8d}")
        print("confusion (gold x pred):", dict(m["conf"]))

    df = pd.DataFrame([e for m in per_model.values() for e in m["flat"]])
    out = f"results/{Path(path).stem}_entities.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
