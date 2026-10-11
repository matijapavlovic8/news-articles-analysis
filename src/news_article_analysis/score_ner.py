import json
from collections import Counter, defaultdict

import pandas as pd

LABELS = ["PER", "LOC", "ORG"]


def span_key(e):
    return (e["sent"], e["start"], e["end"], e["label"])


def overlap(a, b):
    """Token-range overlap between two spans in the same sentence."""
    return a["sent"] == b["sent"] and a["start"] < b["end"] and b["start"] < a["end"]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def main():
    rows = []
    with open("results/ner.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))

    per_model = defaultdict(lambda: {
        "tp": Counter(), "fp": Counter(), "fn": Counter(),
        "boundary_err": 0,      # pred overlaps an unmatched gold span, same label
        "boundary_label": 0,    # pred overlaps an unmatched gold span, wrong label
        "errors": 0,
        "flat": [],
    })

    for r in rows:
        m = per_model[r["model"]]
        if r.get("error"):
            m["errors"] += 1
            continue

        gold = json.loads(r["gold"]) if isinstance(r["gold"], str) else (r["gold"] or [])
        pred = r["ner"] or []

        gc = Counter(span_key(e) for e in gold)
        pc = Counter(span_key(e) for e in pred)

        tp = gc & pc          # multiset intersection
        fp = pc - gc
        fn = gc - tp

        for (s, a, b, lab), n in tp.items():
            m["tp"][lab] += n
            m["flat"].append({"id": r["id"], "model": r["model"], "sent": s,
                              "start": a, "end": b, "label": lab, "status": "tp"})
        for k, n in fn.items():
            m["fn"][k[3]] += n
        for k, n in fp.items():
            m["fp"][k[3]] += n

        # Walk remaining allowance per key so duplicate spans aren't overcounted.
        fn_left = dict(fn)
        unmatched_gold = []
        for e in gold:
            k = span_key(e)
            if fn_left.get(k, 0):
                fn_left[k] -= 1
                unmatched_gold.append(e)
                m["flat"].append({"id": r["id"], "model": r["model"], "sent": e["sent"],
                                  "start": e["start"], "end": e["end"], "label": e["label"],
                                  "status": "fn", "text": e.get("text")})
        fp_left = dict(fp)
        for e in pred:
            k = span_key(e)
            if not fp_left.get(k, 0):
                continue
            fp_left[k] -= 1
            m["flat"].append({"id": r["id"], "model": r["model"], "sent": e["sent"],
                              "start": e["start"], "end": e["end"], "label": e["label"],
                              "status": "fp"})
            for g in unmatched_gold:
                if overlap(e, g):
                    if g["label"] == e["label"]:
                        m["boundary_err"] += 1
                    else:
                        m["boundary_label"] += 1
                    break

    for model, m in per_model.items():
        tp, fp, fn = sum(m["tp"].values()), sum(m["fp"].values()), sum(m["fn"].values())
        p, r_, f1 = prf(tp, fp, fn)

        print(f"\n=== {model} ===")
        print(f"tp={tp}  fp={fp}  fn={fn}  api_errors={m['errors']}")
        print(f"exact-span P/R/F1: {p:.3f} {r_:.3f} {f1:.3f}")
        print(f"boundary errors (same label, offset span): {m['boundary_err']}")
        print(f"overlaps with wrong label:               {m['boundary_label']}")
        print(f"{'label':6s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'gold':>6s}")
        for lab in LABELS:
            p_, r__, f1_ = prf(m["tp"][lab], m["fp"][lab], m["fn"][lab])
            print(f"{lab:6s} {p_:6.3f} {r__:6.3f} {f1_:6.3f} {m['tp'][lab] + m['fn'][lab]:6d}")

    df = pd.DataFrame([e for m in per_model.values() for e in m["flat"]])
    df.to_csv("results/ner_entities.csv", index=False)
    print("\nwrote results/ner_entities.csv")


if __name__ == "__main__":
    main()
