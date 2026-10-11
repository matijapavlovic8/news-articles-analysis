"""
Dump a stratified benchmark set of news articles from the retriever PostgreSQL DB.

  article (partitioned LIST by portal)
    id, portal, title, body, date_published, hidden, features JSONB, ...
    features->'ner'->'data'        per-sentence
    features->'tsa'->'data'        flat list of ints (0=neutral 1=negative 2=positive),
                                  one per HEADLINE entity, aligned with ner.data[0]
    features->'topic_clf'->'data'  list of IPTC-ish topic labels (multi-label)
    features->'tokens'->'data'     per-sentence token lists; index 0 = headline
    features_tsa int[]             flattened copy of tsa.data
    features_topics text[]         flattened copy of topic_clf.data

Stratification
--------------
Eligible articles: hidden = false, date_published within --years years,
non-empty ner/topic_clf/tsa feature keys.

Sentiment bucket (from features_tsa):
    negative  - any headline entity marked 1
    positive  - at least one 2 and no 1
    neutral   - non-empty, all 0
    none      - no headline entities (capped at --max-none-frac of the set)

Primary topic = first label in topic_clf.data (most confident prediction).

Sampling: quotas per (topic, sentiment) stratum; within a stratum we round-robin
over (portal, quarter-of-year) cells so portals and publication dates spread out.
A global per-portal cap (default ~2x equal share) keeps index.hr & friends from
dominating. A top-up pass fills the set if some strata are undersized.

Output TSV columns:
    id, portal, date_published, topics, title, body, tokens, ner, tsa

Usage:
    PG_PASSWORD='...' uv run dump-benchmark --out benchmark.tsv --stats
    PG_PASSWORD='...' uv run dump-benchmark --out benchmark.tsv
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime

import psycopg

SENTIMENT_LABELS = {0: "neutral", 1: "negative", 2: "positive"}

# Only lightweight metadata is pulled for stratification; full rows are fetched
# for the final selection only.
CANDIDATE_SQL = """
SELECT id,
       portal,
       date_published,
       features_topics AS topics,
       features_tsa AS tsa,
       features -> 'ner' -> 'version' AS ner_version,
       features -> 'tsa' -> 'version' AS tsa_version,
       features -> 'topic_clf' -> 'version' AS topic_version
FROM article TABLESAMPLE SYSTEM ({sample_pct}) REPEATABLE ({seed})
WHERE hidden = false
  AND date_published IS NOT NULL
  AND date_published >= now() - make_interval(years => %s)
  AND title IS NOT NULL
  AND body IS NOT NULL
  AND features ? 'ner'
  AND features ? 'tsa'
  AND features ? 'topic_clf'
  AND features -> 'ner' -> 'data' <> '[]'::jsonb
LIMIT %s
"""

FULL_ROW_SQL = """
SELECT a.id,
       a.portal,
       a.date_published::date AS date_published,
       a.title,
       a.body,
       a.features -> 'tokens' -> 'data' AS tokens,
       a.features -> 'ner' -> 'data' AS ner,
       a.features -> 'tsa' -> 'data' AS tsa,
       a.features -> 'topic_clf' -> 'data' AS topics
FROM article a
WHERE a.id = ANY(%s)
ORDER BY a.id
"""

TSV_HEADER = [
    "id",
    "portal",
    "date_published",
    "topics",
    "title",
    "body",
    "tokens",
    "ner",
    "tsa",
]


def sentiment_bucket(tsa):
    """Bucket an article by its headline-entity sentiments (features_tsa)."""
    if not tsa:
        return "none"
    if 1 in tsa:
        return "negative"
    if 2 in tsa:
        return "positive"
    return "neutral"


def quarter_of(dt):
    return f"{dt.year}Q{(dt.month - 1) // 3 + 1}"


def clean_cell(text):
    """Make text safe for a single TSV cell."""
    if text is None:
        return ""
    return " ".join(str(text).split())


def resolve_ner(ner_data, tokens):
    """Flatten per-sentence NER spans into a list of resolved entity dicts.

    Spans are [start, end, label] token indices into tokens.data[sent].
    Entities whose span falls outside the (possibly regenerated) token list
    are skipped.
    """
    out = []
    for sent_idx, sent_entities in enumerate(ner_data or []):
        sent_tokens = (tokens or [None])[sent_idx] if sent_idx < len(tokens or []) else []
        for ent in sent_entities or []:
            try:
                start, end, label = ent
            except (TypeError, ValueError):
                continue
            if sent_tokens and 0 <= start < end <= len(sent_tokens):
                text = " ".join(sent_tokens[start:end])
            else:
                text = None
            out.append(
                {
                    "sent": sent_idx,
                    "start": start,
                    "end": end,
                    "label": label,
                    "text": text,
                }
            )
    return out


def resolve_tsa(tsa_data, ner_data, tokens):
    """Pair headline-entity sentiments with their NER spans.

    tsa.data[i] corresponds to ner.data[0][i] (see tsa processor docs).
    """
    out = []
    headline_entities = (ner_data or [[]])[0] if ner_data else []
    headline_tokens = (tokens or [[]])[0] if tokens else []
    for i, sent_val in enumerate(tsa_data or []):
        ent = headline_entities[i] if i < len(headline_entities) else None
        text = None
        label = None
        if ent:
            start, end, label = ent
            if 0 <= start < end <= len(headline_tokens):
                text = " ".join(headline_tokens[start:end])
        out.append(
            {
                "entity": text,
                "label": label,
                "sentiment": sent_val,
                "sentiment_label": SENTIMENT_LABELS.get(sent_val, "unknown"),
            }
        )
    return out


def fetch_candidates(conn, years, sample_pct, pool_limit, seed):
    """Stream candidate metadata rows into a list of dicts."""
    sql = CANDIDATE_SQL.format(sample_pct=sample_pct, seed=seed)
    rows = []
    with conn.cursor(name="bench_candidates") as cur:
        cur.itersize = 5000
        cur.execute(sql, (years, pool_limit))
        for rec in cur:
            rows.append(
                {
                    "id": rec[0],
                    "portal": rec[1],
                    "date_published": rec[2],
                    "topics": rec[3] or [],
                    "tsa": rec[4] or [],
                    "ner_version": rec[5],
                    "tsa_version": rec[6],
                    "topic_version": rec[7],
                }
            )
            if len(rows) % 25000 == 0:
                print(f"  ...{len(rows):,} candidates")
    return rows


def print_stats(pool, size):
    print(f"\nEligible pool: {len(pool)} articles")
    for name, counter in (
        ("portal", Counter(c["portal"] for c in pool)),
        ("topic", Counter(c["topic"] for c in pool)),
        ("sentiment", Counter(c["sentiment"] for c in pool)),
        ("quarter", Counter(c["quarter"] for c in pool)),
        ("ner_version", Counter(str(c["ner_version"]) for c in pool)),
        ("tsa_version", Counter(str(c["tsa_version"]) for c in pool)),
        ("topic_version", Counter(str(c["topic_version"]) for c in pool)),
    ):
        print(f"\n  {name}:")
        for k, v in counter.most_common():
            print(f"    {k:50s} {v:>9,}")
    n_portals = len({c["portal"] for c in pool})
    print(
        f"\nEqual-share references: {size} target -> "
        f"~{size // max(n_portals, 1)} per portal, "
        f"~{size // max(len({c['topic'] for c in pool}), 1)} per topic"
    )


def stratified_sample(pool, size, seed, portal_cap, max_none_frac):
    """Greedy stratified sampler.

    Strata = (primary_topic, sentiment_bucket); within each stratum,
    candidates are grouped in (portal, quarter) cells and picked round-robin.
    """
    rng = random.Random(seed)
    for c in pool:
        c["topic"] = c["topics"][0] if c["topics"] else "UNTOPICED"
        c["sentiment"] = sentiment_bucket(c["tsa"])
        c["quarter"] = quarter_of(c["date_published"])

    strata = defaultdict(list)
    for c in pool:
        strata[(c["topic"], c["sentiment"])].append(c)

    n_strata = len(strata)
    base_quota = size // n_strata
    remainder = size % n_strata
    none_strata = [k for k in strata if k[1] == "none"]
    none_cap_total = int(size * max_none_frac)
    none_cap_each = none_cap_total // max(len(none_strata), 1) if none_strata else 0

    portal_used = Counter()
    none_used = 0
    selected = []
    selected_ids = set()

    def cell_order(cands):
        cells = defaultdict(list)
        for c in cands:
            cells[(c["portal"], c["quarter"])].append(c)
        for cell in cells.values():
            rng.shuffle(cell)
        return sorted(cells.values(), key=lambda x: -len(x))

    def take(cands, quota):
        nonlocal none_used
        took = 0
        cells = cell_order(cands)
        while took < quota:
            progressed = False
            for cell in cells:
                if took >= quota:
                    break
                while cell:
                    c = cell.pop()
                    if c["id"] in selected_ids:
                        continue
                    if portal_used[c["portal"]] >= portal_cap:
                        continue
                    if c["sentiment"] == "none" and none_used >= none_cap_total:
                        continue
                    selected.append(c)
                    selected_ids.add(c["id"])
                    portal_used[c["portal"]] += 1
                    none_used += c["sentiment"] == "none"
                    took += 1
                    progressed = True
                    break
            if not progressed:
                break
        return took

    # Pass 1: per-stratum quotas.
    for i, (key, cands) in enumerate(sorted(strata.items(), key=lambda kv: kv[0])):
        quota = base_quota + (1 if i < remainder else 0)
        if key[1] == "none":
            quota = min(quota, none_cap_each)
        take(cands, quota)

    # Pass 2: top-up to target size from everything left.
    leftover = [c for c in pool if c["id"] not in selected_ids]
    take(leftover, size - len(selected))

    return selected, portal_used


def dump_tsv(conn, selected, out_path):
    """Fetch full rows for selected ids and write the TSV."""
    ids = [c["id"] for c in selected]
    rows = {}
    chunk = 500
    for i in range(0, len(ids), chunk):
        with conn.cursor() as cur:
            cur.execute(FULL_ROW_SQL, (ids[i : i + chunk],))
            for rec in cur:
                rows[rec[0]] = rec

    missing = set(ids) - set(rows)
    if missing:
        print(f"WARNING: {len(missing)} selected ids not found on fetch: {sorted(missing)[:10]}...")

    written = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(TSV_HEADER)
        for cid in sorted(rows):
            _, portal, date_published, title, body, tokens, ner, tsa, topics = rows[cid]
            ner_flat = resolve_ner(ner, tokens)
            tsa_flat = resolve_tsa(tsa, ner, tokens)
            w.writerow(
                [
                    cid,
                    portal,
                    date_published,
                    json.dumps(topics or [], ensure_ascii=False),
                    clean_cell(title),
                    clean_cell(body),
                    json.dumps(tokens or [], ensure_ascii=False),
                    json.dumps(ner_flat, ensure_ascii=False),
                    json.dumps(tsa_flat, ensure_ascii=False),
                ]
            )
            written += 1
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=os.environ.get("PG_HOST", "suncica.takelab.fer.hr"))
    p.add_argument("--port", type=int, default=int(os.environ.get("PG_PORT", "32742")))
    p.add_argument("--dbname", default=os.environ.get("PG_DBNAME", "retriever"))
    p.add_argument("--user", default=os.environ.get("PG_USER", "readonly"))
    p.add_argument("--password", default=os.environ.get("PG_PASSWORD"))
    p.add_argument("--years", type=int, default=2, help="max article age in years (default 2)")
    p.add_argument("--size", type=int, default=1000, help="target number of articles")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    p.add_argument("--out", default="benchmark.tsv", help="output TSV path")
    p.add_argument(
        "--portal-cap",
        type=int,
        default=None,
        help="max articles per portal (default: ~2x equal share)",
    )
    p.add_argument(
        "--max-none-frac",
        type=float,
        default=0.15,
        help="max fraction of articles with no headline entities (default 0.15)",
    )
    p.add_argument(
        "--exclude-portal",
        action="append",
        default=[],
        help="portal to exclude (repeatable), e.g. --exclude-portal face.ba",
    )
    p.add_argument(
        "--sample-pct",
        type=float,
        default=5.0,
        help="TABLESAMPLE SYSTEM percentage for the candidate pool (default 5 gives ~60k candidates)",
    )
    p.add_argument(
        "--pool-limit",
        type=int,
        default=150000,
        help="hard cap on candidate pool rows (default 150000)",
    )
    p.add_argument("--ner-version", default=None, help="require this ner feature version")
    p.add_argument("--tsa-version", default=None, help="require this tsa feature version")
    p.add_argument("--topic-version", default=None, help="require this topic_clf feature version")
    p.add_argument("--stats", action="store_true", help="print pool stats and exit (no dump)")
    args = p.parse_args()

    if not args.password:
        sys.exit("Set PG_PASSWORD (or --password).")

    conninfo = (
        f"host={args.host} port={args.port} dbname={args.dbname} "
        f"user={args.user} password={args.password} connect_timeout=10"
    )
    print(f"Connecting to {args.host}:{args.port} db={args.dbname} user={args.user} ...")
    with psycopg.connect(conninfo) as conn:
        print(
            f"Fetching candidate metadata "
            f"(TABLESAMPLE {args.sample_pct}%, limit {args.pool_limit:,}) ..."
        )
        pool = fetch_candidates(
            conn, args.years, args.sample_pct, args.pool_limit, args.seed
        )

        # Version filters (versions are compared as strings: ner=1 int, tsa='v1' str).
        if args.ner_version is not None:
            pool = [c for c in pool if str(c["ner_version"]) == args.ner_version]
        if args.tsa_version is not None:
            pool = [c for c in pool if str(c["tsa_version"]) == args.tsa_version]
        if args.topic_version is not None:
            pool = [c for c in pool if str(c["topic_version"]) == args.topic_version]
        if args.exclude_portal:
            pool = [c for c in pool if c["portal"] not in set(args.exclude_portal)]

        for c in pool:
            c["topic"] = c["topics"][0] if c["topics"] else "UNTOPICED"
            c["sentiment"] = sentiment_bucket(c["tsa"])
            c["quarter"] = quarter_of(c["date_published"])

        print_stats(pool, args.size)
        if args.stats:
            return

        n_portals = len({c["portal"] for c in pool})
        portal_cap = args.portal_cap or max(10, math.ceil(args.size * 2 / max(n_portals, 1)))
        print(f"\nSampling {args.size} (portal cap {portal_cap}) ...")
        selected, portal_used = stratified_sample(
            pool, args.size, args.seed, portal_cap, args.max_none_frac
        )
        print(f"Selected {len(selected)} articles.")

        print(f"Fetching full rows and writing {args.out} ...")
        written = dump_tsv(conn, selected, args.out)
        print(f"Wrote {written} rows to {args.out}")

    print("\nFinal distribution:")
    for name, counter in (
        ("portal", Counter(c["portal"] for c in selected)),
        ("topic", Counter(c["topic"] for c in selected)),
        ("sentiment", Counter(c["sentiment"] for c in selected)),
        ("quarter", Counter(c["quarter"] for c in selected)),
    ):
        line = ", ".join(f"{k}={v}" for k, v in counter.most_common())
        print(f"  {name}: {line}")


if __name__ == "__main__":
    main()
