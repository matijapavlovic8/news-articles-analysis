import asyncio
import json
import os
import tomllib

import pandas as pd
from openai import OpenAI
from tqdm.asyncio import tqdm

with open("secrets.toml", "rb") as f:
    secrets = tomllib.load(f)

llm = OpenAI(api_key=secrets["OPENAI_API_KEY"])

MODEL = "gpt-6-luna"  # the only decisions model
OUT = "results/topic_decisions.jsonl"

TOPICS = [
    "ARTS, CULTURE, ENTERTAINMENT AND MEDIA",
    "CONFLICTS, WAR AND PEACE",
    "CRIME, LAW AND JUSTICE",
    "DISASTER, ACCIDENT AND EMERGENCY INCIDENT",
    "ECONOMY, BUSINESS AND FINANCE",
    "EDUCATION",
    "ENVIRONMENT",
    "HEALTH",
    "HOBBY AND PERSONAL INTEREST",
    "LABOUR",
    "LIFESTYLE AND LEISURE",
    "POLITICS",
    "RELIGION",
    "SCIENCE AND TECHNOLOGY",
    "SOCIETY",
    "SPORT",
    "WEATHER",
]

SHARED = (
    "The article is classified into one or more of these categories: "
    + "; ".join(TOPICS)
    + ". Most articles have one or two main topics. Judge the main "
    "subject of the article, not incidental mentions."
)

THRESHOLD = 0.5
MAX_BODY_CHARS = 8000


def inference(title: str, body: str):
    questions = [
        {
            "type": "predicate",
            "name": topic,
            "instructions": (
                f"{SHARED} Is {topic} one of this article's main topics?"
            ),
        }
        for topic in TOPICS
    ]
    questions.append(
        {
            "type": "choice",
            "name": "primary",
            "instructions": (
                f"{SHARED} Which single category best describes this "
                "article's main topic?"
            ),
            "choices": [{"value": t} for t in TOPICS],
        }
    )
    return llm.decisions.create(
        model=MODEL,
        input=f"Title: {title}\n\n{body[:MAX_BODY_CHARS]}",
        questions=questions,
    )


semaphore = asyncio.Semaphore(10)


def load_done() -> set:
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            if "error" not in r:
                done.add(r["id"])
    return done


async def one(article, out) -> dict:
    async with semaphore:
        try:
            resp = await asyncio.to_thread(
                inference,
                article["title"],
                article["body"],
            )
            probs, primary, confidence = {}, None, None
            for topic, ans in zip(TOPICS, resp.answers):
                if ans.type == "predicate":
                    probs[topic] = ans.probability
            choice = resp.answers[-1]
            if choice.type == "choice":
                primary = choice.choice
                confidence = choice.confidence
            rest = sorted(
                (t for t, p in probs.items()
                 if p is not None and p >= THRESHOLD and t != primary),
                key=lambda t: probs[t],
                reverse=True,
            )
            topics = ([primary] if primary else []) + rest
            result = {
                "id": article["id"],
                "model": MODEL,
                "topics": topics,
                "primary": primary,
                "primary_confidence": confidence,
                "probs": probs,
                "gold": article["topics"],
            }
        except Exception as e:
            result = {
                "id": article["id"],
                "model": MODEL,
                "error": str(e),
            }
    out.write(json.dumps(result, ensure_ascii=False) + "\n")
    out.flush()
    return result


async def run_topic():
    articles = pd.read_csv("data/benchmark.tsv", sep="\t")
    done = load_done()
    os.makedirs("results", exist_ok=True)
    with open(OUT, "a") as out:
        tasks = [
            one(article, out)
            for _, article in articles.iterrows()
            if article["id"] not in done
        ]
        print(f"{len(done)} records already done, running {len(tasks)}")
        results = await tqdm.gather(*tasks)
    errors = sum(1 for r in results if "error" in r)
    print(f"topic: {len(results)} records, {errors} errors")


if __name__ == "__main__":
    asyncio.run(run_topic())
