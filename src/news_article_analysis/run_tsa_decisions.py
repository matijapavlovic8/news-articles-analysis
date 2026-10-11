import asyncio
import json
import tomllib

import pandas as pd
from openai import OpenAI
from tqdm.asyncio import tqdm

with open("secrets.toml", "rb") as f:
    secrets = tomllib.load(f)

llm = OpenAI(api_key=secrets["OPENAI_API_KEY"])

MODEL = "gpt-6-luna"  # the only decisions model

SENTIMENTS = ["neutral", "negative", "positive"]

SHARED = (
    "Decide the sentiment the headline expresses toward the target "
    "entity. Judge sentiment toward the entity itself, not the overall "
    "tone of the headline, and only from what the headline states or "
    "directly implies — do not use outside knowledge about the entity."
)


def headline_entities(article) -> list[dict]:
    ner = json.loads(article["ner"])
    return [e for e in ner if e["sent"] == 0]


def inference(title: str, ents: list[dict]):
    questions = [
        {
            "type": "choice",
            "name": f"e{i}",
            "instructions": (
                f"{SHARED} What sentiment does the headline express "
                f"toward '{e['text']}' ({e['label']})?"
            ),
            "choices": [{"value": s} for s in SENTIMENTS],
        }
        for i, e in enumerate(ents)
    ]
    return llm.decisions.create(
        model=MODEL,
        input=f"Title: {title}",
        questions=questions,
    )


semaphore = asyncio.Semaphore(10)


async def one(article) -> dict:
    ents = headline_entities(article)
    async with semaphore:
        try:
            resp = await asyncio.to_thread(
                inference, article["title"], ents
            )
            entities = []
            for i, ans in enumerate(resp.answers):
                e = {"i": i, "sentiment": None}
                if ans.type == "choice":
                    e["sentiment"] = ans.choice
                    e["probs"] = {
                        p.value: p.probability
                        for p in ans.probabilities
                    }
                    e["confidence"] = ans.confidence
                else:
                    e["refused"] = True
                entities.append(e)
            return {
                "id": article["id"],
                "model": MODEL,
                "tsa": {"entities": entities},
                "gold": article["tsa"],
            }
        except Exception as e:
            return {
                "id": article["id"],
                "model": MODEL,
                "error": str(e),
            }


async def run_tsa():
    articles = pd.read_csv("data/benchmark.tsv", sep="\t")
    tasks = [
        one(article)
        for _, article in articles.iterrows()
        if headline_entities(article)
    ]
    results = await tqdm.gather(*tasks)
    with open("results/tsa_decisions.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    errors = sum(1 for r in results if "error" in r)
    print(f"tsa: {len(results)} records, {errors} errors")


if __name__ == "__main__":
    asyncio.run(run_tsa())
