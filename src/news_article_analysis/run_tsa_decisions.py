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
OUT = "results/tsa_decisions.jsonl"

SENTIMENTS = ["neutral", "negative", "positive"]

SHARED = """
Targeted sentiment analysis (TSA) is the task of determining the polarity
of sentiment expressed towards the target entity. The entities mentioned
in the title are provided to you; go through the title text and determine
the sentiment towards each entity in it. The sentiment can be positive,
neutral or negative. Judge sentiment toward the entity itself, not the
overall tone of the headline, and only from what the headline states or
directly implies — do not use outside knowledge about the entity.
Here are a few examples of inputs with the expected output:
INPUT:
	Title: Na današnji dan poginuo je legendarni Andrija Matijaš Pauk: 'Pao je junak da bismo mi mogli živjeti'
	Entities: {1: Andrija Matijaš Pauk}
OUTPUT:
	[1: positive]

INPUT:
	Title: VIDEO Prije 13 godina Paško Dodić se pojavio u spotu HDZ-a. Društvo mu je pravio Andrija Mikulić
	Entities: {1: Paško Dodić, 2: HDZ-a, 3: Andrija Mikulić}
OUTPUT:
	[{1: neutral}, {2: neutral}, {3: neutral}]

INPUT:
	Title: Nekoliko nacionalnih sindikata podržalo štrajk: "Tomašević uvodi konfuziju"
	Entities: {1: nacionalnih sindikata, 2: Tomašević}
OUTPUT:
	[{1: neutral}, {2: negative}]
"""


def headline_entities(article) -> list[dict]:
    ner = json.loads(article["ner"])
    return [e for e in ner if e["sent"] == 0]


def inference(title: str, ents: list[dict]):
    questions = [
        {
            "type": "choice",
            "name": f"e{i}",
            "instructions": (
                f"{SHARED}\nFor the title in the input, what sentiment "
                f"is expressed toward entity {i}: '{e['text']}' "
                f"({e['label']})?"
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


def load_done() -> set:
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            if "error" not in r:
                done.add(r["id"])
    return done


async def one(article, out) -> dict:
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
            result = {
                "id": article["id"],
                "model": MODEL,
                "tsa": {"entities": entities},
                "gold": article["tsa"],
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


async def run_tsa():
    articles = pd.read_csv("data/benchmark.tsv", sep="\t")
    done = load_done()
    os.makedirs("results", exist_ok=True)
    with open(OUT, "a") as out:
        tasks = [
            one(article, out)
            for _, article in articles.iterrows()
            if headline_entities(article) and article["id"] not in done
        ]
        print(f"{len(done)} records already done, running {len(tasks)}")
        results = await tqdm.gather(*tasks)
    errors = sum(1 for r in results if "error" in r)
    print(f"tsa: {len(results)} records, {errors} errors")


if __name__ == "__main__":
    asyncio.run(run_tsa())
