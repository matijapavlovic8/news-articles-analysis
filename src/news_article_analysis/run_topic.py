import asyncio
import json
import os
import tomllib
from typing import Literal

import pandas as pd
from tqdm.asyncio import tqdm as tqdm_asyncio
from openai import OpenAI
from pydantic import BaseModel, ConfigDict

MODELS = ["gpt-6-luna", "gpt-6-sol"]
OUT = "results/topic.jsonl"
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

PROMPT = """
You are a news classification expert. Assign topics to the given Croatian
news article from this fixed taxonomy:

- ARTS, CULTURE, ENTERTAINMENT AND MEDIA
- CONFLICTS, WAR AND PEACE
- CRIME, LAW AND JUSTICE
- DISASTER, ACCIDENT AND EMERGENCY INCIDENT
- ECONOMY, BUSINESS AND FINANCE
- EDUCATION
- ENVIRONMENT
- HEALTH
- HOBBY AND PERSONAL INTEREST
- LABOUR
- LIFESTYLE AND LEISURE
- POLITICS
- RELIGION
- SCIENCE AND TECHNOLOGY
- SOCIETY
- SPORT
- WEATHER

Rules:
- Base the classification on the article's main subject, not on entities or
  topics that are only mentioned in passing.
- Most articles have one or two topics. Assign additional labels only when
  the article genuinely covers several subjects (maximum 5).
- Order topics by importance: the first entry is the primary topic.
- Use only labels from the list above, spelled exactly as written.
"""

MAX_BODY_CHARS = 8000

articles = pd.read_csv("data/benchmark.tsv", sep="\t")
semaphore = asyncio.Semaphore(len(MODELS) * 10)


with open("secrets.toml", "rb") as f:
    secrets = tomllib.load(f)

llm = OpenAI(
    api_key=secrets["OPENAI_API_KEY"],
)


class TopicResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topics: list[Literal[*TOPICS]]  # type: ignore[valid-type]


def inference(model: str, title: str, body: str) -> TopicResponse:
    response = llm.responses.create(
        model=model,
        instructions=PROMPT,
        input=f'Title: "{title}"\n\n{body[:MAX_BODY_CHARS]}',
        text={
            "format": {
                "type": "json_schema",
                "name": "topics",
                "schema": TopicResponse.model_json_schema(),
                "strict": True,
            }
        },
    )
    return TopicResponse.model_validate_json(response.output_text)


def load_done() -> set:
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            if "error" not in r:
                done.add((r["id"], r["model"]))
    return done


async def one(model, article, out) -> dict:
    async with semaphore:
        try:
            resp = await asyncio.to_thread(
                inference, model, article["title"], article["body"]
            )
            result = {
                "id": article["id"],
                "model": model,
                "topics": resp.model_dump()["topics"],
                "gold": article["topics"],
            }
        except Exception as e:
            result = {"id": article["id"], "model": model, "error": str(e)}
    out.write(json.dumps(result, ensure_ascii=False) + "\n")
    out.flush()
    return result


async def run_topic() -> list:
    done = load_done()
    os.makedirs("results", exist_ok=True)
    with open(OUT, "a") as out:
        tasks = []
        for _, article in articles.iterrows():
            topics = article["topics"]
            if not isinstance(topics, str) or topics in ("", "[]"):
                continue
            for model in MODELS:
                if (article["id"], model) not in done:
                    tasks.append(one(model, article, out))
        print(f"{len(done)} records already done, running {len(tasks)}")
        return await tqdm_asyncio.gather(*tasks)


if __name__ == "__main__":
    results = asyncio.run(run_topic())
    errors = sum(1 for r in results if "error" in r)
    print(f"topic: {len(results)} records, {errors} errors")
