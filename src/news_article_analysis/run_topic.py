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
        temperature=0,
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


async def one(model, article) -> dict:
    async with semaphore:
        try:
            resp = await asyncio.to_thread(
                inference, model, article["title"], article["body"]
            )
            return {
                "id": article["id"],
                "model": model,
                "topics": resp.model_dump()["topics"],
                "gold": article["topics"],
            }
        except Exception as e:
            return {"id": article["id"], "model": model, "error": str(e)}


async def run_topic() -> list:
    tasks = []
    for _, article in articles.iterrows():
        topics = article["topics"]
        if not isinstance(topics, str) or topics in ("", "[]"):
            continue
        tasks.extend(one(model, article) for model in MODELS)
    return await tqdm_asyncio.gather(*tasks)


if __name__ == "__main__":
    results = asyncio.run(run_topic())

    os.makedirs("results", exist_ok=True)
    with open("results/topic.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in results)
