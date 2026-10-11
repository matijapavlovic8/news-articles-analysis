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

PROMPT = """
You are an expert at named entity recognition (NER). You are given a Croatian
news article as a token list: each line is one sentence ("S0" is the
headline), each token is prefixed by its index in that sentence.

For every named entity, return its span as token indices:
- "sent": sentence number (S0 -> 0, S1 -> 1, ...)
- "start": index of the entity's first token
- "end": index one past the entity's last token (exclusive)
- "label": one of "PER", "LOC", "ORG"

Entity types:
- PER: person names, full or partial (e.g. "Andrej Plenković", "Tomašević")
- LOC: countries, cities, regions, streets, geographic features
- ORG: companies, institutions, parties, clubs, agencies

Rules:
- Span boundaries must align exactly with token boundaries — do not include
  punctuation tokens (",", ":", "." ...) inside a span.
- Do not merge adjacent entities and do not split one entity into several.
- List every mention, in order of appearance, headline first.
- Only PER, LOC, ORG — ignore dates, amounts, events, products, other types.
"""

MAX_TOKENS = 1500

articles = pd.read_csv("data/benchmark.tsv", sep="\t")
semaphore = asyncio.Semaphore(len(MODELS) * 10)


with open("secrets.toml", "rb") as f:
    secrets = tomllib.load(f)

llm = OpenAI(
    api_key=secrets["OPENAI_API_KEY"],
)


class NerEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sent: int
    start: int
    end: int
    label: Literal["PER", "LOC", "ORG"]


class NerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entities: list[NerEntity]


def format_tokens(tokens) -> str:
    out, total = [], 0
    for s, sent in enumerate(tokens):
        if total + len(sent) > MAX_TOKENS:
            break
        total += len(sent)
        out.append(f"S{s}: " + " ".join(f"[{i}]{t}" for i, t in enumerate(sent)))
    return "\n".join(out)


def inference(model: str, token_block: str) -> NerResponse:
    response = llm.responses.create(
        model=model,
        temperature=0,
        instructions=PROMPT,
        input=token_block,
        text={
            "format": {
                "type": "json_schema",
                "name": "ner",
                "schema": NerResponse.model_json_schema(),
                "strict": True,
            }
        },
    )
    return NerResponse.model_validate_json(response.output_text)


async def one(model, article) -> dict:
    async with semaphore:
        try:
            tokens = json.loads(article["tokens"])
            resp = await asyncio.to_thread(inference, model, format_tokens(tokens))
            return {
                "id": article["id"],
                "model": model,
                "ner": resp.model_dump()["entities"],
                "gold": article["ner"],
            }
        except Exception as e:
            return {"id": article["id"], "model": model, "error": str(e)}


async def run_ner() -> list:
    tasks = []
    for _, article in articles.iterrows():
        if not isinstance(article["tokens"], str):
            continue
        tasks.extend(one(model, article) for model in MODELS)
    return await tqdm_asyncio.gather(*tasks)


if __name__ == "__main__":
    results = asyncio.run(run_ner())

    os.makedirs("results", exist_ok=True)
    with open("results/ner.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in results)
