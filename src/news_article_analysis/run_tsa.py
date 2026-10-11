import asyncio
import os
from tqdm.asyncio import tqdm as tqdm_asyncio
from openai import OpenAI
import pandas as pd
from pydantic import BaseModel, ConfigDict
from typing import Literal
import tomllib
import json

MODELS = ["gpt-6-luna", "gpt-6-sol"]
SENTIMENTS = {0: "neutral", 1: "negative", 2: "positive"}
PROMPT = """
Targeted sentiment analysis (TSA) is the task of determining the polarity of sentiment expressed towards the target entity. The entities mentioned in the title will be provided to you and the only thing you need to do is go through the title text and determine the sentiment towards each entity in it. The sentiment can be positive, neutral or negative. Here are a few examples of inputs with the expected output:
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

with open("secrets.toml", "rb") as f:
    secrets = tomllib.load(f)


articles = pd.read_csv("data/benchmark.tsv", sep="\t")

llm = OpenAI(
    api_key=secrets["OPENAI_API_KEY"],
)

semaphore = asyncio.Semaphore(len(MODELS)*10)

class EntitySentiment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    i: int
    sentiment: Literal["neutral", "negative", "positive"]

class TSAResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entities: list[EntitySentiment]

def inference(model: str, title: str, ner_data: str) -> TSAResponse:
    response = llm.responses.create(
        model=model,
        temperature=0,
        reasoning=None,
        instructions=PROMPT,
        input=f"""
                Title: "{title}"\n\nEntities:\n{ner_data}
                Title: "{title}"\n\nEntities:\n{ner_data}
                """,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "tsa",
                        "schema": TSAResponse.model_json_schema(),
                        "strict": True,
                    }
                },
    )

    return TSAResponse.model_validate_json(response.output_text)

def headline_entities(article) -> str:
    ner = json.loads(article["ner"])
    ents = [e for e in ner if e["sent"] == 0]
    return "\n".join(f'{i}. {e["text"]} ({e["label"]})' for i, e in enumerate(ents))

async def one(model, article, ner_data) -> dict:
    async with semaphore:
        try:
            resp = await asyncio.to_thread(inference, model, article["title"], ner_data)
            return {
                "id": article["id"],
                "model": model,
                "tsa": resp.model_dump(),
                "gold": article["tsa"],
            }
        except Exception as e:
            return {"id": article["id"], "model": model, "error": str(e)}

async def run_tsa() -> list:
    tasks = []
    for _, article in articles.iterrows():
        ner_data = headline_entities(article)
        if not ner_data:
            continue
        tasks.extend(one(model, article, ner_data) for model in MODELS)
    return await tqdm_asyncio.gather(*tasks)



if __name__ == "__main__":
    results = asyncio.run(run_tsa())

    os.makedirs("results", exist_ok=True)
    with open("results/tsa.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in results)
