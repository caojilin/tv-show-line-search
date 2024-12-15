from sentence_transformers import SentenceTransformer
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from rapidfuzz import process, fuzz
import pandas as pd
from fastapi import FastAPI, Query
import pickle
from collections import defaultdict
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


origins = [
    "http://localhost",
    "http://localhost:8889",
    "https://caojilin-playground.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/py/the-office/rapidfuzz")
def rapidfuzz(query: str = "Bears, beets, Battlestar Galactica.", limit: int = 5):
    # Get the folder where this script is located
    current_folder = Path(__file__).resolve().parent
    csv_file_path = current_folder / "schrute.csv"
    df = pd.read_csv(csv_file_path, header=0)
    df['lower'] = df['text'].str.lower()
    df['lower'] = df['lower'].str.strip()
    # Replace empty strings with NaN and then drop those rows
    df.replace(pd.NA, "", inplace=True)
    # Step 3: Unpickle the array
    with open('the_office.pkl', 'rb') as f:
        combined_vertical = pickle.load(f)

    normalized_query = query.lower().strip()
    limit = int(limit)
    exact_match = [(text, 100, i)
                   for i, text in enumerate(df["lower"]) if normalized_query in text][:limit]
    fuzz_match = process.extract(
        normalized_query, choices=df["lower"], scorer=fuzz.partial_ratio, score_cutoff=0.8, limit=limit)

    # Convert the query into a numerical vector that Pinecone can search with
    query_embedding = model.encode(query)

    scores = model.similarity(query_embedding, combined_vertical)[0]
    indices = np.argsort(scores).tolist()[::-1][:limit]
    print(indices, scores[indices[2]])

    sentence_match = []
    for i in range(len(indices)):
        sentence_match.append(
            (indices[i], scores[indices[i]]*100, indices[i]))

    return_body = {}
    counter = 0
    return_body, counter = convert_to_json(df,
                                           return_body, counter, exact_match, 'exact match')
    return_body, counter = convert_to_json(df,
                                           return_body, counter, fuzz_match, 'partial ratio')
    return_body, counter = convert_to_json(df,
                                           return_body, counter, sentence_match, 'sentence embedding')
    return return_body


@app.get("/api/py/the-TBBT/rapidfuzz")
def rapidfuzz(query: str = "I'm as much part of this relationship as you are.", limit: int = 5):
    # Get the folder where this script is located
    current_folder = Path(__file__).resolve().parent
    csv_file_path = current_folder / "TBBT.csv"
    df = pd.read_csv(csv_file_path, header=0)
    df['lower'] = df['text'].str.lower()
    df['lower'] = df['lower'].str.strip()
    # Replace empty strings with NaN and then drop those rows
    df.replace(pd.NA, "", inplace=True)
    print(df.loc[0])
    # Step 3: Unpickle the array
    with open('TBBT.pkl', 'rb') as f:
        combined_vertical = pickle.load(f)

    normalized_query = query.lower().strip()
    limit = int(limit)
    exact_match = [(text, 100, i)
                   for i, text in enumerate(df["lower"]) if normalized_query in text][:limit]
    fuzz_match = process.extract(
        normalized_query, choices=df["lower"], scorer=fuzz.partial_ratio, score_cutoff=0.8, limit=limit)

    # Convert the query into a numerical vector that Pinecone can search with
    query_embedding = model.encode(query)

    scores = model.similarity(query_embedding, combined_vertical)[0]
    indices = np.argsort(scores).tolist()[::-1][:limit]
    print(indices, scores[indices[2]])

    sentence_match = []
    for i in range(len(indices)):
        sentence_match.append(
            (indices[i], scores[indices[i]]*100, indices[i]))

    return_body = {}
    counter = 0
    return_body, counter = convert_to_json(df,
                                           return_body, counter, exact_match, 'exact match')
    return_body, counter = convert_to_json(df,
                                           return_body, counter, fuzz_match, 'partial ratio')
    return_body, counter = convert_to_json(df,
                                           return_body, counter, sentence_match, 'sentence embedding')
    return return_body


def convert_to_json(df, return_body, counter, arr, name):

    for _, result in enumerate(arr):
        _, score, row = result
        _, season, episode, episode_name, director, writer, character, text, _, _ = df.loc[row].tolist(
        )
        return_body[counter] = {
            "season": int(season),  # Ensure it's a Python int
            "episode": int(episode),
            "episode_name": str(episode_name),
            "director": str(director),
            "writer": str(writer),
            "character": str(character),
            "text": str(text),
            "score": float(score),
            "method": name
        }
        counter += 1
    return return_body, counter
