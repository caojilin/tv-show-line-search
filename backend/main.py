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
    csv_file_path = "schrute.csv"
    df = get_df(csv_file_path)
    # Step 3: Unpickle the array
    with open('the_office.pkl', 'rb') as f:
        vector_stacked = pickle.load(f)

    limit = int(limit)
    return_body = core_refactor(query, limit, df, vector_stacked)
    return return_body


@app.get("/api/py/the-TBBT/rapidfuzz")
def rapidfuzz(query: str = "I'm as much part of this relationship as you are.", limit: int = 5):
    csv_file_path = "TBBT.csv"
    df = get_df(csv_file_path)
    # Step 3: Unpickle the array
    with open('TBBT.pkl', 'rb') as f:
        vector_stacked = pickle.load(f)

    limit = int(limit)
    return_body = core_refactor(query, limit, df, vector_stacked)
    return return_body


def get_df(csv_file_path):
    df = pd.read_csv(csv_file_path, header=0)
    df['lower'] = df['text'].str.lower()
    df['lower'] = df['lower'].str.strip()
    # Replace empty strings with NaN and then drop those rows
    df.replace(pd.NA, "", inplace=True)
    return df


def core_refactor(query, limit, df, vector_stacked):
    normalized_query = query.lower().strip()
    fuzz_match = process.extract(
        normalized_query, choices=df["lower"], scorer=partial_ratio_word_level, limit=limit)

    # Convert the query into a numerical vector that Pinecone can search with
    query_embedding = model.encode(query)
    
    #filter out the vectors that are too short
    original_indices = np.arange(len(df))
    a = np.array([x if x is not None else "" for x in df['text'].tolist()])
    mask = np.vectorize(len)(a) >= len(query)  # Boolean mask for valid strings

    filtered_vectors = vector_stacked[mask]
    filtered_indices = original_indices[mask]

    scores = model.similarity(query_embedding, filtered_vectors)[0]
    indices = np.argsort(scores).tolist()[::-1][:limit]

    sentence_match = []
    for i in indices:
        k = filtered_indices[i]
        sentence_match.append(
            (k, float(np.round(scores[i].numpy(), 5)*100), k))

    return_body = {}
    counter = 0
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
            "score": float(round(score, 5)),
            "method": name
        }
        counter += 1
    return return_body, counter


def partial_ratio_word_level(str1, str2, **kwargs):
    words1 = str1.split()
    words2 = str2.split()
    len1 = len(words1)
    len2 = len(words2)
    if len1 > len2:
        return 0.0
    best_score = 0.0
    max_start = len2 - len1 + 1
    ratio = fuzz.ratio
    for i in range(max_start):
        substring_words = words2[i:i+len1]
        score = ratio(" ".join(words1), " ".join(substring_words))
        if score > best_score:
            best_score = score
    return round(best_score, 3)
