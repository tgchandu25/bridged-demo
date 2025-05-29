
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import openai
from pinecone import Pinecone
from pinecone import ServerlessSpec
import ast
import pandas as pd

# Initialize environment
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV", "us-east-1-aws")
index_name = os.getenv("PINECONE_INDEX", "bridged-assignment")

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
if index_name not in pc.list_indexes().names():
    raise Exception(f"Index '{index_name}' not found in Pinecone.")
index = pc.Index(index_name)

# FastAPI app
app = FastAPI()

# Request schema
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

# Filter generation system prompt
SYSTEM_PROMPT = """
You are a smart assistant that converts natural language queries into Pinecone-compatible JSON filters.
Only use these fields:
- author: string
- published_year: integer
- published_month: integer
- tags: list of strings

Return only JSON.
"""

# Helper: embedding
def get_embedding(text: str, model: str = "text-embedding-ada-002") -> List[float]:
    response = openai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# Helper: filter from GPT
def generate_filter_from_query(query: str) -> Dict[str, Any]:
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]
    )
    try:
        return ast.literal_eval(response.choices[0].message.content.strip())
    except Exception as e:
        print("Filter parsing failed:", e)
        return {}

# Main search endpoint
@app.post("/search")
def search_articles(request: QueryRequest):
    try:
        query_embedding = get_embedding(request.query)
        filter_dict = generate_filter_from_query(request.query)

        results = index.query(
            vector=query_embedding,
            top_k=request.top_k,
            filter=filter_dict,
            include_metadata=True
        )

        matches = []
        for match in results.matches:
            matches.append({
                "score": match.score,
                "title": match.metadata.get("title", "N/A"),
                "author": match.metadata.get("author", "N/A"),
                "tags": match.metadata.get("tags", [])
            })

        return {"results": matches}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
