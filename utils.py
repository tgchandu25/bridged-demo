import os
import openai
import ast
from typing import List, Dict, Any
from pinecone import Pinecone

# Environment setup
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV", "us-east-1-aws")
index_name = os.getenv("PINECONE_INDEX", "bridged-assignment")

# Pinecone setup
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)

# Prompt template for GPT-based filter generation
SYSTEM_PROMPT = """
You are a smart assistant that converts natural language search queries into Pinecone-compatible JSON filters.
Use only the following metadata fields:
- author: string
- published_year: integer
- published_month: integer
- tags: list of strings
Return only the JSON.
"""

# Generate embedding using OpenAI
def get_embedding(text: str, model: str = "text-embedding-ada-002") -> List[float]:
    response = openai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# Generate metadata filter using GPT
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
        print("Error parsing GPT filter:", e)
        return {}

# Perform a filtered vector search in Pinecone
def search_articles(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    query_embedding = get_embedding(query)
    filter_dict = generate_filter_from_query(query)

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        filter=filter_dict,
        include_metadata=True
    )

    return [
        {
            "score": match.score,
            "title": match.metadata.get("title", "N/A"),
            "author": match.metadata.get("author", "N/A"),
            "tags": match.metadata.get("tags", [])
        }
        for match in results.matches
    ]