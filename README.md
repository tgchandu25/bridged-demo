# Bridged AI Engineer Project

This project is a production-ready FastAPI application that takes a **natural language query**, generates a **Pinecone metadata filter using OpenAI**, performs **vector search**, and returns the most relevant articles.

---

## 🚀 Features

- Converts natural language into Pinecone-compatible metadata filters using GPT-3.5.
- Uses OpenAI's `text-embedding-ada-002` to generate vector embeddings.
- Performs vector search in Pinecone with metadata filtering.
- Dockerized FastAPI app ready for deployment.
- Accepts environment variables from `.env` file.

---
---

## 📁 Project Structure

```
bridged-demo/
│
├── app/
│   ├── main.py               ← FastAPI app
│   └── utils.py              ← Filter + search logic
│
├── requirements.txt          ← All dependencies
├── Dockerfile                ← For containerization
├── README.md                 ← Setup + architecture + how to run
├── sample_data.csv           ← dataset
└── .env.example              ← Template for API keys
```


## 📦 Setup

### 1. Clone the repository

```bash
git clone https://github.com/tgchandu25/bridged-demo.git
cd bridged-demo
```

### 2. Create `.env` file

Copy and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env` to include your actual API keys.

---

## 🐳 Run with Docker

### Build the Docker image

```bash
docker build -t bridged-demo .
```

### Run the container

```bash
docker run -p 8000:8000 --env-file .env bridged-demo
```

---

## 🧪 Example Query

**POST** `/search`

```json
{
  "query": "Show me articles by Jane Doe from May 2025 about IPL",
  "top_k": 5
}
```

**Response:**

```json
{
  "results": [
    {
      "score": 0.82,
      "title": "IPL 2025: Ruthless MI top table...",
      "author": "Jane Doe",
      "tags": ["#IPL2025", "#MumbaiIndians"]
    },
    ...
  ]
}
```

---

## ✅ Technologies Used

- FastAPI
- OpenAI (GPT + Embedding)
- Pinecone Vector DB
- Docker
- Uvicorn