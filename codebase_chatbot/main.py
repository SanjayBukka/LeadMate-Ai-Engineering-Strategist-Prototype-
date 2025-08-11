"""
LeadMate - Basic RAG Chatbot for a GitHub repo (no `git clone` required)

What this does:
- Downloads a repo ZIP from GitHub (default branch or specific branch),
- Extracts source files (filters by extensions),
- Chunks files into text snippets,
- Builds embeddings (using sentence-transformers locally) and an in-memory nearest-neighbor index,
- Runs a simple chat loop: user query -> retrieve top-K snippets -> calls Gemini to produce the final answer.

Requirements:
- Python 3.9+
- pip install -r requirements.txt

requirements.txt (suggested):
- requests
- sentence-transformers
- scikit-learn
- google-generativeai

Environment variables:
- GEMINI_API_KEY  -> required (for calling Gemini model)

Notes:
- This is a minimal example for prototyping. For production, switch to a vector DB (Pinecone/Chroma/Weaviate), add caching, pagination, error handling, and rate-limit handling for Gemini.
- The script uses the sentence-transformers model `all-MiniLM-L6-v2` for embeddings. It's small & fast.
"""

import os
import io
import zipfile
import requests
from typing import List, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=env_path)

# Embedding & retrieval
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# Gemini client (Google GenAI Python binding)
import google.generativeai as genai

# -----------------------------
# CONFIG
# -----------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Please set GEMINI_API_KEY environment variable with your Gemini API key.")

# configure genai
genai.configure(api_key=GEMINI_API_KEY)

# the sentence-transformers model for embeddings (local)
EMB_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000  # approx characters per chunk
N_NEIGHBORS = 6

# file extensions to include
CODE_EXTS = {".py", ".js", ".ts", ".java", ".go", ".rb", ".php", ".md", ".json", ".yaml", ".yml"}

# -----------------------------
# HELPERS
# -----------------------------

def download_repo_zip(owner: str, repo: str, branch: str = "main") -> bytes:
    """Download repo as zip from GitHub without cloning (default branch or specified branch)."""
    zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
    print(f"Downloading {zip_url} ...")
    r = requests.get(zip_url, stream=True)
    if r.status_code != 200:
        # try to fetch default branch via API if direct zip fails
        raise RuntimeError(f"Failed to download repo ZIP: {r.status_code} {r.text}")
    return r.content


def extract_text_files_from_zip(zip_bytes: bytes, exts=CODE_EXTS) -> List[Tuple[str, str]]:
    """Extract files matching extensions from zip bytes. Returns list of (relative_path, text_content)."""
    files = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            for zi in z.infolist():
                if zi.is_dir():
                    continue
                name = zi.filename
                _, ext = os.path.splitext(name)
                if ext.lower() in exts:
                    try:
                        with z.open(zi) as fh:
                            raw = fh.read()
                            # try utf-8, fallback to latin1
                            try:
                                text = raw.decode("utf-8")
                            except Exception:
                                text = raw.decode("latin-1")
                            files.append((name, text))
                    except Exception as e:
                        print(f"Skipping {name} due to error: {e}")
    return files


def chunk_text(text: str, size: int = CHUNK_SIZE) -> List[str]:
    """Chunk text into approx size-character chunks, tries to split on newlines for cleanliness."""
    if len(text) <= size:
        return [text]
    chunks = []
    lines = text.splitlines(keepends=True)
    cur = []
    cur_len = 0
    for line in lines:
        cur.append(line)
        cur_len += len(line)
        if cur_len >= size:
            chunks.append("".join(cur))
            cur = []
            cur_len = 0
    if cur:
        chunks.append("".join(cur))
    return chunks


# -----------------------------
# BUILD RAG INDEX
# -----------------------------

def build_rag_index(owner: str, repo: str, branch: str = "main"):
    # 1) download repo zip
    zip_bytes = download_repo_zip(owner, repo, branch)

    # 2) extract relevant files
    files = extract_text_files_from_zip(zip_bytes)
    print(f"Found {len(files)} text/code files to index.")

    # 3) chunk files
    docs = []  # list of (doc_text, metadata)
    for path, text in files:
        for i, chunk in enumerate(chunk_text(text)):
            metadata = {"path": path, "chunk_index": i}
            docs.append((chunk, metadata))

    texts = [d[0] for d in docs]
    metadatas = [d[1] for d in docs]

    # 4) embed with sentence-transformers
    print("Loading embedding model...")
    emb_model = SentenceTransformer(EMB_MODEL_NAME)
    embeddings = emb_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # 5) build nearest neighbor index
    nn = NearestNeighbors(n_neighbors=min(N_NEIGHBORS, len(embeddings)), metric="cosine")
    nn.fit(embeddings)

    # return everything needed for retrieval
    return {
        "texts": texts,
        "metadatas": metadatas,
        "embeddings": embeddings,
        "nn": nn
    }


# -----------------------------
# RETRIEVE + CHAT
# -----------------------------

def retrieve_top_k(index, query: str, k: int = 4):
    emb_model = SentenceTransformer(EMB_MODEL_NAME)
    q_emb = emb_model.encode([query], convert_to_numpy=True)
    distances, indices = index["nn"].kneighbors(q_emb, n_neighbors=min(k, len(index["texts"])))
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            "text": index["texts"][idx],
            "metadata": index["metadatas"][idx],
            "score": float(dist)
        })
    return results


def generate_answer_with_gemini(query: str, retrieved: List[dict]) -> str:
    # Build context
    context_blocks = []
    for r in retrieved:
        path = r["metadata"]["path"]
        idx = r["metadata"]["chunk_index"]
        snippet = r["text"]
        header = f"File: {path} (chunk {idx})\n---\n"
        context_blocks.append(header + snippet)
    context = "\n\n----\n\n".join(context_blocks)

    prompt = f"You are an expert programmer who has access to relevant code snippets.\n\nContext:\n{context}\n\nUser question: {query}\n\nPlease answer clearly, reference file paths when helpful, and provide actionable guidance. If unsure, say so."

    # Call Gemini
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()


# -----------------------------
# SIMPLE CLI
# -----------------------------

def parse_github_url(url: str) -> Tuple[str, str]:
    # supports: https://github.com/owner/repo or with .git or trailing slash
    u = url.rstrip("/ ")
    if u.endswith('.git'):
        u = u[:-4]
    parts = u.split("/")
    owner, repo = parts[-2], parts[-1]
    return owner, repo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build a simple RAG index from a GitHub repo and chat over it.")
    parser.add_argument("repo_url", help="GitHub repository URL (https://github.com/owner/repo)")
    parser.add_argument("--branch", default="main", help="Branch to download (default: main)")
    args = parser.parse_args()

    owner, repo = parse_github_url(args.repo_url)
    print(f"Indexing {owner}/{repo} (branch={args.branch}) ... this may take a while")
    index = build_rag_index(owner, repo, args.branch)
    print("Index built. You can now ask questions. Type 'exit' to quit.")

    while True:
        q = input("\nLeadMate> ")
        if q.strip().lower() in {"exit", "quit"}:
            break
        retrieved = retrieve_top_k(index, q, k=4)
        print("Retrieved snippets:")
        for r in retrieved:
            print(f"- {r['metadata']['path']} (chunk {r['metadata']['chunk_index']}) score={r['score']:.3f}")
        answer = generate_answer_with_gemini(q, retrieved)
        print("\nLeadMate Answer:\n")
        print(answer)
