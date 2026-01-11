import os
import numpy as np
from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
from openai import AzureOpenAI
app = FastAPI(title="AI RAG Service")
# ---------- Azure OpenAI Config ----------
API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
EMBED_MODEL = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
client = AzureOpenAI(
   api_key=API_KEY,
   azure_endpoint=ENDPOINT,
   api_version="2024-06-01"
)
# ---------- Vector Store (In-Memory) ----------
vector_store = []  # {text, embedding, source}
# ---------- Utils ----------
def chunk_text(text, chunk_size=500):
   words = text.split()
   for i in range(0, len(words), chunk_size):
       yield " ".join(words[i:i+chunk_size])
def cosine_similarity(a, b):
   return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
# ---------- Upload PDF ----------
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
   vector_store.clear()
   reader = PdfReader(file.file)
   for page in reader.pages:
       text = page.extract_text()
       if not text:
           continue
       for chunk in chunk_text(text):
           emb = client.embeddings.create(
               model=EMBED_MODEL,
               input=chunk
           ).data[0].embedding
           vector_store.append({
               "text": chunk,
               "embedding": emb,
               "source": file.filename
           })
   return {"chunks_indexed": len(vector_store)}
# ---------- Health ----------
@app.get("/health")
def health():
   return {"status": "healthy"}
# ---------- Ask ----------
@app.get("/ask")
def ask(question: str):
   if not vector_store:
       return {"error": "No documents indexed"}
   q_embedding = client.embeddings.create(
       model=EMBED_MODEL,
       input=question
   ).data[0].embedding
   scored = [
       (cosine_similarity(q_embedding, v["embedding"]), v["text"])
       for v in vector_store
   ]
   top_chunks = [t for _, t in sorted(scored, reverse=True)[:3]]
   context = "\n\n".join(top_chunks)
   prompt = f"""
Use ONLY the context below to answer.
CONTEXT:
{context}
QUESTION:
{question}
"""
   response = client.chat.completions.create(
       model=DEPLOYMENT,
       messages=[{"role": "user", "content": prompt}]
   )
   return {"answer": response.choices[0].message.content}
