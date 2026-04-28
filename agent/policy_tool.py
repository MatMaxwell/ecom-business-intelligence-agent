import os
import json
import boto3
from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "returnsense-policy")
bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-2")

def get_embedding(text):
    response = bedrock_runtime.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({"inputText": text}),
        contentType="application/json",
        accept="application/json"
    )
    result = json.loads(response["body"].read())
    return result["embedding"]

def build_policy_index():
    print("Loading policy document...")
    loader = Docx2txtLoader("policy/policy.docx")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    existing = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Created Pinecone index: {INDEX_NAME}")

    index = pc.Index(INDEX_NAME)

    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk.page_content)
        vectors.append({
            "id": f"chunk-{i}",
            "values": embedding,
            "metadata": {"text": chunk.page_content}
        })
        if i % 10 == 0:
            print(f"Embedded {i}/{len(chunks)} chunks...")

    index.upsert(vectors=vectors)
    print(f"Done. Uploaded {len(vectors)} vectors to Pinecone.")

@tool
def query_policy(question: str) -> str:
    """Query the company policy document to answer questions about returns,
    refunds, eligibility, shipping rules, and other policy topics.
    Returns relevant policy text with citations."""

    index = pc.Index(INDEX_NAME)
    query_embedding = get_embedding(question)

    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )

    if not results["matches"]:
        return "No relevant policy found. Please consult your manager for guidance."

    response = "Policy excerpts relevant to your question:\n\n"
    for i, match in enumerate(results["matches"]):
        text = match["metadata"]["text"]
        score = match["score"]
        response += f"[Source {i+1} | Relevance: {score:.2f}]\n{text}\n\n"

    return response