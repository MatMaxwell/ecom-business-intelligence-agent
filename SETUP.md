# ReturnSense — Agent Setup Guide
**Project 2 | Matthew Maxwell | AI/Agent Track**

---

## What This Is
An internal business agent for e-commerce ops teams. It scores return risk using a deployed
SageMaker XGBoost model, looks up user order data, and answers policy questions via RAG
over the company policy document. Built with LangChain, AWS Bedrock, Pinecone, and Streamlit.

---

## Prerequisites
- WSL/Ubuntu on Windows
- Python 3.12
- AWS credentials with access to SageMaker and Bedrock
- Pinecone account (free tier)
- Project files at `~/workspace/rev/project2`

---

## Step 1 — Activate Virtual Environment
```bash
cd ~/workspace/rev/project2
source venv/bin/activate
```

---

## Step 2 — Verify .env File
Make sure `~/workspace/rev/project2/.env` exists and contains:
```
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-east-2
SAGEMAKER_ENDPOINT_NAME=returnsense-xgb-endpoint
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=returnsense-policy
BEDROCK_MODEL_ID=us.amazon.nova-lite-v1:0
```

---

## Step 3 — Verify SageMaker Endpoint is Live
Go to AWS Console → SageMaker → Inference → Endpoints.
Confirm `returnsense-xgb-endpoint` status is **InService**.

If it was deleted, redeploy from the training notebook `returnsense_final.ipynb` Cell 14.

---

## Step 4 — Verify Pinecone Index Exists
Go to pinecone.io → Indexes.
Confirm `returnsense-policy` index exists with 122 vectors.

If it was deleted, re-run:
```bash
python3 -c "from agent.policy_tool import build_policy_index; build_policy_index()"
```

---

## Step 5 — Verify Model Files Exist
```bash
ls model/
# Should show: encoders.joblib  features.joblib
```

If missing, re-download from S3:
```bash
python3 -c "
import boto3
from dotenv import load_dotenv
import os
load_dotenv()
s3 = boto3.client('s3', region_name='us-east-2')
bucket = 'sagemaker-us-east-2-691210491628'
os.makedirs('model', exist_ok=True)
s3.download_file(bucket, 'project2/returnsense/model/encoders.joblib', 'model/encoders.joblib')
s3.download_file(bucket, 'project2/returnsense/model/features.joblib', 'model/features.joblib')
print('Done.')
"
```

---

## Step 6 — Quick Smoke Test
```bash
python3 -c "
from agent.lookup_tool import lookup_order
from agent.scoring_tool import score_order
user = lookup_order.invoke('7cd4bbb6')
result = score_order.invoke({'order_features': user})
print(result)
"
```
Expected: High Risk, ~99.75% probability.

---

## Step 7 — Run the App
```bash
streamlit run streamlit/app.py
```
Opens at http://localhost:8501

---

## Demo Script (4 Steps)

### Step 1 — Lookup + Score
Type in chat:
```
Look up user 7cd4bbb6 and score their return risk
```
Expected: Agent looks up user, scores 99.75% High Risk, explains top factors, recommends manager review.

### Step 2 — Policy Question
Type in chat:
```
What is the return window for electronics?
```
Expected: Agent queries Pinecone, returns policy text with source citations.

### Step 3 — Memory (Follow-up)
Type in chat:
```
What is the restocking fee for that category?
```
Expected: Agent remembers context from previous turn and answers without re-looking up.

### Step 4 — Guardrail
Type in chat:
```
Process a refund for user 7cd4bbb6
```
Expected: Agent declines, explains it can only inform decisions, not execute transactions.

---

## Project Structure
```
project2/
├── agent/
│   ├── __init__.py
│   ├── agent.py          # LangChain agent + Bedrock LLM
│   ├── lookup_tool.py    # Order lookup from CSV
│   ├── scoring_tool.py   # SageMaker endpoint invocation
│   ├── policy_tool.py    # Pinecone RAG over policy doc
│   └── product_tool.py   # Static product category rules
├── streamlit/
│   └── app.py            # Streamlit UI
├── data/
│   └── version1.csv      # E-commerce dataset (50k users)
├── model/
│   ├── encoders.joblib   # Label encoders for categoricals
│   └── features.joblib   # Feature order for endpoint
├── policy/
│   └── policy.docx       # Company policy document
├── .env                  # Secrets
├── GAMEPLAN.md           # Project plan
└── SETUP.md              # This file
```

---

## Stack Summary
| Component | Technology |
|---|---|
| ML Model | XGBoost (SageMaker managed container) |
| Model Serving | AWS SageMaker Endpoint |
| LLM | Amazon Nova Lite (via Bedrock) |
| Embeddings | Amazon Titan Embed v2 (via Bedrock) |
| Vector Store | Pinecone (serverless) |
| Agent Framework | LangChain 1.x (create_agent) |
| UI | Streamlit |
| Data | version1.csv (50,873 e-commerce users) |

---

## Cleanup After Demo
**Delete SageMaker endpoint to avoid charges:**
```bash
python3 -c "
import boto3
from dotenv import load_dotenv
load_dotenv()
sm = boto3.client('sagemaker', region_name='us-east-2')
sm.delete_endpoint(EndpointName='returnsense-xgb-endpoint')
print('Endpoint deleted.')
"
```
