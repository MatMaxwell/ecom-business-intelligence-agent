# E-Commerce Business Intelligence Agent

An internal business intelligence tool built for e-commerce operations teams. Instead of manually reviewing orders and digging through policy docs, ops staff and risk analysts can chat with an AI agent that looks up customer data, scores return risk using a deployed ML model, and pulls relevant policy directly from the company's policy compendium — all in one place.

This was built as part of a data engineering and AI training program. The goal was to connect a real ML pipeline to a LangChain agent and ship something that actually tells a business story.

---

## What It Does

You type something like *"look up user 7cd4bbb6 and score their return risk"* and the agent:

1. Pulls the user's transaction history and behavioral features from the dataset
2. Sends those features to a deployed AWS SageMaker XGBoost endpoint
3. Returns a risk score, tier (Low / Medium / High), and SHAP-based explanation
4. Checks the policy doc if you ask a follow-up like *"what's the restocking fee for electronics?"*
5. Remembers the conversation so follow-up questions work naturally

It won't process refunds or cancel orders — it's a decision support tool, not a system of record. The guardrail fires if you try.

---

## Stack

| Layer | Technology |
|---|---|
| ML Model | XGBoost (trained via SageMaker managed container) |
| Model Serving | AWS SageMaker real-time inference endpoint |
| LLM | Amazon Nova Lite via AWS Bedrock |
| Embeddings | Amazon Titan Embed Text v2 via Bedrock |
| Vector Store | Pinecone (serverless) |
| Agent Framework | LangChain 1.x (`create_agent`) |
| UI | Streamlit |
| Dataset | Synthetic e-commerce dataset (50,873 users, 32 features) |

---

## Project Structure

```
ecom-business-intelligence-agent/
├── agent/
│   ├── agent.py          # LangChain agent wired to Bedrock Nova Lite
│   ├── lookup_tool.py    # Fetches user feature row from dataset by user_id
│   ├── scoring_tool.py   # Calls SageMaker endpoint, returns risk score + explanation
│   ├── policy_tool.py    # RAG over policy doc via Pinecone + Titan embeddings
│   └── product_tool.py   # Static return rules by product category
├── streamlit/
│   └── app.py            # Chat UI + quick action buttons
├── data/                 # Dataset stored in S3 (too large for GitHub)
├── policy/
│   └── policy.docx       # Company policy compendium (chunked + indexed in Pinecone)
├── GAMEPLAN.md           # Original project plan and business problem statement
└── README.md
```

---

## The Model

The XGBoost model predicts whether a user is a high return-risk customer based on their shopping behavior — not whether a specific order will be returned. Features include purchase history, chargeback count, average transaction cost, browsing behavior (page views, cart adds/removes, clicks), device preference, and product category.

Training used the SageMaker managed XGBoost 1.5-1 container with a 70/15/15 train/val/test split, early stopping, and `scale_pos_weight` to handle class imbalance. SHAP values drive the explanation layer in the agent.

The dataset is synthetic so the model achieves near-perfect AUC — in production with real noisy data you'd expect something in the 0.75-0.85 range. The pipeline and explainability are the point, not the number.

---

## Setup

### Prerequisites
- Python 3.12
- AWS account with SageMaker and Bedrock access
- Pinecone account (free tier works)
- Dataset CSV uploaded to S3 at `s3://sagemaker-us-east-2-691210491628/project2/returnsense/data/raw/version1.csv`

### Install dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install langchain langchain-aws langchain-community langchain-text-splitters \
    boto3 pandas scikit-learn streamlit python-dotenv pinecone docx2txt
```

### Environment variables
Create a `.env` file in the project root:
```
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-east-2
SAGEMAKER_ENDPOINT_NAME=returnsense-xgb-endpoint
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=returnsense-policy
BEDROCK_MODEL_ID=us.amazon.nova-lite-v1:0
```

### Download model artifacts from S3
```bash
python3 -c "
import boto3, os
from dotenv import load_dotenv
load_dotenv()
s3 = boto3.client('s3', region_name='us-east-2')
bucket = 'sagemaker-us-east-2-691210491628'
os.makedirs('model', exist_ok=True)
s3.download_file(bucket, 'project2/returnsense/model/encoders.joblib', 'model/encoders.joblib')
s3.download_file(bucket, 'project2/returnsense/model/features.joblib', 'model/features.joblib')
print('Done.')
"
```

### Index the policy document
Only needs to run once. If the Pinecone index already exists with 122 vectors, skip this.
```bash
python3 -c "from agent.policy_tool import build_policy_index; build_policy_index()"
```

### Run the app
```bash
streamlit run streamlit/app.py
```

Opens at http://localhost:8501

---

## Demo

**Score a user's return risk**
```
Look up user 7cd4bbb6 and score their return risk
```

**Ask a policy question**
```
What is the return window for electronics?
```

**Test memory with a follow-up**
```
What is the restocking fee for that category?
```

**Test the guardrail**
```
Process a refund for user 7cd4bbb6
```

---

## Cleanup

Delete the SageMaker endpoint after demo to avoid charges:
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
