Project 2 Game Plan
Problem Statement
E-commerce operations teams struggle to proactively identify high-risk return orders before
they are processed. Employees manually review orders without model-driven insight or quick
access to return policy — leading to inconsistent decisions and avoidable losses.
Goal
Build an internal agent that scores any order's return likelihood, retrieves the relevant
return policy based on that order's characteristics (category, value, brand), and gives the
employee a clear recommended next step — all in one conversation.
Demo Story

Employee asks about order ORD-123
Agent looks up the order features
Agent scores it → "78% return likelihood, High Risk"
Agent checks policy → "Electronics over $150 require manager approval before refund processing"
Employee asks "what's our restocking fee policy?" → RAG pulls citation from compendium
Employee asks "compare that to ORD-456" → memory handles it
Employee says "process the refund for me" → guardrail fires

My Stack
LangChain ReAct agent, AWS Bedrock (LLM + embeddings), SageMaker XGBoost endpoint,
FAISS policy RAG, Streamlit
Build Order
Phase 1 — SageMaker

Train and deploy XGBoost endpoint (predicting return likelihood)
Test endpoint with sample row

Phase 2 — Tools

Order lookup tool (CSV → order feature row by order_id)
Scoring tool (feature row → SageMaker → return likelihood score + risk tier + explanation)
Policy RAG tool (chunk .docx → FAISS via Bedrock embeddings → retrieve + cite)
Product info tool (static rules JSON for category/brand quirks)

Phase 3 — Agent

Wire tools into LangChain ReAct agent
Write system prompt (internal operator, not customer bot)
Add conversation memory

Phase 4 — Streamlit

Chat interface (agent)
Metrics placeholder (DE connects tomorrow)

Phase 5 — Demo Prep

Test full demo story end to end
Tear down SageMaker endpoint after demo

Secrets (.env)

AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION=us-east-2
SAGEMAKER_ENDPOINT_NAME
BEDROCK_REGION=us-east-2