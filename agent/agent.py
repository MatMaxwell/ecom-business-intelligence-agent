import os
from dotenv import load_dotenv
load_dotenv()

from langchain_aws import ChatBedrockConverse
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agent.lookup_tool import lookup_order
from agent.scoring_tool import score_order
from agent.policy_tool import query_policy
from agent.product_tool import get_product_policy

llm = ChatBedrockConverse(
    model=os.getenv("BEDROCK_MODEL_ID", "us.amazon.nova-lite-v1:0"),
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-2"),
    temperature=0.0
)

tools = [lookup_order, score_order, query_policy, get_product_policy]

SYSTEM_PROMPT = """You are an internal business agent for an e-commerce operations team.
You help ops staff, risk analysts, and support leads investigate orders and make informed decisions.

IMPORTANT RULES:
- Always use tools to get real data. Never invent scores or policy.
- You cannot process refunds, cancel orders, or make account changes.
- If asked to execute a transaction, politely decline.
- For high risk orders, always recommend manager review.
- Keep responses concise and business-focused.
- This tool is for internal employees only, not customers.

When scoring an order always:
1. Look up the user first with lookup_order
2. Score it with score_order using the returned features
3. Explain the top risk factors
4. Reference relevant policy if needed"""

def get_agent_executor():
    return create_react_agent(
        llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
    )

def run_agent(user_input: str, agent_executor) -> str:
    try:
        result = agent_executor.invoke({"messages": [{"role": "user", "content": user_input}]})
        return result["messages"][-1].content
    except Exception as e:
        return f"Agent error: {str(e)}"