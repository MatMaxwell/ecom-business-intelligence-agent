import os
import boto3
import joblib
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool

runtime = boto3.client("sagemaker-runtime", region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-2"))
ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT_NAME", "returnsense-xgb-endpoint")

encoders = joblib.load("model/encoders.joblib")
FEATURES = joblib.load("model/features.joblib")

SHAP_CONTEXT = {
    "avg_transaction_cost": "higher average spend strongly increases return risk",
    "chargeback_count": "dispute history is a strong signal of return behavior",
    "unit_price": "expensive items are returned more frequently",
    "quantity": "bulk purchases correlate with higher return likelihood",
    "purchase_count": "more purchases means more exposure to returns",
    "total": "higher total order value increases return risk",
}

def get_risk_tier(prob):
    if prob >= 0.7:
        return "High Risk"
    elif prob >= 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"

@tool
def score_order(order_features: dict) -> dict:
    """Score a user's return risk using the deployed SageMaker XGBoost endpoint.
    Input should be a dict of order features from lookup_order."""

    try:
        for col, le in encoders.items():
            if col in order_features:
                order_features[col] = int(le.transform([str(order_features[col])])[0])

        row = [str(order_features.get(f, 0)) for f in FEATURES]
        payload = ",".join(row)

        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="text/csv",
            Body=payload
        )

        prob = float(response["Body"].read().decode("utf-8").strip())
        tier = get_risk_tier(prob)
        pred = 1 if prob >= 0.5 else 0

        top_factors = [SHAP_CONTEXT[f] for f in SHAP_CONTEXT if f in order_features][:3]

        return {
            "probability": round(prob, 4),
            "prediction": pred,
            "risk_tier": tier,
            "label": "RETURN LIKELY" if pred == 1 else "RETURN UNLIKELY",
            "top_factors": top_factors,
            "suggested_action": (
                "Flag for manager review before processing refund"
                if tier == "High Risk"
                else "Standard processing applies"
            )
        }

    except Exception as e:
        return {"error": str(e)}