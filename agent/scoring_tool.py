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

# Population benchmarks for contextualizing user values
BENCHMARKS = {
    "avg_transaction_cost": 847.0,
    "chargeback_count":     2.1,
    "purchase_count":       150.0,  # updated — aggregated user-level count
    "total":                1200.0,
    "unit_price":           220.0,
    "quantity":             3.0,
}

def get_risk_tier(prob):
    if prob >= 0.65:
        return "High Risk"
    elif prob >= 0.35:
        return "Medium Risk"
    else:
        return "Low Risk"

def build_explanation(order_features: dict) -> list:
    """Generate specific, data-driven risk factor explanations for this user."""
    explanations = []

    cb = order_features.get("chargeback_count", 0)
    cb_bench = BENCHMARKS["chargeback_count"]
    if cb > cb_bench:
        explanations.append(
            f"Chargeback count of {cb} is {cb/cb_bench:.1f}x the average ({cb_bench:.1f}) — "
            f"strong signal of dispute history"
        )

    atc = order_features.get("avg_transaction_cost", 0)
    atc_bench = BENCHMARKS["avg_transaction_cost"]
    if atc > atc_bench:
        explanations.append(
            f"Average transaction cost of \\${atc:,.2f} is above the \\${atc_bench:,.0f} average — "
            f"higher spend correlates with higher return likelihood"
        )

    pc = order_features.get("purchase_count", 0)
    pc_bench = BENCHMARKS["purchase_count"]
    if pc > pc_bench * 1.5:
        explanations.append(
            f"Purchase count of {pc} is {pc/pc_bench:.1f}x the average ({pc_bench:.0f}) — "
            f"high activity increases return exposure"
        )

    total = order_features.get("total", 0)
    total_bench = BENCHMARKS["total"]
    if total > total_bench:
        explanations.append(
            f"Total order value of ${total:,.2f} exceeds the ${total_bench:,.0f} average — "
            f"higher order value increases return risk"
        )

    up = order_features.get("unit_price", 0)
    up_bench = BENCHMARKS["unit_price"]
    if up > up_bench:
        explanations.append(
            f"Average unit price of ${up:,.2f} is above the ${up_bench:,.0f} average — "
            f"expensive items are returned more frequently"
        )

    qty = order_features.get("quantity", 0)
    qty_bench = BENCHMARKS["quantity"]
    if qty > qty_bench:
        explanations.append(
            f"Average quantity per order of {qty:.1f} exceeds the {qty_bench:.1f} average — "
            f"bulk purchases correlate with higher return likelihood"
        )

    # fallback if nothing is elevated
    if not explanations:
        explanations.append(
            f"Chargeback count: {cb}, avg transaction cost: ${atc:,.2f}, "
            f"purchase count: {pc} — all near or below average"
        )

    return explanations[:3]


@tool
def score_order(order_features: dict) -> dict:
    """Score a user's return risk using the deployed SageMaker XGBoost endpoint.
    Input should be a dict of order features from lookup_order."""

    try:
        features_copy = dict(order_features)

        for col, le in encoders.items():
            if col in features_copy:
                features_copy[col] = int(le.transform([str(features_copy[col])])[0])

        row = [str(features_copy.get(f, 0)) for f in FEATURES]
        payload = ",".join(row)

        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="text/csv",
            Body=payload
        )

        prob = float(response["Body"].read().decode("utf-8").strip())
        tier = get_risk_tier(prob)
        pred = 1 if prob >= 0.5 else 0

        explanations = build_explanation(order_features)

        return {
            "user_id": order_features.get("user_id", "unknown"),
            "probability": round(prob, 4),
            "prediction": pred,
            "risk_tier": tier,
            "label": "RETURN LIKELY" if pred == 1 else "RETURN UNLIKELY",
            "top_factors": explanations,
            "user_snapshot": {
                "chargeback_count": order_features.get("chargeback_count"),
                "avg_transaction_cost": order_features.get("avg_transaction_cost"),
                "purchase_count": order_features.get("purchase_count"),
                "primary_category": order_features.get("primary_category"),
                "favorite_device": order_features.get("favorite_device"),
            },
            "suggested_action": (
                "Flag for manager review before processing any refund"
                if tier == "High Risk"
                else "Standard processing applies"
            )
        }

    except Exception as e:
        return {"error": str(e)}
