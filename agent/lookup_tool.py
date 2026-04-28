import pandas as pd
import ast
import numpy as np
from langchain.tools import tool

df = pd.read_csv("data/version1.csv")

def parse_nested_mean(val):
    try:
        parsed = ast.literal_eval(str(val))
        flat = []
        for item in parsed:
            if isinstance(item, list):
                flat.extend([float(x) for x in item])
            else:
                flat.append(float(item))
        return np.mean(flat) if flat else 0.0
    except:
        return 0.0

@tool
def lookup_order(user_id: str) -> dict:
    """Look up a user's order features by user_id from the e-commerce dataset.
    Returns the feature row needed for scoring."""

    row = df[df["user_id"] == user_id]

    if row.empty:
        return {"error": f"No record found for user_id: {user_id}"}

    row = row.iloc[0]

    return {
        "user_id": user_id,
        "primary_category": str(row["primary_category"]),
        "country": str(row["country"]),
        "favorite_device": str(row["favorite_device"]),
        "purchase_count": int(row["purchase_count"]),
        "chargeback_count": int(row["chargeback_count"]),
        "avg_transaction_cost": float(row["avg_transaction_cost"]),
        "page_view_count": int(row["page_view_count"]),
        "add_to_cart_count": int(row["add_to_cart_count"]),
        "remove_from_cart_count": int(row["remove_from_cart_count"]),
        "click_count": int(row["click_count"]),
        "search_count": int(row["search_count"]),
        "login_count": int(row["login_count"]),
        "logout_count": int(row["logout_count"]),
        "total": parse_nested_mean(row["total"]),
        "quantity": parse_nested_mean(row["quantity"]),
        "unit_price": parse_nested_mean(row["unit_price"]),
    }