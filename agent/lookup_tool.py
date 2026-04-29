import ast
import os
import numpy as np
from databricks import sql
from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv()

def get_connection():
    return sql.connect(
        server_hostname=os.getenv("DATABRICKS_HOSTNAME"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=os.getenv("DATABRICKS_TOKEN"),
    )

def parse_nested_mean(val):
    """Parse nested list strings or plain floats into a mean value."""
    try:
        # try as a plain float first (total column)
        return float(val)
    except (ValueError, TypeError):
        pass
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
    """Look up a user's behavioral features by user_id from the e-commerce dataset.
    Returns the feature row needed for chargeback risk scoring."""

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT * FROM read_files(
                'dbfs:/Volumes/project_2/datalake/landing_zone/final_report.csv',
                format => 'csv',
                header => true
            )
            WHERE user_id = '{user_id}'
            LIMIT 1
        """)
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]
        cursor.close()
        conn.close()

        if not rows:
            return {"error": f"No record found for user_id: {user_id}"}

        row = dict(zip(cols, rows[0]))

        def safe_int(val):
            try: return int(float(val))
            except: return 0

        def safe_float(val):
            try: return float(val)
            except: return 0.0

        return {
            "user_id": user_id,
            "primary_category": str(row.get("primary_category", "unknown")),
            "country": str(row.get("country", "unknown")),
            "favorite_device": str(row.get("favorite_device", "unknown")),
            "purchase_count": safe_int(row.get("purchase_count", 0)),
            "chargeback_count": safe_int(row.get("chargeback_count", 0)),
            "avg_transaction_cost": safe_float(row.get("avg_transaction_cost", 0)),
            "page_view_count": safe_int(row.get("page_view_count", 0)),
            "add_to_cart_count": safe_int(row.get("add_to_cart_count", 0)),
            "remove_from_cart_count": safe_int(row.get("remove_from_cart_count", 0)),
            "click_count": safe_int(row.get("click_count", 0)),
            "search_count": safe_int(row.get("search_count", 0)),
            "login_count": safe_int(row.get("login_count", 0)),
            "logout_count": safe_int(row.get("logout_count", 0)),
            "total": parse_nested_mean(row.get("total", 0)),
            "quantity": parse_nested_mean(row.get("quantity", "[]")),
            "unit_price": parse_nested_mean(row.get("unit_price", "[]")),
        }

    except Exception as e:
        return {"error": f"Lookup failed: {str(e)}"}
