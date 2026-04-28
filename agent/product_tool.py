from langchain.tools import tool

PRODUCT_RULES = {
    "electronics": {
        "return_window_days": 15,
        "restocking_fee_pct": 15,
        "requires_manager_approval_above": 150,
        "notes": "Must be in original packaging. No returns on opened software."
    },
    "clothing": {
        "return_window_days": 30,
        "restocking_fee_pct": 0,
        "requires_manager_approval_above": 500,
        "notes": "Items must have tags attached and be unworn."
    },
    "books": {
        "return_window_days": 30,
        "restocking_fee_pct": 0,
        "requires_manager_approval_above": 999999,
        "notes": "No returns on digital or downloadable content."
    },
    "food": {
        "return_window_days": 7,
        "restocking_fee_pct": 0,
        "requires_manager_approval_above": 999999,
        "notes": "Perishable items not eligible for return."
    },
    "toys": {
        "return_window_days": 30,
        "restocking_fee_pct": 0,
        "requires_manager_approval_above": 999999,
        "notes": "Must be unopened and in original packaging."
    },
    "sports": {
        "return_window_days": 30,
        "restocking_fee_pct": 10,
        "requires_manager_approval_above": 300,
        "notes": "Worn or used items not eligible for return."
    },
    "home": {
        "return_window_days": 30,
        "restocking_fee_pct": 10,
        "requires_manager_approval_above": 400,
        "notes": "Large items may require scheduled pickup."
    },
    "beauty": {
        "return_window_days": 14,
        "restocking_fee_pct": 0,
        "requires_manager_approval_above": 999999,
        "notes": "Opened items not eligible for return for hygiene reasons."
    },
}

@tool
def get_product_policy(category: str) -> dict:
    """Get return policy rules for a specific product category.
    Categories: electronics, clothing, books, food, toys, sports, home, beauty."""

    category = category.lower().strip()

    if category not in PRODUCT_RULES:
        return {
            "error": f"Unknown category: {category}",
            "available_categories": list(PRODUCT_RULES.keys())
        }

    rules = PRODUCT_RULES[category]
    return {
        "category": category,
        "return_window_days": rules["return_window_days"],
        "restocking_fee_pct": rules["restocking_fee_pct"],
        "requires_manager_approval_above": rules["requires_manager_approval_above"],
        "notes": rules["notes"],
        "summary": (
            f"{category.title()} items have a {rules['return_window_days']}-day return window. "
            f"Restocking fee: {rules['restocking_fee_pct']}%. "
            f"Manager approval required for orders over ${rules['requires_manager_approval_above']}."
        )
    }