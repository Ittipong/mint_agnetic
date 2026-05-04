from decimal import Decimal
from collections import defaultdict


def _to_decimal(value) -> Decimal:
    if value is None:
        return Decimal("0")
    return Decimal(str(value))


def total_expenses(transactions: list[dict]) -> Decimal:
    """Sum of all expense-type transactions (transaction_effect_on_wallet < 0)."""
    return sum(
        (_to_decimal(t["transaction_amount"]) for t in transactions if int(t.get("transaction_effect_on_wallet", 0)) < 0),
        Decimal("0"),
    )


def total_income(transactions: list[dict]) -> Decimal:
    """Sum of all income-type transactions (transaction_effect_on_wallet > 0)."""
    return sum(
        (_to_decimal(t["transaction_amount"]) for t in transactions if int(t.get("transaction_effect_on_wallet", 0)) > 0),
        Decimal("0"),
    )


def net_change(transactions: list[dict]) -> Decimal:
    """Net change across all transactions (income - expense)."""
    return total_income(transactions) - total_expenses(transactions)


def group_by_category(transactions: list[dict]) -> dict[str, list[dict]]:
    """Group transactions by transaction_display_category."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for t in transactions:
        cat = t.get("transaction_display_category") or "Uncategorized"
        grouped[cat].append(t)
    return dict(grouped)


def group_by_date(transactions: list[dict]) -> dict[str, list[dict]]:
    """Group transactions by transaction_date string (YYYY-MM-DD)."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for t in transactions:
        date_str = str(t.get("transaction_date", ""))[:10]
        grouped[date_str].append(t)
    return dict(grouped)


def group_by_currency(transactions: list[dict]) -> dict[str, list[dict]]:
    """Group transactions by transaction_currency_code."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for t in transactions:
        grouped[t.get("transaction_currency_code", "Unknown")].append(t)
    return dict(grouped)


def sum_by_category(transactions: list[dict]) -> dict[str, Decimal]:
    """Sum amounts grouped by transaction_display_category."""
    totals: dict[str, Decimal] = defaultdict(Decimal)
    for t in transactions:
        cat = t.get("transaction_display_category") or "Uncategorized"
        totals[cat] += _to_decimal(t["transaction_amount"])
    return dict(totals)


def sum_by_currency(transactions: list[dict]) -> dict[str, Decimal]:
    """Sum amounts grouped by transaction_currency_code."""
    totals: dict[str, Decimal] = defaultdict(Decimal)
    for t in transactions:
        totals[t.get("transaction_currency_code", "Unknown")] += _to_decimal(t["transaction_amount"])
    return dict(totals)


def filter_by_tags(transactions: list[dict], tag_names: list[str]) -> list[dict]:
    """Filter transactions that have any of the given tag names."""
    tag_set = set(t.lower() for t in tag_names)
    return [
        t for t in transactions
        if any(tag["name"].lower() in tag_set for tag in t.get("transaction_tags", []))
    ]


def filter_by_type(transactions: list[dict], types: str | list[str]) -> list[dict]:
    """Filter transactions by raw transaction_type."""
    if isinstance(types, str):
        types = [types]
    return [t for t in transactions if t.get("transaction_type") in types]


def balance_from_transactions(initial_balance: Decimal, transactions: list[dict]) -> Decimal:
    """Calculate running balance from initial_balance + list of transactions."""
    total = Decimal(str(initial_balance))
    for t in transactions:
        amount = _to_decimal(t["transaction_amount"])
        effect = int(t.get("transaction_effect_on_wallet", 0))
        total += amount * effect
    return total.quantize(Decimal("0.01"))
