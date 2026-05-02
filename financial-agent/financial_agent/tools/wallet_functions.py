from decimal import Decimal


def _money(v) -> Decimal:
    return Decimal(str(v)).quantize(Decimal("0.01"))


def running_balance(initial_balance: Decimal, transactions: list[dict]) -> Decimal:
    total = Decimal(str(initial_balance))
    for txn in transactions:
        amount = Decimal(str(txn["amount"]))
        effect = int(txn.get("effect_on_wallet", 0))
        total += amount * effect
    return _money(total)


def net_worth(wallet_balances: list[Decimal], debt_balances: list[Decimal]) -> Decimal:
    """Calculate net worth: sum of wallet balances minus sum of debt balances.

    Args:
        wallet_balances: list of Decimal values (e.g., [Decimal("1000"), Decimal("500")])
        debt_balances: list of Decimal values (e.g., [Decimal("200")])
    Returns:
        Decimal: assets - debts, quantized to 2 decimal places
    """
    assets = sum((Decimal(str(b)) for b in wallet_balances), Decimal("0"))
    debts = sum((Decimal(str(b)) for b in debt_balances), Decimal("0"))
    return _money(assets - debts)
