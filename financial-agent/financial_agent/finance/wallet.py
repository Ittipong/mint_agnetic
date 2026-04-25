from decimal import Decimal
from .precision import money


def running_balance(initial_balance: Decimal, transactions: list[dict]) -> Decimal:
    total = Decimal(str(initial_balance))
    for txn in transactions:
        amount = Decimal(str(txn["amount"]))
        effect = int(txn.get("effect_on_wallet", 0))
        total += amount * effect
    return money(total)


def net_worth(wallet_balances: list[Decimal], debt_balances: list[Decimal]) -> Decimal:
    assets = sum((Decimal(str(b)) for b in wallet_balances), Decimal("0"))
    debts = sum((Decimal(str(b)) for b in debt_balances), Decimal("0"))
    return money(assets - debts)
