from decimal import Decimal, ROUND_HALF_UP, getcontext

getcontext().prec = 28

CURRENCY_SCALE = Decimal("0.01")
INTEREST_SCALE = Decimal("0.000001")
PERCENT_SCALE = Decimal("0.0001")


def to_decimal(value) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def money(value) -> Decimal:
    return to_decimal(value).quantize(CURRENCY_SCALE, rounding=ROUND_HALF_UP)


def rate(value) -> Decimal:
    return to_decimal(value).quantize(INTEREST_SCALE, rounding=ROUND_HALF_UP)


def percent(value) -> Decimal:
    return to_decimal(value).quantize(PERCENT_SCALE, rounding=ROUND_HALF_UP)
