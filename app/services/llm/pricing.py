from decimal import Decimal

# USD per 1M tokens: (input, output). Ollama Cloud is subscription-based (no per-token
# price), so cost stays NULL unless you add a model here.
PRICING: dict[str, tuple[Decimal, Decimal]] = {}


def calc_cost(model: str, input_tokens: int, output_tokens: int) -> Decimal | None:
    price = PRICING.get(model)
    if price is None:
        return None
    return (price[0] * input_tokens + price[1] * output_tokens) / Decimal(1_000_000)
