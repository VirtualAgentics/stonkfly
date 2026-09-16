"""Single guarded spot-order action for Coinbase Advanced (an exchange, not CDP wallet).

The action is invoked directly by the fixed neural decoder. No LLM, general
wallet tools, transfers, or agent framework sit between the proposal and the guard.
"""

import time
from typing import Literal

from pydantic import BaseModel, ConfigDict


class Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product: str
    side: Literal["BUY", "SELL"]


class StonkflyActions:
    name = "stonkfly_spot_order"
    description = "Submit a budget-checked, price-bounded Coinbase Advanced spot FOK order from a neural proposal."

    def __init__(self, guard, broker):
        self.guard = guard
        self.broker = broker
        self.quotes = {}

    def invoke(self, args):
        p = Proposal.model_validate(args)
        plan = self.guard.plan(p.product, p.side, self.quotes)
        plan["neural_observation"] = self.guard.l.get("observation")
        plan["checkpoint"] = self.guard.l.get("checkpoint")
        plan = self.guard.l.reserve(plan, time.time())
        return self.broker.execute(plan, self.guard.before_submit)
