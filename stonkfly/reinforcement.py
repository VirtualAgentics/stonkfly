from .config import D


def reinforcement(equity, anchor, deadband):
    """Incremental marked-to-bid portfolio P&L, including booked trading fees.
    The broker rejects external deposits/withdrawals before this is evaluated.
    This is an engineered stimulus, not a statement that a fly understands money.
    """
    delta = D(equity) - D(anchor)
    threshold = D(deadband)
    kind = (
        "reward"
        if delta >= threshold
        else "aversive"
        if delta <= -threshold
        else "none"
    )
    return kind, delta


def due_fill_mark(fill_anchor, quote_timestamp):
    """Return the post-fill equity mark once a newer quote exists, else None.

    The observation right after a fill can share the fill's own quote (in
    replay it always does), which would compare equity with itself. The mark
    waits for the first observation priced later than the fill.
    """
    if not fill_anchor or quote_timestamp <= fill_anchor["timestamp"]:
        return None
    return fill_anchor["equity"]


def stimulus(mode, equity, anchor, fill_anchor, deadband):
    """Select this observation's reinforcement input.

    equity: the upstream rule, portfolio change since the previous observation,
    which while holding is mostly price drift rather than a consequence of the
    last decision.
    fill: a pulse only on the observation right after the fly's own fill,
    comparing equity with the mark taken immediately after that fill. Every
    other observation gets no pulse. The logged delta is the one that decided.
    """
    if mode == "equity":
        return reinforcement(equity, anchor, deadband)
    if mode != "fill":
        raise ValueError("Unknown reward mode")
    if fill_anchor is None:
        return "none", D(equity) - D(anchor)
    return reinforcement(equity, fill_anchor, deadband)
