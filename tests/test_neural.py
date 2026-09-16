import os

import numpy as np
import pandas as pd
import pytest

from stonkfly.config import Settings
from stonkfly.neural.controller import Decoder
from stonkfly.neural.rule import advance
from stonkfly.reinforcement import reinforcement


def test_fixed_neuron_decoder():
    a = pd.DataFrame(
        {"type": ["DNp20", "DNp20", "DNpe017"], "somaSide": ["L", "R", "L"]}
    )
    d = Decoder(np.array([1, 2, 3]), a, 2)
    assert d.decode(np.array([0, 10, 0]), 0.5)["side"] == "HOLD"
    assert d.decode(np.array([0, 10, 1]), 0.5)["side"] == "BUY"
    assert d.decode(np.array([10, 0, 1]), 0.5)["side"] == "SELL"
    assert d.decode(np.array([4, 4, 1]), 0.5)["side"] == "HOLD"


def test_centered_decoder_removes_persistent_bias_without_lookahead():
    a = pd.DataFrame(
        {"type": ["DNp20", "DNp20", "DNpe017"], "somaSide": ["L", "R", "L"]}
    )
    fixed = Decoder(np.array([1, 2, 3]), a, 2)
    centered = Decoder(np.array([1, 2, 3]), a, 2, center="ema", window=5)
    biased = np.array([0, 10, 1])  # Persistent +20 Hz right bias with the gate open.
    first = centered.decode(biased, 0.5)
    assert first["side"] == "BUY" and first["decoder_baseline_hz"] == 0.0
    assert first["difference_hz"] == first["raw_difference_hz"] == 20.0
    sides = [centered.decode(biased, 0.5)["side"] for _ in range(20)]
    assert sides[-1] == "HOLD" and all(
        fixed.decode(biased, 0.5)["side"] == "BUY" for _ in range(3)
    )
    # A genuine deviation from the learned baseline still decodes.
    assert centered.decode(np.array([0, 20, 1]), 0.5)["side"] == "BUY"
    assert centered.decode(np.array([10, 0, 1]), 0.5)["side"] == "SELL"
    with pytest.raises(ValueError):
        Decoder(np.array([1, 2, 3]), a, 2, center="median")


@pytest.mark.parametrize(
    "equity,expected",
    [("100.03", "reward"), ("99.97", "aversive"), ("100.001", "none"), ("100", "none")],
)
def test_explicit_feedback(equity, expected):
    assert reinforcement(equity, "100", ".01")[0] == expected


@pytest.mark.parametrize(
    "mode,equity,anchor,fill_anchor,expected",
    [
        ("equity", "100.05", "100", None, ("reward", "0.05")),
        ("equity", "100.05", "100", "99", ("reward", "0.05")),  # Ignores fill mark.
        ("fill", "100.05", "100", None, ("none", "0.05")),  # No own fill: no pulse.
        ("fill", "100.05", "100", "100.10", ("aversive", "-0.05")),
        ("fill", "100.05", "99", "100.00", ("reward", "0.05")),
        ("fill", "100.005", "99", "100.00", ("none", "0.005")),
    ],
)
def test_stimulus_modes(mode, equity, anchor, fill_anchor, expected):
    from stonkfly.config import D
    from stonkfly.reinforcement import stimulus

    kind, delta = stimulus(mode, equity, anchor, fill_anchor, ".01")
    assert (kind, delta) == (expected[0], D(expected[1]))


def test_fill_mark_waits_for_a_newer_quote():
    from stonkfly.reinforcement import due_fill_mark

    mark = {"equity": "99.9", "timestamp": 120.0}
    assert due_fill_mark(None, 500.0) is None
    assert due_fill_mark(mark, 120.0) is None  # Same candle as the fill.
    assert due_fill_mark(mark, 180.0) == "99.9"


def test_stimulus_rejects_unknown_mode():
    from stonkfly.reinforcement import stimulus

    with pytest.raises(ValueError):
        stimulus("profit", "100", "100", None, ".01")


def trace_protocol(order, frozen=False):
    k = np.zeros(2)
    d = np.zeros(1)
    u = np.zeros(2)
    w = np.zeros(2)
    gain = np.ones((1, 2))
    for phase in order:
        for _ in range(20):
            kh = np.array([20.0, 0.0]) if phase == "cue" else np.zeros(2)
            dh = np.array([30.0]) if phase == "reinforce" else np.zeros(1)
            advance(k, d, u, w, kh, dh, gain, 0.01, 0.001, frozen=frozen)
    return w


def test_memory_rule_temporal_specificity():
    paired = trace_protocol(["cue", "reinforce"])
    reverse = trace_protocol(["reinforce", "cue"])
    assert paired[0] < 0 and reverse[0] > 0
    assert paired[1] == 0 and reverse[1] == 0  # Unactivated input is unchanged.
    assert np.array_equal(trace_protocol(["cue", "reinforce"], True), np.zeros(2))


@pytest.mark.skipif(
    os.environ.get("STONKFLY_FULL_TEST") != "1",
    reason="Downloads/uses full MaleCNS; explicit integration test",
)
def test_full_graph_sensory_reinforcement_checkpoint(tmp_path):
    from stonkfly.data import verify
    from stonkfly.neural.controller import FlyController

    assert verify()["neurons"] == 166700
    c = FlyController(Settings())
    assert len(c.brain.post) == 25582938 and len(c.brain.circuit["edges"]) == 7835
    assert len(c.brain.retina) == 3335 and len(c.brain.r8) == 811
    white = np.full((180, 320, 3), 255, np.uint8)
    for _ in range(3):
        c.observe(white, "none")
    c.save(tmp_path / "before.npz")
    before = c.brain.weight[c.brain.circuit["edges"]].copy()
    reward = c.observe(white, "reward")
    assert reward["reward_spikes"] > 0 and reward["stimulus_ms"] == 200
    assert reward["KC_spikes"] > 0 and reward["memory"]["changed_edges"] > 0
    reward_weights = c.brain.weight[c.brain.circuit["edges"]].copy()
    c.restore(tmp_path / "before.npz")
    control = c.observe(white, "none")
    assert not np.array_equal(reward_weights, c.brain.weight[c.brain.circuit["edges"]])
    c.restore(tmp_path / "before.npz")
    c.brain.weights_frozen = True
    c.observe(white, "reward")
    assert np.array_equal(before, c.brain.weight[c.brain.circuit["edges"]])
    c.restore(tmp_path / "before.npz")
    loss = c.observe(white, "aversive")
    assert loss["aversive_spikes"] > 0 and loss["stimulus_ms"] == 200
    assert np.isfinite(c.brain.weight).all()
    print({"reward": reward, "unpaired_control": control, "loss": loss})
