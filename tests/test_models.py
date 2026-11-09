import math

import numpy as np
import pytest

from vlmsyc.models import MockVLM, find_yes_no_token_ids, load_model, yes_no_from_logits


def test_yes_no_from_logits_hand_computed():
    # vocab: 0,1 = yes spellings; 2 = no; 3 = other.
    logits = np.log(np.array([0.2, 0.1, 0.3, 0.4]))  # already a distribution
    r = yes_no_from_logits(logits, yes_ids=[0, 1], no_ids=[2])
    assert math.isclose(math.exp(r.logp_yes), 0.3, rel_tol=1e-9)
    assert math.isclose(math.exp(r.logp_no), 0.3, rel_tol=1e-9)
    assert math.isclose(r.p_yes, 0.5) and math.isclose(r.mass, 0.6)


def test_exact_tie_goes_to_no():
    assert yes_no_from_logits(np.array([1.0, 1.0, 0.0]), [0], [1]).answer == "no"


def test_yes_no_shift_invariant_and_answer():
    logits = np.array([3.0, 0.0, 1.0, 5.0])
    a = yes_no_from_logits(logits, [0], [2])
    b = yes_no_from_logits(logits + 100.0, [0], [2])  # softmax ignores constant shifts
    assert math.isclose(a.p_yes, b.p_yes) and math.isclose(a.mass, b.mass)
    assert math.isclose(a.p_yes, 1 / (1 + math.exp(-2.0)))
    assert a.answer == "yes"


def test_non_finite_logits_raise():
    with pytest.raises(FloatingPointError):
        yes_no_from_logits(np.array([np.nan, 0.0, 0.0]), [0], [1])


class FakeTokenizer:
    """Mimics a tokenizer where ' Yes' splits into two tokens."""

    vocab = {"Yes": 10, "yes": 11, " yes": 12, "No": 20, "no": 21, " no": 22, " No": 23}

    def encode(self, text, add_special_tokens=False):
        if text == " Yes":
            return [1, 10]  # two tokens: skipped
        return [self.vocab[text]]


def test_find_yes_no_token_ids_skips_multi_token():
    yes, no = find_yes_no_token_ids(FakeTokenizer())
    assert yes == [10, 11, 12] and no == [20, 23, 21, 22]


def test_mock_is_deterministic_and_prompt_dependent():
    m1, m2 = MockVLM(seed=0), MockVLM(seed=0)
    msg = lambda t: [{"role": "user", "content": [{"type": "text", "text": t}]}]  # noqa: E731
    assert m1.yes_no(msg("a"), None) == m2.yes_no(msg("a"), None)
    ps = {round(m1.yes_no(msg(str(i)), None).p_yes, 6) for i in range(20)}
    assert len(ps) > 15  # different prompts, different answers
    assert MockVLM(seed=1).yes_no(msg("a"), None) != m1.yes_no(msg("a"), None)


def test_load_model_mock():
    assert isinstance(load_model({"name": "m", "hf_id": "mock"}), MockVLM)
