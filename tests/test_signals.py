import math

import numpy as np
import pytest
from PIL import Image

from vlmsyc.signals import (
    blank_image, blur_image, conf1, logit, p_of_answer, reliance, shift, shift_to_claim, signal_rows,
)
from vlmsyc.stats import auroc, cluster_bootstrap_ci


def test_conf_and_p_of_answer():
    assert conf1(0.2) == 0.8 and conf1(0.7) == 0.7
    assert p_of_answer(0.2, "yes") == 0.2 and p_of_answer(0.2, "no") == 0.8
    with pytest.raises(ValueError):
        p_of_answer(0.2, "maybe")


def test_reliance_uses_a1_from_real_image():
    # a1 = yes with p 0.9; blurred image gives p(yes) 0.6 -> reliance 0.3
    assert reliance(0.9, 0.6, "yes") == pytest.approx(0.3)
    # a1 = no (p(no) 0.8); blank gives p(no) 0.5 -> reliance 0.3
    assert reliance(0.2, 0.5, "no") == pytest.approx(0.3)
    # degraded image makes the model MORE sure -> negative reliance
    assert reliance(0.9, 0.95, "yes") < 0


def test_shift_signs():
    assert shift(0.5, 0.5) == 0
    assert shift(0.5, 0.9) == pytest.approx(math.log(9))
    # a1 = yes, model moves towards no (the claim): positive shift_to_claim
    assert shift_to_claim(0.9, 0.3, "yes") > 0
    assert shift_to_claim(0.1, 0.7, "no") > 0
    assert shift_to_claim(0.9, 0.95, "yes") < 0
    assert math.isfinite(logit(0.0)) and math.isfinite(logit(1.0))


def test_degradations_keep_size_and_remove_content():
    img = Image.new("RGB", (100, 50), (0, 0, 0))
    img.paste((255, 255, 255), (40, 0, 60, 50))  # a white bar
    blank = blank_image(img)
    assert blank.size == img.size and len(np.unique(np.asarray(blank).reshape(-1, 3), axis=0)) == 1
    blurred = blur_image(img, 0.03)
    assert blurred.size == img.size
    # the sharp edge gets softened: some pixel is now strictly grey
    assert ((np.asarray(blurred) > 0) & (np.asarray(blurred) < 255)).any()


def test_signal_rows_join():
    base = [{
        "uid": "random:1", "question_id": 1, "image_id": 7, "pope_split": "random", "partition": "dev",
        "model": "m", "pushback_type": "assertive", "paraphrase_id": 0, "condition": "base",
        "label": "yes", "p1_yes": 0.9, "a1": "yes", "p2_yes": 0.2, "a2": "no",
        "a1_correct": True, "a2_correct": False, "flip": True, "seed": 0, "git_commit": "x",
    }]
    rows = signal_rows(base, {("random:1", "blur"): 0.6})
    r = rows[0]
    assert r["conf1"] == 0.9 and r["reliance_blur"] == pytest.approx(0.3)
    assert r["reliance_blank"] is None  # not computed yet
    assert r["shift_to_claim"] > 0 and r["good_revision"] is False


def test_auroc_hand_computed():
    assert auroc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert auroc([4, 3, 2, 1], [0, 0, 1, 1]) == 0.0
    assert auroc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5  # all tied
    # pairs (pos, neg): (2 vs 1) win, (2 vs 3) lose, (4 vs 1) win, (4 vs 3) win -> 3/4
    assert auroc([1, 2, 3, 4], [0, 1, 0, 1]) == 0.75
    assert auroc([1, 2], [1, 1]) is None


def test_cluster_bootstrap_ci_contains_estimate():
    rng = np.random.default_rng(0)
    s = rng.normal(size=400)
    y = s + rng.normal(size=400) > 0
    clusters = np.repeat(np.arange(100), 4)
    lo, hi = cluster_bootstrap_ci(lambda idx: auroc(s[idx], y[idx]), clusters, n_boot=200)
    est = auroc(s, y)
    assert lo < est < hi and 0.5 < lo
