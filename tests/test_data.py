import json

import pytest

from vlmsyc.data import image_id_from_filename, load_all_pope, load_image, load_pope, parse_object


def test_parse_object():
    assert parse_object("Is there a snowboard in the image?") == "snowboard"
    assert parse_object("Is there an umbrella in the image?") == "umbrella"
    assert parse_object("Is there a hair drier in the image?") == "hair drier"
    with pytest.raises(ValueError):
        parse_object("Is a dog in the picture?")


def test_image_id_from_filename():
    assert image_id_from_filename("COCO_val2014_000000310196.jpg") == 310196
    with pytest.raises(ValueError):
        image_id_from_filename("310196.jpg")


def test_load_pope_fields(fake_pope):
    pope_dir, image_dir = fake_pope
    qs = load_pope("adversarial", pope_dir, image_dir)
    assert len(qs) == 20 * 6
    q = qs[0]
    assert q.uid == "adversarial:1" and q.pope_split == "adversarial"
    assert q.image_id == 100000 and q.label == "yes" and q.object == "dog"
    assert q.image_path.endswith("COCO_val2014_000000100000.jpg")


def test_uids_unique_across_splits(fake_pope):
    qs = [q for split in load_all_pope(*fake_pope).values() for q in split]
    assert len({q.uid for q in qs}) == len(qs)


def test_load_pope_rejects_bad_rows(tmp_path):
    (tmp_path / "coco_pope_random.json").write_text(
        json.dumps({"question_id": 1, "image": "COCO_val2014_000000000001.jpg", "text": "Is there a dog in the image?", "label": "maybe"})
    )
    with pytest.raises(ValueError, match="label"):
        load_pope("random", tmp_path, tmp_path)


def test_load_image_shrinks(tmp_path):
    from PIL import Image

    Image.new("RGB", (1000, 500)).save(tmp_path / "x.jpg")
    assert load_image(tmp_path / "x.jpg", max_side=200).size == (200, 100)
    assert load_image(tmp_path / "x.jpg").size == (1000, 500)
