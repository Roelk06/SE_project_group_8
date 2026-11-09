"""
Unit tests for sorting results in input_router.py: spam goes to results/spam, everything
else is let through to results/ham. They use temporary folders and a fake route_input, so
they need no saved models.
"""
import json

import pytest

import input_router
from input_router import route_and_sort, save_result

pytestmark = pytest.mark.unit  # every test in this file is a unit test


def read_record(result):
    return json.loads(result["record_path"].read_text(encoding="utf-8"))


def fake_router(monkeypatch, label):
    monkeypatch.setattr(input_router, "route_input", lambda input_type, data: (label, 0.9))


@pytest.mark.parametrize(
    "label, folder, allowed",
    [
        ("spam", "spam", False),
        ("phishing", "spam", False),
        ("ham", "ham", True),
        ("legitimate", "ham", True),
    ],
)
def test_label_goes_to_the_right_folder(label, folder, allowed, tmp_path):
    """
    Purpose: every label the router can return ends up in the right folder.
    Test data: an SMS text with each label.
    Expected: spam and phishing are blocked in spam/, ham and legitimate are allowed in ham/.
    """
    result = save_result("sms", "some text", label, tmp_path)

    assert result["label"] == folder
    assert result["allowed"] is allowed
    assert result["record_path"].parent == tmp_path / folder
    assert read_record(result)["content"] == "some text"


@pytest.mark.parametrize("label", ["unknown", "", None])
def test_unknown_label_is_not_let_through(label, tmp_path):
    """
    Purpose: a broken label must never count as not spam.
    Test data: an unknown label, an empty label and None.
    Expected: a KeyError, and nothing is saved.
    """
    with pytest.raises(KeyError):
        save_result("sms", "some text", label, tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_unsupported_input_type_saves_nothing(tmp_path):
    """
    Purpose: the router's own check stops unsupported types before anything is saved.
    Test data: the input type "video".
    Expected: a ValueError from route_input, and nothing is saved.
    """
    with pytest.raises(ValueError):
        route_and_sort("video", "some text", tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_email_content_is_saved(tmp_path):
    """
    Purpose: an email keeps its subject and message, including non-English characters.
    Test data: an email dict with a pound sign in the message.
    Expected: the record contains the same dict.
    """
    email = {"subject": "YOU HAVE WON", "message": "Claim your £5000 now"}

    result = save_result("email", email, "spam", tmp_path)

    assert read_record(result)["content"] == email


def test_image_is_copied_and_original_is_kept(tmp_path, monkeypatch):
    """
    Purpose: a spam image is copied to spam/ without touching the original, also when the
    input type is written in capitals (the router accepts that).
    Test data: a small fake .JPG file, sent as input type "IMAGE".
    Expected: the original still exists, the copy has the same bytes and is named in the record.
    """
    fake_router(monkeypatch, "spam")
    image = tmp_path / "upload.JPG"
    image.write_bytes(b"fake image bytes")
    results = tmp_path / "results"

    result = route_and_sort("IMAGE", image, results)

    copy = results / "spam" / read_record(result)["stored_file"]
    assert image.exists()
    assert copy.suffix == ".jpg"
    assert copy.read_bytes() == b"fake image bytes"


def test_missing_image_saves_nothing(tmp_path):
    """
    Purpose: a failed image copy must not leave a record without an image.
    Test data: a path to an image that does not exist.
    Expected: a FileNotFoundError, and the folder has no files.
    """
    results = tmp_path / "results"

    with pytest.raises(FileNotFoundError):
        save_result("image", tmp_path / "missing.jpg", "spam", results)

    assert [p for p in results.rglob("*") if p.is_file()] == []


def test_identical_inputs_do_not_overwrite_each_other(tmp_path):
    """
    Purpose: two results saved at the same moment are both kept.
    Test data: the same SMS text saved twice in a row.
    Expected: two different record files.
    """
    first = save_result("sms", "same text", "spam", tmp_path)
    second = save_result("sms", "same text", "spam", tmp_path)

    assert first["record_path"] != second["record_path"]
    assert len(list((tmp_path / "spam").iterdir())) == 2


def test_route_and_sort_sorts_on_the_router_output(tmp_path, monkeypatch):
    """
    Purpose: route_and_sort sorts on what the router returns.
    Test data: a fake route_input that always returns "phishing", for a URL.
    Expected: the URL is blocked in spam/.
    """
    fake_router(monkeypatch, "phishing")

    result = route_and_sort("url", "http://example.com", tmp_path)

    assert result["allowed"] is False
    assert result["record_path"].parent == tmp_path / "spam"
