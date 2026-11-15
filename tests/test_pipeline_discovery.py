from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.discover import evaluate_repository, extract_license
from src.file_discovery import detect_quantization, group_gguf_files


@pytest.mark.parametrize("name, expected", [
    ("Llama-3.1-8B-Instruct-Q4_K_M.gguf", "Q4_K_M"),
    ("Llama-3.1-8B-Instruct-Q5_K_S.gguf", "Q5_K_S"),
    ("Qwen3-8B-UD-Q4_K_XL.gguf", "Q4_K_XL"),
    ("Model-Q8_0.gguf", "Q8_0"),
    ("Model-IQ4_XS.gguf", "IQ4_XS"),
    ("Model-Q6_K_L.gguf", "Q6_K_L"),
    ("Model-BF16.gguf", "BF16"),
    ("Model-F16.gguf", "F16"),
    ("gpt-oss-20b-MXFP4.gguf", "MXFP4"),
    ("sub/dir/Model-Q4_K_M-00001-of-00003.gguf", "Q4_K_M"),
    ("Qwen3-8B.gguf", "UNKNOWN"),
])
def test_quantization_label_is_complete(name, expected):
    assert detect_quantization(name) == expected


def test_shards_are_grouped_and_sizes_summed():
    entries = [
        ("M-Q4_K_M-00002-of-00003.gguf", 20),
        ("M-Q4_K_M-00001-of-00003.gguf", 30),
        ("M-Q4_K_M-00003-of-00003.gguf", 10),
        ("Small-Q8_0.gguf", 5),
    ]

    groups, skipped = group_gguf_files(entries)

    assert skipped == []
    sharded = next(g for g in groups if g["file_name"].startswith("M-"))
    assert sharded["file_name"] == "M-Q4_K_M-00001-of-00003.gguf"
    assert sharded["file_size_bytes"] == 60
    assert len(sharded["files"]) == 3


def test_incomplete_shards_and_auxiliary_files_are_skipped():
    entries = [
        ("M-00001-of-00003.gguf", 1),
        ("M-00003-of-00003.gguf", 1),
        ("mmproj-F16.gguf", 1),
        ("Good-Q4_0.gguf", 1),
    ]

    groups, skipped = group_gguf_files(entries)

    assert [g["file_name"] for g in groups] == ["Good-Q4_0.gguf"]
    assert {reason for _, reason in skipped} == {
        "incomplete shard set", "auxiliary file"
    }


def make_repo(**overrides):
    base = dict(
        id="bartowski/Model-GGUF",
        downloads=1000,
        last_modified=datetime.now(timezone.utc) - timedelta(days=10),
        tags=["license:apache-2.0"],
        card_data=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


FILTERS = dict(
    publishers={"bartowski"},
    min_downloads=100,
    cutoff=datetime.now(timezone.utc) - timedelta(days=365),
    accepted_licenses={"apache-2.0", "mit"},
)


def test_repository_filters_accept_and_reject():
    record, reason = evaluate_repository(make_repo(), **FILTERS)
    assert reason is None and record["license"] == "apache-2.0"

    cases = [
        (make_repo(id="someone/else"), "publisher"),
        (make_repo(downloads=5), "downloads"),
        (make_repo(last_modified=datetime(2020, 1, 1, tzinfo=timezone.utc)),
         "old"),
        (make_repo(tags=["license:other"]), "licence"),
        (make_repo(tags=[]), "licence"),
    ]

    for repo, word in cases:
        record, reason = evaluate_repository(repo, **FILTERS)
        assert record is None and word in reason


def test_multiple_licences_are_not_guessed():
    card = SimpleNamespace(license=["mit", "apache-2.0"])
    assert extract_license(make_repo(card_data=card, tags=[])) is None
