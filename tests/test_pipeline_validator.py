import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from gguf_factory import build_gguf, bytes_reader  # noqa: E402

from src import validator  # noqa: E402
from src.gguf_parser import build_record, extract_header  # noqa: E402
from src.validator import (  # noqa: E402
    ModelEntry, Rejected, Unsupported, normalize_record, validate_all,
)


def parsed_record(tmp_path, name="m.gguf", filename_quant="Q8_0", **kwargs):
    path = build_gguf(tmp_path / name, **kwargs)
    data = path.read_bytes()
    header = extract_header(bytes_reader(data))

    model = {
        "repo_id": "bartowski/Test-GGUF",
        "file_name": name,
        "file_size_bytes": len(data),
        "shard_count": 1,
        "quantization": filename_quant,
        "download_url": f"https://huggingface.co/bartowski/Test-GGUF/{name}",
        "downloads": 500,
        "license": "apache-2.0",
        "last_modified": "2026-10-01T00:00:00+00:00",
    }

    record = build_record(model, [header])
    record["file_size_bytes"] = len(data)
    return record


def test_valid_llama_like_record(tmp_path):
    raw = parsed_record(tmp_path, file_type=7, embedding=64, heads=4,
                        kv_heads=2)

    entry = ModelEntry.model_validate(normalize_record(raw))

    assert (entry.layers, entry.attention_heads, entry.kv_heads) == (2, 4, 2)
    assert entry.quantization == "Q8_0"
    assert entry.expert_count == 0 and entry.expert_tensor_bytes == 0
    assert entry.kv_cache_kind == "standard"
    # head dims were absent from the header: derived, and recorded as such
    assert (entry.head_dim_k, entry.head_dim_v) == (16, 16)
    assert entry.derived_fields == ["head_dim_k", "head_dim_v"]


def test_explicit_header_values_are_never_overridden(tmp_path):
    raw = parsed_record(tmp_path, key_length=32, value_length=24)

    entry = normalize_record(raw)

    assert entry["head_dim_k"] == 32 and entry["head_dim_v"] == 24
    assert "head_dim_k" not in entry["derived_fields"]
    assert "head_dim_v" not in entry["derived_fields"]


def test_missing_kv_heads_uses_loader_default_and_says_so(tmp_path):
    raw = parsed_record(tmp_path, kv_heads=None)

    entry = normalize_record(raw)

    assert entry["kv_heads"] == entry["attention_heads"]
    assert "kv_heads" in entry["derived_fields"]


def test_uniform_per_layer_kv_heads_collapse(tmp_path):
    raw = parsed_record(tmp_path, kv_heads=None, kv_heads_array=[2, 2])

    entry = normalize_record(raw)

    assert entry["kv_heads"] == 2
    assert "kv_heads" not in entry["derived_fields"]


def test_varying_per_layer_kv_heads_are_unsupported(tmp_path):
    raw = parsed_record(tmp_path, kv_heads=None, kv_heads_array=[2, 4])

    with pytest.raises(Unsupported, match="varies per layer"):
        normalize_record(raw)


def test_unreviewed_architecture_is_unsupported_not_guessed(tmp_path):
    raw = parsed_record(tmp_path, architecture="mamba")

    with pytest.raises(Unsupported, match="architecture"):
        normalize_record(raw)


def test_mla_is_unsupported(tmp_path):
    raw = parsed_record(tmp_path)
    raw["kv_lora_rank"] = 512

    with pytest.raises(Unsupported, match="MLA"):
        normalize_record(raw)


def test_sliding_window_is_flagged_as_upper_bound(tmp_path):
    raw = parsed_record(tmp_path, architecture="gemma3", sliding_window=512)

    entry = normalize_record(raw)

    assert entry["sliding_window"] == 512
    assert entry["kv_cache_kind"] == "standard_upper_bound"


def test_sliding_window_with_different_head_dims_is_unsupported(tmp_path):
    raw = parsed_record(tmp_path, architecture="gemma3", sliding_window=512,
                        key_length=16, value_length=16)
    raw["key_length_swa"] = 32

    with pytest.raises(Unsupported, match="sliding-window"):
        normalize_record(raw)


def test_moe_record_keeps_expert_facts(tmp_path):
    raw = parsed_record(tmp_path, architecture="qwen3moe", experts=4,
                        experts_used=2)

    entry = normalize_record(raw)

    assert (entry["expert_count"], entry["expert_used_count"]) == (4, 2)
    assert 0 < entry["expert_tensor_bytes"] < entry["tensor_bytes"]


def test_moe_without_used_count_is_rejected(tmp_path):
    raw = parsed_record(tmp_path, architecture="qwen3moe", experts=4,
                        experts_used=2)
    raw["expert_used_count"] = None

    with pytest.raises(Rejected, match="expert_used_count"):
        normalize_record(raw)


def test_file_size_must_match_tensor_table(tmp_path):
    raw = parsed_record(tmp_path)
    raw["file_size_bytes"] += 1_000_000

    with pytest.raises(Rejected, match="inconsistent with tensor table"):
        normalize_record(raw)

    raw["file_size_bytes"] = raw["expected_file_size_bytes"] - 1  # truncated

    with pytest.raises(Rejected, match="inconsistent with tensor table"):
        normalize_record(raw)


def test_missing_required_field_is_rejected_not_defaulted(tmp_path):
    for field in ("layers", "embedding_length", "context_length",
                  "attention_heads"):
        raw = parsed_record(tmp_path)
        raw[field] = None

        with pytest.raises(Rejected):
            normalize_record(raw)


def test_unknown_tensor_type_is_rejected(tmp_path):
    raw = parsed_record(tmp_path)
    raw["tensor_bytes"] = None
    raw["expected_file_size_bytes"] = None

    with pytest.raises(Rejected, match="tensor sizes unavailable"):
        normalize_record(raw)


def test_quantization_mismatch_is_a_warning_header_wins(tmp_path):
    raw = parsed_record(tmp_path, file_type=7, filename_quant="Q4_K_M")

    entry = normalize_record(raw)

    assert entry["quantization"] == "Q8_0"
    assert entry["quantization_label"] == "Q4_K_M"
    assert entry["warnings"]


def test_inconsistent_gqa_heads_are_rejected(tmp_path):
    raw = parsed_record(tmp_path, heads=4, kv_heads=3)

    with pytest.raises(Rejected, match="inconsistent heads"):
        normalize_record(raw)


def test_validate_all_sorts_records_into_three_buckets(tmp_path):
    good = parsed_record(tmp_path, "good.gguf")
    other = parsed_record(tmp_path, "mamba.gguf", architecture="mamba")
    bad = parsed_record(tmp_path, "bad.gguf")
    bad["file_size_bytes"] = 1
    duplicate = dict(good)
    failed = {"repo_id": "x/y", "file_name": "f.gguf",
              "parse_status": "failed", "parse_error": "boom"}

    path = tmp_path / "parsed.json"
    path.write_text(json.dumps([good, other, bad, duplicate, failed]))

    valid, unsupported, rejected = validate_all(path)

    assert len(valid) == 1
    assert len(unsupported) == 1
    assert len(rejected) == 3
    assert any("Duplicate" in r["reason"] for r in rejected)
    assert any("boom" in r["reason"] for r in rejected)
