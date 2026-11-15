import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from gguf_factory import build_gguf, bytes_reader  # noqa: E402

from src import gguf_parser  # noqa: E402
from src.gguf_parser import (  # noqa: E402
    build_record, extract_header, file_type_name, get_architecture_fields,
)


def parse(path):
    data = Path(path).read_bytes()
    return data, extract_header(bytes_reader(data))


def test_header_reproduces_actual_file_size_exactly(tmp_path):
    path = build_gguf(tmp_path / "m.gguf", tokens=60_000)
    data, header = parse(path)

    # The tensor table accounts for every byte of the real file.
    assert header["expected_file_size_bytes"] == len(data)
    assert header["tensor_count"] == 4 - 1  # three default tensors
    assert header["tensor_bytes"] > 0


def test_header_larger_than_initial_window_is_fetched_incrementally(
        tmp_path, monkeypatch):
    path = build_gguf(tmp_path / "m.gguf", tokens=120_000)
    data = path.read_bytes()
    monkeypatch.setattr(gguf_parser, "INITIAL_RANGE", 4096)

    requested = []

    def reader(start, end):
        requested.append((start, end))
        return data[start:end]

    header = extract_header(reader)

    assert header["expected_file_size_bytes"] == len(data)
    assert len(requested) > 1
    # Each request continues where the previous one stopped.
    assert all(a[0] < b[0] for a, b in zip(requested, requested[1:]))


def test_architecture_fields_and_numeric_arrays_are_kept(tmp_path):
    path = build_gguf(tmp_path / "m.gguf", kv_heads=None,
                      kv_heads_array=[2, 2], key_length=16, value_length=24)
    _, header = parse(path)

    fields = get_architecture_fields(header["metadata"])

    assert fields["layers"] == 2
    assert fields["kv_heads"] == [2, 2]          # array not discarded
    assert fields["key_length"] == 16
    assert fields["value_length"] == 24          # K and V kept separately
    assert fields["expert_count"] is None


def test_expert_tensor_bytes_are_separated(tmp_path):
    path = build_gguf(tmp_path / "m.gguf", experts=4, experts_used=2)
    _, header = parse(path)

    assert 0 < header["expert_tensor_bytes"] < header["tensor_bytes"]
    # 4 experts * 32 * 64 elements in Q8_0 (34 bytes per 32 elements)
    assert header["expert_tensor_bytes"] == 4 * 32 * 64 // 32 * 34


def test_file_type_maps_to_label():
    assert file_type_name(15) == "Q4_K_M"
    assert file_type_name(7) == "Q8_0"
    assert file_type_name(1) == "F16"
    assert file_type_name(32) == "BF16"
    assert file_type_name(1024) is None      # GUESSED is not a real type
    assert file_type_name(9999) is None
    assert file_type_name(None) is None


def test_invalid_magic_and_truncation_fail_loudly(tmp_path):
    with pytest.raises(ValueError, match="magic"):
        extract_header(bytes_reader(b"NOPE" + b"\0" * 64))

    data = build_gguf(tmp_path / "m.gguf").read_bytes()

    with pytest.raises(ValueError, match="Truncated"):
        extract_header(bytes_reader(data[:100]))


def test_sharded_model_totals_all_shards(tmp_path):
    a = build_gguf(tmp_path / "a.gguf", split=(0, 2))
    b = build_gguf(tmp_path / "b.gguf", split=(1, 2))

    headers = [parse(a)[1], parse(b)[1]]
    record = build_record({"quantization": "Q8_0"}, headers)

    single = headers[0]
    assert record["tensor_bytes"] == 2 * single["tensor_bytes"]
    assert record["expected_file_size_bytes"] == (
        2 * single["expected_file_size_bytes"]
    )

    with pytest.raises(ValueError, match="split.count"):
        build_record({}, [headers[0], headers[1], headers[1]])
