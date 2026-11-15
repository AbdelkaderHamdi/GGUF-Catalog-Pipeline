import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_pipeline_validator import parsed_record  # noqa: E402

from src import catalog_builder, validator  # noqa: E402


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    records = [
        parsed_record(tmp_path, "b.gguf"),
        parsed_record(tmp_path, "a.gguf", architecture="qwen3moe",
                      experts=4, experts_used=2),
        parsed_record(tmp_path, "skip.gguf", architecture="mamba"),
    ]

    parsed = tmp_path / "parsed_models.json"
    parsed.write_text(json.dumps(records))

    monkeypatch.setattr(validator, "INPUT_PATH", parsed)
    monkeypatch.setattr(validator, "REPORT_PATH", tmp_path / "report.json")
    monkeypatch.setattr(catalog_builder, "OUTPUT_PATH",
                        tmp_path / "out" / "catalog.json")
    monkeypatch.setattr(catalog_builder, "CHECKSUM_PATH",
                        tmp_path / "out" / "catalog.sha256")
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1800000000")

    return tmp_path


def test_catalog_contents_checksum_and_exclusions(workspace):
    catalog = catalog_builder.build_catalog(min_valid=2)

    out = workspace / "out"
    content = (out / "catalog.json").read_bytes()
    digest = hashlib.sha256(content).hexdigest()

    assert (out / "catalog.sha256").read_text() == (
        f"{digest}  catalog.json\n"
    )
    assert catalog["schema_version"] == "2.0.0"
    assert catalog["model_count"] == 2
    assert [m["file_name"] for m in catalog["models"]] == [
        "a.gguf", "b.gguf"
    ]

    report = json.loads((workspace / "report.json").read_text())
    assert len(report["unsupported"]) == 1
    assert "mamba" in report["unsupported"][0]["reason"]


def test_build_is_byte_for_byte_reproducible(workspace):
    catalog_builder.build_catalog(min_valid=2)
    first = (workspace / "out" / "catalog.json").read_bytes()

    catalog_builder.build_catalog(min_valid=2)
    second = (workspace / "out" / "catalog.json").read_bytes()

    assert first == second


def test_too_few_models_aborts_before_publishing(workspace):
    with pytest.raises(RuntimeError, match="Not enough valid models"):
        catalog_builder.build_catalog(min_valid=3)

    assert not (workspace / "out" / "catalog.json").exists()


def test_catalog_never_stores_kv_cache_precision(workspace):
    catalog = catalog_builder.build_catalog(min_valid=2)

    for model in catalog["models"]:
        assert not any("cache" in key and "kind" not in key
                       for key in model)
        assert not any(key in model for key in (
            "kv_cache_bytes", "kv_cache_type", "bytes_per_element",
        ))
