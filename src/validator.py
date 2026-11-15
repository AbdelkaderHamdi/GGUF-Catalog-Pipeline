"""Validate parsed GGUF records and normalise them into catalog entries.

Every record ends in exactly one of three states:

* valid       - safe for the estimator; goes into the catalog
* unsupported - a real model whose memory behaviour the estimator does not
                model (unreviewed architecture, MLA, per-layer variation).
                Excluded, never guessed.
* rejected    - malformed or inconsistent data.

Where a field is missing from the header, the only fallbacks used are the
documented llama.cpp loader defaults, and each one is listed in the entry's
"derived_fields" so the provenance is never lost.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError


ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "output" / "parsed_models.json"
REPORT_PATH = ROOT / "output" / "validation_report.json"

# Architectures whose KV cache is
#   2-matrix-per-layer: layers * kv_heads * (head_dim_k + head_dim_v) * ctx
# Add an architecture only after checking that formula against llama.cpp.
# Deliberately absent: deepseek2 (MLA), mamba/rwkv/jamba and hybrids
# (qwen3next, nemotron_h, lfm2, falcon-h1, granitehybrid), llama4
# (chunked attention), gemma3n (shared KV), clip/mmproj, embedding models.
SUPPORTED_ARCHITECTURES = {
    "llama", "qwen2", "qwen2moe", "qwen3", "qwen3moe", "phi3",
    "gemma", "gemma2", "gemma3", "granite", "olmo", "olmo2",
    "starcoder2", "glm4", "exaone", "smollm3", "gpt-oss", "cohere2",
    "command-r", "internlm2", "stablelm",
}

QUANT_FAMILY = re.compile(r"^(?:IQ|TQ|Q|BF|F|MXFP)\d+")


class Rejected(Exception):
    """Record is malformed or internally inconsistent."""


class Unsupported(Exception):
    """Record is valid but outside what the estimator models."""


class ModelEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    repo_id: str
    file_name: str
    download_url: HttpUrl
    shard_count: int = Field(ge=1)
    file_size_bytes: int = Field(gt=0)

    quantization: str
    quantization_label: str
    file_type_name: Optional[str] = None

    architecture: str
    model_name: Optional[str] = None
    size_label: Optional[str] = None
    parameters: Optional[int] = None
    tensor_params: int = Field(gt=0)

    layers: int = Field(gt=0)
    embedding_length: int = Field(gt=0)
    context_length: int = Field(gt=0)
    attention_heads: int = Field(gt=0)
    kv_heads: int = Field(gt=0)
    head_dim_k: int = Field(gt=0)
    head_dim_v: int = Field(gt=0)
    sliding_window: Optional[int] = Field(default=None, gt=0)
    kv_cache_kind: str  # "standard" | "standard_upper_bound"

    expert_count: int = Field(ge=0)
    expert_used_count: int = Field(ge=0)
    tensor_bytes: int = Field(gt=0)
    expert_tensor_bytes: int = Field(ge=0)

    derived_fields: List[str] = []
    warnings: List[str] = []

    downloads: int = Field(default=0, ge=0)
    license: Optional[str] = None
    last_modified: Optional[str] = None


def stable_id(repo_id, filename):
    raw = f"{repo_id}\n{filename}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def _int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) \
            or value < minimum:
        raise Rejected(f"{name} missing or invalid: {value!r}")

    return value


def _collapse(value, name, layers):
    """Per-layer array -> scalar when uniform; unsupported when it varies."""

    if not isinstance(value, list):
        return value

    if len(value) != layers:
        raise Rejected(
            f"{name} array has {len(value)} entries for {layers} layers"
        )

    if len(set(value)) != 1:
        raise Unsupported(f"{name} varies per layer")

    return value[0]


def _quant_family(label):
    match = QUANT_FAMILY.match(label or "")

    return match.group(0) if match else None


def normalize_record(raw):
    """Turn one parsed record into a catalog entry dict.

    Raises Unsupported or Rejected; pure, no I/O.
    """

    architecture = raw.get("architecture")

    if not isinstance(architecture, str) or not architecture:
        raise Rejected("general.architecture missing")

    if architecture not in SUPPORTED_ARCHITECTURES:
        raise Unsupported(f"architecture not supported: {architecture}")

    if raw.get("kv_lora_rank") is not None:
        raise Unsupported("multi-head latent attention (MLA) KV cache")

    derived = []
    warnings = []

    layers = _int(raw.get("layers"), "block_count")
    embedding = _int(raw.get("embedding_length"), "embedding_length")
    context = _int(raw.get("context_length"), "context_length")

    heads = _int(
        _collapse(raw.get("attention_heads"), "head_count", layers),
        "attention.head_count",
    )

    # llama.cpp: head_count_kv is optional and defaults to head_count.
    kv_raw = raw.get("kv_heads")

    if kv_raw is None:
        kv_heads = heads
        derived.append("kv_heads")
    else:
        kv_heads = _int(
            _collapse(kv_raw, "head_count_kv", layers),
            "attention.head_count_kv",
        )

    if kv_heads > heads or heads % kv_heads:
        raise Rejected(
            f"inconsistent heads: attention={heads}, kv={kv_heads}"
        )

    # llama.cpp: key/value length default to embedding_length / head_count.
    def head_dim(field, label):
        value = raw.get(field)

        if value is not None:
            return _int(value, label)

        if embedding % heads:
            raise Unsupported(
                f"{label} absent and embedding_length is not divisible "
                f"by head_count"
            )

        derived.append(f"head_dim_{field[0]}")
        return embedding // heads

    head_dim_k = head_dim("key_length", "attention.key_length")
    head_dim_v = head_dim("value_length", "attention.value_length")

    window = raw.get("sliding_window")

    if window is not None:
        window = _int(window, "attention.sliding_window")

        # The "upper bound" reading of a sliding-window model only holds if
        # the windowed layers do not use larger heads than full attention.
        for swa_field, full in (("key_length_swa", head_dim_k),
                                ("value_length_swa", head_dim_v)):
            swa = raw.get(swa_field)

            if swa is not None and swa != full:
                raise Unsupported(
                    f"sliding-window layers use {swa_field}={swa}, "
                    f"different from full attention ({full})"
                )

    # Mixture of experts: all three facts must agree.
    experts = raw.get("expert_count") or 0
    used = raw.get("expert_used_count") or 0
    expert_bytes = raw.get("expert_tensor_bytes")

    if experts:
        _int(experts, "expert_count")
        _int(used, "expert_used_count")

        if used > experts:
            raise Rejected(f"expert_used_count {used} > expert_count {experts}")

        if not expert_bytes:
            raise Rejected("expert_count set but no expert tensors found")

    elif used or expert_bytes:
        raise Rejected("expert tensors or usage present without expert_count")

    # Weights: sizes are computed exactly from the tensor table.
    tensor_bytes = raw.get("tensor_bytes")

    if not tensor_bytes:
        raise Rejected(
            "tensor sizes unavailable (unknown tensor type; "
            "upgrade the gguf package)"
        )

    tensor_params = _int(raw.get("tensor_params"), "tensor_params")

    # The header must account for the whole file. This catches truncated
    # files, mis-parsed headers and mismatched shard sets.
    file_size = _int(raw.get("file_size_bytes"), "file_size_bytes")
    expected = raw.get("expected_file_size_bytes")
    alignment = _int(raw.get("alignment"), "alignment")
    shards = _int(raw.get("shard_count", 1), "shard_count")

    if expected is None:
        raise Rejected("expected file size unavailable")

    if not expected <= file_size <= expected + alignment * shards:
        raise Rejected(
            f"file size {file_size} inconsistent with tensor table "
            f"(expected {expected})"
        )

    # Quantization: header file_type is authoritative; the filename label
    # is kept for display and cross-checked.
    label = raw.get("filename_quantization") or "UNKNOWN"
    header_name = raw.get("file_type_name")
    quantization = header_name or label

    if header_name and label != "UNKNOWN" \
            and _quant_family(header_name) != _quant_family(label):
        warnings.append(
            f"filename says {label} but header says {header_name}"
        )

    parameters = raw.get("parameters")

    if isinstance(parameters, int) and parameters > 0:
        if abs(parameters - tensor_params) / tensor_params > 0.02:
            warnings.append(
                f"general.parameter_count {parameters} differs from "
                f"tensor total {tensor_params}"
            )
    else:
        parameters = None

    return {
        "id": stable_id(raw["repo_id"], raw["file_name"]),
        "repo_id": raw["repo_id"],
        "file_name": raw["file_name"],
        "download_url": raw["download_url"],
        "shard_count": shards,
        "file_size_bytes": file_size,
        "quantization": quantization,
        "quantization_label": label,
        "file_type_name": header_name,
        "architecture": architecture,
        "model_name": raw.get("model_name"),
        "size_label": raw.get("size_label"),
        "parameters": parameters,
        "tensor_params": tensor_params,
        "layers": layers,
        "embedding_length": embedding,
        "context_length": context,
        "attention_heads": heads,
        "kv_heads": kv_heads,
        "head_dim_k": head_dim_k,
        "head_dim_v": head_dim_v,
        "sliding_window": window,
        "kv_cache_kind": (
            "standard_upper_bound" if window else "standard"
        ),
        "expert_count": experts,
        "expert_used_count": used,
        "tensor_bytes": tensor_bytes,
        "expert_tensor_bytes": expert_bytes or 0,
        "derived_fields": sorted(set(derived)),
        "warnings": warnings,
        "downloads": raw.get("downloads", 0),
        "license": raw.get("license"),
        "last_modified": raw.get("last_modified"),
    }


def validate_all(path=None):
    """Return (valid, unsupported, rejected) lists."""

    with Path(path or INPUT_PATH).open("r", encoding="utf-8") as f:
        raw_models = json.load(f)

    valid, unsupported, rejected = [], [], []
    seen = set()

    for raw in raw_models:
        ref = {
            "repo_id": raw.get("repo_id"),
            "file": raw.get("file_name"),
        }

        if raw.get("parse_status") != "success":
            rejected.append({
                **ref,
                "reason": "GGUF parsing failed: "
                          + str(raw.get("parse_error", "unknown")),
            })
            continue

        try:
            entry = ModelEntry.model_validate(normalize_record(raw))

        except Unsupported as error:
            unsupported.append({**ref, "reason": str(error)})
            continue

        except (Rejected, ValidationError, KeyError, TypeError) as error:
            rejected.append({**ref, "reason": str(error)})
            continue

        key = (entry.repo_id, entry.file_name)

        if key in seen:
            rejected.append({**ref, "reason": "Duplicate repository/file"})
            continue

        seen.add(key)
        valid.append(entry.model_dump(mode="json"))

    return valid, unsupported, rejected


def validate_models():
    """Backward-compatible: return only the valid entries."""

    valid, unsupported, rejected = validate_all()

    print(f"Valid models: {len(valid)}")
    print(f"Unsupported models: {len(unsupported)}")
    print(f"Rejected models: {len(rejected)}")

    for item in rejected[:20]:
        print(f"[REJECTED] {item}")

    for item in unsupported[:20]:
        print(f"[UNSUPPORTED] {item}")

    return valid


def write_report(valid, unsupported, rejected, path=None):
    report = {
        "valid": len(valid),
        "unsupported": unsupported,
        "rejected": rejected,
        "warnings": [
            {"repo_id": m["repo_id"], "file": m["file_name"],
             "warnings": m["warnings"]}
            for m in valid if m["warnings"]
        ],
    }

    target = Path(path or REPORT_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    result = validate_all()
    write_report(*result)

    print(f"Valid: {len(result[0])}  "
          f"Unsupported: {len(result[1])}  "
          f"Rejected: {len(result[2])}")
    print(f"Report: {REPORT_PATH}")
