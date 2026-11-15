"""Extract structural metadata from remote GGUF headers.

Only the header (key/value metadata and the tensor table) is fetched, using
HTTP range requests. Everything stored here is either read from the header
or computed exactly from it (tensor byte sizes). Nothing is guessed:
derivations and defaults are applied later, in validator.py, where their
provenance is recorded.
"""

import argparse
import json
import struct
import time
from collections import Counter
from math import prod
from pathlib import Path

import httpx
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType

try:
    from gguf import LlamaFileType
except ImportError:  # pragma: no cover - very old gguf releases
    LlamaFileType = None

ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = ROOT / "output" / "models.json"
OUTPUT_PATH = ROOT / "output" / "parsed_models.json"

INITIAL_RANGE = 1024 * 1024
MAX_RANGE = 64 * 1024 * 1024

DEFAULT_ALIGNMENT = 32
MAX_KEPT_ARRAY = 4096
MAX_ARRAY_ITEMS = 10_000_000

# GGUF metadata value types
UINT8, INT8 = 0, 1
UINT16, INT16 = 2, 3
UINT32, INT32 = 4, 5
FLOAT32, BOOL = 6, 7
STRING, ARRAY = 8, 9
UINT64, INT64, FLOAT64 = 10, 11, 12

SCALAR_FORMATS = {
    UINT8: "B",
    INT8: "b",
    UINT16: "H",
    INT16: "h",
    UINT32: "I",
    INT32: "i",
    FLOAT32: "f",
    BOOL: "?",
    UINT64: "Q",
    INT64: "q",
    FLOAT64: "d",
}

_U64 = struct.Struct("<Q")

# Architecture-prefixed header keys the estimator needs. Values are kept
# raw (int, or list for per-layer arrays); interpretation happens later.
ARCH_KEYS = {
    "layers": "block_count",
    "context_length": "context_length",
    "embedding_length": "embedding_length",
    "attention_heads": "attention.head_count",
    "kv_heads": "attention.head_count_kv",
    "key_length": "attention.key_length",
    "value_length": "attention.value_length",
    "expert_count": "expert_count",
    "expert_used_count": "expert_used_count",
    "sliding_window": "attention.sliding_window",
    "key_length_swa": "attention.key_length_swa",
    "value_length_swa": "attention.value_length_swa",
    "kv_lora_rank": "attention.kv_lora_rank",
}


class NeedMoreData(Exception):
    pass


class GGUFMetadataParser:
    def __init__(self, data):
        self.data = memoryview(data)
        self.offset = 0

    def read(self, fmt):
        size = struct.calcsize("<" + fmt)

        if self.offset + size > len(self.data):
            raise NeedMoreData()

        value = struct.unpack_from("<" + fmt, self.data, self.offset)[0]

        self.offset += size
        return value

    def read_bytes(self, size):
        if size < 0 or self.offset + size > len(self.data):
            raise NeedMoreData()

        result = self.data[self.offset:self.offset + size]
        self.offset += size
        return result

    def read_string(self):
        length = self.read("Q")

        if length > MAX_RANGE:
            raise ValueError("GGUF string exceeds safety limit")

        return bytes(self.read_bytes(length)).decode("utf-8")

    def skip_string(self):
        if self.offset + 8 > len(self.data):
            raise NeedMoreData()

        (length,) = _U64.unpack_from(self.data, self.offset)

        if length > MAX_RANGE:
            raise ValueError("GGUF string exceeds safety limit")

        if self.offset + 8 + length > len(self.data):
            raise NeedMoreData()

        self.offset += 8 + length

    def read_array(self, keep):
        subtype = self.read("I")
        count = self.read("Q")

        if count > MAX_ARRAY_ITEMS:
            raise ValueError("GGUF array exceeds safety limit")

        if subtype in SCALAR_FORMATS:
            fmt = SCALAR_FORMATS[subtype]
            total = struct.calcsize("<" + fmt) * count

            if self.offset + total > len(self.data):
                raise NeedMoreData()

            values = None

            # Small numeric arrays are real metadata: several architectures
            # store per-layer head counts this way. The old parser discarded
            # every array, so those fields silently became None.
            if keep and count <= MAX_KEPT_ARRAY:
                values = list(
                    struct.unpack_from(
                        f"<{count}{fmt}", self.data, self.offset
                    )
                )

            self.offset += total
            return values

        if subtype == STRING:
            for _ in range(count):
                self.skip_string()

            return None

        raise ValueError(f"Unsupported GGUF array element type: {subtype}")

    def read_value(self, value_type, keep_arrays=True):
        if value_type == STRING:
            return self.read_string()

        if value_type == ARRAY:
            return self.read_array(keep_arrays)

        if value_type not in SCALAR_FORMATS:
            raise ValueError(f"Unsupported GGUF value type: {value_type}")

        return self.read(SCALAR_FORMATS[value_type])

    def parse(self):
        if bytes(self.read_bytes(4)) != b"GGUF":
            raise ValueError("Invalid GGUF magic")

        version = self.read("I")

        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version: {version}")

        tensor_count = self.read("Q")
        metadata_count = self.read("Q")

        if tensor_count > 10_000_000 or metadata_count > 1_000_000:
            raise ValueError("GGUF header exceeds safety limits")

        metadata = {}

        for _ in range(metadata_count):
            key = self.read_string()
            value_type = self.read("I")

            # Tokenizer data is large and irrelevant to estimation.
            is_tokenizer = key.startswith("tokenizer.")

            value = self.read_value(value_type, keep_arrays=not is_tokenizer)

            if value is not None and not is_tokenizer:
                metadata[key] = value

        alignment = metadata.get("general.alignment", DEFAULT_ALIGNMENT)

        if not isinstance(alignment, int) or alignment <= 0:
            raise ValueError(f"Invalid general.alignment: {alignment!r}")

        stats = TensorStats()

        for _ in range(tensor_count):
            name = self.read_string()
            n_dims = self.read("I")

            if n_dims < 1 or n_dims > 4:
                raise ValueError(f"Unsupported tensor dimensions: {n_dims}")

            dims = [self.read("Q") for _ in range(n_dims)]
            tensor_type = self.read("I")
            offset = self.read("Q")

            stats.add(name, dims, tensor_type, offset)

        header_size = self.offset
        data_offset = -(-header_size // alignment) * alignment

        return {
            "gguf_version": version,
            "metadata": metadata,
            "alignment": alignment,
            "header_size_bytes": header_size,
            "data_offset": data_offset,
            **stats.summary(data_offset),
        }


def resolve_tensor_type(value):
    try:
        qtype = GGMLQuantizationType(value)
    except ValueError:
        return None, f"UNKNOWN_{value}"

    return qtype, qtype.name


def tensor_nbytes(dims, qtype):
    """Exact byte size of a tensor, or None if it cannot be computed."""

    if qtype is None or qtype not in GGML_QUANT_SIZES:
        return None

    block_size, type_size = GGML_QUANT_SIZES[qtype]

    if dims[0] % block_size:
        return None

    return prod(dims) // block_size * type_size


class TensorStats:
    """Aggregates the tensor table into estimator-relevant totals."""

    def __init__(self):
        self.count = 0
        self.params = 0
        self.nbytes = 0
        self.expert_params = 0
        self.expert_nbytes = 0
        self.bytes_known = True
        self.data_end = 0
        self.type_counts = Counter()
        self.type_bytes = Counter()

    def add(self, name, dims, tensor_type, offset):
        qtype, type_name = resolve_tensor_type(tensor_type)
        elements = prod(dims)
        nbytes = tensor_nbytes(dims, qtype)

        self.count += 1
        self.params += elements
        self.type_counts[type_name] += 1

        # Mixture-of-experts weights are stored as stacked "*_exps" tensors
        # (ffn_gate_exps, ffn_up_exps, ffn_down_exps). Shared experts
        # ("*_shexp") are always active and are deliberately not included.
        is_expert = "_exps." in name

        if is_expert:
            self.expert_params += elements

        if nbytes is None:
            self.bytes_known = False
            return

        self.nbytes += nbytes
        self.type_bytes[type_name] += nbytes
        self.data_end = max(self.data_end, offset + nbytes)

        if is_expert:
            self.expert_nbytes += nbytes

    def summary(self, data_offset):
        known = self.bytes_known

        return {
            "tensor_count": self.count,
            "tensor_params": self.params,
            "tensor_bytes": self.nbytes if known else None,
            "expert_tensor_params": self.expert_params,
            "expert_tensor_bytes": self.expert_nbytes if known else None,
            "tensor_types": dict(self.type_counts),
            "tensor_type_bytes": dict(self.type_bytes) if known else None,
            "expected_file_size_bytes": (
                data_offset + self.data_end if known else None
            ),
        }


def file_type_name(value):
    """Map general.file_type (llama.cpp LLAMA_FTYPE) to a label."""

    if LlamaFileType is None or isinstance(value, bool):
        return None

    if not isinstance(value, int):
        return None

    try:
        name = LlamaFileType(value).name
    except ValueError:
        return None

    if name == "GUESSED":
        return None

    for prefix in ("MOSTLY_", "ALL_"):
        if name.startswith(prefix):
            return name[len(prefix):]

    return name


def get_architecture_fields(metadata):
    architecture = metadata.get("general.architecture")

    if not isinstance(architecture, str) or not architecture:
        return {}

    prefix = architecture + "."

    return {
        field: metadata.get(prefix + key)
        for field, key in ARCH_KEYS.items()
    }


def extract_header(read_range):
    """Fetch and parse a GGUF header.

    read_range(start, end) returns the bytes in [start, end) and may return
    fewer at end-of-file. Already-downloaded bytes are kept, so growing the
    window only transfers the new part.
    """

    data = bytearray()
    size = INITIAL_RANGE

    while True:
        chunk = read_range(len(data), size)
        data.extend(chunk)

        try:
            return GGUFMetadataParser(bytes(data)).parse()

        except NeedMoreData:
            if len(data) < size:
                raise ValueError("Truncated GGUF header")

            if size >= MAX_RANGE:
                raise ValueError("GGUF header exceeds 64 MB")

            size *= 2


def http_range_reader(client, url, attempts=3):
    def read_range(start, end):
        wanted = end - start
        headers = {
            "Range": f"bytes={start}-{end - 1}",
            "Accept-Encoding": "identity",
        }

        last_error = None

        for attempt in range(attempts):
            try:
                buffer = bytearray()

                with client.stream(
                    "GET", url, headers=headers,
                    follow_redirects=True, timeout=60,
                ) as response:
                    if response.status_code == 416:
                        return b""  # start is past end-of-file

                    response.raise_for_status()

                    if response.status_code == 200 and start > 0:
                        raise ValueError("Server ignored the Range header")

                    for chunk in response.iter_bytes(chunk_size=64 * 1024):
                        buffer.extend(chunk)

                        if len(buffer) >= wanted:
                            break

                return bytes(buffer[:wanted])

            except (httpx.TransportError, httpx.HTTPStatusError) as error:
                status = getattr(
                    getattr(error, "response", None), "status_code", None
                )

                if status is not None and status < 500 and status != 429:
                    raise

                last_error = error
                time.sleep(2 ** attempt)

        raise last_error

    return read_range


def build_record(model, headers):
    """Combine the parsed header of every shard into one record."""

    first = headers[0]
    metadata = first["metadata"]

    split_count = metadata.get("split.count")

    if len(headers) > 1 and split_count not in (None, len(headers)):
        raise ValueError(
            f"split.count={split_count} but {len(headers)} shards listed"
        )

    def total(key):
        values = [h[key] for h in headers]

        return None if any(v is None for v in values) else sum(values)

    tensor_types = Counter()
    tensor_type_bytes = Counter()

    for header in headers:
        tensor_types.update(header["tensor_types"])
        tensor_type_bytes.update(header["tensor_type_bytes"] or {})

    bytes_known = all(h["tensor_bytes"] is not None for h in headers)

    architecture = metadata.get("general.architecture")

    return {
        **model,
        "gguf_version": first["gguf_version"],
        "alignment": first["alignment"],
        "shard_headers_parsed": len(headers),
        "architecture": architecture,
        "model_name": metadata.get("general.name"),
        "size_label": metadata.get("general.size_label"),
        "parameters": metadata.get("general.parameter_count"),
        "file_type": metadata.get("general.file_type"),
        "file_type_name": file_type_name(metadata.get("general.file_type")),
        "filename_quantization": model.get("quantization", "UNKNOWN"),
        **get_architecture_fields(metadata),
        "tensor_count": total("tensor_count"),
        "tensor_params": total("tensor_params"),
        "tensor_bytes": total("tensor_bytes"),
        "expert_tensor_params": total("expert_tensor_params"),
        "expert_tensor_bytes": total("expert_tensor_bytes"),
        "expected_file_size_bytes": total("expected_file_size_bytes"),
        "tensor_types": dict(tensor_types),
        "tensor_type_bytes": (
            dict(tensor_type_bytes) if bytes_known else None
        ),
        "parse_status": "success",
    }


def model_files(model):
    files = model.get("files")

    if files:
        return files

    # models.json written by an older file_discovery.py
    return [{
        "file_name": model["file_name"],
        "size_bytes": model["file_size_bytes"],
        "download_url": model["download_url"],
    }]


def parse_models(limit=None):
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        models = json.load(f)

    if limit:
        models = models[:limit]

    results = []

    with httpx.Client(
        follow_redirects=True,
        headers={"User-Agent": "LocalLLMFitChecker/1.0"},
    ) as client:

        for index, model in enumerate(models, start=1):
            print(
                f"[{index}/{len(models)}] "
                f"{model['repo_id']} / {model['file_name']}"
            )

            try:
                headers = [
                    extract_header(http_range_reader(client, f["download_url"]))
                    for f in model_files(model)
                ]

                record = build_record(model, headers)
                results.append(record)

                print(
                    f"  [SUCCESS] {record['architecture']}, "
                    f"{record['tensor_count']} tensors"
                )

            except Exception as error:
                print(f"  [FAILED] {error}")

                results.append({
                    **model,
                    "parse_status": "failed",
                    "parse_error": str(error),
                })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    successful = sum(r["parse_status"] == "success" for r in results)

    print("\nGGUF parsing completed.")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--limit", type=int, default=None)

    args = arg_parser.parse_args()

    parse_models(args.limit)
