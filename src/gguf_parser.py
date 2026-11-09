import json
import struct
import argparse
from collections import Counter
from pathlib import Path

import httpx
from gguf import GGMLQuantizationType

ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = ROOT / "output" / "models.json"
OUTPUT_PATH = ROOT / "output" / "parsed_models.json"

INITIAL_RANGE = 256 * 1024
MAX_RANGE = 64 * 1024 * 1024

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

        value = struct.unpack_from(
            "<" + fmt,
            self.data,
            self.offset
        )[0]

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

    def read_value(self, value_type):
        if value_type == STRING:
            return self.read_string()

        if value_type == ARRAY:
            subtype = self.read("I")
            count = self.read("Q")

            if count > 10_000_000:
                raise ValueError("GGUF array exceeds safety limit")

            for _ in range(count):
                self.skip_value(subtype)

            return None

        if value_type not in SCALAR_FORMATS:
            raise ValueError(
                f"Unsupported GGUF value type: {value_type}"
            )

        return self.read(SCALAR_FORMATS[value_type])

    def skip_value(self, value_type):
        if value_type == STRING:
            length = self.read("Q")
            self.read_bytes(length)

        elif value_type == ARRAY:
            raise ValueError("Nested GGUF arrays are unsupported")

        elif value_type in SCALAR_FORMATS:
            self.read(SCALAR_FORMATS[value_type])

        else:
            raise ValueError(
                f"Unsupported GGUF value type: {value_type}"
            )

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

            value = self.read_value(value_type)

            if value is not None:
                metadata[key] = value

        tensor_types = Counter()

        for _ in range(tensor_count):
            self.read_string()  # Tensor name

            dimensions = self.read("I")

            if dimensions > 4:
                raise ValueError("Unsupported tensor dimensions")

            for _ in range(dimensions):
                self.read("Q")

            tensor_type = self.read("I")
            self.read("Q")  # Relative tensor data offset

            tensor_types[tensor_type] += 1

        return version, metadata, tensor_count, tensor_types


def download_prefix(client, url, size):
    headers = {
        "Range": f"bytes=0-{size - 1}",
        "Accept-Encoding": "identity"
    }

    data = bytearray()

    with client.stream(
        "GET",
        url,
        headers=headers,
        follow_redirects=True,
        timeout=60
    ) as response:

        if response.status_code not in (200, 206):
            response.raise_for_status()

        # If Range is ignored, consume only the requested prefix.
        for chunk in response.iter_bytes(chunk_size=64 * 1024):
            remaining = size - len(data)

            if remaining <= 0:
                break

            data.extend(chunk[:remaining])

            if len(data) >= size:
                break

    return bytes(data)


def extract_metadata(client, url):
    size = INITIAL_RANGE

    while size <= MAX_RANGE:
        data = download_prefix(client, url, size)

        try:
            parser = GGUFMetadataParser(data)

            version, metadata, tensor_count, tensor_types = (
                parser.parse()
            )

            return {
                "gguf_version": version,
                "metadata": metadata,
                "tensor_count": tensor_count,
                "tensor_types": {
                    GGMLQuantizationType(key).name
                    if key in GGMLQuantizationType._value2member_map_
                    else f"UNKNOWN_{key}": value
                    for key, value in tensor_types.items()
                },
                "bytes_inspected": len(data)
            }

        except NeedMoreData:
            if len(data) < size:
                raise ValueError(
                    "Incomplete GGUF file or range response"
                )

            size *= 2

    raise ValueError("GGUF header exceeds 64 MB")


def get_architecture_fields(metadata):
    architecture = metadata.get("general.architecture")

    if not architecture:
        return {}

    prefix = architecture + "."

    def get(key):
        return metadata.get(prefix + key)

    return {
        "layers": get("block_count"),
        "attention_heads": get("attention.head_count"),
        "kv_heads": get("attention.head_count_kv"),
        "context_length": get("context_length"),
        "embedding_length": get("embedding_length")
    }


def parse_models(limit=None):
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        models = json.load(f)

    if limit:
        models = models[:limit]

    results = []

    with httpx.Client(
        follow_redirects=True,
        headers={"User-Agent": "LocalLLMFitChecker/1.0"}
    ) as client:

        for index, model in enumerate(models, start=1):
            print(
                f"[{index}/{len(models)}] "
                f"{model['repo_id']} / {model['file_name']}"
            )

            try:
                parsed = extract_metadata(
                    client,
                    model["download_url"]
                )

                metadata = parsed["metadata"]

                architecture = metadata.get(
                    "general.architecture"
                )

                record = {
                    **model,
                    **parsed,
                    "model_name": metadata.get("general.name"),
                    "architecture": architecture,
                    "parameters": metadata.get(
                        "general.parameter_count"
                    ),
                    "file_type": metadata.get("general.file_type"),
                    "filename_quantization": model.get(
                        "quantization", "UNKNOWN"
                    ),
                    **get_architecture_fields(metadata),
                    "parse_status": "success"
                }

                results.append(record)

                print(
                    f"  [SUCCESS] {architecture}, "
                    f"{parsed['tensor_count']} tensors"
                )

            except Exception as error:
                print(f"  [FAILED] {error}")

                results.append({
                    **model,
                    "parse_status": "failed",
                    "parse_error": str(error)
                })

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )

    successful = sum(
        item["parse_status"] == "success"
        for item in results
    )

    print("\nGGUF parsing completed.")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--limit",
        type=int,
        default=None
    )

    args = arg_parser.parse_args()

    parse_models(args.limit)