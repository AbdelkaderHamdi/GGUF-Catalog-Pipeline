
import json
import re
import argparse
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_url


ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = ROOT / "output" / "discovered.json"
OUTPUT_PATH = ROOT / "output" / "models.json"

# Initial filename-based quantization detection.
QUANT_PATTERN = re.compile(
    r"(?<![A-Z0-9])"
    r"(IQ\d+_[A-Z0-9]+|Q\d+_[A-Z0-9]+|"
    r"Q\d+_\d+|F16|F32|BF16|F8_E4M3FN)"
    r"(?![A-Z0-9])",
    re.IGNORECASE
)


def load_repositories():
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def detect_quantization(filename):
    """
    Extract a preliminary quantization label
    from the GGUF filename.
    """

    match = QUANT_PATTERN.search(filename.upper())

    if match:
        return match.group(1).upper()

    return "UNKNOWN"


def discover_files(limit=None):
    repositories = load_repositories()

    if limit:
        repositories = repositories[:limit]

    api = HfApi()

    results = []
    seen = set()

    for index, repository in enumerate(repositories, start=1):
        repo_id = repository["repo_id"]

        print(
            f"[{index}/{len(repositories)}] "
            f"Scanning {repo_id}"
        )

        try:
            filenames = api.list_repo_files(
                repo_id=repo_id,
                repo_type="model"
            )

            gguf_files = [
                filename
                for filename in filenames
                if filename.lower().endswith(".gguf")
            ]

            if not gguf_files:
                print("  No GGUF files found.")
                continue

            # Retrieve file metadata without downloading weights.
            file_infos = api.get_paths_info(
                repo_id=repo_id,
                paths=gguf_files,
                repo_type="model"
            )

            for file_info in file_infos:
                filename = file_info.path

                key = (repo_id, filename)

                if key in seen:
                    continue

                seen.add(key)

                size = getattr(file_info, "size", None)

                if size is None:
                    print(f"  [SKIPPED] Missing file size: {filename}")
                    continue

                results.append({
                    "repo_id": repo_id,
                    "file_name": filename,
                    "file_size_bytes": size,
                    "quantization": detect_quantization(filename),
                    "download_url": hf_hub_url(
                        repo_id=repo_id,
                        filename=filename,
                        repo_type="model"
                    ),
                    "downloads": repository.get("downloads", 0),
                    "license": repository.get("license"),
                    "last_modified": repository.get("last_modified")
                })

                print(
                    f"  [FOUND] {filename} "
                    f"({size / (1024 ** 3):.2f} GB)"
                )

        except Exception as error:
            print(f"  [ERROR] {repo_id}: {error}")
            continue

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

    print("\nFile discovery completed.")
    print(f"Repositories scanned: {len(repositories)}")
    print(f"GGUF files discovered: {len(results)}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum repositories to scan"
    )

    args = parser.parse_args()

    discover_files(args.limit)
