import argparse
import json
import posixpath
import re
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_url


ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = ROOT / "output" / "discovered.json"
OUTPUT_PATH = ROOT / "output" / "models.json"

# Filename-based quantization label (display only; the GGUF header is the
# authority and is checked later). The suffix group repeats so that
# Q4_K_M, Q5_K_S, Q4_K_XL, IQ4_XS ... are captured whole. The old pattern
# stopped at the second underscore and turned Q4_K_M into Q4_K.
QUANT_PATTERN = re.compile(
    r"(?<![A-Z0-9])"
    r"((?:IQ|TQ|Q)\d+(?:_[A-Z0-9]+)*"
    r"|BF16|F16|F32|MXFP4(?:_MOE)?)"
    r"(?![A-Z0-9])"
)

# "model-00001-of-00003.gguf"
SHARD_PATTERN = re.compile(
    r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$",
    re.IGNORECASE,
)

# Auxiliary GGUF files that are not language models.
SKIPPED_MARKERS = ("mmproj", "imatrix")

PATH_INFO_CHUNK = 100


def load_repositories():
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def detect_quantization(filename):
    """Extract a quantization label from a GGUF filename (last match)."""

    name = posixpath.basename(filename)
    name = re.sub(r"\.gguf$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"-\d{5}-of-\d{5}$", "", name)

    matches = QUANT_PATTERN.findall(name.upper())

    return matches[-1] if matches else "UNKNOWN"


def group_gguf_files(entries):
    """Group (path, size) pairs into one logical model file each.

    Split models (-00001-of-0000N.gguf) become a single entry whose size is
    the sum of all shards; the previous code listed each shard as if it were
    a complete model, which understates memory needs by the shard count.
    Incomplete shard sets and auxiliary files are skipped with a reason.

    Returns (groups, skipped) where skipped is a list of (path, reason).
    """

    groups = {}
    singles = []
    skipped = []

    for path, size in entries:
        base = posixpath.basename(path)

        if any(marker in base.lower() for marker in SKIPPED_MARKERS):
            skipped.append((path, "auxiliary file"))
            continue

        match = SHARD_PATTERN.match(base)

        if not match:
            singles.append({"path": path, "size": size})
            continue

        key = (
            posixpath.dirname(path),
            match.group("stem"),
            int(match.group("total")),
        )

        groups.setdefault(key, []).append(
            (int(match.group("index")), path, size)
        )

    results = []

    for item in singles:
        results.append({
            "file_name": item["path"],
            "file_size_bytes": item["size"],
            "files": [{"file_name": item["path"],
                       "size_bytes": item["size"]}],
        })

    for (directory, stem, total), shards in groups.items():
        shards.sort()

        if [s[0] for s in shards] != list(range(1, total + 1)):
            skipped.append((shards[0][1], "incomplete shard set"))
            continue

        results.append({
            "file_name": shards[0][1],
            "file_size_bytes": sum(s[2] for s in shards),
            "files": [{"file_name": s[1], "size_bytes": s[2]}
                      for s in shards],
        })

    results.sort(key=lambda r: r["file_name"])

    return results, skipped


def discover_files(limit=None):
    repositories = load_repositories()

    if limit:
        repositories = repositories[:limit]

    api = HfApi()

    results = []

    for index, repository in enumerate(repositories, start=1):
        repo_id = repository["repo_id"]

        print(f"[{index}/{len(repositories)}] Scanning {repo_id}")

        try:
            filenames = api.list_repo_files(
                repo_id=repo_id,
                repo_type="model",
            )

            gguf_files = [
                name for name in filenames
                if name.lower().endswith(".gguf")
            ]

            if not gguf_files:
                print("  No GGUF files found.")
                continue

            entries = []

            for start in range(0, len(gguf_files), PATH_INFO_CHUNK):
                infos = api.get_paths_info(
                    repo_id=repo_id,
                    paths=gguf_files[start:start + PATH_INFO_CHUNK],
                    repo_type="model",
                )

                for info in infos:
                    size = getattr(info, "size", None)

                    if size is None:
                        print(f"  [SKIPPED] Missing file size: {info.path}")
                        continue

                    entries.append((info.path, size))

            groups, skipped = group_gguf_files(entries)

            for path, reason in skipped:
                print(f"  [SKIPPED] {path}: {reason}")

            for group in groups:
                for file in group["files"]:
                    file["download_url"] = hf_hub_url(
                        repo_id=repo_id,
                        filename=file["file_name"],
                        repo_type="model",
                    )

                results.append({
                    "repo_id": repo_id,
                    "file_name": group["file_name"],
                    "file_size_bytes": group["file_size_bytes"],
                    "shard_count": len(group["files"]),
                    "files": group["files"],
                    "quantization": detect_quantization(
                        group["file_name"]
                    ),
                    "download_url": group["files"][0]["download_url"],
                    "downloads": repository.get("downloads", 0),
                    "license": repository.get("license"),
                    "last_modified": repository.get("last_modified"),
                })

                print(
                    f"  [FOUND] {group['file_name']} "
                    f"({group['file_size_bytes'] / (1024 ** 3):.2f} GB, "
                    f"{len(group['files'])} file(s))"
                )

        except Exception as error:
            print(f"  [ERROR] {repo_id}: {error}")
            continue

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nFile discovery completed.")
    print(f"Repositories scanned: {len(repositories)}")
    print(f"GGUF models discovered: {len(results)}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum repositories to scan",
    )

    args = parser.parse_args()

    discover_files(args.limit)
