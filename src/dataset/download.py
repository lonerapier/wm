import argparse
import glob
import os
import tarfile
from pathlib import Path
from typing import Any

import zstandard as zstd
from huggingface_hub import hf_hub_download

DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "tworooms": {
        "repo_id": "quentinll/lewm-tworooms",
        "filename": "tworoom.tar.zst",
        "h5_filename": "tworoom.h5",
        "is_tar": True,
    },
    "pusht": {
        "repo_id": "quentinll/lewm-pusht",
        "filename": "pusht_expert_train.h5.zst",
        "h5_filename": "pusht_expert_train.h5",
        "is_tar": False,
    },
    "reacher": {
        "repo_id": "quentinll/lewm-reacher",
        "filename": "reacher.tar.zst",
        "h5_filename": "reacher.h5",
        "is_tar": True,
    },
    "ogbench-cube": {
        "repo_id": "quentinll/lewm-cube",
        "filename": "cube_single_expert.tar.zst",
        "h5_filename": "cube_single_expert.h5",
        "is_tar": True,
    },
}

# Aliases
DATASET_REGISTRY["cube"] = DATASET_REGISTRY["ogbench-cube"]
DATASET_REGISTRY["ogbench"] = DATASET_REGISTRY["ogbench-cube"]

AVAILABLE_DATASETS = list(DATASET_REGISTRY.keys())


def download_dataset(
    dataset: str | dict[str, Any],
    dest_dir: str | None = None,
    force: bool = False,
) -> str:
    """Download and extract a dataset from Hugging Face.

    Args:
        dataset: Either a dataset name key from DATASET_REGISTRY or a dict with
            {"name", "repo_id", "filename", "is_tar", optional "h5_filename"}.
        dest_dir: Target output directory. Defaults to ./data/<dataset_name>.
        force: If True, re-download and re-extract even if .h5 file already exists.

    Returns:
        The path to the extracted .h5 file.
    """
    if isinstance(dataset, str):
        dataset_name = dataset.lower()
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(
                f"Unknown dataset '{dataset}'. Choose from: {list(DATASET_REGISTRY.keys())}"
            )
        info = dict(DATASET_REGISTRY[dataset_name])
        info["name"] = dataset_name
    elif isinstance(dataset, dict):
        assert "name" in dataset, "Dataset dict must contain 'name'"
        assert "repo_id" in dataset, "Dataset dict must contain 'repo_id'"
        assert "filename" in dataset, "Dataset dict must contain 'filename'"
        info = dict(dataset)
        if "is_tar" not in info:
            info["is_tar"] = info["filename"].endswith(".tar.zst") or ".tar" in info["filename"]
    else:
        raise TypeError(f"Invalid dataset specification type: {type(dataset)}")

    dataset_name = info["name"]
    target_dir = dest_dir or f"./data/{dataset_name}"
    os.makedirs(target_dir, exist_ok=True)

    # Check if target HDF5 file already exists
    expected_h5 = info.get("h5_filename")
    if expected_h5 and not force:
        h5_path = os.path.join(target_dir, expected_h5)
        if os.path.exists(h5_path):
            print(f"Dataset '{dataset_name}' already extracted at: {h5_path}")
            return h5_path

    existing_h5 = glob.glob(os.path.join(target_dir, "*.h5"))
    if existing_h5 and not force:
        print(f"Found existing HDF5 file: {existing_h5[0]}")
        return existing_h5[0]

    print(f"Downloading {info['filename']} from {info['repo_id']}...")
    archive_path = hf_hub_download(
        repo_id=info["repo_id"],
        filename=info["filename"],
        repo_type="dataset",
        local_dir=target_dir,
    )

    print(f"Decompressing {archive_path}...")
    dctx = zstd.ZstdDecompressor()

    if info.get("is_tar", True):
        # Extract .tar.zst
        with open(archive_path, "rb") as ifh:
            with dctx.stream_reader(ifh) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as tar:
                    tar.extractall(path=target_dir)
    else:
        # Single .h5.zst file decompression
        out_name = expected_h5 or Path(info["filename"]).stem
        if not out_name.endswith(".h5"):
            out_name = f"{out_name}.h5"
        dest_h5 = os.path.join(target_dir, out_name)
        with open(archive_path, "rb") as ifh, open(dest_h5, "wb") as ofh:
            dctx.copy_stream(ifh, ofh)

    candidates = sorted(glob.glob(os.path.join(target_dir, "*.h5")))
    if not candidates:
        raise RuntimeError(f"Decompression completed but no .h5 file found in {target_dir}")

    print(f"Successfully extracted dataset to: {candidates[0]}")
    return candidates[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract LeWM datasets from HF")
    parser.add_argument(
        "dataset",
        type=str,
        choices=list(DATASET_REGISTRY.keys()),
        help=f"Dataset to download: {list(DATASET_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--dest_dir",
        type=str,
        default=None,
        help="Optional destination directory (defaults to ./data/<dataset_name>)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and re-extraction even if .h5 file exists",
    )
    args = parser.parse_args()
    download_dataset(args.dataset, dest_dir=args.dest_dir, force=args.force)
