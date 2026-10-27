#!/usr/bin/env python3
"""Hugging Face Storage CLI.

Easily list, download, upload, and test checkpoints and artifacts stored on Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hf_storage import (
    HFConfig,
    HFSyncManager,
    download_file,
    is_hf_configured,
    list_repo_files,
    test_hf_connection,
    upload_file,
)


def cmd_test(args: argparse.Namespace) -> int:
    cfg = HFConfig.from_env()
    print("Testing Hugging Face configuration...")
    print(
        f"  Token     : {'*' * 8 + cfg.token[-4:] if cfg.token and len(cfg.token) > 4 else '(Not set)'}"
    )
    print(f"  Repo ID   : {cfg.repo_id or '(Not set)'}")
    print(f"  Repo Type : {cfg.repo_type}")
    print(f"  Private   : {cfg.private}")

    if not cfg.token:
        print("\n[ERROR] HF_TOKEN is missing. Set HF_TOKEN in your .env file.")
        print("Create a write-access token at: https://huggingface.co/settings/tokens")
        return 1

    print("\nVerifying token and repository access...")
    if test_hf_connection(cfg):
        print("\n SUCCESS: Hugging Face authentication and repository access verified!")
        return 0
    else:
        print("\n FAILURE: Hugging Face connection test failed.")
        return 1


def cmd_list(args: argparse.Namespace) -> int:
    cfg = HFConfig.from_env()
    repo_id = args.repo_id or cfg.repo_id
    if not repo_id:
        print(
            "[ERROR] Hugging Face repo_id is not specified. Set HF_REPO_ID in .env or pass --repo-id"
        )
        return 1

    prefix = args.prefix
    if args.run_id:
        prefix = f"{cfg.prefix}/{args.run_id}"

    print(f"Listing files in Hugging Face repo '{repo_id}' (prefix: '{prefix}')...\n")
    try:
        files = list_repo_files(
            prefix=prefix, repo_id=repo_id, repo_type=cfg.repo_type, token=cfg.token
        )
        if not files:
            print("No files found matching prefix.")
            return 0

        print(f"{'Path in Repository':<70}")
        print("-" * 70)
        for f in files:
            print(f"{f['path']:<70}")
        print("-" * 70)
        print(f"Total files: {len(files)}")
        return 0
    except Exception as e:
        print(f"[ERROR] Failed to list repo files: {e}")
        return 1


def cmd_download(args: argparse.Namespace) -> int:
    cfg = HFConfig.from_env()
    repo_id = args.repo_id or cfg.repo_id
    if not repo_id:
        print(
            "[ERROR] Hugging Face repo_id is not specified. Set HF_REPO_ID in .env or pass --repo-id"
        )
        return 1

    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    manager = HFSyncManager(config=cfg, repo_id=repo_id)

    if args.path:
        local_path = dest_dir / Path(args.path).name
        print(f"Downloading hf://{repo_id}/{args.path} -> {local_path} ...")
        download_file(args.path, local_path, repo_id=repo_id, token=cfg.token)
        print("Download complete!")
        return 0

    if not args.run_id:
        print("[ERROR] Must provide either --path or --run-id")
        return 1

    if args.checkpoint:
        print(
            f"Downloading checkpoint '{args.checkpoint}' for run '{args.run_id}' from Hugging Face..."
        )
        path = manager.download_checkpoint(args.run_id, args.checkpoint, dest_dir)
        print(f"Saved to {path}")
    else:
        print(f"Finding latest checkpoint for run '{args.run_id}' on Hugging Face...")
        path = manager.get_latest_checkpoint(args.run_id, dest_dir)
        if path:
            print(f"Downloaded latest checkpoint to: {path}")
        else:
            print(f"No checkpoints found for run '{args.run_id}' on Hugging Face.")
            return 1
    return 0


def cmd_upload(args: argparse.Namespace) -> int:
    cfg = HFConfig.from_env()
    repo_id = args.repo_id or cfg.repo_id
    if not repo_id:
        print(
            "[ERROR] Hugging Face repo_id is not specified. Set HF_REPO_ID in .env or pass --repo-id"
        )
        return 1

    local_path = Path(args.file)
    if not local_path.is_file():
        print(f"[ERROR] File not found: {local_path}")
        return 1

    if args.path:
        path_in_repo = args.path
    elif args.run_id:
        path_in_repo = f"{cfg.prefix}/{args.run_id}/checkpoints/{local_path.name}"
    else:
        path_in_repo = f"{cfg.prefix}/{local_path.name}"

    print(
        f"Uploading {local_path} -> https://huggingface.co/{repo_id}/blob/main/{path_in_repo} ..."
    )
    success = upload_file(local_path, path_in_repo, repo_id=repo_id, token=cfg.token)
    if success:
        print("Upload successful!")
        return 0
    else:
        print("Upload failed.")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Hugging Face checkpoint and artifact manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # test
    subparsers.add_parser("test", help="Test HF token and repository access")

    # list
    list_p = subparsers.add_parser("list", help="List files stored in Hugging Face repository")
    list_p.add_argument("--repo-id", help="Hugging Face repo ID (e.g. username/wm-checkpoints)")
    list_p.add_argument("--prefix", default="", help="Filter by path prefix")
    list_p.add_argument("--run-id", default="", help="Filter by specific run ID")

    # download
    dl_p = subparsers.add_parser("download", help="Download checkpoint or file from Hugging Face")
    dl_p.add_argument("--repo-id", help="Hugging Face repo ID")
    dl_p.add_argument("--path", help="Path inside repository")
    dl_p.add_argument("--run-id", help="Experiment run ID")
    dl_p.add_argument("--checkpoint", help="Checkpoint filename (e.g. checkpoint_final.pt)")
    dl_p.add_argument(
        "--dest", default="checkpoints", help="Destination folder (default: checkpoints)"
    )

    # upload
    up_p = subparsers.add_parser("upload", help="Upload a file to Hugging Face")
    up_p.add_argument("file", help="Path to local file")
    up_p.add_argument("--repo-id", help="Hugging Face repo ID")
    up_p.add_argument("--path", help="Target path in repo")
    up_p.add_argument("--run-id", help="Associate with run ID")

    args = parser.parse_args()

    commands = {
        "test": cmd_test,
        "list": cmd_list,
        "download": cmd_download,
        "upload": cmd_upload,
    }
    sys.exit(commands[args.command](args))


if __name__ == "__main__":
    main()
