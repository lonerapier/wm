from __future__ import annotations

import concurrent.futures
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


@dataclass
class HFConfig:
    token: Optional[str] = None
    repo_id: Optional[str] = None
    repo_type: str = "model"  # "model" or "dataset"
    private: bool = True
    prefix: str = "runs"

    @classmethod
    def from_env(cls) -> HFConfig:
        token = os.getenv("HF_TOKEN") or None
        repo_id = os.getenv("HF_REPO_ID") or None
        repo_type = os.getenv("HF_REPO_TYPE", "model").lower()
        private_str = os.getenv("HF_PRIVATE", "true").lower()
        private = private_str in ("true", "1", "yes")
        prefix = os.getenv("HF_PREFIX", "runs").strip("/")

        return cls(
            token=token,
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            prefix=prefix,
        )

    def is_valid(self) -> bool:
        return bool(self.token and self.repo_id)


def is_hf_configured() -> bool:
    """Check if required Hugging Face environment variables are present."""
    return HFConfig.from_env().is_valid()


def get_hf_api(token: Optional[str] = None):
    """Return an authenticated HfApi client."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise ImportError(
            "huggingface_hub is required. Install it with: pip install huggingface-hub"
        )
    cfg = HFConfig.from_env()
    t = token or cfg.token
    return HfApi(token=t)


def ensure_repo_exists(
    repo_id: str,
    repo_type: str = "model",
    private: bool = True,
    token: Optional[str] = None,
) -> bool:
    """Ensure that the target Hugging Face repository exists, creating it if necessary."""
    api = get_hf_api(token)
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        return True
    except Exception:
        pass

    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            exist_ok=True,
        )
        print(f"[HF] Created repository {repo_type} '{repo_id}' (private={private})")
        return True
    except Exception as e:
        print(f"[HF] Warning: Could not verify/create repo '{repo_id}': {e}")
        return False


def test_hf_connection(config: Optional[HFConfig] = None) -> bool:
    """Validate Hugging Face token and repository access."""
    cfg = config or HFConfig.from_env()
    if not cfg.token:
        print("[HF] HF_TOKEN is not set.")
        return False

    api = get_hf_api(cfg.token)
    try:
        user_info = api.whoami()
        username = user_info.get("name", "Unknown")
        print(f"[HF] Authenticated as Hugging Face user: '{username}'")

        if cfg.repo_id:
            ensure_repo_exists(
                repo_id=cfg.repo_id,
                repo_type=cfg.repo_type,
                private=cfg.private,
                token=cfg.token,
            )
            print(f"[HF] Verified access to repo: '{cfg.repo_id}' ({cfg.repo_type})")
        return True
    except Exception as e:
        print(f"[HF] Connection failed: {e}")
        return False


def upload_file(
    local_path: str | Path,
    path_in_repo: str,
    repo_id: Optional[str] = None,
    repo_type: Optional[str] = None,
    token: Optional[str] = None,
    config: Optional[HFConfig] = None,
) -> bool:
    """Upload a local file to the Hugging Face repository."""
    cfg = config or HFConfig.from_env()
    r_id = repo_id or cfg.repo_id
    r_type = repo_type or cfg.repo_type
    t = token or cfg.token

    if not r_id:
        raise ValueError("HF repo_id is required for uploading.")

    local_path = Path(local_path)
    if not local_path.is_file():
        raise FileNotFoundError(f"Local file does not exist: {local_path}")

    api = get_hf_api(t)
    try:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=r_id,
            repo_type=r_type,
            commit_message=f"Upload {local_path.name}",
        )
        return True
    except Exception as e:
        print(f"[HF] Failed to upload {local_path} to {r_id}/{path_in_repo}: {e}")
        return False


def download_file(
    path_in_repo: str,
    local_path: str | Path,
    repo_id: Optional[str] = None,
    repo_type: Optional[str] = None,
    token: Optional[str] = None,
    config: Optional[HFConfig] = None,
) -> Path:
    """Download a file from the Hugging Face repository."""
    from huggingface_hub import hf_hub_download

    cfg = config or HFConfig.from_env()
    r_id = repo_id or cfg.repo_id
    r_type = repo_type or cfg.repo_type
    t = token or cfg.token

    if not r_id:
        raise ValueError("HF repo_id is required for downloading.")

    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    downloaded = hf_hub_download(
        repo_id=r_id,
        filename=path_in_repo,
        repo_type=r_type,
        token=t,
        local_dir=str(local_path.parent),
    )
    return Path(downloaded)


def list_repo_files(
    prefix: str = "",
    repo_id: Optional[str] = None,
    repo_type: Optional[str] = None,
    token: Optional[str] = None,
    config: Optional[HFConfig] = None,
) -> List[Dict[str, Any]]:
    """List files in the Hugging Face repository."""
    cfg = config or HFConfig.from_env()
    r_id = repo_id or cfg.repo_id
    r_type = repo_type or cfg.repo_type
    t = token or cfg.token

    if not r_id:
        raise ValueError("HF repo_id is required.")

    api = get_hf_api(t)
    try:
        all_files = api.list_repo_files(repo_id=r_id, repo_type=r_type)
        filtered = [f for f in all_files if f.startswith(prefix)]
        return [{"path": f} for f in filtered]
    except Exception as e:
        print(f"[HF] Failed to list repo files: {e}")
        return []


class HFSyncManager:
    """Manages asynchronous checkpoint and experiment artifact uploads to Hugging Face Hub."""

    def __init__(
        self,
        config: Optional[HFConfig] = None,
        repo_id: Optional[str] = None,
        repo_type: Optional[str] = None,
        prefix: Optional[str] = None,
        max_workers: int = 2,
    ):
        self.config = config or HFConfig.from_env()
        self.repo_id = repo_id or self.config.repo_id
        self.repo_type = repo_type or self.config.repo_type
        self.prefix = (prefix or self.config.prefix or "runs").strip("/")
        self.enabled = bool(self.config.token and self.repo_id)

        self._executor = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="hf_sync"
            )
            if self.enabled
            else None
        )
        self._futures: List[concurrent.futures.Future] = []

        if self.enabled:
            # Ensure repo exists on initialization
            ensure_repo_exists(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                private=self.config.private,
                token=self.config.token,
            )

    def get_run_prefix(self, run_id: str) -> str:
        return f"{self.prefix}/{run_id}" if self.prefix else run_id

    def upload_file_async(
        self,
        local_path: str | Path,
        path_in_repo: str,
        on_success_msg: Optional[str] = None,
    ) -> Optional[concurrent.futures.Future]:
        if not self.enabled or self._executor is None:
            return None

        local_path = Path(local_path)
        if not local_path.is_file():
            return None

        def _task():
            success = upload_file(
                local_path=local_path,
                path_in_repo=path_in_repo,
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                token=self.config.token,
                config=self.config,
            )
            if success:
                msg = (
                    on_success_msg
                    or f"[HF] Uploaded {local_path.name} -> hf://{self.repo_id}/{path_in_repo}"
                )
                print(msg)
            return success

        future = self._executor.submit(_task)
        self._futures.append(future)
        return future

    def upload_checkpoint(
        self,
        local_path: str | Path,
        run_id: str,
        filename: Optional[str] = None,
        async_upload: bool = True,
    ) -> Optional[str]:
        """Upload a checkpoint to runs/<run_id>/checkpoints/<filename> on Hugging Face."""
        if not self.enabled:
            return None

        local_path = Path(local_path)
        name = filename or local_path.name
        run_prefix = self.get_run_prefix(run_id)
        path_in_repo = f"{run_prefix}/checkpoints/{name}"
        success_msg = f"[HF] Synced checkpoint {name} -> https://huggingface.co/{self.repo_id}/blob/main/{path_in_repo}"

        if async_upload:
            self.upload_file_async(local_path, path_in_repo, on_success_msg=success_msg)
        else:
            if upload_file(
                local_path=local_path,
                path_in_repo=path_in_repo,
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                token=self.config.token,
                config=self.config,
            ):
                print(success_msg)
            else:
                return None
        return f"hf://{self.repo_id}/{path_in_repo}"

    def upload_run_dir(
        self,
        run_dir: str | Path,
        run_id: str,
        async_upload: bool = True,
    ) -> None:
        """Upload experiment metadata (config.json, summary.json, metrics.jsonl) to Hugging Face."""
        if not self.enabled:
            return

        run_dir = Path(run_dir)
        run_prefix = self.get_run_prefix(run_id)

        for filename in ["config.json", "summary.json", "metrics.jsonl"]:
            p = run_dir / filename
            if p.is_file():
                path_in_repo = f"{run_prefix}/{filename}"
                if async_upload:
                    self.upload_file_async(
                        p,
                        path_in_repo,
                        on_success_msg=f"[HF] Synced {filename} -> https://huggingface.co/{self.repo_id}/blob/main/{path_in_repo}",
                    )
                else:
                    upload_file(
                        local_path=p,
                        path_in_repo=path_in_repo,
                        repo_id=self.repo_id,
                        repo_type=self.repo_type,
                        token=self.config.token,
                        config=self.config,
                    )

    def download_checkpoint(
        self,
        run_id: str,
        filename: str,
        local_dir: str | Path,
    ) -> Path:
        """Download a checkpoint from Hugging Face."""
        run_prefix = self.get_run_prefix(run_id)
        path_in_repo = f"{run_prefix}/checkpoints/{filename}"
        local_path = Path(local_dir) / filename
        download_file(
            path_in_repo=path_in_repo,
            local_path=local_path,
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            token=self.config.token,
            config=self.config,
        )
        return local_path

    def get_latest_checkpoint(
        self,
        run_id: str,
        local_dir: str | Path,
    ) -> Optional[Path]:
        """Find the latest checkpoint for a run in Hugging Face and download it."""
        run_prefix = self.get_run_prefix(run_id)
        ckpt_prefix = f"{run_prefix}/checkpoints/"
        files = list_repo_files(
            prefix=ckpt_prefix,
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            token=self.config.token,
            config=self.config,
        )
        if not files:
            return None

        # Prefer checkpoint_final.pt if present, else highest/last file
        chosen_path = None
        for f in files:
            p = f["path"]
            if p.endswith("checkpoint_final.pt"):
                chosen_path = p
                break
        if not chosen_path:
            chosen_path = sorted([f["path"] for f in files])[-1]

        filename = chosen_path.split("/")[-1]
        local_path = Path(local_dir) / filename
        download_file(
            path_in_repo=chosen_path,
            local_path=local_path,
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            token=self.config.token,
            config=self.config,
        )
        return local_path

    def wait_for_uploads(self, timeout: Optional[float] = None) -> None:
        """Wait for pending asynchronous uploads to finish."""
        if not self._futures:
            return
        done, not_done = concurrent.futures.wait(self._futures, timeout=timeout)
        self._futures = list(not_done)
        for fut in done:
            try:
                fut.result()
            except Exception as e:
                print(f"[HF] Asynchronous upload failed: {e}")

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown background thread pool."""
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            self._executor = None
