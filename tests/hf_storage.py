from pathlib import Path
from unittest.mock import MagicMock, patch

from hf_storage import (
    HFConfig,
    HFSyncManager,
    download_file,
    is_hf_configured,
    upload_file,
)


def test_hf_config_from_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test_token_123")
    monkeypatch.setenv("HF_REPO_ID", "testuser/test-wm-repo")
    monkeypatch.setenv("HF_REPO_TYPE", "model")
    monkeypatch.setenv("HF_PRIVATE", "true")
    monkeypatch.setenv("HF_PREFIX", "experiments")

    cfg = HFConfig.from_env()
    assert cfg.token == "hf_test_token_123"
    assert cfg.repo_id == "testuser/test-wm-repo"
    assert cfg.repo_type == "model"
    assert cfg.private is True
    assert cfg.prefix == "experiments"
    assert cfg.is_valid()
    assert is_hf_configured()


def test_hf_config_incomplete(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_REPO_ID", raising=False)

    cfg = HFConfig.from_env()
    assert not cfg.is_valid()
    assert not is_hf_configured()


def test_hf_upload_and_download(tmp_path: Path):
    mock_api = MagicMock()
    test_file = tmp_path / "checkpoint_test.pt"
    test_file.write_text("model weights")

    cfg = HFConfig(
        token="hf_fake_token",
        repo_id="testuser/repo",
        repo_type="model",
    )

    with patch("hf_storage.get_hf_api", return_value=mock_api):
        # Test upload
        uploaded = upload_file(
            local_path=test_file,
            path_in_repo="runs/exp1/checkpoints/test.pt",
            config=cfg,
        )
        assert uploaded is True
        mock_api.upload_file.assert_called_once()

    dest_file = tmp_path / "downloaded.pt"
    with patch("huggingface_hub.hf_hub_download", return_value=str(dest_file)):
        res = download_file(
            path_in_repo="runs/exp1/checkpoints/test.pt",
            local_path=dest_file,
            config=cfg,
        )
        assert res == dest_file


def test_hf_sync_manager(tmp_path: Path):
    mock_api = MagicMock()
    test_ckpt = tmp_path / "ckpt.pt"
    test_ckpt.write_text("weights")

    cfg = HFConfig(
        token="hf_fake_token",
        repo_id="testuser/repo",
        repo_type="model",
        prefix="runs",
    )

    with patch("hf_storage.get_hf_api", return_value=mock_api):
        manager = HFSyncManager(config=cfg)
        assert manager.enabled

        uri = manager.upload_checkpoint(test_ckpt, run_id="exp_01", async_upload=False)
        assert uri == "hf://testuser/repo/runs/exp_01/checkpoints/ckpt.pt"
