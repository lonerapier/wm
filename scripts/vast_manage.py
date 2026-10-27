#!/usr/bin/env python3
"""Vast.ai Cloud Instance Automation Tool.

Powered by the official Vast.ai Python SDK (https://docs.vast.ai/sdk/python/quickstart).
Search GPU offers, launch instances, sync code, SSH, start training,
and manage instance lifecycles directly from your terminal.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

try:
    from vastai import VastAI
except ImportError:
    print("[ERROR] 'vastai' package is required. Install it with: pip install vastai")
    sys.exit(1)


def get_client() -> VastAI:
    """Initialize VastAI client with API key from environment, .env, or CLI config."""
    api_key = os.getenv("VAST_API_KEY", "").strip() or None
    try:
        return VastAI(api_key=api_key) if api_key else VastAI()
    except Exception as e:
        print(f"[ERROR] Failed to initialize VastAI client: {e}")
        print("Please set VAST_API_KEY in your .env file or run: vastai set api-key <KEY>")
        sys.exit(1)


def ensure_ssh_key(vast: VastAI, instance_id: Optional[int] = None) -> Optional[str]:
    """Ensure local public SSH key is uploaded to Vast.ai account and attached to instance."""
    candidates = [
        Path.home() / ".ssh" / "id_ed25519.pub",
        Path.home() / ".ssh" / "id_rsa.pub",
    ]
    pub_key_path = next((p for p in candidates if p.is_file()), None)
    if not pub_key_path:
        return None

    try:
        pub_key = pub_key_path.read_text().strip()
        user = vast.show_user()
        if not user.get("ssh_key"):
            vast.create_ssh_key(ssh_key=pub_key)
        if instance_id:
            vast.attach_ssh(instance_id=instance_id, ssh_key=pub_key)
        return pub_key
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------


def cmd_user(args: argparse.Namespace) -> int:
    """Show authenticated Vast.ai account details and balance."""
    vast = get_client()
    try:
        user = vast.show_user()
        print("Vast.ai Account Details:")
        print(f"  User ID : {user.get('id', 'N/A')}")
        print(f"  Email   : {user.get('email', 'N/A')}")
        print(f"  Balance : ${user.get('credit', 0.0):.2f}")
        print(f"  SSH Key : {'Configured' if user.get('ssh_key') else 'Not configured'}")
        return 0
    except Exception as e:
        print(f"[ERROR] Failed to fetch user info: {e}")
        return 1


def cmd_search(args: argparse.Namespace) -> int:
    """Search for GPU offers using the official Vast.ai SDK."""
    vast = get_client()

    query_parts = ["rentable=true", "num_gpus=1"]
    if args.verified:
        query_parts.append("verified=true")
    if args.gpu:
        # Normalize e.g. "RTX 4090" -> "RTX_4090"
        gpu_clean = args.gpu.strip().replace(" ", "_")
        query_parts.append(f"gpu_name={gpu_clean}")
    if args.max_dph is not None:
        query_parts.append(f"dph_total<={args.max_dph}")
    if args.min_vram:
        query_parts.append(f"gpu_ram>={args.min_vram * 1024}")
    if args.direct_port:
        query_parts.append("direct_port_count>=1")

    query_str = " ".join(query_parts)
    print(f"Searching Vast.ai with query: '{query_str}' ...")

    try:
        offers = vast.search_offers(
            query=query_str,
            order=args.order,
            limit=args.limit,
        )
    except Exception as e:
        print(f"[ERROR] search_offers failed: {e}")
        return 1

    if not offers or not isinstance(offers, list):
        print(
            "\nNo matching offers found. Try relaxing your filters (e.g. higher max $/hr or removing --verified)."
        )
        return 0

    print("\n" + "=" * 105)
    print(
        f"{'Offer ID':<10} {'GPU Model':<22} {'VRAM':<10} {'$/hr':<8} "
        f"{'DL/UL (Mbps)':<16} {'Host Reliability':<18} {'Location':<10}"
    )
    print("=" * 105)

    for o in offers:
        offer_id = o.get("id")
        gpu_name = o.get("gpu_name", "Unknown")
        vram_mb = o.get("gpu_ram", 0)
        vram = f"{int(vram_mb / 1024)}GB" if vram_mb else "N/A"
        dph = f"${o.get('dph_total', 0.0):.3f}"
        net = f"{int(o.get('inet_down', 0))}/{int(o.get('inet_up', 0))}"
        rel_val = o.get("reliability2", o.get("reliability", 0.0))
        rel = f"{rel_val * 100:.1f}%" if isinstance(rel_val, (int, float)) else str(rel_val)
        loc = str(o.get("geolocation", "Unknown"))[:10]
        print(f"{offer_id:<10} {gpu_name:<22} {vram:<10} {dph:<8} {net:<16} {rel:<18} {loc:<10}")

    print("=" * 105)
    print(f"Showing {len(offers)} offers. To launch an instance:")
    print("  uv run scripts/vast_manage.py launch <OFFER_ID>")
    return 0


def cmd_launch(args: argparse.Namespace) -> int:
    """Launch an instance using VastAI.create_instance()."""
    vast = get_client()
    offer_id = int(args.offer_id)
    disk_gb = float(args.disk)

    ensure_ssh_key(vast)

    create_kwargs: Dict[str, Any] = {
        "id": offer_id,
        "disk": disk_gb,
        "ssh": True,
        "direct": True,
    }

    if args.template:
        template_val = str(args.template).strip()
        # Resolve numeric ID to hash_id if needed
        try:
            my_templates = vast.search_templates(query="private=true")
            for t in my_templates:
                if str(t.get("id")) == template_val or t.get("hash_id") == template_val:
                    template_val = t.get("hash_id") or str(t.get("id"))
                    break
        except Exception:
            pass

        print(f"Creating instance from Offer #{offer_id} using Template '{template_val}'...")
        create_kwargs["template_hash"] = template_val
    else:
        print(f"Creating instance from Offer #{offer_id} using Docker image '{args.image}'...")
        create_kwargs["image"] = args.image

    print(f"  Disk Space  : {disk_gb} GB")

    try:
        resp = vast.create_instance(**create_kwargs)
    except Exception as e:
        print(f"[ERROR] create_instance failed: {e}")
        return 1

    new_id = resp.get("new_contract") if isinstance(resp, dict) else None
    if resp and (resp.get("success") or new_id):
        print(f"\n SUCCESS! Created Instance #{new_id}")
        print("The instance is now provisioning (typically takes 1-3 minutes).")
        print("\nNext steps:")
        print("  1. Check status : uv run scripts/vast_manage.py instances")
        print(f"  2. Sync code    : uv run scripts/vast_manage.py sync {new_id}")
        print(f"  3. Connect SSH  : uv run scripts/vast_manage.py ssh {new_id}")
        return 0
    else:
        print(f"[ERROR] Launch failed: {resp}")
        return 1


def cmd_templates(args: argparse.Namespace) -> int:
    """List templates created on your Vast.ai account."""
    vast = get_client()
    try:
        templates = vast.search_templates(query="private=true")
    except Exception as e:
        print(f"[ERROR] search_templates failed: {e}")
        return 1

    if not templates:
        print("No custom templates found. Create one with:")
        print("  uv run scripts/vast_manage.py create-template --name 'wm-trainer'")
        return 0

    print("\n" + "=" * 105)
    print(f"{'Template ID':<14} {'Hash ID':<34} {'Name':<24} {'Docker Image':<32}")
    print("=" * 105)
    for t in templates:
        t_id = str(t.get("id", "N/A"))
        t_hash = str(t.get("hash_id", "N/A"))
        t_name = str(t.get("name", "Unnamed"))[:23]
        t_image = str(t.get("image_name") or t.get("image") or "N/A")[:31]
        print(f"{t_id:<14} {t_hash:<34} {t_name:<24} {t_image:<32}")
    print("=" * 105)
    print(f"Showing {len(templates)} templates. To launch an offer with a template:")
    print("  uv run scripts/vast_manage.py launch <OFFER_ID> --template <TEMPLATE_ID_OR_HASH>")
    return 0


def cmd_create_template(args: argparse.Namespace) -> int:
    """Create a reusable Vast.ai template via VastAI SDK."""
    vast = get_client()
    print(f"Creating Vast.ai template '{args.name}'...")
    print(f"  Docker Image: {args.image}")
    print(f"  Disk Space  : {args.disk} GB")
    if args.onstart:
        print(f"  Onstart Cmd : {args.onstart}")

    try:
        resp = vast.create_template(
            name=args.name,
            image=args.image,
            disk_space=float(args.disk),
            onstart_cmd=args.onstart,
            desc=args.desc or "World Models training template",
            ssh=True,
            direct=True,
        )
    except Exception as e:
        print(f"[ERROR] create_template failed: {e}")
        return 1

    if resp and resp.get("success"):
        tmpl = resp.get("template", {})
        t_id = tmpl.get("id") or resp.get("id")
        t_hash = tmpl.get("hash_id")
        print(f"\n SUCCESS! Template '{args.name}' created.")
        print(f"  Template ID : {t_id}")
        print(f"  Hash ID     : {t_hash}")
        print("\nYou can now launch instances instantly with:")
        print(f"  uv run scripts/vast_manage.py launch <OFFER_ID> --template {t_id}")
        return 0
    else:
        print(f"[ERROR] Template creation failed: {resp}")
        return 1


def cmd_instances(args: argparse.Namespace) -> int:
    """List instances using VastAI.show_instances()."""
    vast = get_client()
    try:
        instances = vast.show_instances()
    except Exception as e:
        print(f"[ERROR] show_instances failed: {e}")
        return 1

    if not instances:
        print("No active Vast.ai instances found.")
        return 0

    print("\n" + "=" * 115)
    print(
        f"{'ID':<10} {'Status':<14} {'GPU Model':<20} {'$/hr':<8} "
        f"{'SSH Host':<28} {'SSH Port':<10} {'SSH Command'}"
    )
    print("=" * 115)

    for inst in instances:
        inst_id = str(inst.get("id", "N/A"))
        status = str(inst.get("actual_status") or inst.get("status_msg") or "loading")
        num_gpus = inst.get("num_gpus") or 1
        gpu_name = inst.get("gpu_name") or "GPU"
        gpu = f"{num_gpus}x {gpu_name}"
        dph_val = inst.get("dph_total") or 0.0
        dph = f"${dph_val:.3f}"
        host = inst.get("ssh_host") or "N/A"
        port = str(inst.get("ssh_port") or "N/A")
        ssh_cmd = f"ssh -p {port} root@{host}" if host != "N/A" else "Starting up..."
        print(f"{inst_id:<10} {status:<14} {gpu:<20} {dph:<8} {host:<28} {port:<10} {ssh_cmd}")

    print("=" * 115)
    return 0


def get_instance_info(vast: VastAI, instance_id: int | str) -> Dict[str, Any]:
    try:
        inst = vast.show_instance(id=int(instance_id))
        if inst:
            return inst
    except Exception:
        pass

    instances = vast.show_instances()
    for inst in instances:
        if str(inst.get("id")) == str(instance_id):
            return inst

    print(f"[ERROR] Instance #{instance_id} not found.")
    sys.exit(1)


def cmd_sync(args: argparse.Namespace) -> int:
    """Rsync local project code and .env to the remote instance."""
    vast = get_client()
    inst = get_instance_info(vast, args.instance_id)

    host = inst.get("ssh_host")
    port = inst.get("ssh_port")

    if not host or not port:
        print(
            f"[ERROR] Instance #{args.instance_id} is not ready for SSH yet (status: {inst.get('actual_status')})."
        )
        print("Please wait for the container to reach 'running' status.")
        return 1

    ensure_ssh_key(vast, instance_id=int(args.instance_id))

    remote_dest = f"root@{host}:/workspace/wm"
    print(f"Syncing local project to {remote_dest} (Port: {port})...")

    ssh_opts = f"-p {port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
    subprocess.run(
        [
            "ssh",
            "-p",
            str(port),
            "-o",
            "StrictHostKeyChecking=no",
            f"root@{host}",
            "mkdir -p /workspace/wm",
        ],
        check=False,
    )

    excludes = [
        "--exclude=.git",
        "--exclude=.venv",
        "--exclude=__pycache__",
        "--exclude=*.pyc",
        "--exclude=exp",
        "--exclude=wandb",
        "--exclude=data",
        "--exclude=.pytest_cache",
        "--exclude=wheels",
        "--exclude=build",
    ]

    rsync_cmd = [
        "rsync",
        "-avz",
        "-e",
        f"ssh {ssh_opts}",
        *excludes,
        f"{PROJECT_ROOT}/",
        f"root@{host}:/workspace/wm/",
    ]

    res = subprocess.run(rsync_cmd)
    if res.returncode == 0:
        env_file = PROJECT_ROOT / ".env"
        if env_file.is_file():
            print("Syncing .env file...")
            subprocess.run(
                [
                    "scp",
                    "-P",
                    str(port),
                    "-o",
                    "StrictHostKeyChecking=no",
                    str(env_file),
                    f"root@{host}:/workspace/wm/.env",
                ],
                check=False,
            )
        print("\n Code sync complete!")
        print("To setup and run on the instance:")
        print(f"  ssh -p {port} root@{host} 'cd /workspace/wm && bash scripts/setup_instance.sh'")
        return 0
    else:
        print("[ERROR] rsync failed.")
        return res.returncode


def cmd_ssh(args: argparse.Namespace) -> int:
    """Open interactive SSH connection to instance."""
    vast = get_client()
    inst = get_instance_info(vast, args.instance_id)

    host = inst.get("ssh_host")
    port = inst.get("ssh_port")

    if not host or not port:
        print(
            f"[ERROR] Instance #{args.instance_id} does not have an SSH endpoint yet (status: {inst.get('actual_status')})."
        )
        return 1

    cmd = ["ssh", "-p", str(port), "-o", "StrictHostKeyChecking=no", f"root@{host}"]
    print(f"Connecting to Instance #{args.instance_id} via: {' '.join(cmd)}")
    subprocess.run(cmd)
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    """Stop instance via VastAI.stop_instance()."""
    vast = get_client()
    print(f"Stopping instance #{args.instance_id} via VastAI SDK...")
    try:
        resp = vast.stop_instance(id=int(args.instance_id))
        print("Response:", resp)
        return 0
    except Exception as e:
        print(f"[ERROR] stop_instance failed: {e}")
        return 1


def cmd_destroy(args: argparse.Namespace) -> int:
    """Destroy instance via VastAI.destroy_instance()."""
    vast = get_client()
    confirm = input(
        f"Are you sure you want to PERMANENTLY destroy instance #{args.instance_id}? (y/N): "
    )
    if confirm.lower() != "y":
        print("Aborted.")
        return 0
    print(f"Destroying instance #{args.instance_id} via VastAI SDK...")
    try:
        resp = vast.destroy_instance(id=int(args.instance_id))
        print("Response:", resp)
        return 0
    except Exception as e:
        print(f"[ERROR] destroy_instance failed: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Vast.ai Instance Manager using Vast.ai Python SDK"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # user / whoami
    subparsers.add_parser("user", help="Show Vast.ai user info and account balance")

    # search
    search_p = subparsers.add_parser("search", help="Search GPU offers")
    search_p.add_argument(
        "--gpu", default="RTX_4090", help="GPU filter (e.g. RTX_4090, RTX_3090, A100)"
    )
    search_p.add_argument("--max-dph", type=float, help="Maximum price $/hour")
    search_p.add_argument("--min-vram", type=int, help="Minimum VRAM in GB")
    search_p.add_argument(
        "--verified", action="store_true", default=False, help="Filter to verified hosts only"
    )
    search_p.add_argument(
        "--direct-port",
        action="store_true",
        default=False,
        help="Filter to instances with direct port count >= 1",
    )
    search_p.add_argument(
        "--order", default="dlperf_usd-", help="Order query (default: dlperf_usd-)"
    )
    search_p.add_argument("--limit", type=int, default=15, help="Number of offers to display")

    # launch
    launch_p = subparsers.add_parser("launch", help="Launch instance from offer ID")
    launch_p.add_argument("offer_id", help="Offer ID from search results")
    launch_p.add_argument("--disk", type=int, default=60, help="Disk size in GB (default: 60)")
    launch_p.add_argument(
        "--image",
        default="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
        help="Docker image (default: pre-cached PyTorch 2.5.1 CUDA 12.4)",
    )
    launch_p.add_argument("--template", help="Template ID or Hash to use instead of base image")

    # templates
    subparsers.add_parser("templates", help="List your custom Vast.ai templates")

    # create-template
    ct_p = subparsers.add_parser("create-template", help="Create a new reusable Vast.ai template")
    ct_p.add_argument("--name", default="wm-trainer", help="Template name (default: wm-trainer)")
    ct_p.add_argument(
        "--image",
        default="pytorch/pytorch:2.14.1-cuda12.6-cudnn9-devel",
        help="Docker image repository:tag",
    )
    ct_p.add_argument("--disk", type=int, default=60, help="Default disk space in GB (default: 60)")
    ct_p.add_argument("--onstart", help="Startup script/command to execute when instance boots")
    ct_p.add_argument("--desc", help="Template description")

    # instances / status
    subparsers.add_parser("instances", help="List all instances")
    subparsers.add_parser("status", help="Alias for instances")

    # sync
    sync_p = subparsers.add_parser("sync", help="Rsync code and .env to instance")
    sync_p.add_argument("instance_id", help="Instance ID")

    # ssh
    ssh_p = subparsers.add_parser("ssh", help="Connect via SSH to instance")
    ssh_p.add_argument("instance_id", help="Instance ID")

    # stop
    stop_p = subparsers.add_parser("stop", help="Stop instance (pause compute billing)")
    stop_p.add_argument("instance_id", help="Instance ID")

    # destroy
    destroy_p = subparsers.add_parser("destroy", help="Permanently destroy instance")
    destroy_p.add_argument("instance_id", help="Instance ID")

    args = parser.parse_args()

    commands = {
        "user": cmd_user,
        "search": cmd_search,
        "launch": cmd_launch,
        "templates": cmd_templates,
        "create-template": cmd_create_template,
        "instances": cmd_instances,
        "status": cmd_instances,
        "sync": cmd_sync,
        "ssh": cmd_ssh,
        "stop": cmd_stop,
        "destroy": cmd_destroy,
    }

    sys.exit(commands[args.command](args))


if __name__ == "__main__":
    main()
