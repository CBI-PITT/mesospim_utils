#!/usr/bin/env python3
"""
fiji.py (Linux-only)

Run before requiring bigstitcher this module ensures:
  - Fiji is present
  - BigStitcher update site is enabled
  - Updater has applied changes
  - Marks installation as ready

Installation of fiji and bigstitcher is managed automatically unless specified on the config file.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

from constants import FIJI_INSTALL_LOCATION

BIGSTITCHER_SITE_NAME = "BigStitcher"
BIGSTITCHER_SITE_URL = "https://sites.imagej.net/BigStitcher/"

# Stable Build: 20250808-2217
DEFAULT_FIJI_ZIP_URL = (
    "https://downloads.imagej.net/fiji/archive/stable/20250808-2217/"
    "fiji-stable-linux64-jdk.zip"
)


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(
        cmd,
        cwd=str(cwd),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _acquire_lock(lockfile: Path, timeout_s: int = 180) -> int:
    deadline = time.time() + timeout_s
    lockfile.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(str(lockfile), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            return fd
        except FileExistsError:
            if time.time() > deadline:
                raise TimeoutError(f"Timed out waiting for lock: {lockfile}")
            time.sleep(0.2)


def _release_lock(fd: int, lockfile: Path) -> None:
    try:
        os.close(fd)
    finally:
        lockfile.unlink(missing_ok=True)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, dest.open("wb") as f:
        shutil.copyfileobj(r, f)


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)


def _find_repo_root(start: Path) -> Path | None:
    cur = start.resolve()
    for p in [cur, *cur.parents]:
        if (p / ".git").exists() or (p / "pyproject.toml").exists():
            return p
    return None


def resolve_install_root() -> Path:
    """
    Reproducible location:
      1) FIJI_ROOT env var (recommended)
      2) <git repo>/fiji-linux if running from a checkout
      3) current working dir ./fiji-linux
    """
    if os.environ.get("FIJI_ROOT"):
        return Path(os.environ["FIJI_ROOT"]).expanduser().resolve()

    repo = _find_repo_root(Path(__file__).parent)
    if repo is not None:
        return (repo / "fiji-linux").resolve()

    return (Path.cwd() / "fiji-linux").resolve()


def fiji_exe(root: Path) -> Path:
    return root / "Fiji.app" / "ImageJ-linux64"


def is_fiji_ready(root: Path) -> bool:
    return fiji_exe(root).exists() and (root / ".fiji_bigstitcher_ready").exists()


def ensure_fiji_and_bigstitcher(force: bool = False) -> Path:
    root = resolve_install_root()

    print(f'Root: {root}')
    if root != FIJI_INSTALL_LOCATION.parent:
        print(f"[fiji] Warning: Using custom fiji install from the config file")
        print(f"[fiji] Warning: The user is responsible for ensuring fiji and bigstitcher are install at: {FIJI_INSTALL_LOCATION}")
        return FIJI_INSTALL_LOCATION

    if not force and is_fiji_ready(root):
        print(f"[fiji] Fiji + BigStitcher already installed at: {root}")
        return root

    print(f"[fiji] Fiji installation required at: {root}")

    lock = root / ".install.lock"
    fd = _acquire_lock(lock)
    try:
        if force:
            print("[fiji] Force install requested – removing existing installation")
            for name in ("Fiji.app", ".fiji_bigstitcher_ready"):
                p = root / name
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)

        url = os.environ.get("FIJI_ZIP_URL", DEFAULT_FIJI_ZIP_URL)
        print(f"[fiji] Downloading Fiji from:\n        {url}")

        with tempfile.TemporaryDirectory(prefix="fiji_install_") as td:
            td = Path(td)
            zip_path = td / "fiji.zip"
            _download(url, zip_path)
            print("[fiji] Download complete")

            print("[fiji] Extracting Fiji archive")
            stage = td / "stage"
            _extract_zip(zip_path, stage)

            candidates = [stage / "Fiji.app", *stage.rglob("Fiji.app")]
            app = next((c for c in candidates if c.exists() and c.is_dir()), None)
            if app is None:
                raise RuntimeError("Downloaded ZIP did not contain Fiji.app")

            root.mkdir(parents=True, exist_ok=True)
            target = root / "Fiji.app"
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            shutil.move(str(app), str(target))

        exe = fiji_exe(root)
        if not exe.exists():
            raise RuntimeError(f"Fiji executable not found: {exe}")

        print("[fiji] Fixing executable permissions")
        exe.chmod(exe.stat().st_mode | 0o111)

        print("[fiji] Enabling BigStitcher update site")
        _run(
            [str(exe), "--update", "add-update-site",
             BIGSTITCHER_SITE_NAME, BIGSTITCHER_SITE_URL],
            root,
        )

        print("[fiji] Updating Fiji (installing BigStitcher and dependencies)")
        _run([str(exe), "--update", "update"], root)

        db = root / "Fiji.app" / "db.xml.gz"
        stamp = root / ".fiji_bigstitcher_ready"
        stamp.write_text(
            "\n".join(
                [
                    f"fiji_zip_url={url}",
                    f"bigstitcher_site={BIGSTITCHER_SITE_URL}",
                    f"db_xml_gz_sha256={_sha256(db) if db.exists() else 'missing'}",
                ]
            ) + "\n",
            encoding="utf-8",
        )

        print("[fiji] ✅ Fiji + BigStitcher installation complete")
        print(f"[fiji] Installed at: {root}")

        return root

    finally:
        _release_lock(fd, lock)


if __name__ == "__main__":
    root = ensure_fiji_and_bigstitcher(force=("--force" in os.sys.argv))
    print(root)
