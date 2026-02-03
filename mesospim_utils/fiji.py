#!/usr/bin/env python3
"""
ensure_fiji.py (Linux-only)

Run this at package startup. It ensures:
  - Fiji is present
  - BigStitcher update site is enabled
  - Updater has applied changes
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

BIGSTITCHER_SITE_NAME = "BigStitcher"
BIGSTITCHER_SITE_URL = "https://sites.imagej.net/BigStitcher/"

# Stable Build: 20250808-2217
DEFAULT_FIJI_ZIP_URL = "https://downloads.imagej.net/fiji/archive/stable/20250808-2217/fiji-stable-linux64-jdk.zip"


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


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
        try:
            lockfile.unlink(missing_ok=True)
        except Exception:
            pass


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
    exe = fiji_exe(root)
    stamp = root / ".fiji_bigstitcher_ready"
    return exe.exists() and stamp.exists()


def ensure_fiji_and_bigstitcher(force: bool = False) -> Path:
    root = resolve_install_root()
    lock = root / ".install.lock"
    fd = _acquire_lock(lock)
    try:
        if not force and is_fiji_ready(root):
            return root

        # Clean partial installs if forcing or broken
        if force and root.exists():
            for name in ("Fiji.app", ".fiji_bigstitcher_ready"):
                p = root / name
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)

        url = os.environ.get("FIJI_ZIP_URL", DEFAULT_FIJI_ZIP_URL)

        with tempfile.TemporaryDirectory(prefix="fiji_install_") as td:
            td = Path(td)
            zip_path = td / "fiji.zip"
            _download(url, zip_path)

            stage = td / "stage"
            _extract_zip(zip_path, stage)

            # Handle either direct Fiji.app or a top-level directory containing it
            candidates = [stage / "Fiji.app", *stage.rglob("Fiji.app")]
            app = next((c for c in candidates if c.exists() and c.is_dir()), None)
            if app is None:
                raise RuntimeError("Downloaded ZIP did not contain Fiji.app; check FIJI_ZIP_URL")

            root.mkdir(parents=True, exist_ok=True)
            target = root / "Fiji.app"
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            shutil.move(str(app), str(target))

        exe = fiji_exe(root)
        if not exe.exists():
            raise RuntimeError(f"Fiji installed but executable not found: {exe}")

        # Ensure executable bit (zip extraction may drop it)
        try:
            mode = exe.stat().st_mode
            exe.chmod(mode | 0o111)
        except Exception as e:
            raise RuntimeError(f"Failed to set executable bit on {exe}: {e}")


        # Enable BigStitcher update site + update
        _run([str(exe), "--update", "add-update-site", BIGSTITCHER_SITE_NAME, BIGSTITCHER_SITE_URL], root)  # :contentReference[oaicite:5]{index=5}
        _run([str(exe), "--update", "update"], root)  # :contentReference[oaicite:6]{index=6}

        # Stamp with some useful provenance
        db = root / "Fiji.app" / "db.xml.gz"
        stamp = root / ".fiji_bigstitcher_ready"
        stamp.write_text(
            "\n".join(
                [
                    f"fiji_zip_url={url}",
                    f"bigstitcher_site={BIGSTITCHER_SITE_URL}",
                    f"db_xml_gz_sha256={_sha256(db) if db.exists() else 'missing'}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        return root
    finally:
        _release_lock(fd, lock)


if __name__ == "__main__":
    root = ensure_fiji_and_bigstitcher(force=("--force" in os.sys.argv))
    print(str(root))
