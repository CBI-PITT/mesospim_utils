#!/usr/bin/env python3
"""
fiji.py (Linux-only)

Run before requiring bigstitcher this module ensures:
  - Fiji is present
  - BigStitcher update site is enabled
  - Updater has applied changes
  - Classic BigStitcher is installed from pinned Maven artifacts: the update
    site removed classic BigStitcher on 2026-09-04 and now only ships the
    new-generation multiview_reconstruction, and the updater ignores files
    that are not already installed locally
  - Jars shipped by base Fiji that cannot load on the bundled Java 8 are
    disabled in place rather than deleted
  - BigStitcher is verified loadable headless before the install is marked ready

Installation of fiji and bigstitcher is managed automatically unless specified on the config file.
"""

from __future__ import annotations

import hashlib
import gzip
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

# Classic BigStitcher is no longer published on the BigStitcher update site, so
# the last Java-8-compatible set is pinned from the SciJava Maven repository.
# Base Fiji additionally ships a multiview-reconstruction jar compiled for
# Java > 8 that cannot load on the bundled Java 8 and would shadow the pinned
# Java 8 jar; conflicting jars are disabled in place, not deleted.
MAVEN_BASE_URL = "https://maven.scijava.org/content/groups/public"
BIGSTITCHER_PINNED_JARS = [
    # (maven path under MAVEN_BASE_URL, target path under the Fiji root,
    #  conflict globs relative to the Fiji root, disabled before installing)
    (
        "net/preibisch/BigStitcher/2.5.3/BigStitcher-2.5.3.jar",
        "plugins/Big_Stitcher-2.5.3.jar",
        (),
    ),
    (
        "net/preibisch/multiview-reconstruction/7.0.7/multiview-reconstruction-7.0.7.jar",
        "plugins/multiview_reconstruction-7.0.7.jar",
        ("jars/multiview-reconstruction*.jar",),
    ),
    (
        "net/preibisch/multiview-simulation/0.2.0/multiview-simulation-0.2.0.jar",
        "jars/multiview-simulation-0.2.0.jar",
        ("jars/multiview-simulation*.jar", "plugins/multiview-simulation*.jar"),
    ),
]
BIGSTITCHER_IMPORT_CHECK = "from net.preibisch.stitcher.plugin import BigStitcher"


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
        # Force the data to the storage server before returning: on network
        # filesystems a buffered write can be visible to this process while
        # other processes still see nothing (or a truncated file).
        f.flush()
        os.fsync(f.fileno())


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


def bigstitcher_jar(root: Path) -> Path | None:
    for p in sorted((root / "Fiji.app" / "plugins").glob("Big_[Ss]titcher*.jar")):
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def _jar_is_valid(path: Path) -> bool:
    """True only if the file opens as a zip with no corrupt entries.

    A bare is_file() is not trusted here: network filesystems can report
    stale directory entries for files that are unreadable or missing."""
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None
    except Exception:
        return False


def _disable_conflicting_jars(root: Path, globs: tuple[str, ...], keep: Path) -> None:
    """Rename jars that would shadow a pinned install so the Fiji classloader
    ignores them. Renamed in place with a .disabled suffix, not deleted."""
    for pattern in globs:
        for p in sorted(root.glob(pattern)):
            if not p.is_file() or p == keep:
                continue
            print(f"[fiji] Disabling conflicting jar: {p.name}")
            shutil.move(str(p), str(p.with_name(p.name + ".disabled")))


def _copy_jar_into_place(src: Path, target: Path) -> None:
    """Copy a jar onto the network filesystem and verify it landed.

    Writes to this filesystem have been observed to silently drop (the file
    is visible to the writing process but missing for fresh processes), so
    the copy is verified from a fresh process and retried before giving up."""
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        shutil.copy2(str(src), str(target))
        check = subprocess.run(
            ["unzip", "-t", str(target)],
            capture_output=True,
            timeout=120,
        )
        if check.returncode == 0:
            return
        print(f"[fiji] Jar not readable after copy (attempt {attempt + 1} of 3); retrying in 20s")
        time.sleep(20)
    raise RuntimeError(f"Jar copy did not persist to the filesystem: {target}")


def _install_pinned_bigstitcher(root: Path) -> None:
    print("[fiji] Installing pinned classic BigStitcher jars from Maven")
    # resolve_install_root() returns the directory that CONTAINS Fiji.app;
    # jar paths must resolve inside Fiji.app or the classloader never sees
    # them (and the install would silently land outside the application).
    app = root / "Fiji.app"
    for maven_path, target_rel, conflict_globs in BIGSTITCHER_PINNED_JARS:
        target = app / target_rel
        _disable_conflicting_jars(app, conflict_globs, keep=target)
        # Always download fresh into /tmp and copy into place: writing jars
        # directly onto the network filesystem has silently dropped, and a
        # stale filesystem view can make a missing file look present.
        with tempfile.TemporaryDirectory(prefix="fiji_pinned_") as td:
            tmp = Path(td) / target.name
            _download(f"{MAVEN_BASE_URL}/{maven_path}", tmp)
            if not _jar_is_valid(tmp):
                raise RuntimeError(f"Downloaded jar failed zip integrity check: {tmp}")
            _copy_jar_into_place(tmp, target)
        print(f"[fiji] Installed pinned jar: {target_rel}")


def _ensure_bigstitcher_site(exe: Path, root: Path) -> None:
    """Register the BigStitcher update site unless it is already in the db.

    Re-adding an already-registered site makes the updater exit non-zero
    ('Site ... was already added!'), which is not an error for us."""
    db = root / "Fiji.app" / "db.xml.gz"
    if db.is_file():
        with gzip.open(db, "rt", errors="replace") as f:
            if 'name="BigStitcher"' in f.read():
                print("[fiji] BigStitcher update site already registered")
                return
    _run(
        [str(exe), "--update", "add-update-site",
         BIGSTITCHER_SITE_NAME, BIGSTITCHER_SITE_URL],
        root,
    )


def _verify_bigstitcher_headless(root: Path) -> None:
    exe = fiji_exe(root)
    print("[fiji] Verifying BigStitcher loads headless")
    with tempfile.TemporaryDirectory(prefix="fiji_verify_") as td:
        script = Path(td) / "verify_bigstitcher.py"
        script.write_text(
            BIGSTITCHER_IMPORT_CHECK + '\nprint("BIGSTITCHER_IMPORT_OK")\n',
            encoding="utf-8",
        )
        last_output = ""
        for attempt in range(3):
            if attempt:
                # Freshly written jars can stay invisible to a new process for
                # a while on network filesystems; give the metadata a chance.
                print(f"[fiji] Verification attempt {attempt + 1} of 3 after 20s wait")
                time.sleep(20)
            result = subprocess.run(
                [str(exe), "--headless", "--console", "--run", str(script)],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=600,
            )
            last_output = result.stdout + result.stderr
            if "BIGSTITCHER_IMPORT_OK" in last_output:
                return
        raise RuntimeError(
            "BigStitcher failed to load headless after installation. "
            "Output tail:\n" + last_output[-2000:]
        )


def is_fiji_ready(root: Path) -> bool:
    return (
        fiji_exe(root).exists()
        and (root / ".fiji_bigstitcher_ready").exists()
        and bigstitcher_jar(root) is not None
    )


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

        exe = fiji_exe(root)
        url = os.environ.get("FIJI_ZIP_URL", DEFAULT_FIJI_ZIP_URL)
        if not force and exe.exists():
            print("[fiji] Fiji already present; refreshing base files and BigStitcher only")
        else:
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
        _ensure_bigstitcher_site(exe, root)

        print("[fiji] Updating Fiji (base files from update sites)")
        _run([str(exe), "--update", "update"], root)

        _install_pinned_bigstitcher(root)

        _verify_bigstitcher_headless(root)

        db = root / "Fiji.app" / "db.xml.gz"
        stamp = root / ".fiji_bigstitcher_ready"
        stamp.write_text(
            "\n".join(
                [
                    f"fiji_zip_url={url}",
                    f"bigstitcher_site={BIGSTITCHER_SITE_URL}",
                    "bigstitcher_pinned_jars=" + ",".join(t for _, t, _ in BIGSTITCHER_PINNED_JARS),
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
