#!/usr/bin/env python3
import os
import re
import sys
from pathlib import Path

import yaml


APP_USER = os.environ.get("MESOSPIM_CONTAINER_USER", "docker")
WINE_DOSDIR = Path(f"/home/{APP_USER}/.wine/dosdevices")
MESOSPIM_UTILS_CONFIG = Path(os.environ.get("MESOSPIM_CONFIG", "/data/config/main.yaml"))


def normalize_letter(s: str):
    s = (s or "").strip().lower()
    if s.endswith(":"):
        s = s[:-1]
    return s if len(s) == 1 and "a" <= s <= "z" else None


def prune_all_but_c(dosdir: Path):
    if not dosdir.is_dir():
        return
    for entry in dosdir.iterdir():
        if re.fullmatch(r"[A-Za-z]:", entry.name) and entry.name.lower() != "c:":
            try:
                if entry.is_symlink() or entry.is_file():
                    entry.unlink()
            except FileNotFoundError:
                pass


def main():
    try:
        with MESOSPIM_UTILS_CONFIG.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"[wine-yaml] Config not found: {MESOSPIM_UTILS_CONFIG}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"[wine-yaml] Failed to read YAML: {e}", file=sys.stderr)
        return 2

    mappings = {}
    general = data.get("general") or {}
    wm = general.get("wine_mappings")
    if isinstance(wm, dict):
        for linux_path, drive_letter in wm.items():
            if not isinstance(linux_path, str) or not isinstance(drive_letter, str):
                continue
            letter = normalize_letter(drive_letter)
            if not letter or letter == "c":
                continue
            mappings[linux_path] = letter

    WINE_DOSDIR.mkdir(parents=True, exist_ok=True)
    prune_all_but_c(WINE_DOSDIR)

    if not mappings:
        print("[wine-yaml] No wine_mappings entries; kept only C:.")
        return 0

    for linux_path, letter in mappings.items():
        target = Path(linux_path)
        if not target.is_absolute():
            print(f"[wine-yaml] Skipping non-absolute path: {linux_path}", file=sys.stderr)
            continue
        if not target.exists():
            print(f"[wine-yaml] Warning: target not found; skipping {letter.upper()}: -> {linux_path}", file=sys.stderr)
            continue
        link = WINE_DOSDIR / f"{letter}:"
        try:
            if link.exists() or link.is_symlink():
                link.unlink()
        except FileNotFoundError:
            pass
        link.symlink_to(target)
        print(f"[wine-yaml] {letter.upper()}: -> {linux_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
