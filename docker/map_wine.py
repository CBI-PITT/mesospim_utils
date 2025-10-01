#!/opt/miniconda/bin/python
import os, re, sys
import yaml

WINE_DOSDIR = "/home/docker/.wine/dosdevices"
MESOSPIM_UTILS_CONFIG = "/opt/mesospim_utils/mesospim_utils/config/main.yaml"

def normalize_letter(s: str):
    s = (s or "").strip().lower()
    if s.endswith(":"):
        s = s[:-1]
    return s if len(s) == 1 and "a" <= s <= "z" else None

def prune_all_but_c(dosdir: str):
    if not os.path.isdir(dosdir):
        return
    for name in os.listdir(dosdir):
        if re.fullmatch(r"[A-Za-z]:", name) and name.lower() != "c:":
            p = os.path.join(dosdir, name)
            try:
                # Only remove files or symlinks; never touch directories
                if os.path.islink(p) or os.path.isfile(p):
                    os.remove(p)
            except FileNotFoundError:
                pass

def main():

    # Load YAML and extract mappings
    try:
        with open(MESOSPIM_UTILS_CONFIG, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[wine-yaml] Failed to read YAML: {e}", file=sys.stderr)
        return 2

    mappings = {}
    general = data.get("general") or {}
    wm = general.get("wine_mappings")
    if isinstance(wm, dict):
        # Normalize {linux_path: drive_letter}
        for k, v in wm.items():
            if not isinstance(k, str) or not isinstance(v, str):
                continue
            letter = normalize_letter(v)
            if not letter or letter == "c":
                # skip invalid letters and never allow overriding C:
                continue
            mappings[k] = letter

    dosdir = WINE_DOSDIR
    os.makedirs(dosdir, exist_ok=True)

    # Always prune everything except C:
    prune_all_but_c(dosdir)

    if not mappings:
        print("[wine-yaml] No wine_mappings entries; kept only C:.")
        return 0

    # Create requested letters
    for linux_path, letter in mappings.items():
        if not linux_path.startswith("/"):
            print(f"[wine-yaml] Skipping non-absolute path: {linux_path}", file=sys.stderr)
            continue
        if not os.path.exists(linux_path):
            print(f"[wine-yaml] Warning: target not found; skipping {letter.upper()}: -> {linux_path}",
                  file=sys.stderr)
            continue
        link = os.path.join(dosdir, f"{letter}:")
        try:
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
        except FileNotFoundError:
            pass
        os.symlink(linux_path, link)
        print(f"[wine-yaml] {letter.upper()}: -> {linux_path}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
