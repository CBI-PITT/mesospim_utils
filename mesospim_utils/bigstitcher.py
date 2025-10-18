# STD library imports
from pathlib import Path

# Local imports
from metadata import collect_all_metadata, get_first_entry, get_all_tile_entries
from utils import ensure_path, sort_list_of_paths_by_tile_number, make_directories, dict_to_json_file
from constants import (
    ALIGNMENT_DIRECTORY,
    CORRELATION_THRESHOLD_FOR_ALIGN,
    RESOLUTION_LEVEL_FOR_ALIGN,
    VERBOSE,
    OFFSET_METRIC,
    REMOVE_OUTLIERS,
    ALIGN_ALL_OUTPUT_FILE_NAME,
    ALIGN_METRIC_OUTPUT_FILE_NAME,
    OVERIDE_STAGE_DIRECTION
)

# # INIT typer cmdline interface
# app = typer.Typer()


'''
This module has functions to orchestrate bigstitcher for mesospim data using.
'''

def write_big_stitcher_config_file(path):
    path = ensure_path(path)
    tile_cfg_path = path / 'big_stitcher_config.txt'
    macro_path = path / 'big_stitcher_macro.ijm'
    # BigStitcher Configuration file

    # Write tile_configuration.txt
    lines = [
        "# Define the number of dimensions we are working on",
        "dim = 3",
        "",
        "# Define the image coordinates (in microns)",
        "# Format: image_path; ; ; x y z",
    ]

    metadata_by_channel = collect_all_metadata(path)
    for ch in metadata_by_channel:
        for tile in metadata_by_channel[ch]:
            path = tile.get("file_path")

            # Extract tile origin in um from affine_microns
            am = tile.get("affine_microns")
            z = float(am[0][3])
            y = float(am[1][3])
            x = float(am[2][3])
            lines.append(f"{path}; ; ; {x:.6f} {y:.6f} {z:.6f}")

            tile_cfg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    pixel_z_um, pixel_y_um, pixel_x_um = get_first_entry(metadata_by_channel)['resolution']
    # Create a minimal headless macro that embeds pixel sizes and runs a safe pipeline
    macro = """// BigStitcher headless macro (args parsed manually)

    // Read options passed after --run "...":
    args = getArgument();
    config_path = call("ij.Macro.getValue", args, "config_path", "");
    output_dir  = call("ij.Macro.getValue", args, "output_dir", "");

    if (config_path == "" || output_dir == "") {
      exit("Missing required args: config_path and/or output_dir");
    }

    // If your paths may contain spaces, wrap them in [] for IJ command arguments
    file_arg = "file=[" + config_path + "]";

    setBatchMode(true);

    // Import tiles via configuration (positions in microns)
    run("BigStitcher - Import Tiles from Configuration",
        file_arg + " pixel_size_x=1 pixel_size_y=1 pixel_size_z=5.0 unit=micron");

    // Detect & match interest points (SIFT)
    run("BigStitcher - Detect Interest Points (SIFT)",
        "initialSigma=1.6 steps=3 minOctaveSize=64 maxOctaveSize=1024 featureDescriptorSize=8 featureDescriptorOrientationBins=8 threshold=0.01 findSmallerScale=true");

    // Match
    run("BigStitcher - Match Interest Points",
        "ratioOfDistance=0.92 maxAllowedError=10.0 inlierRatio=0.05 model=Rigid");

    // Optimize globally
    run("BigStitcher - Optimize Globally", "regularize=false");

    // Export to BDV N5 (multiscale)
    run("BigStitcher - Export to BDV N5",
        "path=[" + output_dir + "] export_type=FUSED multires=true blending=LINEAR downsamplingFactors=\\"[1,1,1];[2,2,2];[4,4,4];[8,8,8]\\" compression=RAW");

    setBatchMode(false);
    print("BigStitcher headless run completed.");
    """

    macro_path.write_text(macro, encoding="utf-8")

    tile_cfg_path.as_posix(), macro_path.as_posix(), {"pixel_size_x_um": pixel_x_um, "pixel_size_y_um": pixel_y_um, "pixel_size_z_um": pixel_z_um}#, "tiles": len(tiles)}




def _make_fiji_install(script_path):
    # Retry creating the installer script since the previous Python session was reset.
    from pathlib import Path
    script_path = ensure_path(script_path)

    script = r"""#!/usr/bin/env bash
# install_fiji_bigstitcher.sh
# Headless installer for Fiji (ImageJ2) + BigStitcher on Linux.
# Usage:
#   bash install_fiji_bigstitcher.sh [/path/to/install/dir]
#
# If no path is provided, installs to $HOME/apps/fiji
#
# After installation, Fiji with BigStitcher can be run headlessly:
#   $INSTALL_DIR/ImageJ-linux64 --ij2 --headless --run "BigStitcher - Import Tiles from Configuration"

set -euo pipefail

RED=$'\e[31m'; GRN=$'\e[32m'; YLW=$'\e[33m'; BLU=$'\e[34m'; RST=$'\e[0m'

die() { echo "${RED}ERROR:${RST} $*" >&2; exit 1; }
log() { echo "${BLU}==>${RST} $*"; }
ok()  { echo "${GRN}✔${RST} $*"; }
warn(){ echo "${YLW}⚠${RST} $*"; }

INSTALL_DIR="${1:-$HOME/apps/fiji}"
FIJI_ZIP_URL="https://downloads.imagej.net/fiji/archive/stable/20250808-2217/fiji-stable-linux64-jdk.zip"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

# 1) Prepare install directory
log "Installing to: $INSTALL_DIR"
mkdir -p "$(dirname "$INSTALL_DIR")"

# 2) Download Fiji
DL="$WORKDIR/fiji-linux64.zip"
if command -v wget >/dev/null 2>&1; then
  log "Downloading Fiji (wget) ..."
  wget -q -O "$DL" "$FIJI_ZIP_URL" || die "wget failed to download Fiji."
elif command -v curl >/dev/null 2>&1; then
  log "Downloading Fiji (curl) ..."
  curl -fsSL -o "$DL" "$FIJI_ZIP_URL" || die "curl failed to download Fiji."
else
  die "Need wget or curl installed to download Fiji."
fi
ok "Downloaded Fiji zip"

# 3) Unpack
log "Unpacking Fiji ..."
if command -v unzip >/dev/null 2>&1; then
    unzip -"$DL" -d "$WORKDIR"
elif command -v python3 >/dev/null 2>&1; then
    python3 - "$DL" "$WORKDIR" <<'PY'
import sys, zipfile, os
zip_path, dest = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(dest)
PY
else
    echo "No extractor available (install unzip/bsdtar/7z or ensure Java/Python present)" >&2

fi

[ -d "$WORKDIR/Fiji.app" ] || die "Fiji.app not found after unzip."
rm -rf "$INSTALL_DIR"
mv "$WORKDIR/Fiji.app" "$INSTALL_DIR"
ok "Fiji unpacked at $INSTALL_DIR"

FIJI_BIN="$INSTALL_DIR/ImageJ-linux64"
chmod 755 "$FIJI_BIN"

# 4) First headless run to initialize update system
log "Initializing Fiji update system (headless) ..."
"$FIJI_BIN" --ij2 --headless --eval 'System.exit(0);' >/dev/null 2>&1 || true
ok "Initialization complete"

# 5) Enable update sites (BigStitcher + BigDataViewer)
UPD_DIR="$INSTALL_DIR/update-sites"
mkdir -p "$UPD_DIR"

log "Enabling BigStitcher update site ..."
cat > "$UPD_DIR/bigstitcher.properties" <<'EOF'
name=BigStitcher
url=https://sites.imagej.net/BigStitcher/
enabled=true
EOF
ok "BigStitcher site enabled"

log "Enabling BigDataViewer update site ..."
cat > "$UPD_DIR/bdv.properties" <<'EOF'
name=BigDataViewer
url=https://sites.imagej.net/BigDataViewer/
enabled=true
EOF
ok "BigDataViewer site enabled"

# Optional: add other useful sites (uncomment if desired)
# cat > "$UPD_DIR/bigdataprocessor2.properties" <<'EOF'
# name=BigDataProcessor2
# url=https://sites.imagej.net/BigDataProcessor2/
# enabled=true
# EOF

# 6) Run Fiji updater headlessly
log "Running Fiji updater (headless). This may take a few minutes ..."
set +e
"$FIJI_BIN" --update update > "$WORKDIR/update.log" 2>&1
RET=$?
set -e
if [ $RET -ne 0 ]; then
  warn "Updater exited with code $RET. Printing tail of log:"
  tail -n 50 "$WORKDIR/update.log" || true
  die "Fiji updater failed. See full log at: $WORKDIR/update.log"
fi
ok "Fiji updated with BigStitcher + dependencies"

# 7) Verify BigStitcher command is available
log "Verifying BigStitcher availability (headless) ..."
set +e
OUT="$("$FIJI_BIN" --ij2 --headless --run "BigStitcher - Import Tiles from Configuration" 2>&1)"
RET=$?
set -e
if echo "$OUT" | grep -qiE "No such command|Unknown command"; then
  echo "$OUT"
  die "BigStitcher command not found. Update site may not have installed correctly."
fi
ok "BigStitcher detected"

# 8) Print usage tips
cat <<EOF

${GRN}Installation successful!${RST}

Fiji with BigStitcher is installed at:
  $INSTALL_DIR

Headless usage example:
  $INSTALL_DIR/ImageJ-linux64 --ij2 --headless \\
    --run /path/to/bigstitcher_headless_auto.ijm \\
    "config_path='/path/to/tile_configuration.txt',output_dir='/scratch/bs_out'"

# To add Fiji to your PATH:
#   echo 'export PATH=$INSTALL_DIR:\$PATH' >> ~/.bashrc
#   source ~/.bashrc

EOF
"""

    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(0o755)

    str(script_path)


if __name__ == '__main__':
    write_big_stitcher_config_file('/CBI_FastStore/test_data/mesospim/04CL18_Stujenske_injection')
    _make_fiji_install('/CBI_FastStore/test_data/mesospim/04CL18_Stujenske_injection/fiji_install.sh')

