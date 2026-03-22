# Docker Appliance

This directory contains a Windows-hosted Docker Desktop setup for running the full CPU-first
`mesospim_utils` workflow inside a single Linux container with local SLURM services.

## What it includes

- source-built SLURM pinned by `docker/.env`
- source-built Wine pinned by `docker/.env`
- Miniconda-based Python runtime
- preinstalled Fiji + BigStitcher
- bind mounts rooted at `/data`
- startup script that brings up `munged`, `slurmd`, and `slurmctld`
- optional GPU override for deconvolution hosts

## Prerequisites

- Docker Desktop on Windows
- WSL2 backend enabled
- Drive sharing enabled for any Windows paths you plan to mount

## Default mount layout

- `/data/c`
- `/data/z`
- `/data/h`
- `/data/config`
- `/data/work`
- optional `/data/share`

The default Wine mappings are:

- `/data/z -> z:`
- `/data/h -> h:`
- `/data/share -> y:`

Wine `c:` is left untouched.

## Configure `.env`

Edit `docker/.env` and set the host mount sources you want Docker Desktop to expose.

The checked-in defaults are for launching Docker from `cmd.exe` or PowerShell on Windows.
If you launch Docker from WSL instead, override them with `/mnt/c/...`, `/mnt/z`, and `/mnt/h`
style paths that are actually visible inside WSL.

`/data/z` and `/data/h` are mounted through Docker-managed CIFS volumes so the Docker backend can
access the SMB shares directly. The checked-in defaults assume open guest shares. If your shares
require authentication, replace the CIFS option strings in `docker/.env` with username/password
based options.

Default Windows-style example:

```env
MESO_C_SRC=C:\data
MESO_Z_SRC=//faststore.cbi.pitt.edu/CBI_FastStore
MESO_H_SRC=//tn.cbi.pitt.edu/h20
MESO_CONFIG_SRC=./config
MESO_WORK_SRC=./work
MESO_Z_CIFS_OPTS=guest,vers=3.0,file_mode=0777,dir_mode=0777,noperm,iocharset=utf8
MESO_H_CIFS_OPTS=guest,vers=3.0,file_mode=0777,dir_mode=0777,noperm,iocharset=utf8
MESO_SHARE_SRC=./disabled-share
```

Docker does not reliably pass Windows-mapped network drives into Linux containers, so the
recommended setup is to keep `C:` as a normal bind mount and let Docker mount the network shares
directly from their UNC paths.

Example if launching from WSL instead:

```env
MESO_C_SRC=/mnt/c/data
MESO_Z_SRC=//faststore.cbi.pitt.edu/CBI_FastStore
MESO_H_SRC=//tn.cbi.pitt.edu/h20
MESO_CONFIG_SRC=./config
MESO_WORK_SRC=./work
MESO_Z_CIFS_OPTS=guest,vers=3.0,file_mode=0777,dir_mode=0777,noperm,iocharset=utf8
MESO_H_CIFS_OPTS=guest,vers=3.0,file_mode=0777,dir_mode=0777,noperm,iocharset=utf8
MESO_SHARE_SRC=./disabled-share
```

Example using a UNC-backed share for the optional share mount:

```env
MESO_SHARE_SRC=\\server\share\mesospim
```

## Build and run

```bash
cd docker
mkdir -p config work disabled-share
docker compose build
docker compose up -d
docker compose exec mesospim_utils bash
```

## GPU hosts

On Windows + WSL2 hosts with NVIDIA GPU passthrough, or on Linux hosts with NVIDIA GPUs,
start the container with the GPU override file:

```bash
cd docker
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

The GPU override requests all visible GPUs from Docker. At container startup, `startup.sh`
detects either `/dev/nvidia*` or `/dev/dxg` and advertises GPU GRES to SLURM automatically.
Docker decon config requests a full GPU per decon job via `gpu:1`.

Recommended GPU validation inside the container:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
sinfo
scontrol show node
```

## Config file

The container reads `MESOSPIM_CONFIG`, which defaults to `/data/config/main.yaml`.

If that file does not exist on first startup, the container copies:

`mesospim_utils/config/docker-example.yaml`

into `/data/config/main.yaml`.

## Useful checks inside the container

```bash
python /opt/src/mesospim_utils/mesospim_utils/metadata.py --help
python /opt/src/mesospim_utils/mesospim_utils/omezarr.py --help
python /opt/src/mesospim_utils/mesospim_utils/bigstitcher.py --help
python /opt/src/mesospim_utils/mesospim_utils/automated.py automated-method-slurm --help
```

SLURM checks:

```bash
service munge status
service slurmd status
service slurmctld status
sinfo
```
