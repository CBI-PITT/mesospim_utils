#!/bin/bash

# Determine whether script is running as root
sudo_cmd=""
if [ "$(id -u)" != "0" ]; then
    sudo_cmd="sudo"
    sudo -k
fi

# Configure Slurm to use maximum available processors and memory
# and start required services
${sudo_cmd} bash <<'SCRIPT'
GRES=/etc/slurm/gres.conf
SLURM_CONF=/etc/slurm/slurm.conf

#######  AUTO CONFIGURE SLURM FOR EACH NEW CONTAINER  #######
# Edit slurm.conf
sed -i "s/<<HOSTNAME>>/$(hostname)/" "$SLURM_CONF"
sed -i "s/<<CPU>>/$(nproc)/" "$SLURM_CONF"
sed -i "s/<<MEMORY>>/$(if [[ "$(slurmd -C)" =~ RealMemory=([0-9]+) ]]; then echo "${BASH_REMATCH[1]}"; else exit 100; fi)/" "$SLURM_CONF"

# Edit gres.conf
## For Linux Hosts
if compgen -G "/dev/nvidia[0-9]*" >/dev/null; then
  # --- Linux + NVIDIA devices ---
  # 1) enable LinuxAuto line, keep wsl2 template commented
  sed -i -E 's/^[[:space:]]*#<<LinuxAuto>>(.*)$/\1/' "$GRES"
  # 2) refresh per-GPU File= lines
  sed -i -E '/^Name=gpu File=\/dev\/nvidia[0-9]+/d' "$GRES"
  for d in /dev/nvidia[0-9]*; do
    [ -e "$d" ] || break
    echo "Name=gpu File=$d" >> "$GRES"
  done
  # 3) determine gpu and shard number
  GPU_COUNT=$(ls /dev/nvidia[0-9]* 2>/dev/null | wc -l)
  SHARDS=$(( GPU_COUNT * 2 ))
  sed -i.bak "/^[[:space:]]*#<<shard>>/ { s/^[[:space:]]*#<<shard>>//; s/<<SHARD_COUNT>>/$SHARDS/ }" "$GRES"

  # Add GPUs to slurm.conf
  grep -q '^GresTypes=gpu,shard' "$SLURM_CONF" || sed -i '/^SelectTypeParameters/a GresTypes=gpu,shard' "$SLURM_CONF"
  #sed -i -E "s/^(NodeName=[^ ]+ [^#]*)( Gres=gpu:[0-9]+)?(.*)$/\1 Gres=gpu:${GPU_COUNT}\3/" "$SLURM_CONF"
  sed -i -E "/^NodeName=/ s/$/ Gres=gpu:${GPU_COUNT},shard:${SHARDS}/" "$SLURM_CONF"

## For Windows WSL-2 Hosts
elif [ -e /dev/dxg ]; then
  # --- WSL2 / dxg bridge ---
  # 1) enable the wsl2 line; comment the LinuxAuto template
  sed -i -E 's/^[[:space:]]*#<<wsl2>>(.*)$/\1/' "$GRES"

  GPU_COUNT=1
  SHARDS=2
  sed -i.bak "/^[[:space:]]*#<<shard>>/ { s/^[[:space:]]*#<<shard>>//; s/<<SHARD_COUNT>>/$SHARDS/ }" "$GRES"

  # Add GPU to slurm.conf
  grep -q '^GresTypes=gpu,shard' "$SLURM_CONF" || sed -i '/^SelectTypeParameters/a GresTypes=gpu,shard' "$SLURM_CONF"
  #sed -i -E "s/^(NodeName=[^ ]+ [^#]*)( Gres=gpu:[0-9]+)?(.*)$/\1 Gres=gpu:1\3/" "$SLURM_CONF"
  sed -i -E "/^NodeName=/ s/$/ Gres=gpu:${GPU_COUNT},shard:${SHARDS}/" "$SLURM_CONF"
fi

service munge start
service slurmd start
service slurmctld start

# Map volumes to wine letter drives
/etc/map_wine.py
chown -R docker:docker /home/docker/.wine
SCRIPT

# Revoke sudo permissions
if [[ ${sudo_cmd} ]]; then
    sudo -k
fi