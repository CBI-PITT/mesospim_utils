#!/bin/bash

set -euo pipefail

APP_USER="${MESOSPIM_CONTAINER_USER:-docker}"
SLURM_CONF_TEMPLATE=/etc/slurm/slurm.conf.template
SLURM_CONF=/etc/slurm/slurm.conf
GRES_TEMPLATE=/etc/slurm/gres.conf.template
GRES=/etc/slurm/gres.conf

sudo_cmd=""
if [ "$(id -u)" != "0" ]; then
    sudo_cmd="sudo"
    sudo -k
fi

${sudo_cmd} mkdir -p /data/c /data/z /data/h /data/config /data/work /var/log/slurmd /var/log/slurmctld /var/spool/slurmd /var/spool/slurmctld

if [ ! -f /data/config/main.yaml ] && [ -f /opt/src/mesospim_utils/mesospim_utils/config/docker-example.yaml ]; then
    ${sudo_cmd} cp /opt/src/mesospim_utils/mesospim_utils/config/docker-example.yaml /data/config/main.yaml
fi

${sudo_cmd} env \
    APP_USER="${APP_USER}" \
    SLURM_CONF_TEMPLATE="${SLURM_CONF_TEMPLATE}" \
    SLURM_CONF="${SLURM_CONF}" \
    GRES_TEMPLATE="${GRES_TEMPLATE}" \
    GRES="${GRES}" \
    bash <<'SCRIPT'
set -euo pipefail

cp "${SLURM_CONF_TEMPLATE}" "${SLURM_CONF}"
cp "${GRES_TEMPLATE}" "${GRES}"

sed -i "s/<<HOSTNAME>>/$(hostname)/" "${SLURM_CONF}"
sed -i "s/<<CPU>>/$(nproc)/" "${SLURM_CONF}"
sed -i "s/<<MEMORY>>/$(if [[ "$(slurmd -C)" =~ RealMemory=([0-9]+) ]]; then echo "${BASH_REMATCH[1]}"; else exit 100; fi)/" "${SLURM_CONF}"

if compgen -G "/dev/nvidia[0-9]*" >/dev/null; then
  sed -i -E 's/^[[:space:]]*#<<LinuxAuto>>(.*)$/\1/' "${GRES}"
  sed -i -E '/^Name=gpu File=\/dev\/nvidia[0-9]+/d' "${GRES}"
  for d in /dev/nvidia[0-9]*; do
    [ -e "$d" ] || break
    echo "Name=gpu File=$d" >> "${GRES}"
  done
  GPU_COUNT=$(ls /dev/nvidia[0-9]* 2>/dev/null | wc -l)
  SHARDS=$(( GPU_COUNT * 2 ))
  echo "Detected ${GPU_COUNT} NVIDIA GPU(s) via /dev/nvidia*"
  sed -i -E "/^[[:space:]]*#<<shard>>/ { s/^[[:space:]]*#<<shard>>//; s/<<SHARD_COUNT>>/$SHARDS/ }" "${GRES}"
  grep -q '^GresTypes=gpu,shard' "${SLURM_CONF}" || sed -i '/^SelectTypeParameters/a GresTypes=gpu,shard' "${SLURM_CONF}"
  sed -i -E "/^NodeName=/ s/$/ Gres=gpu:${GPU_COUNT},shard:${SHARDS}/" "${SLURM_CONF}"
elif [ -e /dev/dxg ]; then
  sed -i -E 's/^[[:space:]]*#<<wsl2>>(.*)$/\1/' "${GRES}"
  GPU_COUNT=1
  SHARDS=2
  echo "Detected WSL2 GPU passthrough via /dev/dxg"
  sed -i -E "/^[[:space:]]*#<<shard>>/ { s/^[[:space:]]*#<<shard>>//; s/<<SHARD_COUNT>>/$SHARDS/ }" "${GRES}"
  grep -q '^GresTypes=gpu,shard' "${SLURM_CONF}" || sed -i '/^SelectTypeParameters/a GresTypes=gpu,shard' "${SLURM_CONF}"
  sed -i -E "/^NodeName=/ s/$/ Gres=gpu:${GPU_COUNT},shard:${SHARDS}/" "${SLURM_CONF}"
else
  echo "No GPU device nodes detected; starting SLURM without GPU GRES"
fi

echo "Effective slurm.conf NodeName line:"
grep '^NodeName=' "${SLURM_CONF}"

mkdir -p /home/${APP_USER}/.wine/dosdevices
service munge start
service slurmd start
service slurmctld start
/opt/miniconda/bin/python /etc/map_wine.py
chown -R ${APP_USER}:${APP_USER} /home/${APP_USER}/.wine /data
SCRIPT

if [[ -n "${sudo_cmd}" ]]; then
    sudo -k
fi
