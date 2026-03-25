ARG UBUNTU_VERSION=22.04
ARG TIMEZONE=America/New_York

ARG USERNAME=docker
ARG PASSWORD=ubuntu
ARG UID=1000

ARG CONDA_DIR=/opt/miniconda
ARG CONDA_VERSION=py312_24.7.1-0
ARG CONDA_OS=Linux
ARG CONDA_ARCH=x86_64

ARG SLURM_VER=25.05.3
ARG WINE_GITHUB=https://github.com/wine-mirror/wine.git
ARG WINE_COMMIT=2bca8cb236b7d4d5f4e0e35d6fcad6a1d8a2c5e3
ARG MESOSPIM_SRC=/opt/src/mesospim_utils
ARG FIJI_ROOT=/opt/fiji-linux
ARG FIJI_ZIP_URL=https://downloads.imagej.net/fiji/archive/stable/20250808-2217/fiji-stable-linux64-jdk.zip


FROM ubuntu:${UBUNTU_VERSION} AS slurm-build

ARG SLURM_VER

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    bash-completion \
    bison \
    build-essential \
    ca-certificates \
    cmake \
    debhelper \
    devscripts \
    dh-exec \
    dh-autoreconf \
    dpkg-dev \
    flex \
    g++ \
    git \
    hdf5-helpers \
    libcurl4-openssl-dev \
    libdbus-1-dev \
    libfreeipmi-dev \
    libgtk2.0-dev \
    libhdf5-dev \
    libhttp-parser-dev \
    libhwloc-dev \
    libipmimonitoring-dev \
    libjson-c-dev \
    libjwt-dev \
    liblua5.3-dev \
    liblz4-dev \
    libmariadb-dev \
    libmunge-dev \
    libncurses-dev \
    libnvidia-ml-dev \
    libpam0g-dev \
    libperl-dev \
    libpmix-dev \
    librdkafka-dev \
    librrd-dev \
    libyaml-dev \
    man2html-base \
    po-debconf \
    python3 \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q "https://download.schedmd.com/slurm/slurm-${SLURM_VER}.tar.bz2" \
    && tar -xjf "slurm-${SLURM_VER}.tar.bz2" \
    && cd "slurm-${SLURM_VER}" \
    && debuild -b -uc -us -j"$(nproc)" \
    && mkdir -p /tmp/slurm-install \
    && cp /slurm*.deb /tmp/slurm-install/


FROM ubuntu:${UBUNTU_VERSION} AS wine-build

ARG WINE_GITHUB
ARG WINE_COMMIT

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    autoconf \
    automake \
    bison \
    build-essential \
    ca-certificates \
    flex \
    gcc \
    git \
    libasound2-dev \
    libcups2-dev \
    libdbus-1-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libglu1-mesa-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    libncurses-dev \
    libosmesa6-dev \
    libpulse-dev \
    libsdl2-dev \
    libssl-dev \
    libtool \
    libudev-dev \
    libusb-1.0-0-dev \
    libx11-dev \
    libxcomposite-dev \
    libxcursor-dev \
    libxext-dev \
    libxi-dev \
    libxinerama-dev \
    libxrandr-dev \
    libxrender-dev \
    make \
    pkg-config \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV SRC_DIR=/opt

RUN set -eux; \
    cd "${SRC_DIR}"; \
    git init wine; \
    cd wine; \
    git remote add origin "${WINE_GITHUB}"; \
    git fetch --depth 1 origin "${WINE_COMMIT}"; \
    git checkout --detach FETCH_HEAD; \
    ./configure --enable-win64; \
    CPUS=$(nproc || echo 1); \
    make -j"${CPUS}"; \
    make install DESTDIR=/tmp/wine-install


FROM ubuntu:${UBUNTU_VERSION} AS runtime

ARG TIMEZONE
ARG USERNAME
ARG PASSWORD
ARG UID
ARG CONDA_DIR
ARG CONDA_VERSION
ARG CONDA_OS
ARG CONDA_ARCH
ARG SLURM_VER
ARG MESOSPIM_SRC
ARG FIJI_ROOT
ARG FIJI_ZIP_URL

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=${CONDA_DIR}/bin:${PATH}
ENV FIJI_ROOT=${FIJI_ROOT}
ENV MESOSPIM_CONTAINER_USER=${USERNAME}
ENV MESOSPIM_SRC=${MESOSPIM_SRC}
ENV MESOSPIM_CONFIG=/data/config/main.yaml
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_MESOSPIM_UTILS=0.0.0
ENV WINEPREFIX=/home/${USERNAME}/.wine

COPY --from=slurm-build /tmp/slurm-install/ /opt/slurm-install/
COPY --from=wine-build /tmp/wine-install/ /

RUN apt-get update && apt-get install -y \
    bzip2 \
    ca-certificates \
    build-essential \
    curl \
    g++ \
    git \
    locales \
    munge \
    nano \
    openssl \
    sudo \
    tmux \
    tzdata \
    unzip \
    wget \
    htop \
    libasound2 \
    libcups2 \
    libdbus-1-3 \
    libfontconfig1 \
    libfreetype6 \
    libglu1-mesa \
    libgstreamer-plugins-base1.0-0 \
    libgstreamer1.0-0 \
    libhwloc15 \
    libncurses6 \
    libnss3 \
    libosmesa6 \
    libpmix2 \
    libpulse0 \
    libsdl2-2.0-0 \
    libssl3 \
    libudev1 \
    libusb-1.0-0 \
    libx11-6 \
    libxcomposite1 \
    libxcursor1 \
    libxext6 \
    libxi6 \
    libxinerama1 \
    libxrandr2 \
    libxrender1 \
    python3 \
    && locale-gen en_US.UTF-8 \
    && update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
    && ln -snf "/usr/share/zoneinfo/${TIMEZONE}" /etc/localtime \
    && echo "${TIMEZONE}" >/etc/timezone \
    && groupadd -r -g 64030 slurm \
    && useradd -M slurm -u 64030 -g 64030 -s /usr/sbin/nologin --home /nonexistent \
    && mkdir -p /var/spool/slurmd /var/spool/slurmctld /var/log/slurmd /var/log/slurmctld /etc/slurm \
    && chown -R slurm:slurm /var/spool/slurmd /var/spool/slurmctld /var/log/slurmd /var/log/slurmctld /etc/slurm \
    && chmod -R 775 /var/spool/slurmd /var/spool/slurmctld /var/log/slurmd /var/log/slurmctld /etc/slurm \
    && apt-get install -y /opt/slurm-install/slurm-smd_${SLURM_VER}-1_amd64.deb \
    && apt-get install -y /opt/slurm-install/slurm-smd-slurmctld_${SLURM_VER}-1_amd64.deb \
    && apt-get install -y /opt/slurm-install/slurm-smd-slurmd_${SLURM_VER}-1_amd64.deb \
    && apt-get install -y /opt/slurm-install/slurm-smd-client_${SLURM_VER}-1_amd64.deb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /opt/slurm-install

RUN useradd --create-home --uid="${UID}" "${USERNAME}" -p "$(openssl passwd -1 ${PASSWORD})" \
    && usermod -aG sudo "${USERNAME}" \
    && printf "%s ALL=(ALL) NOPASSWD: /etc/startup.sh\n" "${USERNAME}" >/etc/sudoers.d/startup \
    && chmod 0440 /etc/sudoers.d/startup \
    && visudo -c

RUN set -eux; \
    installer="Miniconda3-${CONDA_VERSION}-${CONDA_OS}-${CONDA_ARCH}.sh"; \
    wget -q "https://repo.anaconda.com/miniconda/${installer}"; \
    bash "${installer}" -b -p "${CONDA_DIR}"; \
    rm -f "${installer}"; \
    mkdir -p /etc/profile.d; \
    printf '. %s/etc/profile.d/conda.sh\nconda activate base\n' "${CONDA_DIR}" >/etc/profile.d/conda.sh; \
    "${CONDA_DIR}"/bin/conda config --system --set always_yes true; \
    "${CONDA_DIR}"/bin/conda update -n base -c defaults conda; \
    "${CONDA_DIR}"/bin/conda clean -afy; \
    chown -R "${USERNAME}:${USERNAME}" "${CONDA_DIR}"; \
    chmod -R 775 "${CONDA_DIR}"

WORKDIR /opt/src
COPY . ${MESOSPIM_SRC}

RUN set -eux; \
    python -m pip install --no-cache-dir -U pip setuptools wheel; \
    python -m pip install --no-cache-dir -e "${MESOSPIM_SRC}[psfmodels]"; \
    cp -R "${MESOSPIM_SRC}"/docker /opt/docker-src; \
    rm -rf /root/.cache/pip /tmp/*

RUN set -eux; \
    mkdir -p "${FIJI_ROOT}"; \
    wget -q -O /tmp/fiji.zip "${FIJI_ZIP_URL}"; \
    unzip -q /tmp/fiji.zip -d /tmp/fiji-stage; \
    if [ -d /tmp/fiji-stage/Fiji.app ]; then mv /tmp/fiji-stage/Fiji.app "${FIJI_ROOT}/Fiji.app"; else mv /tmp/fiji-stage/*/Fiji.app "${FIJI_ROOT}/Fiji.app"; fi; \
    "${FIJI_ROOT}/Fiji.app/ImageJ-linux64" --update add-update-site BigStitcher https://sites.imagej.net/BigStitcher/; \
    "${FIJI_ROOT}/Fiji.app/ImageJ-linux64" --update update; \
    python -c 'from pathlib import Path; import hashlib; root = Path("/opt/fiji-linux"); db = root / "Fiji.app" / "db.xml.gz"; h = hashlib.sha256(db.read_bytes()).hexdigest() if db.exists() else "missing"; (root / ".fiji_bigstitcher_ready").write_text("\n".join(["fiji_zip_url=https://downloads.imagej.net/fiji/archive/stable/20250808-2217/fiji-stable-linux64-jdk.zip", "bigstitcher_site=https://sites.imagej.net/BigStitcher/", f"db_xml_gz_sha256={h}"]) + "\n", encoding="utf-8")'; \
    rm -rf /tmp/fiji.zip /tmp/fiji-stage

RUN rm -rf /opt/ImarisFileConverter_wine \
    && git clone --depth 1 --branch main https://github.com/CBI-PITT/ImarisFileConverter_wine.git /opt/ImarisFileConverter_wine \
    && rm -rf /opt/ImarisFileConverter_wine/.git \
    && chown -R "${USERNAME}:${USERNAME}" /opt/ImarisFileConverter_wine

COPY docker/etc/init.d/slurmctld /etc/init.d/slurmctld
COPY docker/etc/init.d/slurmd /etc/init.d/slurmd
COPY docker/etc/slurm /etc/slurm
COPY docker/startup.sh /etc/startup.sh
COPY docker/map_wine.py /etc/map_wine.py

RUN sed -i 's/\r$//' /etc/init.d/slurmctld /etc/init.d/slurmd /etc/startup.sh /etc/map_wine.py /etc/slurm/slurm.conf /etc/slurm/cgroup.conf /etc/slurm/gres.conf \
    && chmod 555 /etc/init.d/slurmctld /etc/init.d/slurmd /etc/startup.sh /etc/map_wine.py \
    && chmod 644 /etc/slurm/slurm.conf /etc/slurm/cgroup.conf /etc/slurm/gres.conf \
    && cp /etc/slurm/slurm.conf /etc/slurm/slurm.conf.template \
    && cp /etc/slurm/gres.conf /etc/slurm/gres.conf.template

RUN mkdir -p /data/c /data/z /data/config /data/work \
        /home/${USERNAME}/.wine/dosdevices \
        /home/${USERNAME}/.wine/drive_c \
    && ln -sf ../drive_c /home/${USERNAME}/.wine/dosdevices/c: \
    && chown -R "${USERNAME}:${USERNAME}" /data /home/${USERNAME} /opt/fiji-linux /opt/src /opt/docker-src

RUN HOME=/home/${USERNAME} \
    WINEPREFIX=/home/${USERNAME}/.wine \
    sudo -E -u "${USERNAME}" wineboot --init 2>&1; \
    test -d /home/${USERNAME}/.wine/drive_c/windows \
    || { echo "ERROR: Wine prefix not initialised – drive_c/windows missing"; exit 1; }

USER ${USERNAME}
WORKDIR /home/${USERNAME}

RUN sed -i -e 's/#force_color_prompt=yes/force_color_prompt=yes/' "/home/${USERNAME}/.bashrc"

CMD ["sudo", "/etc/startup.sh"]
