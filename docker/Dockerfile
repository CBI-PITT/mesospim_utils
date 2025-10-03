## SETTINGS ------------------------------------------------------------------
ARG USER_PRIVILEGE_LEVEL

# Settings for all images
ARG TIMEZONE
ARG UBUNTU_VERSION

# Settings for standard user images
ARG USERNAME
ARG PASSWORD
ARG UID

# Settings for python
ARG CONDA_DIR
ARG CONDA_VERSION
ARG CONDA_OS
ARG CONDA_ARCH

# --- SLURM builder from source with nvml support---
FROM ubuntu:${UBUNTU_VERSION} AS slurm-wine-build
ARG SLURM_VER=${SLURM_VER}

RUN apt-get update && apt-get install -y \
    build-essential \
    fakeroot \
    devscripts \
    equivs \
    dpkg-dev \
    debhelper \
    dh-autoreconf \
    ca-certificates \
    wget \
    git \
    python3 \
    libnvidia-ml-dev \
    libpmix-dev \
    libhwloc-dev \
    g++ \
    cmake \
    flex \
    bison \
    && apt autoremove -y && apt clean -y

RUN wget https://download.schedmd.com/slurm/slurm-${SLURM_VER}.tar.bz2; \
    tar -xjf slurm-${SLURM_VER}.tar.bz2; \
    cd slurm-${SLURM_VER}; \
    mk-build-deps -i -r -t "apt-get -y" debian/control; \
    debuild -b -uc -us -j"$(nproc)"; \
    cd ..; \
    mkdir -p /tmp/slurm-install; \
    cp slurm*.deb /tmp/slurm-install

#mk-build-deps -i -r -t "apt-get -y" debian/control; \

# --- Clone and install wine from source---
# Compile wine from source and install in /usr/local/bin
# Binary used for mesospim_utils /usr/local/bin/wine64
# Final step, initialize wine for the primary user: default docker
ARG WINE_GITHUB=${WINE_GITHUB}
ARG WINE_COMMIT=${WINE_COMMIT}
ENV SRC_DIR="/opt"
RUN set -eux; \
    #mkdir -p "${SRC_DIR}"; \
    cd "${SRC_DIR}"; \
    git init wine; \
    cd wine; \
    git remote add origin ${WINE_GITHUB}; \
    git fetch --depth 1 origin ${WINE_COMMIT}; \
    git checkout --detach FETCH_HEAD; \
    ./configure --enable-win64 --without-x --without-freetype; \
    CPUS=$(nproc || echo 1); MAX=$((CPUS-2)); [ "$MAX" -lt 1 ] && MAX=1; echo "$MAX"; \
    make -j${CPUS}; \
    make install DESTDIR=/tmp/wine-install


## BUILD IMAGE: mesospim_utils ----------------------------------------------------
FROM ubuntu:${UBUNTU_VERSION} as setup-mesospim-utils

# Copy over SLURM binaries
COPY --from=slurm-wine-build /tmp/slurm-install/ /opt/slurm-install
COPY --from=slurm-wine-build /tmp/wine-install/ /
RUN ls -lh /opt/slurm-install

# Run system updates
RUN apt update && apt -y upgrade && apt autoremove -y && apt clean -y

# Set time zone
RUN ln -sn /usr/share/zoneinfo/${TIMEZONE} /etc/localtime \
    && echo ${TIMEZONE} > /etc/timezone \
    && apt -y install tzdata && apt autoremove -y && apt clean -y

# Set locale
RUN apt -y install locales \
    && locale-gen en_US.UTF-8 \
    && update-locale LANG=en_US.UTF-8 LC_MESSAGES=POSIX \
    && apt autoremove -y && apt clean -y

# Install system packages
RUN DEBIAN_FRONTEND=noninteractive \
    apt -y install \
    sudo \
    openssl \
    munge \
    tmux \
    htop \
    nano \
    git \
    wget \
    build-essential \
    cmake \
    && apt autoremove -y && apt clean -y

ARG SLURM_VER=${SLURM_VER}
RUN apt-get update; \
    apt-get install libpmix-dev -y; \
    groupadd -r -g 64030 slurm; \
    useradd -M slurm -u 64030 -g 64030 -s /usr/sbin/nologin --home /nonexistent; \
    mkdir -p /var/spool/slurmd; \
    mkdir -p /var/spool/slurmctld; \
    chown -R slurm:slurm /var/spool/slurm*; \
    chmod -R 775 /var/spool/slurm*; \
    mkdir -p /var/log/slurmd; \
    mkdir -p /var/log/slurm; \
    chown -R slurm:slurm /var/log/slurm*; \
    chmod -R 775 /var/log/slurm*; \
    mkdir -p /etc/slurm; \
    chown -R slurm:slurm /etc/slurm*; \
    chmod -R 775 /etc/slurm*; \
    chmod 777 -R /opt/slurm-install; \
    apt install -y /opt/slurm-install/slurm-smd_${SLURM_VER}-1_amd64.deb; \
    apt install -y /opt/slurm-install/slurm-smd-slurmctld_${SLURM_VER}-1_amd64.deb; \
    apt install -y /opt/slurm-install/slurm-smd-slurmd_${SLURM_VER}-1_amd64.deb; \
    apt install -y /opt/slurm-install/slurm-smd-client_${SLURM_VER}-1_amd64.deb; \
    systemctl disable slurmd; \
    systemctl disable slurmctld; \
    apt autoremove -y && apt clean -y
#rm -r /opt/slurm-install

# Settings for standard user images
ARG USERNAME
ARG PASSWORD
ARG UID

# Settings for python
ARG CONDA_DIR
ARG CONDA_VERSION
ARG CONDA_OS
ARG CONDA_ARCH


# Set working directory and default command when running container
WORKDIR /root
#CMD /etc/startup.sh ; /bin/bash -l
CMD /bin/bash -l


# Create user account and add to "sudo" group
# Configure user permissions for startup script
RUN useradd --create-home --uid=${UID} ${USERNAME} -p "$(openssl passwd -1 ${PASSWORD})" \
    && usermod -a -G sudo ${USERNAME} \
    && printf "${USERNAME} ALL=(root:root) NOPASSWD: /etc/startup.sh\n" >> /etc/sudoers.d/startup \
    && chmod 0440 /etc/sudoers.d/startup \
    && visudo -c

# ---------------- Miniconda (system-wide) ----------------
# You can override these at build time if desired
ENV PATH=${CONDA_DIR}/bin:${PATH}

# Install Miniconda silently to ${CONDA_DIR} and enable globally
RUN set -eux; \
    installer=Miniconda3-${CONDA_VERSION}-${CONDA_OS}-${CONDA_ARCH}.sh; \
    wget -q https://repo.anaconda.com/miniconda/${installer}; \
    bash ${installer} -b -p "${CONDA_DIR}"; \
    rm -f ${installer}; \
    # Global activation script for login shells
    mkdir -p /etc/profile.d; \
    echo '. '"${CONDA_DIR}"'/etc/profile.d/conda.sh' > /etc/profile.d/conda.sh; \
    echo 'conda activate base' >> /etc/profile.d/conda.sh; \
    # Basic house-keeping
    "${CONDA_DIR}"/bin/conda config --system --set always_yes true; \
    "${CONDA_DIR}"/bin/conda update -n base -c defaults conda; \
    "${CONDA_DIR}"/bin/conda clean -afy; \
    rm -rf /root/.cache/conda /tmp/*; \
    chown -R ${USERNAME}:${USERNAME} ${CONDA_DIR}; \
    chmod -R 775 ${CONDA_DIR}

# --- Clone and editable-install mesospim_utils for ${USERNAME} ---
ARG MESOSPIM_UTILS_REF=main
ENV SRC_DIR="/opt"
RUN set -eux; \
    mkdir -p "${SRC_DIR}" && cd "${SRC_DIR}"; \
    git clone --depth 1 --branch "${MESOSPIM_UTILS_REF}" "https://github.com/CBI-PITT/mesospim_utils.git"; \
    cd mesospim_utils; \
    # Use the python from /opt/conda (already on PATH); editable install
    python -m pip install --no-cache-dir -U pip setuptools wheel; \
    python -m pip install --no-cache-dir -e . psfmodels; \
    #rm -rf /home/${USERNAME}/.cache/pip/* /tmp/*; \
    rm -rf /root/.cache/pip/* /tmp/*; \
    rm -r "${SRC_DIR}/mesospim_utils/.git"; \
    rm -r "${SRC_DIR}/mesospim_utils/.eggs"; \
    rm -r "${SRC_DIR}/mesospim_utils/images"; \
    rm -r "${SRC_DIR}/mesospim_utils/notes"; \
    chown -R ${USERNAME}:${USERNAME} ${SRC_DIR}/mesospim_utils; \
    chmod -R 775 ${SRC_DIR}/mesospim_utils



# --- Clone ImarisConverter for ${USERNAME} ---
ENV SRC_DIR="/opt"
RUN set -eux; \
    mkdir -p "${SRC_DIR}" && cd "${SRC_DIR}"; \
    git clone --depth 1 --branch "main" "https://github.com/CBI-PITT/ImarisFileConverter_wine.git"; \
    cd ImarisFileConverter_wine; \
    rm -r "${SRC_DIR}/ImarisFileConverter_wine/.git"; \
    chown -R ${USERNAME}:${USERNAME} ${SRC_DIR}/ImarisFileConverter_wine; \
    chmod -R 775 ${SRC_DIR}/ImarisFileConverter_wine

# Copy slurm init.d service scripts
COPY etc/init.d/slurmctld /etc/init.d/slurmctld
COPY etc/init.d/slurmd /etc/init.d/slurmd
RUN sed -i 's/\r$//' /etc/init.d/slurmctld && chmod 555 /etc/init.d/slurmctld
RUN sed -i 's/\r$//' /etc/init.d/slurmd && chmod 555 /etc/init.d/slurmd
RUN chmod 755 /etc/init.d/slurm*

# Copy SLURM configuration
COPY etc/slurm /etc/slurm
RUN chmod 777 /etc/slurm/*
RUN sed -i 's/\r$//' /etc/slurm/slurm.conf
RUN sed -i 's/\r$//' /etc/slurm/cgroup.conf
RUN sed -i 's/\r$//' /etc/slurm/gres.conf
#COPY etc/slurm/cgroup.conf /etc/slurm/cgroup.conf
#COPY etc/slurm/gres.conf /etc/slurm/gres.conf

# Startup configuration
COPY startup.sh /etc/startup.sh
RUN sed -i 's/\r$//' /etc/startup.sh && chmod 555 /etc/startup.sh

COPY map_wine.py /etc/map_wine.py
RUN sed -i 's/\r$//' /etc/map_wine.py && chmod 555 /etc/map_wine.py

# Initialize wine for ${USERNAME}
RUN sudo -u ${USERNAME} winecfg

# Set standard user as default

USER ${USERNAME}

# Enable colored Bash terminal prompt
RUN sed -i -e 's/#force_color_prompt=yes/force_color_prompt=yes/' /home/${USERNAME}/.bashrc

# Set working directory and default command when running container
WORKDIR /home/${USERNAME}
CMD sudo /etc/startup.sh


## SELECT DESIRED USER PRIVILEGE LEVEL ----------------------------------------
#FROM stage-${USER_PRIVILEGE_LEVEL}
