# syntax=docker/dockerfile:1.6
#
# RT-COSMIK image: everything the toolbox needs, at the versions it is validated
# with -- CUDA 12.1, PyTorch 2.4.1, TensorRT 10.15, ROS 2 Humble, and CasADi,
# fatrop, Pinocchio and acados built from source.
#
# The repository is not copied in. docker/run.sh (or the VS Code dev container)
# mounts your checkout at /root/workspace/rt-cosmik, so code, downloaded models,
# generated solvers and results all stay on the host.
#
# Building compiles CasADi, Pinocchio and acados from source: expect about an
# hour. Compile jobs are capped by available memory (see build_jobs below); on a
# machine with little RAM, add swap rather than raising them.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]

ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    RCUTILS_COLORIZED_OUTPUT=1 \
    CMAKE_BUILD_TYPE=RelWithDebInfo \
    ROS_DISTRO=humble \
    DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------
# Locales and ROS 2 Humble (for the rtcosmik_ros overlay)
# -----------------------------------------------------------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      locales software-properties-common \
      curl gnupg2 lsb-release ca-certificates \
    && locale-gen en_US.UTF-8 \
    && update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

RUN add-apt-repository universe \
    && apt-get update \
    && apt-get install -y --no-install-recommends curl python3 ca-certificates \
    && ROS_APT_SOURCE_VERSION="$(curl -fsSL -H 'User-Agent: docker' -H 'Accept: application/vnd.github+json' \
      https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
      | python3 -c 'import sys,json; print(json.load(sys.stdin)["tag_name"])')" \
    && test -n "${ROS_APT_SOURCE_VERSION}" \
    && CODENAME="$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})" \
    && curl -fsSL -o /tmp/ros2-apt-source.deb \
      "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.${CODENAME}_all.deb" \
    && dpkg -i /tmp/ros2-apt-source.deb \
    && rm -f /tmp/ros2-apt-source.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      ros-${ROS_DISTRO}-desktop \
      ros-dev-tools \
      python3-rosdep python3-colcon-common-extensions python3-vcstool \
    && rosdep init \
    && rosdep update

# -----------------------------------------------------------------------------
# System packages, and Python packages pinned to numpy 1.x
# -----------------------------------------------------------------------------
# ffmpeg opens the cameras and records them; v4l-utils (v4l2-ctl) lists them.
RUN --mount=type=cache,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,sharing=locked,target=/var/lib/apt \
    --mount=type=cache,sharing=locked,target=/root/.cache \
    set -euxo pipefail \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        bash-completion build-essential cmake gdb git htop nano iputils-ping net-tools sudo wget vim \
        ffmpeg v4l-utils usbutils \
        clang-format ccache liburdfdom-tools \
        libxcb1 libx11-6 libxext6 libxrender1 libxkbcommon0 libxfixes3 libxi6 libxrandr2 libxcursor1 libxcomposite1 libxdamage1 \
        libglib2.0-0 libgl1 libsm6 libgtk-3-0 \
        libnvinfer-bin \
        gfortran liblapack-dev libblas-dev libopenblas-dev coinor-libipopt-dev swig pkg-config libmetis-dev libgmp-dev \
        python3 python3-venv python3-distutils python3-dev python-is-python3 python3-flake8 python3-pip python3-setuptools ipython3 \
        \
        # eigenpy / coal / pinocchio, built from source below
        libeigen3-dev libboost-all-dev libboost-python-dev \
        liburdfdom-dev libconsole-bridge-dev libtinyxml2-dev \
        pybind11-dev \
        libassimp-dev liboctomap-dev libqhull-dev \
    && rm -rf /var/lib/apt/lists/* \
    \
    # ---- pip constraints: force numpy<2 everywhere ----
    && python -m pip install -U pip \
    && printf "numpy==1.26.4\n" > /tmp/constraints.txt \
    && python -m pip install -c /tmp/constraints.txt "numpy==1.26.4" \
    \
    # torch stack
    && python -m pip install -c /tmp/constraints.txt \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        "torch==2.4.1+cu121" "torchvision==0.19.1+cu121" "torchaudio==2.4.1+cu121" \
    \
    # the rest (also constrained)
    && python -m pip install -c /tmp/constraints.txt --upgrade-strategy only-if-needed \
        argcomplete pre-commit \
        quadprog ultralytics opencv-python meshcat pandas \
        onnxruntime-gpu onnx \
        looseversion pynput \
    && python -m pip check \
    \
    # sanity: make sure numpy is <2 right now
    && python -c "import numpy as np; assert int(np.__version__.split('.')[0]) < 2, np.__version__; print('numpy ok:', np.__version__, np.__file__)"

# TensorRT, for the detector engines fetch_models.sh exports. Pinned: an engine
# only loads with the TensorRT version that built it.
RUN python -m pip install -c /tmp/constraints.txt "tensorrt-cu12==10.15.1.29"

# -----------------------------------------------------------------------------
# Record of what the image contains
# -----------------------------------------------------------------------------
RUN mkdir -p /opt/manifest; \
  python -m pip freeze > /opt/manifest/pip_freeze.txt; \
  dpkg-query -W > /opt/manifest/dpkg.txt; \
  apt-mark showmanual > /opt/manifest/apt_manual.txt

# -----------------------------------------------------------------------------
# Workspace layout
# -----------------------------------------------------------------------------
RUN mkdir -p /root/workspace/deps /root/workspace/ros_ws
WORKDIR /root/workspace
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc
# The checkout is mounted from the host and owned by the host user, while the
# container runs as root: without this, git refuses to work in it ("dubious
# ownership"), and fetch_models.sh locates the repository with git.
RUN git config --global --add safe.directory '*'

# -----------------------------------------------------------------------------
# Build-job limiter: min(nproc-4, RAM/3GB). The heavy template compiles below
# (eigenpy/pinocchio/casadi/coal) are MEMORY-bound (~2-4 GB per job); limiting by
# cores alone OOM-freezes machines with modest RAM. Used as make -j"$(build_jobs)".
# -----------------------------------------------------------------------------
RUN printf '%s\n' \
    '#!/bin/bash' \
    'cpu=$(( $(nproc) > 4 ? $(nproc) - 4 : 1 ))' \
    'mem=$(( $(grep MemAvailable /proc/meminfo | tr -dc 0-9) / 4194304 ))' \
    '[ "$mem" -lt 1 ] && mem=1' \
    'echo "$(( cpu < mem ? cpu : mem ))"' \
    > /usr/local/bin/build_jobs && chmod +x /usr/local/bin/build_jobs

# -----------------------------------------------------------------------------
# Build BLASFEO, FATROP, CasADi from source (installed into /usr/local)
# -----------------------------------------------------------------------------
WORKDIR /root/workspace/deps

RUN git clone https://github.com/giaf/blasfeo.git; \
    cd blasfeo; \
    git checkout 9923ac8a63d710d894a3d2987d885002df69db74; \
    mkdir -p build; \
    cd build; \
    cmake .. \
      -DTARGET=X64_AUTOMATIC \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DCMAKE_BUILD_TYPE=Release; \
    make -j"$(build_jobs)"; \
    make install; \
    ldconfig

RUN git clone https://github.com/meco-group/fatrop.git; \
    cd fatrop; \
    git checkout 45ee388750ba2b4cf7ba603203d4c12431e485fe; \
    mkdir -p build; \
    cd build; \
    cmake .. \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTS=OFF; \
    make -j"$(build_jobs)"; \
    make install; \
    ldconfig

RUN git clone https://github.com/casadi/casadi.git; \
    cd casadi; \
    git checkout 3.7.2; \
    mkdir -p build; \
    cd build; \
    cmake .. \
      -DWITH_IPOPT=ON -DWITH_BUILD_IPOPT=ON \
      -DWITH_FATROP=ON \
      -DWITH_OPENMP=ON -DWITH_THREAD=ON \
      -DWITH_BUILD_MUMPS=ON -DWITH_BUILD_METIS=ON \
      -DWITH_PYTHON=ON -DWITH_PYTHON3=ON \
      -DPYTHON_PREFIX="$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')" \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DCMAKE_BUILD_TYPE=Release; \
    make -j1 mumps-external; \
    make -j"$(build_jobs)"; \
    make install; \
    ldconfig

# -----------------------------------------------------------------------------
# Build eigenpy from source (pinned) -> installs /usr/local/lib{,64}/cmake/eigenpy + libeigenpy.so
# -----------------------------------------------------------------------------
RUN git clone https://github.com/stack-of-tasks/eigenpy.git; \
    cd eigenpy; \
    git checkout v3.12.0; \
    mkdir -p build; \
    cd build; \
    cmake .. \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=OFF \
      -DPython_EXECUTABLE=/usr/bin/python \
      -DCMAKE_PREFIX_PATH="/usr/local"; \
    make -j"$(build_jobs)"; \
    make install; \
    ldconfig

# -----------------------------------------------------------------------------
# Build Pinocchio from source WITH CasADi + Python bindings + coal (pinned),
# and example-robot-data, which provides RT-COSMIK's human model
# -----------------------------------------------------------------------------
RUN git clone https://github.com/coal-library/coal.git; \
    cd coal; \
    git checkout v3.0.2; \
    mkdir -p build; \
    cd build; \
    cmake .. \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=OFF \
      -DBUILD_PYTHON_INTERFACE=ON \
      -DCOAL_BACKWARD_COMPATIBILITY_WITH_HPP_FCL=ON \
      -DPYTHON_EXECUTABLE=/usr/bin/python \
      -DPython_EXECUTABLE=/usr/bin/python \
      -Deigenpy_DIR=/usr/local/lib/cmake/eigenpy \
      -DCMAKE_PREFIX_PATH="/usr/local"; \
    make -j"$(build_jobs)"; \
    make install; \
    ldconfig

RUN git clone --recursive https://github.com/stack-of-tasks/pinocchio.git; \
    cd pinocchio; \
    git checkout v3.9.0; \
    git submodule update --init --recursive; \
    mkdir -p build; \
    cd build; \
    cmake .. \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_BENCHMARK=OFF \
      -DBUILD_PYTHON_INTERFACE=ON \
      -DBUILD_WITH_CASADI_SUPPORT=ON \
      -DBUILD_WITH_COLLISION_SUPPORT=ON \
      -DPython_EXECUTABLE=/usr/bin/python \
      -Deigenpy_DIR=/usr/local/lib/cmake/eigenpy \
      -DCMAKE_PREFIX_PATH="/usr/local"; \
    make -j"$(build_jobs)"; \
    make install; \
    ldconfig

RUN git clone --recursive https://github.com/Gepetto/example-robot-data.git; \
    cd example-robot-data; \
    # The human model's trunk fix (PR #387), without the later loader changes
    # that need a Pinocchio newer than 3.9 (pin.WITH_COLLISION).
    git checkout 9c202b1f38ba38b02e2c70e19364ea9e3f4407d5; \
    git submodule update --init --recursive; \
    mkdir -p build; \
    cd build; \
    cmake ..; \
    make -j"$(build_jobs)"; \
    make install; \
    ldconfig

# -----------------------------------------------------------------------------
# Build acados from source (MHE-IK "acados" backend) + Python template + tera.
# Pinned to the commit validated with RT-COSMIK (v0.5.4-20-g8e1a6f856).
# -----------------------------------------------------------------------------
RUN git clone https://github.com/acados/acados.git; \
    cd acados; \
    git checkout 8e1a6f856e063c423de583e03c691e3b2b7fc0a0; \
    git submodule update --init --recursive; \
    mkdir -p build; \
    cd build; \
    cmake .. \
      -DACADOS_INSTALL_DIR=/root/workspace/deps/acados \
      -DACADOS_WITH_OPENMP=OFF \
      -DBLASFEO_TARGET=X64_AUTOMATIC \
      -DCMAKE_BUILD_TYPE=Release; \
    make -j"$(build_jobs)" install; \
    echo "/root/workspace/deps/acados/lib" > /etc/ld.so.conf.d/acados.conf; \
    ldconfig

# tera renderer (acados codegen template engine) — download non-interactively
RUN mkdir -p /root/workspace/deps/acados/bin; \
    curl -fsSL -o /root/workspace/deps/acados/bin/t_renderer \
      https://github.com/acados/tera_renderer/releases/download/v0.2.0/t_renderer-v0.2.0-linux-amd64; \
    chmod +x /root/workspace/deps/acados/bin/t_renderer

# acados Python interface. --no-deps is CRITICAL: it prevents pip from pulling a
# casadi wheel that would shadow the from-source casadi and break pinocchio.casadi.
# Its other deps (numpy/scipy already present) are installed explicitly.
RUN python -m pip install -c /tmp/constraints.txt matplotlib cython Deprecated; \
    python -m pip install --no-deps -e /root/workspace/deps/acados/interfaces/acados_template; \
    python -c "import acados_template; print('acados_template OK')"

# -----------------------------------------------------------------------------
# Environment. Set here rather than in .bashrc so that non-interactive commands
# (docker/run.sh <command>, the dev container's tasks) see it too.
# -----------------------------------------------------------------------------
# Pinocchio, coal and example-robot-data install into /usr/local/.../site-packages,
# which Ubuntu's Python does not search by default; rtcosmik is imported from
# the mounted checkout. Paths are prepended to the CUDA base image's own values.
ENV ACADOS_SOURCE_DIR=/root/workspace/deps/acados \
    PYTHONPATH=/root/workspace/rt-cosmik/src:/usr/local/lib/python3.10/site-packages${PYTHONPATH:+:${PYTHONPATH}} \
    LD_LIBRARY_PATH=/usr/local/lib:/root/workspace/deps/acados/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} \
    LIBRARY_PATH=/usr/local/lib${LIBRARY_PATH:+:${LIBRARY_PATH}} \
    CPATH=/usr/local/include${CPATH:+:${CPATH}} \
    PKG_CONFIG_PATH=/usr/local/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}

WORKDIR /root/workspace/rt-cosmik

CMD ["bash"]
