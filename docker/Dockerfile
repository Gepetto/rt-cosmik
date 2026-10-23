# syntax=docker/dockerfile:1.6
# RT-COSMIK development image. The version pins preserve a known-working
# environment from development; they do not identify the newest compatible
# releases. Upgrade them together with the checks in docker/check.sh, especially
# Pinocchio/CasADi and the CUDA/PyTorch/TensorRT stack.
#
# Sources are downloaded by Docker, separately from compilation. Release
# tarballs include their CMake submodules; generated GitHub archives do not,
# so the required submodules are fetched explicitly at their parent's revisions.
# When changing a release version, update its checksum as well.
FROM scratch AS sources

ARG ROS_APT_SOURCE_VERSION=1.3.0
ADD --checksum=sha256:110b9a462d55252decb8b7c816f61c2ba0d9890ce5fb93ac504e97cae5860d76 \
    https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.jammy_all.deb /ros2-apt-source.deb

ARG BLASFEO_REV=9923ac8a63d710d894a3d2987d885002df69db74
ADD https://github.com/giaf/blasfeo/archive/${BLASFEO_REV}.tar.gz /blasfeo.tar.gz
ARG FATROP_REV=45ee388750ba2b4cf7ba603203d4c12431e485fe
ADD https://github.com/meco-group/fatrop/archive/${FATROP_REV}.tar.gz /fatrop.tar.gz
ARG CASADI_VERSION=3.7.2
ADD https://github.com/casadi/casadi/archive/refs/tags/${CASADI_VERSION}.tar.gz /casadi.tar.gz

ARG PINOCCHIO_VERSION=3.9.0
ADD --checksum=sha256:60553630d83de492bc0cf1126add2acc591c87f1bc8ea7f70693e7563fc103a3 \
    https://github.com/stack-of-tasks/pinocchio/releases/download/v${PINOCCHIO_VERSION}/pinocchio-${PINOCCHIO_VERSION}.tar.gz /pinocchio.tar.gz

# Includes the human model's trunk fix (PR #387), before loader changes that
# require a newer Pinocchio. Keep its CMake submodule at the matching revision.
ARG EXAMPLE_ROBOT_DATA_REV=9c202b1f38ba38b02e2c70e19364ea9e3f4407d5
ADD https://github.com/Gepetto/example-robot-data/archive/${EXAMPLE_ROBOT_DATA_REV}.tar.gz /example-robot-data.tar.gz
ARG EXAMPLE_ROBOT_DATA_CMAKE_REV=330567ca8c9b342952222a1311a54dbb7ed6608f
ADD https://github.com/jrl-umi3218/jrl-cmakemodules/archive/${EXAMPLE_ROBOT_DATA_CMAKE_REV}.tar.gz /example-robot-data-cmake.tar.gz

# acados uses its own BLASFEO/HPIPM revisions; these are the gitlinks from the
# pinned acados commit, rather than the standalone BLASFEO used by fatrop.
ARG ACADOS_REV=8e1a6f856e063c423de583e03c691e3b2b7fc0a0
ADD https://github.com/acados/acados/archive/${ACADOS_REV}.tar.gz /acados.tar.gz
ARG ACADOS_BLASFEO_REV=d6251233923c9b475fe894fb729fb63ab693e301
ADD https://github.com/giaf/blasfeo/archive/${ACADOS_BLASFEO_REV}.tar.gz /acados-blasfeo.tar.gz
ARG ACADOS_HPIPM_REV=e3a56c1caddd7f12d125d84f337b9a9e5c186271
ADD https://github.com/giaf/hpipm/archive/${ACADOS_HPIPM_REV}.tar.gz /acados-hpipm.tar.gz
ARG TERA_VERSION=0.2.0
ADD https://github.com/acados/tera_renderer/releases/download/v${TERA_VERSION}/t_renderer-v${TERA_VERSION}-linux-amd64 /t_renderer

FROM ghcr.io/astral-sh/uv:0.13.5 AS uv

# The toolbox builds C/C++, but no CUDA kernels. Keep the CUDA runtime without
# nvcc, Nsight and the CUDA/cuDNN development archives from the devel image.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    ROS_DISTRO=humble \
    RCUTILS_COLORIZED_OUTPUT=1

# Ubuntu normally deletes downloaded .deb files after installation. Cache
# mounts keep them outside the image and reuse them on subsequent builds.
RUN rm -f /etc/apt/apt.conf.d/docker-clean \
    && printf 'Binary::apt::APT::Keep-Downloaded-Packages "true";\n' > /etc/apt/apt.conf.d/keep-cache

RUN --mount=type=cache,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,sharing=locked,target=/var/lib/apt \
    apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -qqy --no-install-recommends \
        ca-certificates locales software-properties-common \
    && locale-gen en_US.UTF-8 \
    && update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
    && add-apt-repository -y -n universe

# Pinocchio 3.9 needs the hpp-fcl interface. ROS's coal binary does not enable
# that compatibility layer, so use the packaged HPP-FCL instead.
RUN --mount=type=cache,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,sharing=locked,target=/var/lib/apt \
    --mount=type=bind,from=sources,source=/ros2-apt-source.deb,target=/tmp/ros2-apt-source.deb \
    dpkg -i /tmp/ros2-apt-source.deb \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -qqy --no-install-recommends \
        ros-${ROS_DISTRO}-ros-base ros-dev-tools \
        ros-${ROS_DISTRO}-rviz2 ros-${ROS_DISTRO}-robot-state-publisher \
        ros-${ROS_DISTRO}-eigenpy ros-${ROS_DISTRO}-hpp-fcl \
        python3-rosdep python3-colcon-common-extensions python3-vcstool \
    && printf '/opt/ros/%s/lib\n' "${ROS_DISTRO}" > /etc/ld.so.conf.d/ros.conf \
    && ldconfig \
    && rosdep init \
    && rosdep update

# ffmpeg/v4l-utils open, record and enumerate cameras. The remaining development
# tools are also needed when generating solvers or building the ROS overlay.
RUN --mount=type=cache,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,sharing=locked,target=/var/lib/apt \
    apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -qqy --no-install-recommends \
        bash-completion build-essential cmake gdb git htop nano iputils-ping net-tools sudo wget vim \
        curl gnupg2 lsb-release ffmpeg v4l-utils usbutils \
        clang-format ccache liburdfdom-tools \
        libxcb1 libx11-6 libxext6 libxrender1 libxkbcommon0 libxfixes3 libxi6 libxrandr2 libxcursor1 libxcomposite1 libxdamage1 \
        libglib2.0-0 libgl1 libsm6 libgtk-3-0 \
        gfortran liblapack-dev libblas-dev libopenblas-dev coinor-libipopt-dev swig pkg-config libmetis-dev libgmp-dev \
        python3 python3-venv python3-distutils python3-dev python-is-python3 python3-flake8 python3-setuptools python3-cairo ipython3 \
        libeigen3-dev libboost-all-dev libboost-python-dev \
        liburdfdom-dev libconsole-bridge-dev libtinyxml2-dev pybind11-dev \
        libassimp-dev liboctomap-dev libqhull-dev

# One prefix for pip packages and compiled Python bindings. System packages
# remain visible for ROS. The dist-packages alias lets ROS scripts with a
# /usr/bin/python3 shebang find the same packages without PYTHONPATH juggling.
RUN find /opt/ros/${ROS_DISTRO}/lib -maxdepth 1 -type d -name '*-linux-gnu' \
        >> /etc/ld.so.conf.d/ros.conf \
    && ldconfig
ENV VIRTUAL_ENV=/usr/local
ENV PATH=${VIRTUAL_ENV}/bin:${PATH}
RUN /usr/bin/python3 -m venv --system-site-packages "${VIRTUAL_ENV}" \
    && python - <<'PY'
from pathlib import Path
import os
import sysconfig

site_packages = Path(sysconfig.get_path("purelib"))
dist_packages = site_packages.with_name("dist-packages")
if dist_packages.exists():
    dist_packages.rmdir()  # Fail rather than replace a populated directory.
dist_packages.symlink_to("site-packages", target_is_directory=True)
(site_packages / "rtcosmik.pth").write_text("/root/workspace/rt-cosmik/src\n")
ros_prefix = Path("/opt/ros") / os.environ["ROS_DISTRO"]
ros_paths = sorted(ros_prefix.glob("lib/python*/site-packages"))
ros_paths += sorted(ros_prefix.glob("local/lib/python*/dist-packages"))
assert ros_paths, "ROS Python packages were not found"
(site_packages / "ros.pth").write_text("".join(f"{path}\n" for path in ros_paths))
PY

# NumPy is a native-build prerequisite. Application requirements come after the
# compiled libraries, so changing a detector or utility does not rebuild them.
COPY constraints.txt /opt/rt-cosmik/
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -c /opt/rt-cosmik/constraints.txt numpy

FROM base AS native

# Build settings belong to the build, rather than every running container.
# The default limiter allows roughly 4 GiB per compile job and reserves 4 CPUs.
# Use --build-arg MAKE_JOBS=2 to choose a smaller (or larger) explicit limit.
ARG MAKE_JOBS
ARG CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_PREFIX_PATH=/usr/local;/opt/ros/${ROS_DISTRO} -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
RUN <<'SH'
cat > /usr/local/bin/build_jobs <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${MAKE_JOBS:-}" ]]; then
    [[ "${MAKE_JOBS}" =~ ^[1-9][0-9]*$ ]] || { echo "MAKE_JOBS must be a positive integer" >&2; exit 1; }
    echo "${MAKE_JOBS}"
    exit
fi
cpu=$(( $(nproc) > 4 ? $(nproc) - 4 : 1 ))
mem=$(awk '/MemAvailable:/ {print int($2 / 4194304)}' /proc/meminfo)
if [[ "${mem}" -lt 1 ]]; then mem=1; fi
echo "$(( cpu < mem ? cpu : mem ))"
EOF
chmod +x /usr/local/bin/build_jobs
SH

# Source archives are mounted for each build, so neither the downloads nor
# temporary build trees are retained in the final image.
RUN --mount=type=bind,from=sources,source=/blasfeo.tar.gz,target=/tmp/source.tar.gz \
    --mount=type=cache,id=rt-cosmik-ccache,target=/root/.cache/ccache \
    mkdir /tmp/blasfeo \
    && tar -xzf /tmp/source.tar.gz -C /tmp/blasfeo --strip-components=1 \
    && cmake -S /tmp/blasfeo -B /tmp/blasfeo/build ${CMAKE_ARGS} -DTARGET=X64_AUTOMATIC \
    && cmake --build /tmp/blasfeo/build --parallel "$(build_jobs)" \
    && cmake --install /tmp/blasfeo/build \
    && ldconfig \
    && rm -rf /tmp/blasfeo

RUN --mount=type=bind,from=sources,source=/fatrop.tar.gz,target=/tmp/source.tar.gz \
    --mount=type=cache,id=rt-cosmik-ccache,target=/root/.cache/ccache \
    mkdir /tmp/fatrop \
    && tar -xzf /tmp/source.tar.gz -C /tmp/fatrop --strip-components=1 \
    && cmake -S /tmp/fatrop -B /tmp/fatrop/build ${CMAKE_ARGS} -DBUILD_TESTS=OFF \
    && cmake --build /tmp/fatrop/build --parallel "$(build_jobs)" \
    && cmake --install /tmp/fatrop/build \
    && ldconfig \
    && rm -rf /tmp/fatrop

RUN --mount=type=bind,from=sources,source=/casadi.tar.gz,target=/tmp/source.tar.gz \
    --mount=type=cache,id=rt-cosmik-ccache,target=/root/.cache/ccache \
    mkdir /tmp/casadi \
    && tar -xzf /tmp/source.tar.gz -C /tmp/casadi --strip-components=1 \
    && cmake -S /tmp/casadi -B /tmp/casadi/build ${CMAKE_ARGS} \
        -DWITH_EXAMPLES=OFF \
        -DWITH_IPOPT=ON -DWITH_BUILD_IPOPT=ON -DWITH_FATROP=ON \
        -DWITH_OPENMP=ON -DWITH_THREAD=ON \
        -DWITH_BUILD_MUMPS=ON -DWITH_BUILD_METIS=ON \
        -DWITH_PYTHON=ON -DWITH_PYTHON3=ON \
        -DPYTHON_EXECUTABLE="${VIRTUAL_ENV}/bin/python3" \
        -DPYTHON_PREFIX="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')" \
        -DCASADI_PYTHON_PIP_METADATA_INSTALL=ON \
    && cmake --build /tmp/casadi/build --target mumps-external --parallel 1 \
    && cmake --build /tmp/casadi/build --parallel "$(build_jobs)" \
    && cmake --install /tmp/casadi/build \
    && ldconfig \
    && rm -rf /tmp/casadi

# ccache's PCH settings also cache Pinocchio's Python binding compilations.
RUN --mount=type=bind,from=sources,source=/pinocchio.tar.gz,target=/tmp/source.tar.gz \
    --mount=type=cache,id=rt-cosmik-ccache,target=/root/.cache/ccache \
    mkdir /tmp/pinocchio \
    && tar -xzf /tmp/source.tar.gz -C /tmp/pinocchio --strip-components=1 \
    && cmake -S /tmp/pinocchio -B /tmp/pinocchio/build ${CMAKE_ARGS} \
        -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF \
        -DBUILD_PYTHON_INTERFACE=ON -DBUILD_WITH_CASADI_SUPPORT=ON \
        -DBUILD_WITH_COLLISION_SUPPORT=ON -DPython_EXECUTABLE="${VIRTUAL_ENV}/bin/python3" \
    && CCACHE_SLOPPINESS=pch_defines,time_macros \
        cmake --build /tmp/pinocchio/build --parallel "$(build_jobs)" \
    && cmake --install /tmp/pinocchio/build \
    && ldconfig \
    && rm -rf /tmp/pinocchio

RUN --mount=type=bind,from=sources,source=/example-robot-data.tar.gz,target=/tmp/source.tar.gz \
    --mount=type=bind,from=sources,source=/example-robot-data-cmake.tar.gz,target=/tmp/cmake.tar.gz \
    --mount=type=cache,id=rt-cosmik-ccache,target=/root/.cache/ccache \
    mkdir -p /tmp/example-robot-data/cmake \
    && tar -xzf /tmp/source.tar.gz -C /tmp/example-robot-data --strip-components=1 \
    && tar -xzf /tmp/cmake.tar.gz -C /tmp/example-robot-data/cmake --strip-components=1 \
    && cmake -S /tmp/example-robot-data -B /tmp/example-robot-data/build ${CMAKE_ARGS} \
        -DBUILD_TESTING=OFF -DPython_EXECUTABLE="${VIRTUAL_ENV}/bin/python3" \
    && cmake --build /tmp/example-robot-data/build --parallel "$(build_jobs)" \
    && cmake --install /tmp/example-robot-data/build \
    && ldconfig \
    && rm -rf /tmp/example-robot-data

# acados headers, libraries, templates and sources remain available for solver
# generation. The other QP backends are disabled, so only HPIPM/BLASFEO are needed.
ENV ACADOS_SOURCE_DIR=/root/workspace/deps/acados
RUN --mount=type=bind,from=sources,source=/acados.tar.gz,target=/tmp/source.tar.gz \
    --mount=type=bind,from=sources,source=/acados-blasfeo.tar.gz,target=/tmp/blasfeo.tar.gz \
    --mount=type=bind,from=sources,source=/acados-hpipm.tar.gz,target=/tmp/hpipm.tar.gz \
    --mount=type=cache,id=rt-cosmik-ccache,target=/root/.cache/ccache \
    mkdir -p "${ACADOS_SOURCE_DIR}" \
    && tar -xzf /tmp/source.tar.gz -C "${ACADOS_SOURCE_DIR}" --strip-components=1 \
    && mkdir -p "${ACADOS_SOURCE_DIR}/external/blasfeo" "${ACADOS_SOURCE_DIR}/external/hpipm" \
    && tar -xzf /tmp/blasfeo.tar.gz -C "${ACADOS_SOURCE_DIR}/external/blasfeo" --strip-components=1 \
    && tar -xzf /tmp/hpipm.tar.gz -C "${ACADOS_SOURCE_DIR}/external/hpipm" --strip-components=1 \
    && cmake -S "${ACADOS_SOURCE_DIR}" -B /tmp/acados-build \
        -DCMAKE_BUILD_TYPE=Release -DACADOS_INSTALL_DIR="${ACADOS_SOURCE_DIR}" \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DACADOS_WITH_OPENMP=OFF -DBLASFEO_TARGET=X64_AUTOMATIC \
        -DACADOS_WITH_QPOASES=OFF -DACADOS_WITH_OSQP=OFF \
    && cmake --build /tmp/acados-build --parallel "$(build_jobs)" \
    && cmake --install /tmp/acados-build \
    && printf '%s\n' "${ACADOS_SOURCE_DIR}/lib" > /etc/ld.so.conf.d/acados.conf \
    && ldconfig \
    && rm -rf /tmp/acados-build

RUN --mount=type=bind,from=sources,source=/t_renderer,target=/tmp/t_renderer \
    install -Dm755 /tmp/t_renderer "${ACADOS_SOURCE_DIR}/bin/t_renderer"

FROM native AS development

COPY --from=uv /uv /usr/local/bin/uv
# Keep the large GPU wheels ahead of application requirements, so changing a
# Python utility does not reinstall or re-export PyTorch and TensorRT.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python "${VIRTUAL_ENV}/bin/python" --link-mode=copy \
        -c /opt/rt-cosmik/constraints.txt \
        --index-url https://download.pytorch.org/whl/cu121 \
        "torch==2.4.1+cu121" "torchvision==0.19.1+cu121" "torchaudio==2.4.1+cu121"

# Engines are generated on the target GPU. Keep the full Python TensorRT API
# for export and inference; an unrelated APT libnvinfer-bin is unnecessary.
ARG TENSORRT_VERSION=10.15.1.29
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python "${VIRTUAL_ENV}/bin/python" --link-mode=copy \
        -c /opt/rt-cosmik/constraints.txt "tensorrt-cu12==${TENSORRT_VERSION}"

COPY requirements.txt /opt/rt-cosmik/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python "${VIRTUAL_ENV}/bin/python" --link-mode=copy \
        -c /opt/rt-cosmik/constraints.txt -r /opt/rt-cosmik/requirements.txt

# --no-deps preserves the CasADi compiled above. uv's isolated packaging build
# supports editable installs without upgrading ROS's system setuptools.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python "${VIRTUAL_ENV}/bin/python" --link-mode=copy \
        --no-deps -e "${ACADOS_SOURCE_DIR}/interfaces/acados_template" \
    && python -m pip check

# Native libraries use ldconfig; compilers and pkg-config already search
# /usr/local. Preserve the CUDA image's own library environment unchanged.
# Both the venv and ROS's system interpreter must load the compiled bindings.
RUN python -c "import numpy, scipy.signal, casadi, eigenpy, hppfcl, pinocchio.casadi, acados_template, tensorrt, torch, torchaudio, onnxruntime; assert numpy.__version__ == '1.26.4'" \
    && /usr/bin/python3 -c "import numpy, casadi, pinocchio.casadi, acados_template; assert numpy.__version__ == '1.26.4'"

RUN --mount=type=bind,from=sources,target=/tmp/sources \
    mkdir -p /opt/manifest \
    && python -m pip freeze > /opt/manifest/pip_freeze.txt \
    && dpkg-query -W > /opt/manifest/dpkg.txt \
    && apt-mark showmanual > /opt/manifest/apt_manual.txt \
    && sha256sum /tmp/sources/* > /opt/manifest/sources.sha256

# The checkout, models, generated solvers and results live on the host.
RUN mkdir -p /root/workspace/ros_ws \
    && echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc \
    && git config --global --add safe.directory /root/workspace/rt-cosmik \
    && git config --global --add safe.directory /root/workspace/cams_calibration \
    && git config --global --add safe.directory /root/workspace/ros_ws/src/rtcosmik_ros
WORKDIR /root/workspace/rt-cosmik
CMD ["bash"]
