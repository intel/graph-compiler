FROM python:3.9-bullseye as python_builder

WORKDIR /app

RUN set -ex; \
    git clone https://github.com/dirkcgrunwald/jupyter_codeserver_proxy-.git; \
    cd jupyter_codeserver_proxy-; \
    make build-package

FROM ubuntu:22.04 as conda_builder

RUN set -ex; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get update -y; \
    apt-get install -y curl xz-utils pixz

ENV USER=jovyan
ENV HOME=/home/${USER}
ENV SHELL=/bin/bash

ARG CONDA_PREFIX=${HOME}/.conda
ARG CONDA=${CONDA_PREFIX}/bin/conda

RUN adduser \
        --disabled-password \
        --gecos "Default user" \
        --uid 1000 \
        --home ${HOME} \
        --force-badname \
        ${USER}

USER ${USER}

WORKDIR ${HOME}

# install miniforge to CONDA_PREFIX
RUN set -ex; \
    curl -sSL -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh; \
    /bin/bash /tmp/miniforge.sh -b -p ${CONDA_PREFIX}; \
    rm -f /tmp/miniforge.sh

RUN echo "conda prefix after initial installation: $(du --human-readable --summarize ${CONDA_PREFIX})" > /dev/stderr

# install mamba solver
RUN ${CONDA} install -n base conda-libmamba-solver

RUN echo "conda prefix after installing conda-libmamba-solver: $(du --human-readable --summarize ${CONDA_PREFIX})" > /dev/stderr

# set libmamba as the default resolver
# https://conda.github.io/conda-libmamba-solver/getting-started/
RUN ${CONDA} config --set solver libmamba

# update to conda-forge
RUN ${CONDA} update -n base -c conda-forge conda

RUN echo "conda prefix after updating to conda-forge: $(du --human-readable --summarize ${CONDA_PREFIX})" > /dev/stderr

COPY --chown=${USER}:${USER} requirements.txt /tmp/requirements.txt
COPY --from=python_builder --chown=${USER}:${USER} /app/jupyter_codeserver_proxy-/dist/*.whl /opt/jupyter_codeserver_proxy/dist/

# install nb_conda_kernels for discovering kernels (https://towardsdatascience.com/get-your-conda-environment-to-show-in-jupyter-notebooks-the-easy-way-17010b76e874)
# install all dependencies
# disable JupyterLab announcement pop-up (https://jupyterlab.readthedocs.io/en/stable/user/announcements.html)
RUN set -e; \
    . ${CONDA_PREFIX}/etc/profile.d/conda.sh; \
    conda activate base; \
    conda install nb_conda_kernels nodejs; \
    pip install --no-cache-dir -r /tmp/requirements.txt /opt/jupyter_codeserver_proxy/dist/*.whl; \
    jupyter labextension disable "@jupyterlab/apputils-extension:announcements"

RUN echo "conda prefix after installing jupyterhub: $(du --human-readable --summarize ${CONDA_PREFIX})" > /dev/stderr

# create new conda environment python-3.9
# install infractl, lit
RUN set -e; \
    . ${CONDA_PREFIX}/etc/profile.d/conda.sh; \
    conda create --name python-3.9 python=3.9 ipykernel s3cmd dvc; \
    conda activate python-3.9; \
    pip --no-cache-dir install infractl lit

RUN echo "conda prefix after installing infractl: $(du --human-readable --summarize ${CONDA_PREFIX})" > /dev/stderr

RUN ${CONDA} clean --all --force-pkgs-dirs --yes

RUN echo "conda prefix after cleaning: $(du --human-readable --summarize ${CONDA_PREFIX})" > /dev/stderr

# archive conda prefix
RUN tar c -I"pixz -9" -f conda.tar.xz .conda

RUN echo "conda archive: $(du --human-readable conda.tar*)" > /dev/stderr

FROM ubuntu:22.04

ENV USER=jovyan
ENV HOME=/home/${USER}
ENV SHELL=/bin/bash

RUN set -ex; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get update -y; \
    apt-get install -y software-properties-common curl; \
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg > /usr/share/keyrings/githubcli-archive-keyring.gpg; \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list; \
    apt-get update -y; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gh jq \
        git \
        bash-completion \
        cmake gcc g++ ninja-build git clang-format \
        sudo \
        pixz \
        ; \
    rm -rf /var/lib/apt/lists/*

RUN set -ex; \
    adduser \
        --disabled-password \
        --gecos "Default user" \
        --uid 1000 \
        --home ${HOME} \
        --force-badname \
        ${USER}; \
    usermod -aG sudo ${USER}; \
    echo "%sudo   ALL=(ALL:ALL) NOPASSWD:ALL" > /etc/sudoers

COPY --from=conda_builder ${HOME}/conda.tar.xz /template/

COPY entrypoint.sh /template/

USER ${USER}

WORKDIR ${HOME}

RUN curl -fsSL https://code-server.dev/install.sh | sh && rm -rf "${HOME}/.cache"

EXPOSE 8888

ENTRYPOINT ["/bin/bash", "/template/entrypoint.sh"]

