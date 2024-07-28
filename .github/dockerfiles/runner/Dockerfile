FROM summerwind/actions-runner:ubuntu-22.04

USER root

RUN gpasswd --delete runner sudo

RUN set -ex; \
    export DEBIAN_FRONTEND=noninteractive; \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list; \
    curl -sSL https://cli.github.com/packages/githubcli-archive-keyring.gpg > /usr/share/keyrings/githubcli-archive-keyring.gpg; \
    apt-get update -y; \
    apt-get install -y --no-install-recommends --fix-missing \
      python3-pip python3-dev \
      cmake gcc g++ ninja-build git clang-format \
      gh \
      libomp-dev \
    ; \
    rm -rf /var/lib/apt/lists/*

RUN pip install lit numpy

USER runner
WORKDIR $HOME

SHELL ["/bin/bash", "-xec"]

