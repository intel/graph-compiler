FROM ubuntu:latest

SHELL ["/bin/bash", "-xec"]

RUN set -ex; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get update -y; \
    apt-get install -y --no-install-recommends --fix-missing \
      curl \
      bash-completion \
      python3-pip \
      cmake gcc g++ ninja-build git clang-format; \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list;\
    curl -sSL https://cli.github.com/packages/githubcli-archive-keyring.gpg > /usr/share/keyrings/githubcli-archive-keyring.gpg; \
    apt-get update -y; \
    apt-get install -y --no-install-recommends --fix-missing gh; \
    rm -rf /var/lib/apt/lists/*

RUN pip install lit

RUN useradd -m dev

USER dev
WORKDIR /home/dev
