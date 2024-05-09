FROM ubuntu:latest

RUN set -ex; \
    export DEBIAN_FRONTEND=noninteractive; \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list;\
    curl -sSL https://cli.github.com/packages/githubcli-archive-keyring.gpg > /usr/share/keyrings/githubcli-archive-keyring.gpg; \
    apt-get update -y; \
    apt-get install -y --no-install-recommends --fix-missing \
      bash-completion \
      python3-pip \
      cmake gcc g++ ninja-build git clang-format \
      gh \
    ; \
    rm -rf /var/lib/apt/lists/*

RUN pip install lit

SHELL ["/bin/bash", "-xec"]
