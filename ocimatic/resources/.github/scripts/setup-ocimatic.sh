#!/bin/bash
# setup-ocimatic.sh: Setup script for ocimatic on Ubuntu and Alpine Linux
# Usage: ./setup-ocimatic.sh [--with-langs]

set -e

WITH_LANGS=0
for arg in "$@"; do
    if [[ "$arg" == "--with-langs" ]]; then
        WITH_LANGS=1
    fi
done

# Detect distro
if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO=$ID
else
    echo "Cannot detect Linux distribution. Exiting."
    exit 1
fi

export PATH="$HOME/.local/bin:$PATH"

case "$DISTRO" in
    ubuntu|debian)
        echo "Detected $DISTRO. Installing dependencies..."
        apt-get update
        if [ $WITH_LANGS -eq 1 ]; then
            echo "Installing language dependencies..."
            apt-get install -y build-essential g++
            apt-get install -y openjdk-21-jdk
            apt-get install -y rustc
        fi
        ;;
    *)
        echo "Unsupported distribution: $DISTRO"
        exit 1
        ;;
esac

curl -LsSf https://astral.sh/uv/install.sh | sh

uv tool install git+https://github.com/OCIoficial/ocimatic
