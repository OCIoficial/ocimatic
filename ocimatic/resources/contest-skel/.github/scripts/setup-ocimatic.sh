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

case "$DISTRO" in
    ubuntu)
        echo "Detected Ubuntu. Installing dependencies..."
        sudo apt-get update
        # TODO: Add Ubuntu-specific dependencies installation here
        if [ $WITH_LANGS -eq 1 ]; then
            echo "Installing language dependencies for Ubuntu..."
            sudo apt-get install -y build-essential g++
            sudo apt-get install -y openjdk-21-jdk
            sudo apt-get install -y rustc
        fi
        ;;
    alpine)
        echo "Detected Alpine. Installing dependencies..."
        apk add --update --no-cache git
        apk add --update --no-cache python3 && ln -sf python3 /usr/bin/python
        if [ $WITH_LANGS -eq 1 ]; then
            echo "Installing language dependencies for Alpine..."
            apk add --update --no-cache build-base
            apk add --update --no-cache openjdk21-jdk
            apk add --update --no-cache rust
        fi
        ;;
    *)
        echo "Unsupported distribution: $DISTRO"
        exit 1
        ;;
esac

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and setuptools inside the virtual environment
pip install --no-cache --upgrade pip setuptools

# Install ocimatic using pip inside the virtual environment
pip install git+https://github.com/OCIoficial/ocimatic

echo "ocimatic setup complete and venv activated."



