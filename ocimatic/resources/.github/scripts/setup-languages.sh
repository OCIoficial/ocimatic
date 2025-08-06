#!/bin/bash
# Install compilers and runtimes

set -e

apt-get update
apt-get install -y build-essential g++
apt-get install -y openjdk-21-jdk
apt-get install -y rustc
