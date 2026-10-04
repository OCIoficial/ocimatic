#!/usr/bin/env bash
# Release the ocimatic-testplan VS Code extension.
#
# Usage: scripts/release.sh [--dry-run]
#
# Releases the version currently in package.json, so bump it and add a matching
# `## [x.y.z]` entry to CHANGELOG.md before running this.
#
#   --dry-run  lint and package the .vsix, but don't publish.
#
# Publishing uses the Marketplace token stored with `npx vsce login nlehmann`.
set -euo pipefail

cd "$(dirname "$0")/.."

dry_run=false
case "${1-}" in
  --dry-run) dry_run=true ;;
  "") ;;
  *) echo "usage: $0 [--dry-run]" >&2; exit 2 ;;
esac

die() { echo "error: $*" >&2; exit 1; }
step() { echo; echo "==> $*"; }

version=$(node -p 'require("./package.json").version')
vsix="dist/ocimatic-testplan-$version.vsix"

grep -q "^## \[$version\]" CHANGELOG.md || die "CHANGELOG.md has no '## [$version]' entry"

step "Installing dependencies"
npm ci

step "Linting"
npm run lint

step "Packaging $vsix"
mkdir -p dist
npx vsce package --out "$vsix"

if $dry_run; then
  echo
  echo "Dry run done. Package is at $PWD/$vsix"
  exit 0
fi

echo
read -rp "Publish ocimatic-testplan $version to the VS Code Marketplace? [y/N] " answer
[ "$answer" = y ] || [ "$answer" = Y ] || die "aborted"

step "Publishing to VS Code Marketplace"
npx vsce publish --packagePath "$vsix"

echo
echo "Released ocimatic-testplan $version"
