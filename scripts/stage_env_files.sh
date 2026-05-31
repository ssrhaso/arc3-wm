#!/usr/bin/env bash
#
# scripts/stage_env_files.sh - bundle local arc_agi environment_files
# into a tarball and (optionally) upload to B2.
#
# Pattern mirrors the Phase-3 replay tarball staging: the laptop owns
# the source of truth, the tarball on B2 is the deploy artifact, and
# every Vast/5070 instance pulls via curl (read-only) rather than
# running `gdown` or any per-instance arc_agi download.
#
# Background - arc_agi.Arcade resolves env_files from the *current
# working directory* (`arc.environments_dir == 'environment_files'`),
# not from inside the wheel. A clean Vast install of arc_agi has no
# cached env_files, so `Arcade.make('vc33')` fails. The Phase-3
# pretrain ran offline against replay JSONLs and never tripped this;
# Phase-4 per-game runs do.
#
# Usage:
#
#   # Bundle the local environment_files dir into a tarball.
#   ./scripts/stage_env_files.sh bundle
#
#   # Upload (Haso only; CLAUDE.md restricts B2 uploads to the user).
#   ./scripts/stage_env_files.sh upload v1
#
#   # On Vast: pull + extract.
#   ./scripts/stage_env_files.sh fetch v1
#
# Versioning convention: `v1` is the pilot-3 bundle (vc33/tu93/cd82,
# ~20 KB). Full-25-game bundles bump the version label.

set -euo pipefail

BUCKET="${B2_BUCKET:-arc-agi-3-replays-hasaan}"
B2_HOST="${B2_HOST:-https://f003.backblazeb2.com}"
KEY_PREFIX="env-files"
TARBALL_NAME="environment_files-pilot.tar.gz"

cmd="${1:-help}"

case "$cmd" in
  bundle)
    # Bundle the local environment_files/ into a gzipped tarball next
    # to the script's working directory. Idempotent - overwrites.
    if [ ! -d environment_files ]; then
      echo "error: ./environment_files/ not found in $(pwd); run from repo root." >&2
      exit 1
    fi
    tar czf "$TARBALL_NAME" environment_files/
    ls -la "$TARBALL_NAME"
    echo
    echo "Contents:"
    tar tzf "$TARBALL_NAME"
    ;;

  upload)
    version="${2:?usage: stage_env_files.sh upload <version>  e.g. v1}"
    if [ ! -f "$TARBALL_NAME" ]; then
      echo "error: $TARBALL_NAME not found; run 'bundle' first." >&2
      exit 1
    fi
    target_key="$KEY_PREFIX/$version/$TARBALL_NAME"
    echo "Uploading $TARBALL_NAME -> b2://$BUCKET/$target_key"
    b2 file upload "$BUCKET" "$TARBALL_NAME" "$target_key"
    echo
    echo "Public URL:"
    echo "  $B2_HOST/file/$BUCKET/$target_key"
    echo
    echo "Verify reachability with:"
    echo "  curl -I '$B2_HOST/file/$BUCKET/$target_key'"
    ;;

  fetch)
    version="${2:?usage: stage_env_files.sh fetch <version>  e.g. v1}"
    target_key="$KEY_PREFIX/$version/$TARBALL_NAME"
    url="$B2_HOST/file/$BUCKET/$target_key"
    echo "Fetching $url ..."
    curl -fSL "$url" | tar xz
    echo
    echo "Extracted:"
    ls -la environment_files/
    ;;

  help|*)
    sed -n '1,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac
