#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../data/sift1m" && pwd)"
mkdir -p "${DEST_DIR}"
TAR_PATH="${DEST_DIR}/sift.tar.gz"

if [ ! -f "${DEST_DIR}/sift_base.fvecs" ]; then
    if [ ! -f "${TAR_PATH}" ]; then
        echo "Downloading SIFT1M corpus from TEXMEX..."
        curl -L --progress-bar -o "${TAR_PATH}" ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
    fi
    echo "Extracting SIFT1M archive..."
    tar -xzf "${TAR_PATH}" -C "${DEST_DIR}"
    # The tarball unpacks into a "sift" folder; move files up if needed
    if [ -d "${DEST_DIR}/sift" ]; then
        mv "${DEST_DIR}/sift"/* "${DEST_DIR}/"
        rmdir "${DEST_DIR}/sift"
    fi
    echo "SIFT1M dataset successfully staged at ${DEST_DIR}"
else
    echo "SIFT1M dataset already staged at ${DEST_DIR}"
fi
