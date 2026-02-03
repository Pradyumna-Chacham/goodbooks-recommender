#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="data"
ZIP_PATH="${DATA_DIR}/goodbooks-10k.zip"
URL="https://github.com/zygmuntz/goodbooks-10k/releases/download/v1.0/goodbooks-10k.zip"

mkdir -p "${DATA_DIR}"

echo "Downloading Goodbooks dataset into ./${DATA_DIR}"

# Skip download if files already exist
if [ -f "${DATA_DIR}/books.csv" ] && [ -f "${DATA_DIR}/ratings.csv" ]; then
  echo "Dataset already present. Skipping download."
  exit 0
fi

echo "Fetching dataset from:"
echo "  ${URL}"

curl -L --fail "${URL}" -o "${ZIP_PATH}"

echo "Extracting dataset..."
unzip -oq "${ZIP_PATH}" -d "${DATA_DIR}"

echo "Cleaning up..."
rm -f "${ZIP_PATH}"

echo "Done."
echo "Files in ${DATA_DIR}:"
ls -lh "${DATA_DIR}"
