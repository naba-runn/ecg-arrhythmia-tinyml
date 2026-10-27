#!/usr/bin/env bash
# Download the MIT-BIH Arrhythmia Database (records, headers, beat annotations)
# from PhysioNet's AWS open-data mirror, 8 files in parallel.
set -euo pipefail

BASE="https://physionet-open.s3.amazonaws.com/mitdb/1.0.0"
OUT="$(cd "$(dirname "$0")/.." && pwd)/data/mitdb"
mkdir -p "$OUT"

curl -sSf --retry 5 -o "$OUT/RECORDS" "$BASE/RECORDS"

while read -r rec; do
  for ext in hea dat atr; do
    echo "$rec.$ext"
  done
done < "$OUT/RECORDS" |
  xargs -P 8 -I{} curl -sSf --retry 5 --retry-delay 2 -o "$OUT/{}" "$BASE/{}"

echo "Downloaded $(ls "$OUT" | grep -c '\.dat$') records to $OUT"
