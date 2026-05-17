#!/bin/bash
# Local validation for tz_fornacis_*.pbs (syntax + CLI flag parity).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

for pbs in "${DIR}/tz_fornacis_lightcurves.pbs" "${DIR}/tz_fornacis_spectra.pbs"; do
    echo "=== bash -n ${pbs} ==="
    bash -n "${pbs}"
done

echo "=== lightcurves CLI ==="
"${PYTHON_BIN}" "${DIR}/tz_fornacis_lightcurves.py" \
    --mode both --num-times 1 --num-eclipse-times 2 --num-wavelengths 10 \
    --wl-min 4800 --wl-max 6800 --orbit-chunk 2 --output-dir /tmp/tz_val \
    --model RozanskiT/TPayne-spice-harps --help >/dev/null
echo "OK"

echo "=== spectra CLI ==="
"${PYTHON_BIN}" "${DIR}/tz_fornacis_spectra.py" \
    --num-times 1 --num-eclipse-times 2 --num-wavelengths 10 \
    --wl-min 4800 --wl-max 6800 --orbit-chunk 2 --output-dir /tmp/tz_val \
    --model RozanskiT/TPayne-spice-harps --help >/dev/null
echo "OK"

echo "All PBS checks passed."
