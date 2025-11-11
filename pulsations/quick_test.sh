#!/bin/bash
# Quick test with minimal settings for rapid verification

set -e

echo "=========================================="
echo "Quick Test - Minimal Settings"
echo "=========================================="
echo ""
echo "This runs with very low resolution for quick testing:"
echo "  - 1 mode: (2,6)"
echo "  - 5000 faces (low resolution)"
echo "  - 10 time steps"
echo ""
echo "Expected runtime: ~1-2 minutes"
echo ""

OUTPUT_DIR="pulsations/test_output"
mkdir -p "${OUTPUT_DIR}"

python pulsations/probe_pulsation_modes.py \
    --modes "2,6" \
    --amplitude 1e-3 \
    --n-times 10 \
    --n-faces 5000 \
    --output "${OUTPUT_DIR}/test_data.pkl"

echo ""
echo "Data generated successfully!"
echo ""
echo "Now generating visualizations..."

python pulsations/visualize_pulsation_modes.py \
    "${OUTPUT_DIR}/test_data.pkl" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
echo "Check the outputs in: ${OUTPUT_DIR}"
echo ""

