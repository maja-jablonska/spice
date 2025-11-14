#!/bin/bash
# Example workflow for pulsation mode analysis

set -e  # Exit on error

echo "=========================================="
echo "Pulsation Mode Analysis - Example Run"
echo "=========================================="
echo ""

# Configuration
OUTPUT_DIR="pulsations/example_output"
DATA_FILE="${OUTPUT_DIR}/pulsation_data.pkl"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "Step 1: Generating pulsation mode data..."
echo "------------------------------------------"
echo "This will probe modes (2,6), (3,4), and (4,3)"
echo "Using amplitude=1e-3, 20 time steps, and 10000 faces (moderate resolution)"
echo ""

python pulsations/probe_pulsation_modes.py \
    --modes "2,6;3,4;4,3" \
    --amplitude 1e-3 \
    --n-times 20 \
    --n-faces 10000 \
    --period 2.0 \
    --output "${DATA_FILE}"

echo ""
echo "Step 2: Visualizing results..."
echo "------------------------------------------"
echo "Generating all plots and saving to ${OUTPUT_DIR}"
echo ""

python pulsations/visualize_pulsation_modes.py \
    "${DATA_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --plots all

echo ""
echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo "- Data file: ${DATA_FILE}"
echo "- Plots: ${OUTPUT_DIR}/*.png"
echo ""
echo "To view the data interactively:"
echo "  python pulsations/visualize_pulsation_modes.py ${DATA_FILE}"
echo ""

