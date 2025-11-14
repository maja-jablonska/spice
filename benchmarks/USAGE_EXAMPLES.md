# Benchmark Usage Examples

Quick reference for running the GaussianLineEmulator benchmark with different configurations.

## Quick Start

```bash
cd benchmarks
JAX_PLATFORMS=cpu python benchmark_gaussian_emulator.py
```

## Common Use Cases

### 1. Quick Test (Fast)
Test with small meshes and few runs for rapid iteration:
```bash
python benchmark_gaussian_emulator.py \
    --mesh-resolutions 100 300 1000 \
    --n-runs 2 \
    --n-wavelengths 500
```

### 2. Standard Benchmark (Default)
Full benchmark with default settings:
```bash
python benchmark_gaussian_emulator.py
```

### 3. High Precision (Publication Quality)
More runs for better statistics:
```bash
python benchmark_gaussian_emulator.py \
    --mesh-resolutions 300 1000 5000 20000 \
    --n-runs 10 \
    --n-wavelengths 2000 \
    --output-prefix publication
```

### 4. Test Different Wavelength Counts
Compare performance with different wavelength resolutions:
```bash
# Test 1: Low resolution
python benchmark_gaussian_emulator.py \
    --n-wavelengths 500 \
    --output-prefix low_wl

# Test 2: Medium resolution
python benchmark_gaussian_emulator.py \
    --n-wavelengths 2000 \
    --output-prefix med_wl

# Test 3: High resolution
python benchmark_gaussian_emulator.py \
    --n-wavelengths 5000 \
    --output-prefix high_wl
```

### 5. Compare Emulators
Benchmark both GaussianLineEmulator and TransformerPayne:
```bash
# GaussianLineEmulator
python benchmark_gaussian_emulator.py \
    --mesh-resolutions 300 1000 5000 \
    --n-wavelengths 1000 \
    --output-prefix gaussian_1000wl

# TransformerPayne
python benchmark_gaussian_emulator.py \
    --use-transformer-payne \
    --mesh-resolutions 300 1000 5000 \
    --n-wavelengths 1000 \
    --output-prefix tpayne_1000wl
```

### 6. Single Mesh Resolution (Deep Dive)
Focus on one resolution with many runs:
```bash
python benchmark_gaussian_emulator.py \
    --mesh-resolutions 1000 \
    --n-runs 20 \
    --output-prefix mesh_1000_detailed
```

### 7. Large Mesh Test (if you have time/memory)
Test very high resolution meshes:
```bash
python benchmark_gaussian_emulator.py \
    --mesh-resolutions 10000 20000 50000 \
    --n-runs 3 \
    --output-prefix large_mesh
```

## All Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--use-transformer-payne` | Use TransformerPayne instead of GaussianLineEmulator | False |
| `--n-wavelengths N` | Number of wavelength points | 2000 |
| `--mesh-resolutions N1 N2 ...` | Mesh resolutions to test | [300, 1000, 5000, 20000] |
| `--n-runs N` | Number of timing runs per configuration | 5 |
| `--output-prefix PREFIX` | Prefix for output files | Auto-generated |

## Output Files

The benchmark creates:
- `<prefix>_benchmark_results.csv` - Raw timing data
- `<prefix>_performance.png` - Performance plots

## Tips

1. **Start small**: Use `--mesh-resolutions 100 300` and `--n-runs 2` for quick testing
2. **Use CPU**: Prefix with `JAX_PLATFORMS=cpu` to avoid GPU issues
3. **Monitor memory**: Large meshes (>20000) may require significant RAM
4. **Consistent conditions**: Close other applications for more consistent timing
5. **Multiple runs**: Use `--n-runs 10` or more for publication-quality statistics

## Interpreting Results

- **Mesh Creation + Spectrum**: Total time including mesh construction
- **Spectrum Only**: Pure spectrum computation (after JIT and mesh creation)
- **Rotation + Spectrum**: Additional cost of rotation transformation
- **Pulsation + Spectrum**: Most expensive due to multiple time evaluations

The CSV file contains all raw data for further analysis in Python, R, or Excel.

