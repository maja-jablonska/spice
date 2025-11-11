#!/usr/bin/env julia
# ================================
# synth_korg.jl
# Generate synthetic spectra at multiple resolutions
# Usage: julia synth_korg.jl 5850 5900
# ================================

using Korg
using ProgressMeter
using NPZ

# --- parse command-line arguments ---
if length(ARGS) < 2
    println("Usage: julia synth_korg.jl <start_wavelength> <end_wavelength>")
    exit(1)
end

λ_start = parse(Float64, ARGS[1])
λ_end   = parse(Float64, ARGS[2])

println("Generating spectra from $λ_start Å to $λ_end Å")

# --- Base parameters ---
Teff = 5000.0
logg = 4.32
M_H  = -1.1
C    = -0.5
linelist = Korg.get_GALAH_DR3_linelist()
resolutions = [2000, 10_000, 30_000, 100_000]

# --- Compute base (no-resolution) spectrum ---
wls, flux_no_R, _ = Korg.synth(
    Teff = Teff,
    logg = logg,
    M_H = M_H,
    C = C,
    linelist = linelist,
    wavelengths = (λ_start, λ_end)
)

# --- Compute for different resolutions ---
korg_wavelengths = Dict{Int, Vector{Float64}}()
korg_flux = Dict{Int, Vector{Float64}}()

@showprogress 1 "Synthesizing spectra..." for R in resolutions
    _wls, _flux, _ = Korg.synth(
        Teff = Teff,
        logg = logg,
        M_H = M_H,
        C = C,
        linelist = linelist,
        wavelengths = (λ_start, λ_end),
        R = R
    )
    korg_wavelengths[R] = _wls
    korg_flux[R] = _flux
end

# --- Include base spectrum ---
korg_flux[0] = flux_no_R
korg_wavelengths[0] = wls

# --- Convert to Python-compatible dicts ---
flux_dict = Dict("$(R)" => korg_flux[R] for R in sort(collect(keys(korg_flux))))
wls_dict  = Dict("$(R)" => korg_wavelengths[R] for R in sort(collect(keys(korg_wavelengths))))

# --- Save to .npz files ---
npzwrite("korg_flux_$(λ_start)_$(λ_end).npz", flux_dict)
npzwrite("korg_wavelengths_$(λ_start)_$(λ_end).npz", wls_dict)

println("Saved output files:")
println("  korg_flux_$(λ_start)_$(λ_end).npz")
println("  korg_wavelengths_$(λ_start)_$(λ_end).npz")
