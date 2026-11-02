# shared settings for the interpolated-atmosphere literature models (sourced by the PBS scripts)
export PYTHONPATH=/scratch/y89/mj8805/spice_v2/src
export JAX_ENABLE_X64=1 JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 JAX_PLATFORMS=cuda,cpu
PY=/g/data/y89/mj8805/miniforge/envs/aemu2/bin/python
OUTD=/scratch/y89/mj8805/litrepro/interp/out; mkdir -p $OUTD
V2B=/scratch/y89/mj8805/emu_train/cno_v2b/runs/harps_cno_v2_ram_tpayne_12p_2ch_192x6_500k/bundle_released
PILOT=/scratch/y89/mj8805/emu_train/interp_pilot/runs/pilot_30k_11p_192x6/bundle_released
P1=/scratch/y89/mj8805/emu_train/interp_t02/runs/interp_t02_11p_192x6_500k/bundle_released
P3=/scratch/y89/mj8805/emu_train/interp_t02_v2/runs/interp_t02_11p_192x6_1m25_blue/bundle_released   # phase 3: v2 grid (blue CN/CH), 1.25M steps
SCAN=/scratch/y89/mj8805/litrepro/interp/emulator_jump_scan.py
DC_ARGS="--fits /scratch/y89/mj8805/dcep_v2b/delta_cep.fits --harpsn /scratch/y89/mj8805/dcep_v2b/harpsn_windows_log.npz --bands VT --ext-law g23 --ebv0 0.092 --distance-kpc 0.2795 --fix-dphi --dv-mode season --zp-mode zero-mean --spec-floor 0.02 --no-hessian --feh 0.08 --cno0 -0.13 0.35 0.01 --vrot 0 --vmicro-curve /scratch/y89/mj8805/exactfit/dcep/literature/dcep_luck2018_table6.txt --vmacro-curve /scratch/y89/mj8805/exactfit/dcep/literature/dcep_luck2018_table6.txt --vmacro-order 4 --fix-teff --fix-ebv --fix-vmacro --skip-validation --maxiter 0 --exact-all --logg-curve /scratch/y89/mj8805/exactfit/dcep/literature/dcep_luck2018_table6.txt"
TZ_ARGS="--harps /scratch/y89/mj8805/tzfor_v2b/run/harps_fullrange_log.npz --spec-meshes /scratch/y89/mj8805/tzfor_v2b/run/tzfor_aemu_roche_n1500_spectra_meshes.pkl --lc-meshes /scratch/y89/mj8805/tzfor_v2b/run/tzfor_aemu_roche_n1200_lc6650_meshes.pkl --photometry /scratch/y89/mj8805/tzfor_v2b/run/tzfor_lightcurve.csv --mask-in /scratch/y89/mj8805/tzfor_v2b/out/fr_mask_all_cno.npz --mu-interp cubic --mu-spacing sqrt --shift-interp cubic --coord-interp cubic --n-g 4 --phot-clip 15 --phot-nightly-sigma 3 --skip-validation --maxiter 0 --stall-retries 0 --exact-dump --set Teff1=4930 Teff2=6350 feh=-0.30 dphi=0"
DC_PY=/scratch/y89/mj8805/exactfit/dcep/dcep_joint_fit.py
TZ_PY=/scratch/y89/mj8805/litrepro/tzfor/tzfor_fullrange_fit.py
