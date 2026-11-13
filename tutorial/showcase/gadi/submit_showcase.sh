#!/bin/bash
# On a Gadi login node:  bash submit_showcase.sh smoke     -> all compute scripts with --quick (one job)
#                        bash submit_showcase.sh full      -> the four full jobs, submitted together
# (two stages so the quick GIFs can be looked at before GPU hours are spent on the full runs). GPU cap via qsub_gpu.sh (MAX_GPU=5: one dg97 GPU stays free).
set -euo pipefail
export PATH=$PATH:/opt/pbs/default/bin
ROOT=/scratch/y89/mj8805/showcase; cd $ROOT/tutorial/showcase/gadi
Q="env MAX_GPU=5 /scratch/y89/mj8805/spice_docsexp/run/qsub_gpu.sh"
GLOG=/scratch/y89/mj8805/exactfit/logs/jobs.txt
note() { echo "$(date '+%F %T') [session showcase] $*" | tee -a $ROOT/logs/jobids.txt >> $GLOG; }
ALL="s01_rotation s02_doppler_imaging s03_eclipse s04_cepheid s05_gradients s06_fit_loop s08_star_builder_grid s09_photometry_grid s10_rossiter_mclaughlin s11_solar_oscillations"
STAGE=${1:?usage: submit_showcase.sh smoke|full}
if [ "$STAGE" = smoke ]; then
smoke=$($Q -N sc_smoke -o $ROOT/logs -l walltime=01:30:00 -v "QUICK=1,SCRIPTS=$ALL" showcase.pbs)
note "qsub $smoke sc_smoke (all scripts --quick)"
exit 0
fi
sub() {  # name walltime scripts [extra qsub args]
  local id; id=$($Q -N $1 -o $ROOT/logs -l walltime=$2 ${4:-} -v "SCRIPTS=$3" showcase.pbs)
  note "qsub $id $1 ${4:-} SCRIPTS=$3"; echo $id
}
sub sc_single 04:00:00 "s01_rotation s02_doppler_imaging s04_cepheid s11_solar_oscillations" >/dev/null
sub sc_binary 04:00:00 "s03_eclipse s10_rossiter_mclaughlin" >/dev/null
sub sc_grids  04:00:00 "s08_star_builder_grid s09_photometry_grid" >/dev/null
# the gradient scripts failed the first smoke test (traced radius in Binary.from_bodies): re-test them quickly first
gq=$($Q -N sc_gradq -o $ROOT/logs -l walltime=00:45:00 -v "QUICK=1,SCRIPTS=s05_gradients s06_fit_loop" showcase.pbs)
note "qsub $gq sc_gradq (s05/s06 --quick re-test)"
sub sc_grad 06:00:00 "s05_gradients s06_fit_loop" "-W depend=afterok:$gq" >/dev/null
