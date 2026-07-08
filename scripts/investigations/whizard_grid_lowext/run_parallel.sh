#!/usr/bin/env bash
# Sub-154 GeV extension of the WHIZARD anchor grid (2026-07-02 morph-edge fix
# follow-up): extend grid_fine/wings below the 154 GeV knot edge so the morph
# anchor rests on real 4f Born data instead of the frozen edge calibration.
#
# Layout mirrors mnuqq_lo_scan/run_parallel.sh: one self-contained WHIZARD dir
# per (sqrts, mW, gW) point, 1 core each, all concurrent up to a slot cap.
# Lattice = the morph's uniform-Gamma_W sublattice of grid_fine (9 mW x 5 gW),
# sqrts = 150.0 .. 153.5 step 0.5 (8 slices, 360 points).  Model params match
# grid_fine/provenance.json exactly (mZ 91.188, mtop 174.2, mH 115, GF PDG).
# Integration stats are lighter than grid_fine (~few 1e-4 rel): below the WW
# threshold the anchor enters only through the radiator tail, so sub-1e-3 MC
# is ample (the artifact being replaced was a -1.6% effect).
#
# NOTE: no `set -u` -- the cvmfs LCG view setup.sh is not unbound-var clean.

WZ=/afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold/whizard
PAR="$WZ/work/grid_lowext"
SLOTS=${SLOTS:-44}

source "$WZ/setup.sh" >/dev/null 2>&1
export OMP_NUM_THREADS=1

SQRTS=(150.0 150.5 151.0 151.5 152.0 152.5 153.0 153.5)
MWS=(80.279 80.304 80.329 80.354 80.379 80.404 80.429 80.454 80.479)
GWS=(2.045 2.065 2.085 2.105 2.125)

rm -rf "$PAR"
mkdir -p "$PAR"
njob=0
for E in "${SQRTS[@]}"; do for MW in "${MWS[@]}"; do for GW in "${GWS[@]}"; do
    d="$PAR/p_${E}_${MW}_${GW}"
    mkdir -p "$d"
    cat > "$d/job.sin" <<EOF
model = "SM"
mZ = 91.188
mW = $MW
wW = $GW
mtop = 174.2
mH = 115
GF = 1.16637E-5
?fatal_beam_decay = false
process mnuqq = e1, E1 => e2, N2, u, D
compile
beams = e1, E1
sqrts = $E GeV
integrate (mnuqq) { iterations = 5:30000:"gw", 3:60000 }
show (integral(mnuqq), error(mnuqq))
EOF
    njob=$((njob+1))
done; done; done
echo "prepared $njob jobs in $PAR; running with $SLOTS slots"

# Slot-capped parallel launch (bash job control).
running=0
for d in "$PAR"/p_*; do
    ( cd "$d" && whizard job.sin > run.log 2>&1; echo "$?" > exit.code ) &
    running=$((running+1))
    if [ "$running" -ge "$SLOTS" ]; then wait -n; running=$((running-1)); fi
done
wait

ok=0; tot=0
for d in "$PAR"/p_*; do
    tot=$((tot+1))
    [ "$(cat "$d/exit.code" 2>/dev/null)" = 0 ] && ok=$((ok+1))
done
echo "ALL DONE: exit-0 points $ok / $tot"
