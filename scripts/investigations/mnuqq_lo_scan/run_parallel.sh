#!/usr/bin/env bash
# Parallel LO mnuqq sqrts scan: one independent WHIZARD job per sqrts point,
# all running concurrently.  Each job is self-contained in its own dir (own
# compile + grids + log) so there are no file collisions.  1 core/point;
# concurrency comes from running the points in parallel.
#
# NOTE: no `set -u` -- the cvmfs LCG view setup.sh is not unbound-var clean.

WZ=/afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold/whizard
PAR="$WZ/work/mnuqq_par"

# Source the environment FIRST (before any strictness).
source "$WZ/setup.sh" >/dev/null 2>&1
export OMP_NUM_THREADS=1

# Dense grid: 1 GeV steps through the WW threshold (157-170), 2.5 GeV above.
# 163 and 240 are exact grid points (ratio request).
POINTS=(157 158 159 160 161 162 163 164 165 166 167 168 169 170
        172.5 175 177.5 180 182.5 185 187.5 190 192.5 195 197.5 200
        202.5 205 207.5 210 212.5 215 217.5 220 222.5 225 227.5 230
        232.5 235 237.5 240 242.5 245 247.5 250)

rm -rf "$PAR"
mkdir -p "$PAR"
echo "launching ${#POINTS[@]} points into $PAR"

# Write all job cards first (cheap, serial), then launch all in parallel.
for E in "${POINTS[@]}"; do
    d="$PAR/p_$E"
    mkdir -p "$d"
    cat > "$d/job.sin" <<EOF
model = "SM"
mZ = 91.188
mW = 80.377
wW = 2.04483
mtop = 174.2
mH = 115
GF = 1.16637E-5
?fatal_beam_decay = false
process mnuqq = e1, E1 => e2, N2, u, D
compile
beams = e1, E1
sqrts = $E GeV
integrate (mnuqq) { iterations = 5:60000:"gw", 3:150000 }
show (integral(mnuqq), error(mnuqq))
EOF
done

for E in "${POINTS[@]}"; do
    d="$PAR/p_$E"
    ( cd "$d" && whizard job.sin > run.log 2>&1; echo "$?" > exit.code ) &
done
wait

echo "ALL DONE"
ok=0; for E in "${POINTS[@]}"; do [ "$(cat "$PAR/p_$E/exit.code" 2>/dev/null)" = 0 ] && ok=$((ok+1)); done
echo "points with exit 0: $ok / ${#POINTS[@]}"
