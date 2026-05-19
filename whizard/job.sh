#!/bin/bash
# HTCondor wrapper for a single WHIZARD integration job.
#
# Args:
#   $1  LABEL        identifier (used to name the output directory)
#   $2  MW           W mass in GeV
#   $3  GW           W width in GeV
#   $4  SQRTS_CSV    comma-separated sqrt(s) values in GeV, e.g. "155.0,158.0,161.0"
#   $5  MODE         iteration preset: "bfs" (<0.1%) or "grid" (~0.05%)
#   $6  OUTPUT_DIR   absolute AFS path where results.csv + whizard.log go
#
# Channel: e+ e- -> mu- nu_mu_bar u d_bar (BFS Tables 1+2 reference channel).
# The job runs in $_CONDOR_SCRATCH_DIR on the worker, then copies results back.

# Note: LCG view's setup.sh references unbound vars (COMPILER, ...), so we
# can't use `set -u`. The remaining `-e` + `-o pipefail` still catch real failures.
set -eo pipefail

LABEL="$1"
MW="$2"
GW="$3"
SQRTS_CSV="$4"
MODE="$5"
OUTPUT_DIR="$6"

case "$MODE" in
    bfs)       ITER_SPEC='8:200000:"gw",5:1000000'  ;;  # target <0.1% MC stat
    grid)      ITER_SPEC='6:100000:"gw",3:300000'   ;;  # target ~0.05% MC stat
    highstats) ITER_SPEC='6:500000:"gw",5:5000000'  ;;  # target ~0.02% (~20x events); needs workday queue
    *)         echo "Unknown MODE: $MODE (expected bfs|grid|highstats)" >&2; exit 2 ;;
esac

# OpenMP threads — match the condor request_cpus = 4 in the submit file.
# WHIZARD OpenMP scales near-linearly to ~4-8 cores; past that it sub-linears.
export OMP_NUM_THREADS=4

WHIZARD_TOP=/afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold/whizard
# shellcheck disable=SC1091
source "$WHIZARD_TOP/setup.sh"

# Renew Kerberos token (best-effort; if it fails, the AFS copy at the end
# will fail loudly and condor will hold the job).
kinit -R 2>/dev/null || true

# Use a per-job temp dir. Prefer condor's scratch when set; otherwise mktemp
# under /tmp (avoids dumping WHIZARD's compile artifacts into the submit dir
# on pools that don't populate $_CONDOR_SCRATCH_DIR).
WORK="${_CONDOR_SCRATCH_DIR:-$(mktemp -d -t whizard_job.XXXXXX)}"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK" || exit 1

# Build the Sindarin file. WHIZARD's scan-over-sqrts reuses the VAMP phase-
# space grid between adjacent points, which is why we batch sqrt(s) per job.
SQRTS_LIST=$(echo "$SQRTS_CSV" | sed 's/,/ GeV, /g')
cat > job.sin <<EOF
model = "SM"
mZ = 91.188
mW = $MW
wW = $GW
mtop = 174.2
mH = 115
GF = 1.16637E-5

?fatal_beam_decay = false

process p_$LABEL = e1, E1 => e2, N2, u, D
compile

iterations = $ITER_SPEC

scan sqrts = ($SQRTS_LIST GeV) {
    integrate (p_$LABEL)
    printf "RESULT %g %g %g %g %g" (sqrts/1 GeV, integral(p_$LABEL), error(p_$LABEL), $MW, $GW)
}
EOF

# Run. Capture the exit code without letting set -e abort the rest of the
# wrapper — we still want to stage whatever we have (whizard.log etc.) so
# failures are debuggable from the submitter side.
RC=0
whizard job.sin > whizard.log 2>&1 || RC=$?

# Distill results — one RESULT line per scan point.
# Format: RESULT <sqrts/GeV> <sigma/fb> <err/fb> <mW> <gammaW>
# WHIZARD's printf output is *not* prefixed with the usual `| ` decoration.
# WHIZARD also prints the trailing scan-point's RESULT twice (once inside the
# scan loop, once in the run-finished tail), so we dedupe on the fly.
{
    echo "# sqrts_GeV sigma_fb err_fb mW_GeV gammaW_GeV"
    grep '^RESULT ' whizard.log 2>/dev/null | sed 's/^RESULT //' | awk '!seen[$0]++' || true
} > results.csv

# Stage back to AFS — unconditionally, so a failed whizard still leaves
# whizard.log behind for inspection.
mkdir -p "$OUTPUT_DIR"
cp job.sin whizard.log results.csv "$OUTPUT_DIR/" || true

exit "$RC"
