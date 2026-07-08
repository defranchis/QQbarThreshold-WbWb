# Python + scientific stack (numpy/scipy/iminuit/uncertainties/matplotlib)
# from a read-only, versioned cvmfs LCG view -- replaces the flaky AFS
# ~/.local user-site.  Sourced FIRST so the QQBAR_INSTALL prepends below
# stay at the front of LD_LIBRARY_PATH for the eMELA / QQbar dlopen.
# Guarded so re-sourcing in the same shell (e.g. nested pipeline scripts)
# is a no-op, and so a node without cvmfs falls back to the system python.
_WW_LCG_VIEW=/cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
if [ -z "${WW_LCG_SOURCED:-}" ] && [ -r "$_WW_LCG_VIEW" ]; then
    # The LCG view's setup.sh references unset vars bare (e.g. $COMPILER), so it
    # aborts under ``set -u``.  Disable -u only around the source so this script
    # stays safe under ``set -u`` (the guarantee from commit 8af161b); restore
    # the caller's -u state afterwards.
    case "$-" in *u*) _WW_HAD_U=1;; *) _WW_HAD_U=0;; esac
    set +u
    source "$_WW_LCG_VIEW"
    [ "$_WW_HAD_U" = 1 ] && set -u
    unset _WW_HAD_U
    export WW_LCG_SOURCED=1
fi
unset _WW_LCG_VIEW

# Prepend the QQbar_threshold install/{lib,include} dirs to the
# linker / loader / compiler search paths. ``${VAR:-}`` keeps the
# script safe under ``set -u`` when the var is unset.
export QQBAR_INSTALL=/afs/cern.ch/user/m/mdefranc/work/private/FCC/QQbar_threshold/install
export LIBRARY_PATH="$QQBAR_INSTALL/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$QQBAR_INSTALL/lib:${LD_LIBRARY_PATH:-}"
export CPLUS_INCLUDE_PATH="$QQBAR_INSTALL/include:${CPLUS_INCLUDE_PATH:-}"
