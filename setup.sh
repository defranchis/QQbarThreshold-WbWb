# Prepend the QQbar_threshold install/{lib,include} dirs to the
# linker / loader / compiler search paths. ``${VAR:-}`` keeps the
# script safe under ``set -u`` when the var is unset.
export QQBAR_INSTALL=/afs/cern.ch/user/m/mdefranc/work/private/FCC/QQbar_threshold/install
export LIBRARY_PATH="$QQBAR_INSTALL/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$QQBAR_INSTALL/lib:${LD_LIBRARY_PATH:-}"
export CPLUS_INCLUDE_PATH="$QQBAR_INSTALL/include:${CPLUS_INCLUDE_PATH:-}"
