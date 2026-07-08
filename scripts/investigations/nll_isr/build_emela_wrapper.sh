#!/usr/bin/env bash
# Build the thin C wrapper (emela_c_wrapper.cpp) into libeMELApy.so and
# install it into the shared install prefix alongside libeMELA.so.
#
# Usage: bash build_emela_wrapper.sh
# Requires: eMELA already installed (libeMELA.so + headers in $INSTALL).

set -euo pipefail

INSTALL=/afs/cern.ch/user/m/mdefranc/work/private/FCC/QQbar_threshold/install
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

g++ -std=c++11 -O2 -fPIC -shared \
    -I"${INSTALL}/include" \
    -L"${INSTALL}/lib" -leMELA \
    -Wl,-rpath,"${INSTALL}/lib" \
    "${SCRIPT_DIR}/emela_c_wrapper.cpp" \
    -o "${INSTALL}/lib/libeMELApy.so"

echo "Installed: ${INSTALL}/lib/libeMELApy.so"
