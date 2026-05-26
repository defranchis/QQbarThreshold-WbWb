"""Print each term of c_p,LR^(1,ct) finite to compare against bare."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from c1fin import (
    _c_p_LR_bare_finite, _c_p_LR_ct_finite,
    L_MW2_m_MW, L_0_m1_m2, dB0_MW2_MW_m,
)


def main():
    m_W = 80.377
    M_Z = 91.188
    M_H = 115.0
    m_t = 174.2
    M_W2 = m_W * m_W
    M_Z2 = M_Z * M_Z
    M_H2 = M_H * M_H
    mt2 = m_t * m_t

    cw2 = M_W2 / M_Z2
    sw2 = 1.0 - cw2
    sw4 = sw2 * sw2
    cw4 = cw2 * cw2
    cw6 = cw4 * cw2
    cw8 = cw4 * cw4

    L_MW_MH_MW = L_MW2_m_MW(M_W2, M_H2)
    L_0_MH_MW = L_0_m1_m2(M_H2, M_W2)
    L_0_MW_MZ = L_0_m1_m2(M_W2, M_Z2)
    L_MW_MW_MZ = L_MW2_m_MW(M_W2, M_Z2) + np.log(M_Z2 / M_W2)
    dB0_MW_MW_MH = dB0_MW2_MW_m(M_W2, M_H2)
    dB0_MW_MW_MZ = dB0_MW2_MW_m(M_W2, M_Z2)

    print(f"L(M_W², M_H², M_W²)              = {L_MW_MH_MW:+.6e}")
    print(f"L(0, M_H², M_W²)                  = {L_0_MH_MW:+.6e}")
    print(f"L(0, M_W², M_Z²)                  = {L_0_MW_MZ:+.6e}")
    print(f"L(M_W², M_W², M_Z²)               = {L_MW_MW_MZ:+.6e}")
    print(f"∂B_0(M_W², M_W², M_H²)            = {dB0_MW_MW_MH:+.6e}")
    print(f"∂B_0(M_W², M_W², M_Z²)            = {dB0_MW_MW_MZ:+.6e}")
    print()

    terms = []
    add = lambda label, val: terms.append((label, complex(val)))

    add("L_H term",   -(M_H2 * M_H2 - 3.0 * M_W2 * M_H2 + 6.0 * M_W2 * M_W2) * L_MW_MH_MW
        / (12.0 * M_W2 * M_W2 * sw2))
    add("L_0_H term", -(M_H2 - 5.0 * M_W2) * L_0_MH_MW / (12.0 * M_W2 * sw2))
    add("L_0_Z term", -(8.0 * cw4 + 27.0 * cw2 - 5.0) * L_0_MW_MZ / (12.0 * cw2 * sw2))
    add("L_Z term",   (42.0 * cw4 - 11.0 * cw2 - 1.0) * L_MW_MW_MZ / (12.0 * cw4 * sw2))
    add("dB0_H",      -(M_H2 * M_H2 - 4.0 * M_W2 * M_H2 + 12.0 * M_W2 * M_W2) * dB0_MW_MW_MH
        / (24.0 * M_W2 * sw2))
    add("dB0_Z",      (48.0 * cw6 + 68.0 * cw4 - 16.0 * cw2 - 1.0) * M_W2 * dB0_MW_MW_MZ
        / (24.0 * cw4 * sw2))
    add("ln(M_H/M_W)", (2.0 * M_H2 * M_H2 - 3.0 * M_H2 * M_W2 + 2.0 * M_W2 * M_W2)
        * np.log(M_H2 / M_W2) / (24.0 * M_W2 * (M_H2 - M_W2) * sw2))
    add("M_H⁴ const", M_H2 * M_H2 / (12.0 * M_W2 * M_W2 * sw2))
    add("M_H² const", -3.0 * M_H2 / (16.0 * M_W2 * sw2))
    add("m_t log",    -3.0 * mt2 * (mt2 * mt2 - M_W2 * M_W2) * np.log(1.0 - M_W2 / mt2)
        / (4.0 * M_W2 ** 3 * sw2))
    add("m_t² const", -3.0 * mt2 / (8.0 * M_W2 * sw2))
    add("m_t⁴ const", -3.0 * mt2 * mt2 / (4.0 * M_W2 * M_W2 * sw2))
    add("ln(Z/W)",    -(12.0 * cw8 - 72.0 * cw6 + 26.0 * cw4 - 15.0 * cw2 - 2.0)
        * np.log(M_Z2 / M_W2) / (24.0 * cw4 * sw4))
    add("ln 2",       (4.0 * cw4 - 22.0 * cw2 - 1.0) * np.log(2.0) / (4.0 * cw2 * sw2))
    add("rational",   (2.0 * (35.0 - 6.0j * np.pi) * cw6
                       + (-112.0 + 66.0j * np.pi) * cw4
                       + (13.0 + 3.0j * np.pi) * cw2 + 2.0)
                      / (24.0 * cw4 * sw2))

    print(f"{'term':<16}{'Re':>14}{'Im':>14}")
    print(f"{'-'*44}")
    total = 0j
    for label, val in terms:
        print(f"{label:<16}{val.real:+14.4f}{val.imag:+14.4f}")
        total += val
    print(f"{'-'*44}")
    print(f"{'TOTAL ct':<16}{total.real:+14.4f}{total.imag:+14.4f}")
    print()

    bare = _c_p_LR_bare_finite(M_W2, M_Z2, M_H2)
    print(f"{'bare check':<16}{bare.real:+14.4f}{bare.imag:+14.4f}")
    print(f"{'sum':<16}{(bare + total).real:+14.4f}{(bare + total).imag:+14.4f}")
    print(f"{'BFS target':<16}{-10.076:+14.4f}{0.205:+14.4f}")


if __name__ == "__main__":
    main()
