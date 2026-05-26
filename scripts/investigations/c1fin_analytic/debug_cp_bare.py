"""Print each term of c_p,LR^(1,bare) finite separately to locate the discrepancy."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from c1fin import (
    C0_zero_MW2_mMW2_0_m_MW, C0_zero_4MW2_0_0_m_m,
    C0_mMW2_MW2_0_0_m_MW, C0_MW2_mMW2_0_0_0_m,
    C0_4MW2_0_0_0_0_m, L_MW2_m_MW, L_pp2_m_m,
)


def main():
    m_W = 80.377
    M_Z = 91.188
    M_H = 115.0
    M_W2 = m_W * m_W
    M_Z2 = M_Z * M_Z
    M_H2 = M_H * M_H

    cw2 = M_W2 / M_Z2
    sw2 = 1.0 - cw2
    sw4 = sw2 * sw2
    cw4 = cw2 * cw2
    cw6 = cw4 * cw2
    cw8 = cw4 * cw4
    cw10 = cw8 * cw2

    C0_a = C0_zero_MW2_mMW2_0_m_MW(M_W2, M_Z2)
    C0_b = C0_zero_4MW2_0_0_m_m(M_W2, M_Z2)
    C0_c1 = C0_mMW2_MW2_0_0_m_MW(M_W2, 0.0)
    C0_c2 = C0_mMW2_MW2_0_0_m_MW(M_W2, M_H2)
    C0_c3 = C0_mMW2_MW2_0_0_m_MW(M_W2, M_Z2)
    C0_d1 = C0_MW2_mMW2_0_0_0_m(M_W2, M_W2)
    C0_d2 = C0_MW2_mMW2_0_0_0_m(M_W2, M_Z2)
    C0_e1 = C0_4MW2_0_0_0_0_m(M_W2, M_Z2)
    C0_e2 = C0_4MW2_0_0_0_0_m(M_W2, M_W2)

    L_MW_MW_MH = L_MW2_m_MW(M_W2, M_H2) + np.log(M_H2 / M_W2)
    L_MW_MW_MZ = L_MW2_m_MW(M_W2, M_Z2) + np.log(M_Z2 / M_W2)
    L_4MW_MZ_MZ = L_pp2_m_m(4.0 * M_W2, M_Z2)

    print(f"C_0(0, M_W², -M_W², 0, M_Z², M_W²) [a]    = {C0_a:+.6e}")
    print(f"C_0(0, 4M_W², 0, 0, M_Z², M_Z²) [b]       = {C0_b:+.6e}")
    print(f"C_0(-M_W², M_W², 0, 0, 0, M_W²) [c1]      = {C0_c1:+.6e}")
    print(f"C_0(-M_W², M_W², 0, 0, M_H², M_W²) [c2]   = {C0_c2:+.6e}")
    print(f"C_0(-M_W², M_W², 0, 0, M_Z², M_W²) [c3]   = {C0_c3:+.6e}")
    print(f"C_0(M_W², -M_W², 0, 0, 0, M_W²) [d1]      = {C0_d1:+.6e}")
    print(f"C_0(M_W², -M_W², 0, 0, 0, M_Z²) [d2]      = {C0_d2:+.6e}")
    print(f"C_0(4M_W², 0, 0, 0, 0, M_Z²) [e1]         = {C0_e1:+.6e}")
    print(f"C_0(4M_W², 0, 0, 0, 0, M_W²) [e2]         = {C0_e2:+.6e}")
    print(f"L(M_W², M_W², M_H²)                       = {L_MW_MW_MH:+.6e}")
    print(f"L(M_W², M_W², M_Z²)                       = {L_MW_MW_MZ:+.6e}")
    print(f"L(4M_W², M_Z², M_Z²)                      = {L_4MW_MZ_MZ:+.6e}")
    print()

    terms = []
    add = lambda label, val: terms.append((label, complex(val)))

    add("C0_a term", (2.0 * cw2 - 1.0) * (24.0 * cw4 + 16.0 * cw2 - 1.0) * M_W2 * C0_a
        / (8.0 * cw4 * sw4))
    add("C0_b term", -(2.0 * cw2 - 1.0) * M_W2 * C0_b / (2.0 * cw4 * sw2))
    add("C0_c1 term", -((cw4 + 17.0 * cw2 - 16.0) * M_H2 + M_W2) * M_W2 * C0_c1
        / (4.0 * M_H2 * sw2))
    add("C0_c2 term", (M_H2 + M_W2) * M_W2 * C0_c2 / (4.0 * M_H2 * sw2))
    add("C0_c3 term", -(2.0 * cw8 + 32.0 * cw6 + 32.0 * cw4 - 11.0 * cw2 - 16.0) * M_W2 * C0_c3
        / (8.0 * cw2 * sw4))
    add("C0_d1 term", 3.0 * (33.0 - 46.0 * cw2) * M_W2 * C0_d1 / (8.0 * sw4))
    add("C0_d2 term", (4.0 * cw4 - 1.0) * (14.0 * cw6 + 15.0 * cw4 - 2.0 * cw2 - 1.0) * M_W2 * C0_d2
        / (16.0 * cw8 * sw4))
    add("C0_e1 term", -((1.0 - 2.0 * cw2) ** 2) * (cw2 + 1.0) * ((4.0 * cw2 + 1.0) ** 2) * M_W2 * C0_e1
        / (16.0 * cw8 * sw2))
    add("C0_e2 term", -25.0 * M_W2 * C0_e2 / (4.0 * sw2))
    add("L_H term",  M_W2 * L_MW_MW_MH / (4.0 * M_H2 * sw2))
    add("L_Z term", (-168.0 * cw8 - 214.0 * cw6 + 56.0 * cw4 + 32.0 * cw2 - 3.0) * L_MW_MW_MZ
        / (24.0 * cw2 * (1.0 - 4.0 * cw2) * sw2))
    add("L_4Z term", (1.0 - 2.0 * cw2) * (8.0 * cw4 + cw2 + 3.0) * L_4MW_MZ_MZ
        / (6.0 * cw2 * sw2))
    add("ln(W/Z+1)", 3.0 * (cw2 + 1.0) * np.log(M_W2 / M_Z2 + 1.0) / (16.0 * cw6))
    add("ln(4W/Z-1)", (1.0 - 2.0 * cw2) * (64.0 * cw4 + 4.0 * cw2 + 1.0) * np.log(4.0 * M_W2 / M_Z2 - 1.0)
        / (24.0 * cw4))
    add("ln(Z/W)", (-512.0 * cw10 + 1536.0 * cw8 - 672.0 * cw6 + 44.0 * cw4 + 3.0 * cw2 - 3.0)
        * np.log(M_Z2 / M_W2) / (48.0 * cw4 * (1.0 - 4.0 * cw2) * sw2))
    add("ln 2", (-128.0 * cw10 + 304.0 * cw8 + 144.0 * cw6 - 38.0 * cw4 + 9.0 * cw2 + 3.0)
        * np.log(2.0) / (24.0 * cw6 * sw2))
    add("rational", (96.0 * cw6 - (10.0 - 2.0 * sw2 * np.pi ** 2) * cw4 - 9.0 * cw2 - 6.0)
        / (24.0 * cw4 * sw2))
    add("iπ explicit", -(128.0 * cw8 - 64.0 * cw6 + 4.0 * cw4 + 23.0 * cw2 + 5.0) * 1.0j * np.pi
        / (48.0 * cw4 * sw2))

    print(f"{'term':<16}{'Re':>14}{'Im':>14}")
    print(f"{'-'*44}")
    total = 0j
    for label, val in terms:
        print(f"{label:<16}{val.real:+14.4f}{val.imag:+14.4f}")
        total += val
    print(f"{'-'*44}")
    print(f"{'TOTAL':<16}{total.real:+14.4f}{total.imag:+14.4f}")


if __name__ == "__main__":
    main()
