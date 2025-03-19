import numpy as np
import mplhep as hep
import matplotlib.pyplot as plt
plt.style.use(hep.style.CMS)


def total_error(x, factor = 1., factor_EM = 1.):
    return ((1.49831 / 250 * x) **2 + factor*( (0.49/factor_EM)**2 + 0.126114**2 + 0.065042**2) )**0.5


x_values = np.linspace(0, 300, 500)

y_values = total_error(x_values)
y_values_mt = total_error(x_values, factor=0)
y_values_EM = total_error(x_values, factor_EM=3)


plt.plot(x_values, y_values, label=r'All parametric uncertainties (FCC-ee)', linewidth=2)
plt.plot(x_values, y_values_EM, label=r'Improved $\alpha_\mathrm{EM}$ (FCC-ee)', linewidth=2, linestyle='--', color='C0')
plt.plot(x_values, y_values_mt, label=r'$m_\mathrm{t}$ uncertainty only', linewidth=2)

plt.axhline(y=0.18, color='gray', linestyle='--', label=r'$m_\mathrm{W}$ experimental (FCC-ee)', linewidth=2)
plt.axvline(x=250, color='r', linestyle='--', label=r'$m_\mathrm{t}$ experimental (HL-LHC)', linewidth=2)
                                                                                                                            

plt.ylabel(r'$m_\mathrm{W}$ uncertainty [MeV]')
plt.xlabel(r'$m_\mathrm{t}$ uncertainty [MeV]')
plt.legend(loc='upper left')

plt.savefig('plots/mt_mW_uncert.png')
plt.savefig('plots/mt_mW_uncert.pdf')
