// Thin C wrapper over the eMELA C++ library, exposing the minimal interface
// needed by the Python ctypes layer in emela_wrapper.py.
//
// Build (after eMELA is installed):
//   see build_emela_wrapper.sh in this directory.

#include <string>
#include "eMELA/eMELA.hh"

extern "C" {

void emela_quick_initialize(const char* pert_order,
                            const char* fac_scheme,
                            const char* ren_scheme,
                            double      alpha_value)
{
    eMELA::QuickInitialize(
        std::string(pert_order),
        std::string(fac_scheme),
        std::string(ren_scheme),
        alpha_value
    );
}

// Returns x * D(x, Q) at NLL (or LL, depending on initialisation).
// pdg_id=11 selects the electron PDF.
// omx = 1-x, passed explicitly for numerical accuracy near x→1.
double emela_code_pdf(int pdg_id, double x, double omx, double Q)
{
    return eMELA::CodePdf(pdg_id, x, omx, Q);
}

// Returns x * D_LL(x, Q) using eMELA's built-in LL radiator.
// ll_index: 0=collinear, 1=beta, 2=eta, 3=mixed  (use 1 for BETA scheme).
double emela_ll_pdf(int ll_index, double x, double omx, double Q)
{
    return eMELA::LLPDF(ll_index, x, omx, Q);
}

// Returns alpha_QED at scale squared Q2.
double emela_alpha_qed(double q2)
{
    return eMELA::aQED(q2);
}

} // extern "C"
