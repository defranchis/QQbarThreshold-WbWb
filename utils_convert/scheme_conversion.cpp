#include <pybind11/pybind11.h>
#include "QQbar_threshold/scheme_conversion.hpp"
#include "QQbar_threshold/width.hpp"

namespace py = pybind11;
namespace QQt = QQbar_threshold;

double calculate_mt_Pole(double mt_PS, double mu){
    return QQt::top_pole_mass(mu, mt_PS, QQt::N3LO);
}

double calculate_mt_MS(double mt_PS, double mu){
    const double precision = 1e-4;
    QQt::options opt = QQt::top_options();
    double mt_MS = mt_PS;
    double delta_M;
    do{
        opt.mass_scheme = {QQt::MSshift, mt_MS};
        delta_M = QQt::top_pole_mass(mu, mt_MS, QQt::N3LO, opt) - calculate_mt_Pole(mt_PS, mu);
        mt_MS -= delta_M;
    } while(std::abs(delta_M) > precision);
    return mt_MS;
}

double calculate_width(double mt_PS, double mu){
    return QQt::top_width(mu, mt_PS, QQt::N2LO);
}

PYBIND11_MODULE(scheme_conversion, m) {
    m.def("calculate_mt_Pole", &calculate_mt_Pole, py::arg("mt_PS"), py::arg("mu") = 80.,
    "A function which converts PS mass to pole mass");
    m.def("calculate_mt_MS", &calculate_mt_MS, py::arg("mt_PS"), py::arg("mu") = 80.,
    "A function which converts PS mass to MS mass");
    m.def("calculate_width_N2LO", &calculate_width, py::arg("mt_PS"), py::arg("mu") = 80.,
    "A function which calculates top width");
}