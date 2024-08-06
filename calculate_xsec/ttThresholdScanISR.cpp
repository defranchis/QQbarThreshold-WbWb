#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <array>
#include <algorithm>
#include <filesystem>


#include "QQbar_threshold/parameters.hpp"
#include "QQbar_threshold/xsection.hpp"
#include "QQbar_threshold/load_grid.hpp"
#include "QQbar_threshold/integrate.hpp"
#include "QQbar_threshold/structure_function.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace QQt = QQbar_threshold;


void create_directory(const std::string& dir) {
    if (!std::filesystem::exists(dir)) {
        std::filesystem::create_directory(dir);
    }
}

std::string order_to_string(QQt::pert_order::order order){
    switch (order){
        case QQt::LO: return "LO";
        case QQt::NLO: return "NLO";
        case QQt::NNLO: return "NNLO";
        case QQt::N3LO: return "N3LO";
        default: return "Unknown";
    }
}

QQt::pert_order::order int_to_order(int order){
    switch (order){
        case 0: return QQt::LO;
        case 1: return QQt::NLO;
        case 2: return QQt::NNLO;
        case 3: return QQt::N3LO;
        default: return QQt::N3LO;
    }
}

std::string floatToString(float value, int n_decimals) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(n_decimals) << value;
    return oss.str();
}

constexpr std::array<double, 102> sqrt_s_values = { //yes, it is this stupid...
                                                340.0, 340.1, 340.2, 340.3, 340.4, 340.5, 340.6, 340.7, 340.8, 340.9, 
                                                341.0, 341.1, 341.2, 341.3, 341.4, 341.5, 341.6, 341.7, 341.8, 341.9, 
                                                342.0, 342.1, 342.2, 342.3, 342.4, 342.5, 342.6, 342.7, 342.8, 342.9, 
                                                343.0, 343.1, 343.2, 343.3, 343.4, 343.5, 343.6, 343.7, 343.8, 343.9, 
                                                344.0, 344.1, 344.2, 344.3, 344.4, 344.5, 344.6, 344.7, 344.8, 344.9, 
                                                345.0, 345.1, 345.2, 345.3, 345.4, 345.5, 345.6, 345.7, 345.8, 345.9, 
                                                346.0, 346.1, 346.2, 346.3, 346.4, 346.5, 346.6, 346.7, 346.8, 346.9, 
                                                347.0, 347.1, 347.2, 347.3, 347.4, 347.5, 347.6, 347.7, 347.8, 347.9, 
                                                348.0, 348.1, 348.2, 348.3, 348.4, 348.5, 348.6, 348.7, 348.8, 348.9, 
                                                349.0, 349.1, 349.2, 349.3, 349.4, 349.5, 349.6, 349.7, 349.8, 349.9, 
                                                350.0, 365.0};

template<size_t Index, size_t N>
struct sqrt_s_loop {
    static void compute_xsec(QQt::pert_order::order order, float mass, float width, 
                    float mass_scale, float width_scale, float Yukawa, float as_var, std::ofstream & outputFile){
                      
        if (Index < N){
            constexpr double sqrt_s = sqrt_s_values[Index];
            QQt::options opt = QQt::top_options();
            opt.Yukawa_factor = Yukawa;
            opt.alpha_s_mZ = opt.alpha_s_mZ + as_var;
            opt.ISR_const = true;
            const double beta = QQt::ISR_log(sqrt_s, QQt::alpha_mZ);
            const auto integrand = [=](double t){
                const double x = 1 - std::pow(t, 1/beta);
                const double L = QQt::modified_luminosity_function(t, beta);
                const double sigma = QQt::ttbar_xsection(
                    std::sqrt(x)*sqrt_s, {mass_scale, width_scale}, {mass, width}, order,
                    opt
                 );
                return L*sigma;
            };
            constexpr double x_min = 330.*330./(sqrt_s*sqrt_s);
            const double t_max = std::pow(1 - x_min, beta);
            double xsec = QQt::integrate(integrand, 0, t_max);
            outputFile << sqrt_s << ", " << xsec << '\n';  
            outputFile.flush();
        }
        sqrt_s_loop<Index+1, N>::compute_xsec(order, mass, width, mass_scale, width_scale, Yukawa, as_var, outputFile);
    
    }
};

template<size_t N>
struct sqrt_s_loop<N, N> {
    static void compute_xsec(QQt::pert_order::order, float, float, float, float, float, float, std::ofstream &){
        // Base case: do nothing when Index equals N (end of recursion)
    }
};

std::string formFileName (std::string outdir, QQt::pert_order::order order, float mass, float width, float Yukawa, float as_var, float mass_scale, float width_scale){
    std::string filename = outdir + "/" + order_to_string(order) + "_scan_PS_ISR";
    filename += "_mass" + floatToString(mass,2) + "_width" + floatToString(width, 2) + "_yukawa" + floatToString(Yukawa, 2) + "_asVar" + floatToString(as_var, 4) + 
    "_scaleM" + floatToString(mass_scale, 1) + "_scaleW" + floatToString(width_scale, 1);
    filename += ".txt";
    return filename;
}

void do_scan(int order, float PS_mass, float width, float mass_scale, float width_scale, float Yukawa, float as_var, std::string outdir){
    
    QQt::load_grid(QQt::grid_directory() + "ttbar_grid.tsv");

    create_directory(outdir);

    if (order < 0 || order > 3){
        std::cerr << "Invalid order: " << order << std::endl;
        return;
    }
    QQt::pert_order::order pert_ord = int_to_order(order);
    constexpr size_t sqrt_s_count = sqrt_s_values.size();
    std::cout << "Scan for order: " << order_to_string(int_to_order(order)) << ", PS mass: " << PS_mass << ", width: " << width << ", mass scale: " << mass_scale << ", width scale: " << width_scale << ", Yukawa: " << Yukawa << ", alpha_s variation: " << as_var << std::endl;

    std::string filename = formFileName(outdir, pert_ord, PS_mass, width, Yukawa, as_var, mass_scale, width_scale);
    std::ofstream outputFile(filename);
    sqrt_s_loop<0, sqrt_s_count>::compute_xsec(pert_ord, PS_mass, width, mass_scale, width_scale, Yukawa, as_var, outputFile);
    outputFile.close();

    return;
}

PYBIND11_MODULE(xsec_calc, m) {
    m.def("do_scan", &do_scan, py::arg("order") = 3, py::arg("PS_mass") = 171.5, py::arg("width") = 1.33,  py::arg("mass_scale") = 80., py::arg("width_scale") = 350., 
            py::arg("yukawa") = 1., py::arg("as_var") = 0., py::arg("outdir")="output");
}