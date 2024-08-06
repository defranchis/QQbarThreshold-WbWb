#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <array>

#include "QQbar_threshold/parameters.hpp"
#include "QQbar_threshold/xsection.hpp"
#include "QQbar_threshold/load_grid.hpp"

namespace QQt = QQbar_threshold;

std::string order_to_string(QQt::pert_order::order order){
    switch (order){
        case QQt::LO: return "LO";
        case QQt::NLO: return "NLO";
        case QQt::NNLO: return "NNLO";
        case QQt::N3LO: return "N3LO";
        default: return "Unknown";
    }
}

std::string floatToString(float value, int n_decimals) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(n_decimals) << value;
    return oss.str();
}

int scan_ttbar_threshold(QQt::pert_order::order order, std::vector<double> ecm_scan, float mass, float width, float mass_scale, float width_scale, 
                        bool ISR_const, bool MS_scheme, float Yukawa, float as_var = 0.){

        std::string filename = "output/" + order_to_string(order) + "_scan";
        if (MS_scheme) filename += "_MS";
        else filename += "_PS";

        if (ISR_const) filename += "_ISR";
        else filename += "_noISR";

        filename += "_mass" + floatToString(mass,2);
        filename += "_width" + floatToString(width, 2);
        filename += "_yukawa" + floatToString(Yukawa, 1);
        
        if (as_var > 1E-6) filename += "_asUp";
        else if (as_var < -1E-6) filename += "_asDown";
        else filename += "_asNominal";

        std::string param_filename = filename + "_params.txt";
        filename += ".txt";

        QQt::options opt = QQt::top_options();
        if (MS_scheme) {
            opt.mass_scheme = {QQt::MSshift, mass};
            opt.beyond_QCD = QQt::SM::none;
        }
        opt.Yukawa_factor = Yukawa;
        opt.alpha_s_mZ = opt.alpha_s_mZ + as_var;

        std::ofstream param_outputFile(param_filename);
            param_outputFile << QQt::top_options() << '\n';
            param_outputFile.close();

        std::ofstream outputFile(filename);
        for (double ecm : ecm_scan){
            outputFile << ecm << ", " << QQt::ttbar_xsection(ecm, {mass_scale, width_scale}, {mass, width}, order, opt) << '\n';
        }
        outputFile.close(); 

    return 0;
}

int main(){
    QQt::load_grid(QQt::grid_directory() + "ttbar_grid.tsv");

    std::vector<double> ecm_scan = {};
    for (float e = 335.; e <= 360.; e += .1){
        ecm_scan.push_back(e);
    }

    //constexpr std::array<double, 3> ecm_scan = {340., 345., 350.};

    std::ofstream outputFile("output/default_params.txt");
    outputFile << QQt::top_options() << '\n';
    outputFile.close();

    bool doVariations = true;

    float default_PS_mass = 171.5;
    float default_width = 1.33;
    float default_PS_mass_scale = 80.;
    float default_width_scale = 350.;

    float default_MS_mass = 163.;

    bool ISR_const = false;
    bool MS_scheme = false;

    float default_Yukawa = 1.;

    float mt_var = 0.03;
    float width_var = 0.05;
    float Yukawa_var = 0.1;
    float as_var = 0.0002;

    //std::vector<QQt::pert_order::order> orders = {QQt::NLO, QQt::NNLO, QQt::N3LO};
    std::vector<QQt::pert_order::order> orders = {QQt::N3LO};

    for (auto order : orders){
        float default_mt = MS_scheme ? default_MS_mass : default_PS_mass;

        scan_ttbar_threshold(order, ecm_scan, default_mt, default_width, default_PS_mass_scale, default_width_scale, ISR_const, MS_scheme, default_Yukawa);

        if (!doVariations) continue;

        // mass variation
        scan_ttbar_threshold(order, ecm_scan, default_mt+mt_var, default_width, default_PS_mass_scale, default_width_scale, ISR_const, MS_scheme, default_Yukawa);
        scan_ttbar_threshold(order, ecm_scan, default_mt-mt_var, default_width, default_PS_mass_scale, default_width_scale, ISR_const, MS_scheme, default_Yukawa);

        // width variation
        scan_ttbar_threshold(order, ecm_scan, default_mt, default_width+width_var, default_PS_mass_scale, default_width_scale, ISR_const, MS_scheme, default_Yukawa);
        scan_ttbar_threshold(order, ecm_scan, default_mt, default_width-width_var, default_PS_mass_scale, default_width_scale, ISR_const, MS_scheme, default_Yukawa);

        // Yukawa variation
        scan_ttbar_threshold(order, ecm_scan, default_mt, default_width, default_PS_mass_scale, default_width_scale, ISR_const, MS_scheme, default_Yukawa+Yukawa_var);
        scan_ttbar_threshold(order, ecm_scan, default_mt, default_width, default_PS_mass_scale, default_width_scale, ISR_const, MS_scheme, default_Yukawa-Yukawa_var);

        // alphaS variation
        scan_ttbar_threshold(order, ecm_scan, default_mt, default_width, default_PS_mass_scale, default_width_scale, ISR_const, MS_scheme, default_Yukawa, as_var);
        scan_ttbar_threshold(order, ecm_scan, default_mt, default_width, default_PS_mass_scale, default_width_scale, ISR_const, MS_scheme, default_Yukawa, -1.*as_var);
    }

    return 0;
}



