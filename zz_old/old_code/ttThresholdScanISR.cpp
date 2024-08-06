#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <array>
#include <algorithm>

#include "QQbar_threshold/parameters.hpp"
#include "QQbar_threshold/xsection.hpp"
#include "QQbar_threshold/load_grid.hpp"
#include "QQbar_threshold/integrate.hpp"
#include "QQbar_threshold/structure_function.hpp"




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

void symmetriseShifts(std::vector<float>& shifts) {
    size_t originalSize = shifts.size();
    for (size_t i = 0; i < originalSize; ++i){
        shifts.push_back(-shifts[i]);
    }
    shifts.push_back(0.);
    std::sort(shifts.begin(), shifts.end());
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
                    float mass_scale, float width_scale, float Yukawa, float as_var = 0.){
                      
        if (Index < N){ 
            constexpr double sqrt_s = sqrt_s_values[Index];
            std::string filename = "output_ISR/" + order_to_string(order) + "_scan_PS_ISR";
            filename += "_ecm" + floatToString(sqrt_s, 1);
            filename += "_mass" + floatToString(mass,2);
            filename += "_width" + floatToString(width, 2);
            filename += "_yukawa" + floatToString(Yukawa, 1);
            if (as_var > 1E-6) filename += "_asUp";
            else if (as_var < -1E-6) filename += "_asDown";
            else filename += "_asNominal";
            filename += ".txt";
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
            std::ofstream outputFile(filename);
            double xsec = QQt::integrate(integrand, 0, t_max);
            //std::cout << sqrt_s << ", " << xsec << '\n';
            outputFile << sqrt_s << ", " << xsec << '\n';  
            outputFile.close();
        }
        sqrt_s_loop<Index+1, N>::compute_xsec(order, mass, width, mass_scale, width_scale, Yukawa, as_var);
    
    }
};

template<size_t N>
struct sqrt_s_loop<N, N> {
    static void compute_xsec(QQt::pert_order::order, float, float, float, float, float, float) {
        // Base case: do nothing when Index equals N (end of recursion)
    }
};

int main(){
    QQt::load_grid(QQt::grid_directory() + "ttbar_grid.tsv");


    bool doVariations = false;
    bool doShifts = true;

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

    std::vector<QQt::pert_order::order> orders = {QQt::NLO, QQt::NNLO, QQt::N3LO};

    constexpr size_t sqrt_s_count = sqrt_s_values.size();

    for (auto order : orders){
        float default_mt = default_PS_mass;
        continue;
        
        sqrt_s_loop<0, sqrt_s_count>::compute_xsec(order, default_mt, default_width, default_PS_mass_scale, default_width_scale, default_Yukawa);
        
        if (!doVariations) continue;

        // mass variation
        sqrt_s_loop<0, sqrt_s_count>::compute_xsec(order, default_mt+mt_var, default_width, default_PS_mass_scale, default_width_scale, default_Yukawa);
        sqrt_s_loop<0, sqrt_s_count>::compute_xsec(order, default_mt-mt_var, default_width, default_PS_mass_scale, default_width_scale, default_Yukawa);

        // width variation
        sqrt_s_loop<0, sqrt_s_count>::compute_xsec(order, default_mt, default_width+width_var, default_PS_mass_scale, default_width_scale, default_Yukawa);
        sqrt_s_loop<0, sqrt_s_count>::compute_xsec(order, default_mt, default_width-width_var, default_PS_mass_scale, default_width_scale, default_Yukawa);

        // Yukawa variation
        sqrt_s_loop<0, sqrt_s_count>::compute_xsec(order, default_mt, default_width, default_PS_mass_scale, default_width_scale, default_Yukawa+Yukawa_var);
        sqrt_s_loop<0, sqrt_s_count>::compute_xsec(order, default_mt, default_width, default_PS_mass_scale, default_width_scale, default_Yukawa-Yukawa_var);


        // alphaS variation
        sqrt_s_loop<0, sqrt_s_count>::compute_xsec(order, default_mt, default_width, default_PS_mass_scale, default_width_scale, default_Yukawa, as_var);
        sqrt_s_loop<0, sqrt_s_count>::compute_xsec(order, default_mt, default_width, default_PS_mass_scale, default_width_scale, default_Yukawa, -1.*as_var);
        
    }

    if (!doShifts) return 0;

    QQt::pert_order::order order = QQt::N3LO;
    std::vector<float> mass_shifts = {0.02};
    std::vector<float> width_shifts = {0.03};
    std::vector<float> Yukawa_shifts = {0.1};
    std::vector<float> as_shifts = {0.0003};

    symmetriseShifts(mass_shifts);
    symmetriseShifts(width_shifts);
    symmetriseShifts(Yukawa_shifts);
    symmetriseShifts(as_shifts);

    Yukawa_shifts = {0.};
    as_shifts = {0.};
    /*
    for (auto mass_shift : mass_shifts){
        for (auto width_shift : width_shifts){
            for (auto Yukawa_shift : Yukawa_shifts){
                for (auto as_shift : as_shifts){
                    std::cout << "mass shift: " << mass_shift << ", width shift: " << width_shift << ", Yukawa shift: " << Yukawa_shift << ", as shift: " << as_shift << std::endl;
                    sqrt_s_loop<0, sqrt_s_count>::compute_xsec(order, default_PS_mass+mass_shift, default_width+width_shift, default_PS_mass_scale, 
                                                                default_width_scale, default_Yukawa+Yukawa_shift, as_shift);
                }
            }
        }
    }
    */

    float mass_shift = 0.03;
    float width_shift = 0.05;
    float Yukawa_shift = 0.1;
    float as_shift = 0.0003;

    /*
    std::cout << "nominal" << std::endl;
    sqrt_s_loop<0, sqrt_s_count>::compute_xsec(order, default_PS_mass, default_width, default_PS_mass_scale, 
                                                                default_width_scale, default_Yukawa);

    std::cout << "mass shift: " << mass_shift << std::endl;
    sqrt_s_loop<0, sqrt_s_count>::compute_xsec(order, default_PS_mass+mass_shift, default_width, default_PS_mass_scale, 
                                                                default_width_scale, default_Yukawa);

    std::cout << "width shift: " << width_shift << std::endl;
    sqrt_s_loop<0, sqrt_s_count>::compute_xsec(order, default_PS_mass, default_width+width_shift, default_PS_mass_scale, 
                                                                default_width_scale, default_Yukawa);

    std::cout << "Yukawa shift: " << Yukawa_shift << std::endl;
    sqrt_s_loop<0, sqrt_s_count>::compute_xsec(order, default_PS_mass, default_width, default_PS_mass_scale, 
                                                                default_width_scale, default_Yukawa+Yukawa_shift);
    
    std::cout << "as shift: " << as_shift << std::endl;
    sqrt_s_loop<0, sqrt_s_count>::compute_xsec(order, default_PS_mass, default_width, default_PS_mass_scale, 
                                                                default_width_scale, default_Yukawa, as_shift);

    */

    mass_shift = 0.01;
    width_shift = -0.03;
    Yukawa_shift = 0.05;

    std::cout << "mass shift: " << mass_shift << ", width shift: " << width_shift << ", Yukawa shift: " << Yukawa_shift << ", as shift: " << as_shift << std::endl;
    sqrt_s_loop<0, sqrt_s_count>::compute_xsec(order, default_PS_mass+mass_shift, default_width+width_shift, default_PS_mass_scale, 
                                                                default_width_scale, default_Yukawa+Yukawa_shift);


    return 0;
}



