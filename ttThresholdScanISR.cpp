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


constexpr std::array<double, 251> sqrt_s_values = { //yes, it is this stupid...
                                                335., 335.1, 335.2, 335.3, 335.4, 335.5, 335.6, 335.7, 335.8, 335.9, 
                                                336.0, 336.1, 336.2, 336.3, 336.4, 336.5, 336.6, 336.7, 336.8, 336.9,
                                                337.0, 337.1, 337.2, 337.3, 337.4, 337.5, 337.6, 337.7, 337.8, 337.9, 
                                                338.0, 338.1, 338.2, 338.3, 338.4, 338.5, 338.6, 338.7, 338.8, 338.9, 
                                                339.0, 339.1, 339.2, 339.3, 339.4, 339.5, 339.6, 339.7, 339.8, 339.9, 
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
                                                350.0, 350.1, 350.2, 350.3, 350.4, 350.5, 350.6, 350.7, 350.8, 350.9, 
                                                351.0, 351.1, 351.2, 351.3, 351.4, 351.5, 351.6, 351.7, 351.8, 351.9, 
                                                352.0, 352.1, 352.2, 352.3, 352.4, 352.5, 352.6, 352.7, 352.8, 352.9, 
                                                353.0, 353.1, 353.2, 353.3, 353.4, 353.5, 353.6, 353.7, 353.8, 353.9, 
                                                354.0, 354.1, 354.2, 354.3, 354.4, 354.5, 354.6, 354.7, 354.8, 354.9, 
                                                355.0, 355.1, 355.2, 355.3, 355.4, 355.5, 355.6, 355.7, 355.8, 355.9, 
                                                356.0, 356.1, 356.2, 356.3, 356.4, 356.5, 356.6, 356.7, 356.8, 356.9, 
                                                357.0, 357.1, 357.2, 357.3, 357.4, 357.5, 357.6, 357.7, 357.8, 357.9, 
                                                358.0, 358.1, 358.2, 358.3, 358.4, 358.5, 358.6, 358.7, 358.8, 358.9, 
                                                359.0, 359.1, 359.2, 359.3, 359.4, 359.5, 359.6, 359.7, 359.8, 359.9, 
                                                360.0};

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
            std::cout << sqrt_s << ", " << xsec << '\n';
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

    std::vector<QQt::pert_order::order> orders = {QQt::NLO, QQt::NNLO, QQt::N3LO};

    constexpr size_t sqrt_s_count = sqrt_s_values.size();

    for (auto order : orders){
        float default_mt = default_PS_mass;
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

    return 0;
}



