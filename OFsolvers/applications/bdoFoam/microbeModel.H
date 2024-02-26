#include<vector>
#include<scalar.H>
//#include <boost/math/special_functions/gamma.hpp>
//#include <gamma.hpp>

namespace microbemodel
{
    extern const int X;
    extern const int O2;
    extern const int G;
    extern const int Xy;
    extern const int A;
    extern const int B;
    extern const int nvars;
    extern const double y_xs;
    extern const double y_as;
    extern const double y_bs;
    extern const double y_os;
    extern const double x_max;
    extern const double qs_max;
    extern const double o2_max;
    extern const double K_e;
    extern const double K_s;
    extern const double alpha_s;
    extern const double beta_s;
    extern const double alpha_e;
    extern const double beta_e;
    extern const double kLa;
    extern const double chi_p;

    extern double X_avg;
    extern double G_avg;
    extern double Xy_avg;
    extern double A_avg;
    extern double B_avg;

    void get_rhs(std::vector<double>& rhs,std::vector<double> solnvec,double t,int nvars);
    void advance(std::vector<double>& solnvec,int nvars,double t_now,double t_adv,double dt);
    double get_our(std::vector<double> solnvec,int nvars);
}
