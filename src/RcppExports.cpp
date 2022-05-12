// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// solveMMP
Rcpp::List solveMMP(arma::mat dims, arma::mat Phi1, arma::mat Phi2, arma::mat Phi3, Rcpp::NumericVector resp, std::string penalty, double kappa, arma::vec lambda, int nlambda, int makelamb, double lambdaminratio, arma::mat penaltyfactor, double tol, int maxiter, std::string alg, std::string stopcond, double orthval, double gamma0, double gmh, double gmg, double minval, double epsiloncor, int Tf, std::string wf, int J, int dim, double tauk, double gamk, double eta);
RcppExport SEXP _FRESHD_solveMMP(SEXP dimsSEXP, SEXP Phi1SEXP, SEXP Phi2SEXP, SEXP Phi3SEXP, SEXP respSEXP, SEXP penaltySEXP, SEXP kappaSEXP, SEXP lambdaSEXP, SEXP nlambdaSEXP, SEXP makelambSEXP, SEXP lambdaminratioSEXP, SEXP penaltyfactorSEXP, SEXP tolSEXP, SEXP maxiterSEXP, SEXP algSEXP, SEXP stopcondSEXP, SEXP orthvalSEXP, SEXP gamma0SEXP, SEXP gmhSEXP, SEXP gmgSEXP, SEXP minvalSEXP, SEXP epsiloncorSEXP, SEXP TfSEXP, SEXP wfSEXP, SEXP JSEXP, SEXP dimSEXP, SEXP taukSEXP, SEXP gamkSEXP, SEXP etaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type dims(dimsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Phi1(Phi1SEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Phi2(Phi2SEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Phi3(Phi3SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type resp(respSEXP);
    Rcpp::traits::input_parameter< std::string >::type penalty(penaltySEXP);
    Rcpp::traits::input_parameter< double >::type kappa(kappaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< int >::type nlambda(nlambdaSEXP);
    Rcpp::traits::input_parameter< int >::type makelamb(makelambSEXP);
    Rcpp::traits::input_parameter< double >::type lambdaminratio(lambdaminratioSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type penaltyfactor(penaltyfactorSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type maxiter(maxiterSEXP);
    Rcpp::traits::input_parameter< std::string >::type alg(algSEXP);
    Rcpp::traits::input_parameter< std::string >::type stopcond(stopcondSEXP);
    Rcpp::traits::input_parameter< double >::type orthval(orthvalSEXP);
    Rcpp::traits::input_parameter< double >::type gamma0(gamma0SEXP);
    Rcpp::traits::input_parameter< double >::type gmh(gmhSEXP);
    Rcpp::traits::input_parameter< double >::type gmg(gmgSEXP);
    Rcpp::traits::input_parameter< double >::type minval(minvalSEXP);
    Rcpp::traits::input_parameter< double >::type epsiloncor(epsiloncorSEXP);
    Rcpp::traits::input_parameter< int >::type Tf(TfSEXP);
    Rcpp::traits::input_parameter< std::string >::type wf(wfSEXP);
    Rcpp::traits::input_parameter< int >::type J(JSEXP);
    Rcpp::traits::input_parameter< int >::type dim(dimSEXP);
    Rcpp::traits::input_parameter< double >::type tauk(taukSEXP);
    Rcpp::traits::input_parameter< double >::type gamk(gamkSEXP);
    Rcpp::traits::input_parameter< double >::type eta(etaSEXP);
    rcpp_result_gen = Rcpp::wrap(solveMMP(dims, Phi1, Phi2, Phi3, resp, penalty, kappa, lambda, nlambda, makelamb, lambdaminratio, penaltyfactor, tol, maxiter, alg, stopcond, orthval, gamma0, gmh, gmg, minval, epsiloncor, Tf, wf, J, dim, tauk, gamk, eta));
    return rcpp_result_gen;
END_RCPP
}
// solveMag
Rcpp::List solveMag(arma::mat& B);
RcppExport SEXP _FRESHD_solveMag(SEXP BSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type B(BSEXP);
    rcpp_result_gen = Rcpp::wrap(solveMag(B));
    return rcpp_result_gen;
END_RCPP
}
// WT
arma::mat WT(arma::mat x, int dim, std::string wf, int J, int p1, int p2, int p3);
RcppExport SEXP _FRESHD_WT(SEXP xSEXP, SEXP dimSEXP, SEXP wfSEXP, SEXP JSEXP, SEXP p1SEXP, SEXP p2SEXP, SEXP p3SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type dim(dimSEXP);
    Rcpp::traits::input_parameter< std::string >::type wf(wfSEXP);
    Rcpp::traits::input_parameter< int >::type J(JSEXP);
    Rcpp::traits::input_parameter< int >::type p1(p1SEXP);
    Rcpp::traits::input_parameter< int >::type p2(p2SEXP);
    Rcpp::traits::input_parameter< int >::type p3(p3SEXP);
    rcpp_result_gen = Rcpp::wrap(WT(x, dim, wf, J, p1, p2, p3));
    return rcpp_result_gen;
END_RCPP
}
// IWT
arma::mat IWT(arma::mat x, int dim, std::string wf, int J, int p1, int p2, int p3);
RcppExport SEXP _FRESHD_IWT(SEXP xSEXP, SEXP dimSEXP, SEXP wfSEXP, SEXP JSEXP, SEXP p1SEXP, SEXP p2SEXP, SEXP p3SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type dim(dimSEXP);
    Rcpp::traits::input_parameter< std::string >::type wf(wfSEXP);
    Rcpp::traits::input_parameter< int >::type J(JSEXP);
    Rcpp::traits::input_parameter< int >::type p1(p1SEXP);
    Rcpp::traits::input_parameter< int >::type p2(p2SEXP);
    Rcpp::traits::input_parameter< int >::type p3(p3SEXP);
    rcpp_result_gen = Rcpp::wrap(IWT(x, dim, wf, J, p1, p2, p3));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_FRESHD_solveMMP", (DL_FUNC) &_FRESHD_solveMMP, 29},
    {"_FRESHD_solveMag", (DL_FUNC) &_FRESHD_solveMag, 1},
    {"_FRESHD_WT", (DL_FUNC) &_FRESHD_WT, 7},
    {"_FRESHD_IWT", (DL_FUNC) &_FRESHD_IWT, 7},
    {NULL, NULL, 0}
};

RcppExport void R_init_FRESHD(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
