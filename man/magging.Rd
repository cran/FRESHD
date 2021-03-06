% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/magging.R
\name{magging}
\alias{magging}
\title{Maximin Aggregation}
\usage{
magging(B)
}
\arguments{
\item{B}{array of size \eqn{p \times G } containing the group parameter
estimates where \eqn{p} is the number of model  parameters and \eqn{G} is the
number of groups.}
}
\value{
An object with S3 Class "FRESHD".
\item{...}{A \eqn{p} vector containing the maximin aggregated parameter estimates.}
}
\description{
R wrapper for a C++ implementation of the generic maximin 
aggregation procedure.
}
\details{
Following \cite{Buhlmann 2016}  this function computes the maximin 
aggregation estimator for given group estimates. This entails solving a 
convex quadratic optimization problem. The function wraps a C++ implementation
of an algorithm by Goldfarb and Idnani for solving a (convex) quadratic 
programming problem by means of a dual method.

The underlying C++ program solving the convex quadratic optimization problem, 
eiquadprog.hpp, copyright (2011) Benjamin Stephens, GPL v2 see 
https://www.cs.cmu.edu/~bstephe1/eiquadprog.hpp, is based on previous 
libraries:

QuadProg++, Copyright (C) 2007-2016 Luca Di Gaspero, MIT License. See 
https://github.com/liuq/QuadProgpp

uQuadProg, Copyright (C) 2006 - 2017 Angelo Furfaro, LGPL v3, 
a port  of QuadProg++  working with ublas data structures. See
https://github.com/fx74/uQuadProg/blob/master/README.md
  
QuadProg Copyright (C) 2014-2015 Gael Guennebaud, LGPL v3, a modification of 
uQuadProg, working with Eigen data structures. See
http://www.labri.fr/perso/guenneba/code/QuadProg/.
}
\examples{
##size of example
set.seed(42)
G <- 15; n <- c(50, 20, 13); p <- c(7, 5, 4)
nlambda <- 10

##marginal design matrices (Kronecker components)
x <- list()
for(i in 1:length(n)){
x[[i]] <- matrix(rnorm(n[i] * p[i], 0, 1), n[i], p[i])
}

##common features and effects
common_features <- rbinom(prod(p), 1, 0.1)
common_effects <- rnorm(prod(p), 0, 1) * common_features
system.time({
##group response and fit
lambda <- exp(seq(-1, -4, length.out = nlambda))
magbeta <- matrix(0, prod(p), nlambda)
B <- array(NA, c(prod(p), G, nlambda))
y <- array(NA, c(n, G))
for(g in 1:G){
bg <- rnorm(prod(p), 0, 0.1) * (1 - common_features) + common_effects
Bg <- array(bg, p)
mu <- RH(x[[3]], RH(x[[2]], RH(x[[1]], Bg)))
y[,,, g] <- array(rnorm(prod(n)), dim = n) + mu
B[, g, ] <- glamlasso::glamlasso(x, y[,,, g], lambda = lambda)$coef
}
})

##maximin aggregation for all lambdas (models)
for(l in 1:dim(B)[3]){
magbeta[, l] <- magging(B[, , l])
}

##estimated common effects for specific lambda
modelno <- 10
betafit <- magbeta[, modelno]
plot(common_effects, type = "h", ylim = range(betafit, common_effects), col = "red")
lines(betafit, type = "h")

}
\references{
Buhlmann, Peter and Meinshausen, Nicolai (2016). Magging: maximin aggregation for
 inhomogeneous large-scale data. Proceedings of the IEEE, 1, 104, 126-135

D. Goldfarb, A. Idnani. A numerically stable dual method for solving strictly
convex quadratic programs (1983). Mathematical Programming, 27,  1-33.
}
\author{
Adam Lund, Benjamin Stephens, Gael Guennebaud, Angelo Furfaro, Luca Di Gaspero

Maintainer: Adam Lund, \email{adam.lund@math.ku.dk}
}
