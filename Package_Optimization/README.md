# Optimisation_algorithms
This package contains Algorithms to minimise functions (1D or multivariate) with algorithms of decomposition and inversing matrix

## How to use :
_After installing the package use following import :_ <br>

For 1 dimensional minimisation : <br>
**from Optimisation_tools.Minimisation_Methods_1D import 
  SearchWithFixedStepSize,
  SearchWithAcceleratedStepSize,
  ExhaustiveSearchMethod,
  IntervalHalvingSearchMethod,
  DichotomousSearchMethod,
  FibonacciSearchMethod,
  GoldenSectionSearchMethod,
  Newton_1D,
  Quasi_Newton_1D,
  Secant_1D**

For multivariate minimisation : <br>
**from Optimisation_tools.Multivariate_Minimisation_Methods import
  Methode_Newton,
  Methode_Newton2,
  Quasi_Newton,
  descent_gradient,
  Conjugate_gradient**

For Matrix decomposition and inverse : <br>
**from Optimisation_tools.matrix_decomposition_inverse import
  decomp_LU,
  LU_inv,
  Choleski_decomp,
  Choleski_inv,
  Gauss_inv**

For Search of the step : <br>
**from Optimisation_tools.Search_of_step import
  Armijo,
  Goldstein**
  
