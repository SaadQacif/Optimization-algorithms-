# Importer les bibliothèques ---------------------------------------

from time import *
import matplotlib.pylab as plt
from Optimisation_tools.Minimisation_Methods_1D import *
from Optimisation_tools.Multivariate_Minimisation_Methods import *
from Optimisation_tools.matrix_decomposition_inverse import *
from Optimisation_tools.Search_of_step import *


# Tester les fonctions du module Minimisation_Methods_1D ------------------------------

#La fonction objective #
f=lambda x:x**4-2*x**3+(3*x-1)**2-x
fp=lambda x: 4*x**3-6*x**2+6*(3*x-1)-1
fpp=lambda x: 12*x**2-12*x+18

#Temps d'éxecution#
print("\t#1 dimension minimisation exemple :#\n")

t1=time() ; xopt1=SearchWithFixedStepSize(f ,0,1,100) ; t2=time()          
print(f"Avec FS \t xopt={xopt1} \t\t en: {t2-t1}s") ;

t1=time() ; xopt2=SearchWithAcceleratedStepSize(f ,0,1,100) ; t2=time()
print(f"Avec AS \t xopt={xopt2} \t\t en: {t2-t1}s") ;

t1=time() ; xopt3=ExhaustiveSearchMethod(f ,0,1,100) ; t2=time()
print(f"Avec EX \t xopt={xopt3} \t\t en: {t2-t1}s") ; 

t1=time() ; xopt4=IntervalHalvingSearchMethod(f ,0,1,1e-3) ; t2=time()
print(f"Avec IH \t xopt={xopt4} \t\t en: {t2-t1}s") ; 

t1=time()  ; xopt5=DichotomousSearchMethod(f ,0,1,1e-3)  ;  t2=time()
print(f"Avec Dicho \t xopt={xopt5} \t\t en: {t2-t1}s") ; 

t1=time() ; xopt6=FibonacciSearchMethod(f ,0,1,1e-3) ; t2=time() 
print(f"Avec Fibo \t xopt={xopt6} \t\t en: {t2-t1}s") ;

t1=time() ; xopt7=GoldenSectionSearchMethod(f ,0,1,1e-3) ; t2=time()
print(f"Avec GS \t xopt={xopt7} \t\t en: {t2-t1}s") ; 

t1=time() ; xopt8=Newton_1D(fpp,fp,f,0,1e-3) ; t2=time()              
print(f"Avec Newton \t xopt={xopt8} \t\t en: {t2-t1}s") ;                        

t1=time() ; xopt9=Quasi_Newton_1D(f,0,1e-3) ; t2=time()
print(f"Avec Quasi Newton  xopt={xopt9} \t\t en: {t2-t1}s") ;

t1=time() ; xopt10=Secant_1D(fp,f,0.1,1e-3) ; t2=time()
print(f"Avec Secant \t xopt={xopt10} \t\t en: {t2-t1}s") ;

# Tester les fonctions du module Multivariate_Minimisation_Methods ------------------------------

#La fonction objective #
f=lambda x : (x[0]-1)**2+(x[1]-5)**2
Q=np.array([[2,0],
            [0,2]])

x0=np.array([2.,1])
eps=10**-2

#Temps d'éxecution#

print("\n\n \t#Minimisation multivariable exemple :#\n")

Methods=["DescentGradient","Newton","QuasiNewton","ConjugateGradient"]
Temps_dexe=[]

t1=time();xopt1=descent_gradient(f,x0,eps);t2=time();Temps_dexe.append(t2-t1)
print(f"Avec Descent gradient \txopt={xopt1} \t\ten {t2-t1}s");
t1=time();xopt2=Methode_Newton2(f,x0,eps);t2=time();Temps_dexe.append(t2-t1)
print(f"Avec Newton \t\txopt={xopt2} \t\ten {t2-t1}s");
t1=time();xopt3=Quasi_Newton(f,x0,eps);t2=time();Temps_dexe.append(t2-t1)
print(f"Avec Newton \t\txopt={xopt3} \t\ten {t2-t1}s");
t1=time();xopt4=Conjugate_gradient(f,Q,x0,eps);t2=time();Temps_dexe.append(t2-t1)
print(f"Avec Conjugate gradient xopt={xopt4} \t\ten {t2-t1}s");

#Histogramme#
      
plt.bar(Methods,Temps_dexe)
plt.title("Temps d'execution des differents methods de minimisations multivariable")
plt.show()

# Tester les fonctions du module matrix_decomposition_inverse -------------------------------------

#La matrice objective #

C=np.array([[4.,1,1],
            [1,5,3],
            [1,3,6] ])

#Temps d'éxecution#

print("\n\n \t#Inverse de matrice exemple:#",end="\n"+"-"*50+"\n")

t1=time() ; Cp=Gauss_inv(C) ; t2=time()
print(f"L'inverse de \n{C}\n avec Gauss Jordan est :\n{Cp}\n en {t2-t1}s",end="\n"+"-"*50)

t1=time() ; Cp=LU_inv(C) ; t2=time()
print(f"\nL'inverse de \n{C}\n avec la decomposition LU est :\n{Cp}\n en {t2-t1}s",end="\n"+"-"*50)

t1=time() ; Cp=Choleski_inv(C) ; t2=time()
print(f"\nL'inverse de \n{C}\n avec la decomposition Choleski est :\n{Cp}\n en {t2-t1}s",end="\n"+"-"*50)

