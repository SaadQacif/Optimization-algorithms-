import numpy as np
import numdifftools as nd
from Minimisation_Methods_1D import DichotomousSearchMethod
from matrix_decomposition_inverse import Choleski_inv
from Search_of_step import Goldstein

                #-----------------------------------#
                #            Newton methods         #
                #-----------------------------------#

#Methode_Newton1:-------------------------------------------------------------------------------

def Methode_Newton(f,x0,eps):
    
    hessian_f=nd.Hessian(f)
    gradient_f=nd.Gradient(f)
    shape=len(hessian_f(x0))
    
    x1=x0
    x2=x1
    N_d=100

    while N_d>eps:
        
        vap=np.linalg.eig(hessian_f(x2))[1]
        
        if np.all(vap>0):
            d=-np.linalg.inv(hessian_f(x2))@gradient_f(x2)                          #Le calcul de l'inverse de matrice par np.linalg.inv

            
        else:
            d=-np.linalg.inv((eps*np.identity(shape)+hessian_f(x2)))@gradient_f(x2)
    
        N_d=np.linalg.norm(d)
        
        phi=lambda alpha : f(x1+alpha*d)
        alpha=DichotomousSearchMethod(phi,0,10,eps)
        x1=x2
        x2=x1+alpha*d

    return x2
        
#Methode_Newton2:-------------------------------------------------------------------------------------------------------------------

def Methode_Newton2(f,x0,eps):
    hessian_f=nd.Hessian(f)
    gradient_f=nd.Gradient(f)
    shape=len(hessian_f(x0))
    
    x1=x0
    x2=x1
    N_d=100

    while N_d>eps:
        
        vap=np.linalg.eig(hessian_f(x2))[1]
        
        if np.all(vap>0):
            d=-Choleski_inv(hessian_f(x2))@gradient_f(x2)                           #Le calcul de l'inverse par la fonction Choleski_inv du fichier Decomposition_Matrice

            
        else:
            d=-Choleski_inv((eps*np.identity(shape)+hessian_f(x2)))@gradient_f(x2)
    
        N_d=np.linalg.norm(d)
        
        phi=lambda alpha : f(x1+alpha*d)
        alpha=DichotomousSearchMethod(phi,0,10,eps)
        x1=x2
        x2=x1+alpha*d

    return x2

#Quasi_Newton----------------------------------------------------------------

def DFP(alpha,d,H,y):

    A=alpha*d@d.T/(d.T@y)
    P=H@y
    B=-P@P.T/(y.T@P)
    
    return H+A+B

def Quasi_Newton(f,x0,eps):

    x1=x0
    x2=x1
    
    grad_f=nd.Gradient(f)
    g=grad_f(x0).T
    H=np.identity(len(g))
    g_norm=np.linalg.norm(g)

    while g_norm > eps:
        
        d=H@g
        d_norm=np.linalg.norm(d)
        phi=lambda alpha:f(x1-alpha*d)
        
        alpha=Goldstein(phi,d_norm,alpha0=1,gho=eps)
        x1=x2
        x2=x1-alpha*d
        
        g=grad_f(x2).T
        g_norm=np.linalg.norm(g)
        
        y=grad_f(x2)-grad_f(x1)
        H=DFP(alpha,d,H,y)

    return x2

#--------------------------------------------------------------------------

                #-----------------------------------#
                #        Gradient methods           #
                #-----------------------------------#

#Descent Gradient------------------------------------------------------

def descent_gradient(f,x0,eps):
    
    D=nd.Gradient(f)
    x1=x0
    N_f=100
    
    while (N_f>eps or N_x>eps):
        
        N_f=np.linalg.norm(D(x1))
        phi=lambda alpha : f(x1-alpha*D(x1))
        alpha=DichotomousSearchMethod(phi,0,10,eps)
        x2=x1-alpha*D(x1)
        N_x=np.linalg.norm(x2-x1)
        x1=x2

    return x2

    
#Gradient_Conjugué----------------------------------------------------------------------------------------------------------------------

def Conjugate_gradient(f,Q,x0,eps):
    D=nd.Gradient(f)
    x1=x0
    x2=x1 
    d=-D(x0)
    N_d=100

    while N_d>eps:
        
        alpha=(d.T@d)/(d.T@Q@d)
        x1=x2
        x2=x1+alpha*d
        beta=-D(x1).T@d/(d.T@Q@d)
        d=-D(x2)+beta*d
        N_d=np.linalg.norm(d)

    return x2

#---------------------------------------------------------------------------------------------------------------------
