import numpy as np
import numdifftools as nd


#---------------------------------------------------------------------------

def Armijo(phi,d,alpha0,eps,eta):
    alpha=alpha0
    phi_p0=-d**2
    phi_cap=lambda x:phi(0)+eps*phi_p0*x

    while (phi(alpha)>phi_cap(alpha) or phi(eta*alpha)<=phi_cap(eta*alpha)):

        if phi(alpha)>phi_cap(alpha):
            alpha=alpha/eta
            
        elif phi(eta*alpha)<=phi_cap(eta*alpha):
            alpha=eta*alpha
            

    return alpha

#------------------------------------------------------------------------------

def Goldstein(phi,d,alpha0,gho):
    
    phi_p0=-d**2
    phi_0=phi(0)
    phi_cap1=lambda x:phi_0+gho*phi_p0*x
    phi_cap2=lambda x:phi_0+(1-gho)*phi_p0*x
    
    a=0; b=np.inf; alpha=alpha0; t=1/gho;

    while (phi(alpha)>phi_cap1(alpha) or phi(alpha)<phi_cap2(alpha)):
        if phi(alpha)>phi_cap1(alpha):
            b=alpha
            alpha=(a+b)/2
        elif phi(alpha)<phi_cap2(alpha):
            a=alpha
            if b<np.inf:
                alpha=(a+b)/2
            else:
                alpha=t*alpha
    
    return alpha

#--------------------------------------------------------------------------
