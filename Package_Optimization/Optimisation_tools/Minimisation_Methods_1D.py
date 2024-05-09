from math import *
from numdifftools import *


                #-----------------------------------#
                #       Elimination Methods         #
                #-----------------------------------#


#Recherche à pas fixe ----------------------------------

def SearchWithFixedStepSize(f,a,b,d):#f is unimodel function on [a,b] with a minimum (user can just use -f instead of f in the case of a function f with a maximum)
    op=a
    while f(op+d)<f(op) and op<=b :
        op=op+d
    return op


#Recherche à pas acceleré--------------------------------------------

def SearchWithAcceleratedStepSize(f,a,b,d):
    op=a
    initialStep=d
    while (f(op+d)<f(op) or d!=initialStep) :
        if f(op+d)>f(op) :
            a=op-d/2
            d=initialStep
        else :
            d=2*d
        op=a+d
    return op


#Recherche Exhaustive-----------------------------------------------------

def ExhaustiveSearchMethod(f,a,b,p):
    n=int((b-a)/p)
    d=(b-a)/(n+1)
    op=a
    while f(op+d)<f(op) and op<=b:
        op+=d
    return op


#Interval halving--------------------------------------------

def IntervalHalvingSearchMethod(f,xs,xf,d):
    p=(xf-xs)/4
    x1,x0,x2=xs+p,xs+2*p,xf-p
    L=[(xs,xf),(x1,x2)]
    while (x2-x1)>d :
        p=p/2
        if f(x1)<f(x0) :
            x1,x0,x2=x1-p,x1,x1+p
        elif f(x2)<f(x0) :
            x1,x0,x2=x2-p,x2,x2+p
        else :
            x1,x2=x0-p,x0+p
        L.append((round(x1,2),round(x2,2)))
    return (x1+x2)/2


#Dichotomie----------------------------------------------------


def DichotomousSearchMethod(f,xs,xf,d):
    while ((xf-xs)/2)>=d :
        m=(xs+xf)/2
        x1=m-d/2
        x2=m+d/2
        if f(x1)<f(x2) :
            xf=x2
        else :
            xs=x1
    return (xs+xf)/2



#Fibonacci------------------------------------------------------

def F(n):                              #Definition de la suite Fibonacci
    L=[1,1]
    for i in range(2,n+1):
        L.append(L[i-1]+L[i-2])
    return L


#----------------------------------------------
    
def FibonacciSearchMethod(f,a,b,d):
    n=1
    while(F(n)[n-1]<((b-a)/d)):
        n+=1
    Fn=F(n)
    L0=Lj=b-a
    j=1
    while (Lj>d):
        j+=1
        Lj=(Fn[n-j]/Fn[n])*L0
        x1=a+Lj
        x2=b-Lj
        if(f(x1)<f(x2)):
            b=x2
        elif (f(x1)>f(x2)):
            a=x1
        else :
            a,b=x1,x2
    return (x1+x2)/2



#Golden section-----------------------------------------------


def GoldenSectionSearchMethod(f,a,b,d):
    L0=Lj=b-a
    j=1
    while (Lj>d):
        j+=1
        Lj=((1/phi)**(j))*L0              #Où phi est le nombre d'or
        x1=a+Lj
        x2=b-Lj
        if(f(x1)<f(x2)):
            b=x2
        elif (f(x1)>f(x2)):
            a=x1
        else :
            a,b=x1,x2
    return (x1+x2)/2

#---------------------------------------------------------------------------

                #-----------------------------------#
                #   Root finding (interpolation)     #
                #-----------------------------------#


#Newton-Rapson---------------------------------------------------------------

def fp(x):
    d=0.0000001
    return (f(x+d)-f(x-d))/(2*d)

def fpp(x) :
    d=0.0000001
    return (f(x+d)-2*f(x)+f(x-d))/(d**2)

def Newton_1D(f,a,b,d):
    op=a
    L=[]
    while(abs(fp(op))>d and op < b ):
          op=op-fp(op)/fpp(op)
          L.append(op)
    return op


#Quasi-Newton-----------------------------------------------------------------


s=0.00001

def fp(f,x):
    return (f(x+s)-f(x-s))/(2*s)

def fpp(f,x) :
    return (f(x+s)-2*f(x)+f(x-s))/(s**2)

def Quasi_Newton_1D(f,a,b,d):
    global L
    L=[]
    op=a
    while(abs(fp(f,op))>d and op<b) :
          op=op-(fp(f,op))/(fpp(f,op))
          L.append(round(op,2))
    return op


#Secant-------------------------------------------------------------------------

s=0.00001

def fp(f,x):
    return (f(x+s)-f(x-s))/(2*s)

def fpp(f,x) :
    return (f(x+s)-2*f(x)+f(x-s))/(s**2)

def Secant_1D(f,a,b,d):
    global L
    L=[]
    op=a
    f1=nd.Derivative(f)
    s=(f1(b)-f1(a))/(b-a)
    while(abs(f1(op))>d and op < b ):
        op=op-f1(op)/s
        L.append(round(op,2))
    return op

#------------------------------------------------------------------------------------
