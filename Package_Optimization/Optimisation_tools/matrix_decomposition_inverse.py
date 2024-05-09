import numpy as np
from math import *

                #-----------------------------------#
                #    Decomposition LU Methods       #
                #-----------------------------------#

#Décomposition_matrice_LU:-----------------------------------------------------------------

def decomp_LU(A):
    U=A.copy();L=np.identity(len(A))                            #U doit etre triangulaire supérieure ; L doit etre triangulaire inferieur avec des 1 dans la diagonal
    for i in range(len(A)-1):
        m=U[i,i]
        if m==0:
            print("Erreur zero en diagonal") ; return           #Si un coef dans la diagonale est null alors la matrice n'est pas decomposable en LU
            
        for j in range(i+1,len(A)):
            if U[j,i]==0 : continue                             #si le coef de la colone i au dessou de de la ligne j est null on passe à l'iteration suivante
            
            coef=U[j,i]/m
            L[j,i]=coef
            U[j,:]=U[j,:]-coef * U[i,:]

    return L,U 

#Resolution_AX=b_Decomposition_LU:-----------------------------------------------

def LU_solve(A,b):
    L,U=decomp_LU(A)                                #On décompose A en LU
    Z=np.zeros(len(A))
    for i in range(len(A)):
        Z[i]=b[i]-L[i,:i]@Z[:i]                     #L[i,i] vaut 1 donc aucun interêt a diviser par cet element
    X=np.zeros(len(A))
    for i in range(len(A)-1,-1,-1):
        X[i]=(Z[i]-U[i,i:]@X[i:])/U[i,i]
        
    return X


#Inverse_Matrice_LU:---------------------------------------------------------------

def LU_inv(A):
    I=np.identity(len(A))
    Ap=np.zeros((len(A),len(A)))
    for i in range(len(A)):
        
        Ap[:,i]=LU_solve(A,I[:,i])                      #On resout AX=b ( avec b est pendant chaque iteration une colonne de l'identité et X est une colone de l'inverse de A)
                         
    return Ap

#----------------------------------------------------------------------------------

                #-----------------------------------#
                #        Choleski Methods           #
                #-----------------------------------#

#Inverse_Choleski-------------------------------------------------------------------

def Choleski_decomp(A):                                 #La matrice A doit etre definie positive pour etre Choleski-decomposable        
    L=np.zeros((len(A),len(A)))                         #L est triangulaire inferieur
    
    for i in range(len(A)):
        L[i,i]=sqrt(A[i,i]-L[i,:i]@L[i,:i])
        for j in range(i+1,len(A)):
            L[j,i]=(A[j,i]-L[j,:i]@L[i,:i])/L[i,i]

    return L                                            #on a donc A=L@transposée(L)
        
def Choleski_solve(A,b):
    
    L=Choleski_decomp(A)
    Z=np.zeros(len(A))
    
    for i in range(len(A)):
        Z[i]=(b[i]-L[i,:i]@Z[:i])/L[i,i]                #On resout LZ=b
        
    X=np.zeros(len(A))
    for i in range(len(A)-1,-1,-1):
        X[i]=(Z[i]-L.T[i,i:]@X[i:])/L.T[i,i]            #Apres on resout (L.T)X=Z (avec L.T est la transposée de L)
        
    return X

def Choleski_inv(A):
    
    I=np.identity(len(A))
    Ap=np.zeros((len(A),len(A)))
    for i in range(len(A)):
        
        Ap[:,i]=Choleski_solve(A,I[:,i])                #On resout AX=b ( avec b est a chaque iteration une colone de l'identité et X est une colone de l'inverse de A)
                         
    return Ap

#---------------------------------------------------------------------------------------

                #-----------------------------------#
                #            Gauss Methods          #
                #-----------------------------------#

#Elimination_Gauss_to_solve_(S):AX=b --------------------------------------------------------------
def Gauss_eli(B,b):                                 #B:la matrice 

    A=B.copy()
    for i in range(len(A)-1):
        
        l=i+np.argmax(np.abs(A[i:,i]))          #l'indice de la ligne avec la valeur maximal dans la colone i
        A[[i,l]]=A[[l,i]]                       #On permute les lignes
        b[i],b[l]=b[l],b[i]
        pivot=A[i,i]
        if pivot==0:                            #Si le pivot est null le systeme n'admet pas de solution
            print("Erreur pivot null")
            return
        for j in range(i+1,len(A)):             #On parcoure les lignes d'indice superieur a i+1
            coef=A[j,i]/pivot
            A[j,:]=A[j,:]-coef * A[i,:]
            b[j]=b[j]-coef * b[i]
        
    X=np.zeros(len(b))
    for k in range(len(b)-1,-1,-1):
        X[k] = (b[k]- A[k,k+1:]@X[k+1:])/A[k,k]
    return X
        
        
#Elimination_Gauss_to_calculate_Inverse_Matrix------------------------------------------------------------

def Gauss_inv(B):
    
    A=B.copy() ; Ap=np.identity(len(B))                         #Ap va contenir l'inverse de la matrice A
    
    for i in range(len(A)-1):
        
        l=i+np.argmax(np.abs(A[i:,i]))
        A[[i,l]]=A[[l,i]] ; Ap[[i,l]]=Ap[[l,i]]                       #On permute les lignes
        pivot=A[i,i]
        
        if pivot==0:
            print("Erreur pivot null") ; return                 #Si le pivot est null la matrice n'est pas inversible
            
        for j in range(i+1,len(A)):
            if A[j,i]==0 : continue                             #Si un coeficient est au dessou du pivot de la colonne i il n'y a pas besoin de faire l'operation
            
            coef=pivot/A[j,i]
            A[j,:]=A[j,:]*coef - A[i,:]                         #Ici la matrice est triangulaire superieur
            Ap[j,:]=Ap[j,:]*coef - Ap[i,:]                      #On répete les même opérations pour la matrice Ap
        
    for i in range(len(A)-1,-1,-1):
        pivot=A[i,i]
        
        for j in range(i-1,-1,-1):
            if A[j,i]==0 : continue
            
            coef=pivot/A[j,i]
            A[j,:]=A[j,:]*coef - A[i,:]                         #Ici la matrice est triangulaire inferieure donc diagonale
            Ap[j,:]=Ap[j,:]*coef - Ap[i,:]                          

        A[i,:]=A[i,:]/pivot                                     #On divise par le pivot pour avoir A une matrice identité
        Ap[i,:]=Ap[i,:]/pivot

    return Ap

#----------------------------------------------------------------

