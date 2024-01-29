# Calcul de la distribution des valeurs propres des matrices de l'ensemble unitaire gaussien (GUE).
# Comparaison du resultat avec la loi du demi-cercle de Wigner.

import numpy as np
import matplotlib.pyplot as plt

def W_SCircle(x): # Wigner semicircle law
    n = np.size(x)
    rez = np.zeros(n)
    for i in range(n):
        if 4-x[i]**2 >= 0:
            rez[i] = np.sqrt(4 - x[i]**2)/(2*np.pi)            
    return rez
    
def dag(m):
    return np.transpose(np.conjugate(m))    

N = 200 # nombre de tirages (on etudie la convergence en distribution)
m = 50 # taille de la matrice
evals = np.array([]) # contiendra N*m elements (valeurs propres des matrices aleatoires de taille mxm sur N tirages)
for i in range(N):
    # cas GUE
    g = np.random.normal(0,1,(m,m))+1j*np.random.normal(0,1,(m,m))
    h = (g + dag(g))/2. # h matrice hermitienne
    # cas GOE
    #g = np.random.normal(0,1,(m,m)) 
    #h = (g + g.T)/np.sqrt(2.) # h reelle symetrique (cas GOE)

    evals = np.append(evals, np.linalg.eigvalsh(h)) # valeurs propres de h
    
    
# Mise a l'echelle du spectre : eval <- eval/sqrt(n*sigma²) ou
# n: taille de la matrice et sigma: variance des elements non diagonaux de la matrice
evals = np.sort(np.real(evals))
evals = evals/np.sqrt(m) 

x = np.linspace(-3, 3, num=200, endpoint=True)
proba = W_SCircle(x)


plt.figure(1)
plt.plot(x, proba, 'r-', linewidth=2, label='Wigner semicircle law')
plt.hist(evals, bins=35, range=(-2.3, 2.3), density=True, facecolor='green', alpha=0.3, edgecolor = 'white')
plt.legend(fontsize=13, loc='upper right')
plt.xlabel(r'$\lambda/\sqrt{n \sigma^2}$', size=14)
plt.ylabel(r'$\langle \rho \rangle$', size=20) 
plt.ylim(ymax=0.37)
plt.xlim(-2.5, 2.5)
plt.title('Spectral density of GUE (n=%s)\n average over %s samples'%(m, N), size=16)
plt.tight_layout()
plt.show()

# On obtient le même resultat pour le tirage d'une seule matrice h (N=1) pourvu que m soit grand.
# Une seule realisation d'une grande matrice est representative de l'ensemble,
# parce que les fluctuations diminuent avec la taille de la matrice.
# On observe la convergence vers la loi du demi-cercle egalement pour GOE et GSE.
#
# En pratique, il vaut mieux travailler avec des matrices pas trop grandes et en faire plusieurs tirages.
