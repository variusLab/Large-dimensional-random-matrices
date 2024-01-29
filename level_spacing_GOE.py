'''
Les matrices de l'ensemble orthogonal gaussien (GOE) sont symetriques.
Leurs entrees sont des variables aleatoires gaussiennes reelles
independantes identiquement distribuees. Le programme:
- calcule la distribution d'ecarts entre les valeurs propres d'une telle matrice.
- compare le resultat (moyennage sur plusieurs tirages de matrices)
  avec l'expression analytique : 'Wigner surmise' pour GOE
'''

import numpy as np
import matplotlib.pyplot as plt

def expected_p(s):
    return 0.5*np.pi*s*np.exp(-np.pi*s*s/4.) # Wigner surmise pour GOE (beta = 1)

N = 200 # nombre de tirages aleatoires
n = 50  # taille de la matrice
spacing = np.array([])

for i in range(N):
    U = np.random.normal(0,1,(n,n)) # matrice nxn gaussienne reelle
    # for i in range(n): U[i,i] = np.random.normal(0,2)
    U = (U + U.T)/np.sqrt(2.)   # U matrice symetrique
    evals = np.linalg.eigvalsh(U)  # valeurs propres de U (facultatif: UPLO='U')
    evals = np.sort(np.real(evals))
    s = np.diff(evals) # ecart entre les valeurs propres consecutives
    s = s/np.mean(s)
    spacing = np.append(spacing, s)
    
x = np.linspace(0, np.max(spacing), num=200, endpoint=True) # grille plus fine pour l'expression analytique

plt.figure(1)
plt.hist(spacing, bins='rice', density=True, color='green', alpha=0.3, edgecolor = 'white') # bins 16 ok
plt.plot(x, expected_p(x), 'r-', linewidth=1.5, label='Wigner surmise')
plt.xlim(0,3.5)
plt.legend(fontsize=14, loc=0)
plt.xlabel(r'$s$', size=20)
plt.ylabel(r'$p(s)$', size=20)
plt.title('Level spacing distribution, GOE (n=%s)'%n, size=16)
plt.show()
