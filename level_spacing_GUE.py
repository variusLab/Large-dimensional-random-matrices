'''
Les matrices de l'ensemble unitaire gaussien (GUE) sont hermitiennes.
Leurs entrees sont des variables aleatoires gaussiennes complexes (partie reelle et imaginaire ~ N(0,1))
independantes identiquement distribuees. Le programme:
- calcule la distribution d'ecarts entre les valeurs propres d'une telle matrice.
- compare le resultat (moyennage sur plusieurs tirages de matrices)
  avec l'expression analytique : 'Wigner surmise' pour GUE
'''

import numpy as np
import matplotlib.pyplot as plt

def dag(m):
    return np.transpose(np.conjugate(m))

def expected_p(s):
    return 32*(s**2)*np.exp(-4*s*s/np.pi)/(np.pi**2) # Wigner surmise pour GUE (beta = 2)

N = 200 # nombre de tirages aleatoires
n = 50  # taille de la matrice
spacing = np.array([])
for i in range(N):    
    U = np.random.normal(0,1,(n,n))+1j*np.random.normal(0,1,(n,n)) # matrice nxn complexe gaussienne
    U = (U + dag(U))/np.sqrt(2.)   # U matrice hermitienne
    evals = np.linalg.eigvalsh(U)  # valeurs propres de U
    evals = np.sort(np.real(evals))
    s = np.diff(evals) # espacement entre les valeurs propres
    s = s/np.mean(s)
    spacing = np.append(spacing, s)
    
x = np.linspace(0, np.max(spacing), num=200, endpoint=True) # grille plus fine pour l'expression analytique

plt.figure(1)
plt.hist(spacing, bins='rice', range=(0, 3), density=True, color='green', alpha=0.3, edgecolor = 'white')
plt.plot(x, expected_p(x), 'r-', linewidth=1.5, label='Wigner surmise')
plt.xlim(0,3.1)
plt.legend(fontsize=14, loc=0)
plt.xlabel(r'$s$', size=20)
plt.ylabel(r'$p(s)$', size=20)
plt.title('Level spacing distribution, GUE (n=%s)'%n, size=16)
plt.show()
