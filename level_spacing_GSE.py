'''
Les matrices de l'ensemble symplectique gaussien (GSE) sont quaternioniques hermitiennes.
Leurs entrees sont des variables aleatoires gaussiennes iid (independantes identiquement distribuees).
Le programme:
- calcule la distribution d'ecarts entre les valeurs propres d'une telle matrice.
- compare le resultat (moyennage sur plusieurs tirages de matrices)
  avec l'expression analytique : 'Wigner surmise' pour GSE
'''

import numpy as np
import matplotlib.pyplot as plt

def dag(m):
    return np.transpose(np.conjugate(m))
    
def p_4(s):
    return (2**18)*(s**4)*np.exp(-64*s**2/(9*np.pi))/((3**6)*(np.pi**3)) # Wigner surmise pour GSE (beta = 4)

N = 2000 # nombre de tirages
n = 8 # taille de la matrice 
spacing = np.array([])

# {e1, e2, e3} une base formee par les matrices de Pauli
e1=np.array([[0,1.],[1., 0]])
e2=np.array([[0,-1j],[1j,0]])
e3=np.array([[1, 0],[0, -1]])
I2=np.eye(2) 

for i in range(N):    

    # variante 1
    x = np.random.normal(0,1,(n,n)) + 1j*np.random.normal(0,1,(n,n))
    y = np.random.normal(0,1,(n,n)) + 1j*np.random.normal(0,1,(n,n))
    q = np.vstack((np.hstack((x, y)), np.hstack((-np.conjugate(y), np.conjugate(x))))) # q matrice quaternionique

    '''
    # variante 2 (un peu moins rapide)
    # idee: q matrice n*n quaternionique => q=q0*I2 + 1j(q1*e1 + q2*e2 + q3*e3) 
    # ou "*" designe le produit tensoriel et q0, q1, q2, q3 sont des v.a. reelles (iid selon N(0,1))
    # e1, e2, e3 sont les matrices de Pauli
    
    # Tableaux de coefficients reels
    q0=np.random.normal(0,1,(n,n))
    q1=np.random.normal(0,1,(n,n))
    q2=np.random.normal(0,1,(n,n))
    q3=np.random.normal(0,1,(n,n))    

    q = np.kron(q0, I2) + 1j*(np.kron(q1, e1) + np.kron(q2, e2) + np.kron(q3, e3)) # q matrice quaternionique
    '''    
    
    h = (q + dag(q))/np.sqrt(2.) # h matrice hermitienne symplectique
    
    ev = np.linalg.eigvalsh(h)   # valeurs propres de h (compose de doublons [v1, v1, v2, v2, ...])
    e = np.sort(np.real(ev[::2]))# spectre [v1, v2, ...] ordonne
    s = np.diff(e)               # espacement entre les valeurs propres 
    s = s/np.mean(s)
    spacing = np.append(spacing, s)
    
x = np.linspace(0, np.max(spacing), 200, endpoint=True) # grille pour l'expression analytique

plt.figure(1)
plt.hist(spacing, bins='rice', density=True, color='green', alpha=0.3, edgecolor = 'white')
plt.plot(x, p_4(x), 'r-', linewidth=1.5, label='Wigner surmise')
plt.xlim(0,3.1)
plt.legend(fontsize=14, loc=0)
plt.xlabel(r'$s$', size=20)
plt.ylabel(r'$p(s)$', size=20)
plt.title('Level spacing distribution of GSE (n=%s, %s samples)'%(n, N), size=13)
plt.show()

# Plus n est grand moins la convergence est bonne (pic de l'histogramme se deplace vers
# la gauche et la "queue" est moins evanaissante).
