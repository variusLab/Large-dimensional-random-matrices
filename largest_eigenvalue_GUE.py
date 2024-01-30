''' Ce programme : 
 - resout l'equation diff. de Painleve II : solution q(t)
 - calcule la fonction f_2(x) (loi de Tracy-Widom pour GUE, beta = 2)
 - effectue N tirages de matrices hermitiennes (n,n) aleatoires et calcule leurs valeurs propres maximales
 - trace la ditribution des val. propres maximales (normalisées) et la compare a la loi de Tracy-Widom
 - (pour verification technique: trace la solution q(t) trouvee et verifie la coherence avec la condition limite imposee)

Distribution de Tracy-Widom pour GUE:
f_2(x) := F_2'(x)
ou F_2(x) := exp\left( -\int_x ^{+\infty} (t-x)q^2(t)dt \right)
ou q(t) est la solution de l'equation differentielle de Painleve II:
q''(t) = t*q(t)+2*q^3(t)
avec la condition asymtotique : q(t) tend vers la fonction d'Airy lorsque t tend vers + l'infini

remarque: F_2(x) := limite de Proba( (\lambda_{max} - \sqrt{4n})*n^{1/6} <= x ) lorsque n tend vers + l'infini
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, simpson, solve_ivp
from scipy.special import airy

def dag(m):      # conjugue hermitien de la matrice m
    return np.transpose(np.conjugate(m))

def funct(t, y): # vecteur dy/dt a deux composantes, avec y=(q, dq/dt), q est la sol de eq. diff. de Painleve II  
    return [y[1], t*y[0] + 2.*y[0]**3]

N = 3500    # nombre de tirages de matrices 
n = 10      # taille de la matrice 

tmin = -17. # borne inferieure de la grille pour la fonction q(t) (tmin = -8 ok)
tmax = 6.   # jour le role de +infinite (q(tmax)=Ai(tmax)) (tmax = 5 ok)
dt = 0.01   # pas 
t = np.arange(tmin, tmax+dt, dt) # grille pour la fonction q(t)

m = np.size(t)
x = t[0: int(4.*m/5)] # grille pour la fonction f_2(x)

l = np.size(x)
integ = np.zeros(l) # valeurs de l'integrale \int_x ^{+\infty} (t-x)q^2(t)dt en chaque point x, initialisation
max_eval = np.zeros(N) # valeurs propres maximales, initialisation

y0 = airy(tmax)[0:2]  # conditions au bord: q(t -> +infty)=Ai(t), dq/dt(t -> infty)=derivee de Ai(t)
trevesed = t[::-1]

solve_ivp_output = solve_ivp(funct, (t[-1],t[0]), y0, dense_output=True) # integration de +infty a tmin (t etant inverse)
q = solve_ivp_output.sol(trevesed)[0]   # solution dans le sens de t decroissant
q = np.flip(q) # solution dans le sens de t croissant

for i in range(l):
    g = (t[i:]-x[i])*q[i:]**2    # integrande 
    integ[i] = simpson(g, t[i:]) # integration de g selon t

F2 = np.exp(-integ)      # fonction F_2(x)
f2 = np.gradient(F2)     # fonction f_2(x)=F_2'(x)
f2 = f2/simpson(f2, x)   # normalisation


for i in range(N): # Tirage de N matrices aleatoires hermitiennes (n,n)
    U = np.random.normal(0,1,(n,n))+1j*np.random.normal(0,1,(n,n))
    U = (U + dag(U))/2. # U est hermitienne    
    evals = np.linalg.eigvalsh(U) # valeurs propres de U
    max_eval[i] = np.max(evals)   # on retient la valeur propre maximale de U

max_eval = (max_eval-2*np.sqrt(n))*(n**(1./6)) # normalisation (d'apres un theoreme)

# ---------- Plots --------------------------------------
plt.figure(1)
plt.hist(max_eval, bins='rice', density=True, color='green', alpha=0.3, edgecolor = 'white')
plt.plot(x, f2, 'r-', linewidth=1.5, label='Tracy-Widom law')
plt.legend(fontsize=13, loc='upper right')
plt.xlabel(r'$\left(\lambda_{max}-2\sqrt{n}\right)n^{1/6}$', size=13)
plt.title('Distribution of the largest eigenvalue of GUE (n=%s, %s samples)'%(n, N), size=12, pad=10)
plt.xlim(-5.2, 2.2)
#plt.tight_layout()

plt.figure(2)
plt.plot(t, q, label='q(x)', linestyle=':', color='blue')
plt.plot(x, airy(x)[0], label='Ai(x)')
plt.legend(loc='best')
plt.grid()
#plt.xlim(-5,2)

plt.show()

# remarque: la version
# sol = odeint(funct, y0, trevesed) # integration de infty a tmin (t etant inverse) avec signature funct(y, t)
# q = sol[:,0]   # dans le sens de t décroissant
# q = np.flip(q) # dans le sens de t croissant
# produit un warning. A reexaminer la version avec odeint.
