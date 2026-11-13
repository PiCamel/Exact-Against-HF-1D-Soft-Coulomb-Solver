# Everything is in Hartree atomic units (hbar = m_e = e = 1): energies in Ha, lengths in bohr.
# All interactions use the SAME softening a = 1, so the exact and Kohn-Sham solvers describe the same physical system.

import numpy as np
import scipy as s
import scipy.sparse as sp
import scipy.sparse.linalg as spla

A_SOFT = 1.0 # softening of Coulomb, used everywhere


def make_grid(L, h): # uniform grid from -L to L with spacing h, odd N so x = 0 is on the grid
    N = int(round(2*L/h)) + 1
    x = np.linspace(-L, L, N, endpoint = True)
    return x, N


# Exact Solver Functions

def w(x, a=A_SOFT):
    return 1.0 / np.sqrt(x**2 + a**2)

def v_1atom(x, Z, a=A_SOFT):# just realized it is better to keep z separate as compared to within the w function
    return -Z * w(x, a)

def v_2atom(x, z, a=A_SOFT, R = 3.5): # nuclei at -R/2 and +R/2, so the internuclear distance is R
    return -z*w(x+R/2,a)- z*w(x-R/2,a)

def kinetic_matrix(N, h):
    return sp.diags([-0.5 / h**2 * np.ones(N - 1),
                     1.0 / h**2 * np.ones(N),
                     -0.5 / h**2 * np.ones(N - 1)], [-1, 0, 1], format="csc")# csc is good for storage as we dont have to store the entire matrix
                     # I have used finite difference operator here

def H_two_electron(x, h, v, a=A_SOFT):
    N = len(x)
    T = kinetic_matrix(N, h)# basically our kinetic energy operator on the combined sci vector
    I = sp.identity(N, format="csc")
    X1, X2 = np.meshgrid(x, x, indexing="ij")# meshgrid is an oddly weird function
    V = np.repeat(v, N) + np.tile(v, N) + w(X1 - X2, a).ravel()# basically converting the 2D grid of all possible coordinates into a single vector, very concisely written indeed
    return (sp.kron(T, I) + sp.kron(I, T) + sp.diags(V)).tocsc()


# Kohn-Sham Solver Functions

a_KS = A_SOFT # softening of Coloumb

def w_KS(x, a=a_KS): # the 1/|x| term equivalent in the soft-Coulomb case.
    return 1.0 / np.sqrt(x**2 + a**2)

def v_ext(Z, x, a=a_KS):
  return -Z*w_KS(x,a)

def v_ext_H2(x, R, a=a_KS): # two nuclei of charge 1 at -R/2 and +R/2
  return -w_KS(x - R/2, a) - w_KS(x + R/2, a)

def T_1e(N, h): # Tridiagonal matrix of KE
  return sp.diags([-0.5/h**2*np.ones(N-1),
                   1.0/h**2*np.ones(N),
                   -0.5/h**2*np.ones(N-1)],
                    [-1, 0, 1], format="csc")

def W_grid(N, h, x, a=a_KS): # creates a matrix of all i-j pair distances' w(x)
  W = 1.0/np.sqrt((x[:,None] - x[None,:])**2 + a**2)
  # W is an N x N matrix
  return W

def V_H(n, h, x, a=a_KS): # Hartree Potential
  W = W_grid(len(x), h, x, a) # W matrix formed, now to dot it with n
  return h*(W @ n)

def one_electron(V, h, N): # lowest eigenpair of T + diag(V): one electron, no other electron around
  T = T_1e(N, h)
  # select i selects from indices, and select_range chooses which indices. 0,0 is the lowest of each.
  min_eps, min_phi = s.linalg.eigh_tridiagonal(T.diagonal()+V, T.diagonal(k=1), select="i", select_range=(0,0))
  min_phi = min_phi[:,0]/np.sqrt(h)
  return min_eps[0], min_phi


# Grid convergence tools

def richardson(E_h, E_2h): # Richardson extrapolation for an O(h^2) method
  # E(h) = E0 + c h^2 and E(2h) = E0 + 4 c h^2, so E0 = E(h) + (E(h) - E(2h))/3
  return E_h + (E_h - E_2h)/3

def convergence_order(E_4h, E_2h, E_h): # observed order p from three spacings 4h, 2h, h
  # the error shrinks by 2^p each time h is halved, so p = log2[(E_2h - E_4h)/(E_h - E_2h)]
  return np.log2((E_2h - E_4h)/(E_h - E_2h))
