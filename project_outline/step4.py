import numpy as np
from itertools import product
 
def twist_grid(n_theta, offset=0.5):
    th = 2*np.pi/n_theta*(np.arange(n_theta)+offset)
    return [(tx,ty) for tx,ty in product(th,th)]   # halve by time-reversal
 
def twist_average_pairing(solve_at_twist, twists):
    """solve_at_twist(theta) -> (psi, basis, Gup, Gdn)."""
    Vd=[]
    for theta in twists:
        psi,basis,Gup,Gdn = solve_at_twist(theta)
        Pd   = raw_dwave_pairing(psi, basis)     # <Delta_d^dag Delta_d>
        Pbar = uncorrelated_pairing(Gup, Gdn)    # this twist's Green's fns
        Vd.append(Pd - Pbar)                     # connected vertex part
    Vd=np.array(Vd)
    return Vd.mean(axis=0), Vd.std(axis=0)
