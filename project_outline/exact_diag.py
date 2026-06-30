import numpy as np
from quspin.basis import spinful_fermion_basis_general
from quspin.operators import hamiltonian
 
def build_hubbard(coords, hoppings, nnn, U, V, n_up, n_dn,
                  t=1.0, tprime=0.0, twist=(0.0,0.0)):
    N = len(coords)
    basis = spinful_fermion_basis_general(N, Nf=(n_up, n_dn))
    def ph(bv): tx,ty=twist; return np.exp(1j*(tx*bv[0]+ty*bv[1]))
    pm, mp = [], []
    for (i,j,bv) in hoppings:
        a=-t*ph(bv);      pm+=[[a,i,j]]; mp+=[[np.conj(a),j,i]]
    for (i,j,bv) in nnn:
        a=-tprime*ph(bv); pm+=[[a,i,j]]; mp+=[[np.conj(a),j,i]]
    static=[["+-|",pm],["-+|",mp],["|+-",pm],["|-+",mp],
            ["n|n",[[U,i,i] for i in range(N)]],
            ["nn|",[[V,i,j] for (i,j,_) in hoppings]],
            ["|nn",[[V,i,j] for (i,j,_) in hoppings]]]
    nc=dict(check_pcon=False,check_symm=False,check_herm=False)
    H=hamiltonian(static,[],basis=basis,dtype=np.complex128,**nc)
    E0,psi0=H.eigsh(k=1,which="SA")
    #E, psi = H.eigh() 
    #E0, psi0 = E[0], psi[:, 0]
    return E0[0], psi0[:,0], basis
    #return E0, psi0, basis

 
def test_two_site():
    for U in [0.,4.,8.]:
        E,_,_=build_hubbard([(0,0),(1,0)],[(0,1,(1,0))],[],U,0.,1,1)
        assert abs(E-0.5*(U-np.sqrt(U**2+16)))<1e-9
    print("oracle validation: PASS")


test_two_site()
