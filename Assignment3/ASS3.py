# %%
import matplotlib.pyplot as plt
import matplotlib.colors as color
import numpy as np
import matplotlib
import tikzplotlib
import scipy.integrate as integrate
matplotlib.use('Qt5Agg')

def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)

def isNotebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
    
def mlayer(n,L,lam0,theta=0,pol='TE'):
    # compute the input reflectivity for a multilayer stack of materials (first and last mediums infinetely thick)
    # imputs:
    #   n     : array-like,       array of reflactive indeces (constant, no chromatic aberration considered)
    #   L     : array-like,       array of thickneses of layers (len(L)=len(n)-2)
    #   lam0  : array-like,       wavelength in vacuum to be tested
    #   theta : float,            angle of incidence (deg), default=0 deg
    #   pol   : str,              "TE" if perpendicular polarization, "TM" for parallel polarization, default = "TE"
    # return:
    #   R     : array-like        input reflectivity
    
    if not isinstance(n, np.ndarray):        # convert to np.array if needed
        n   =   np.array(n)
    if not isinstance(L, np.ndarray):        # convert to np.array if needed
        L   =   np.array(L) 
    if not isinstance(lam0, np.ndarray):     # convert to np.array if needed
        lam0   =   np.array(lam0) 
    

    theta=theta*np.pi/180           # from deg to rad

    if not len(L)==(len(n)-2):      # check consistency of dimensions
        raise Exception("len(L)!=len(n)-2")
    
    Z0  =   120*np.pi               # vacuum impedence
    R=[]                            # initialize empty reflectivity
    for lam in lam0:                # iterate for all wavelength of interest
        k0      =   2*np.pi/lam     # k0 inital wave number
        k_vect  =   k0*(n[1:-1]**2-n[0]**2*np.sin(theta)**2)**0.5
        for ii in range(0,len(k_vect)):
            if np.imag(k_vect[ii])>0:        # fix k_z=-j*alpha
                k_vect[ii]=k_vect[ii].conjugate()
        
        # compute the z_infs
        match pol:
            case "TE":
                zinf = Z0/(n**2-n[0]**2*np.sin(theta)**2)**(0.5)
            case "TM":
                zinf = Z0/(n**2)*(n**2-n[0]**2*np.sin(theta)**2)**(0.5)
            case _ :
                raise Exception('pol is neither "TE" nor "TM"')
            
        for ii in range(0,len(zinf)):
            if np.imag(zinf[ii])<0:        # fix z_inf
                k_vect[ii]=k_vect[ii].conjugate()

        rho = (zinf[1:]-zinf[0:-1])/(zinf[1:]+zinf[0:-1])

        gamma=[rho[-1]]                   # initialize gamma array - last reflective coefficient is rho
        for r,k,l in zip(reversed(rho[:-1]),reversed(k_vect),reversed(L)):# reverse to propagate backward
            gamma.append((r+gamma[-1]*np.exp(-2j*k*l))/(1+r*gamma[-1]*np.exp(-2j*k*l)))
        R.append(abs(gamma[-1]**2*100))
    return R  # reflectivity [%]

def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)

def isNotebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter   

if isNotebook(): # run widget only if in interactive mode
    get_ipython().run_line_magic('matplotlib', 'widget')

# %% c
