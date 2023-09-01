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

#  Function for plotting the beam diameter
def BeamExpander(lam0,w0,d0,d1,d2,f1,f2,f3,npoint):
    # this function aim to produce a plot of a gausiann beam that passes thru three thin lenses, the approach used is tho compute the complex beam parameter q and propagate that thru air and lenses, then compute the diameter and show a plot
    # lam0      wavelength considered                           [mm]
    # w0        initial beam waist                              [mm]
    # d0        from initial waist to first thin lens           [mm]
    # d1        between first and second thin lenses            [mm]
    # d2        between second and third thin lenses            [mm]
    # f1        focal length first lens                         [mm]
    # f2        focal length first lens                         [mm]
    # f3        focal length first lens                         [mm]
    # npoint    number of points of the plot (resolution)       [--]
    L1      =   d0+d1                           # second length position
    L2      =   d0+d1+d2                        # third length position
    zr0     =   np.pi*w0**2/lam0                # Rayleigh range
    q0      =   1j*zr0                          # complex beam parameter
    th0     =   lam0/np.pi/w0                   # divergence at the left of the first lens
    
    M1      =   f1/((d0-f1)**2+zr0**2)**0.5     # magnification first lens
    w1      =   M1*w0                           # weist of second beam
    zr1     =   zr0*M1**2                       # Rayleigh range right first lens
    th1     =   th0/M1                          # Divergence right first lens
    q1minus =   q0+d0                           # propagate left side first lens
    A,B,C,D =   (1,0,-1/f1,1)                   # matrix entries of first lens
    q1plus  =   (A*q1minus+B)/(C*q1minus+D)     # propagate right side first lens
    
    M2      =   f2/((d0-f2)**2+zr1**2)**0.5     # magnification second lens
    w2      =   M2*w1                           # weist of third beam
    zr2     =   zr1*M2**2                       # Rayleigh range right second lens
    th2     =   th1/M2                          # Divergence right second lens
    q2minus =   q1plus+d1                       # propagate left side second lens
    A,B,C,D =   (1,0,-1/f2,1)                   # matrix entries of second lens
    q2plus  =   (A*q2minus+B)/(C*q2minus+D)     # propagate right side second lens
    
    q3minus =   q2plus+d2                       # propagate left side third lens
    A,B,C,D =   (1,0,-1/f3,1)                   # matrix entries of third lens
    q3plus  =   (A*q3minus+B)/(C*q3minus+D)     # propagate right side third lens
    z_vect  =   np.linspace(0,L2+2*f3,npoint)   # points of z axis
    w       =   []                              # initialize beam diameter along z

    for z in z_vect:
        if      0<=z<d0:
            q   = q0+(z-0)                      # propagate q to z position
            aux = 1/q                           # auxilliary for diameter calculation
            w.append((-lam0/(np.pi*aux.imag))**0.5)  # beam diameter along z axis
        elif    d0<=z<L1:
            q   = q1plus+(z-d0)                 # propagate q to z position
            aux = 1/q                           # auxilliary for diameter calculation
            w.append((-lam0/(np.pi*aux.imag))**0.5)  # beam diameter along z axis
        elif    L1<=z<L2:
            q   = q2plus+(z-L1)                 # propagate q to z position
            aux = 1/q                           # auxilliary for diameter calculation
            w.append((-lam0/(np.pi*aux.imag))**0.5)  # beam diameter along z axis
        elif    L2<=z:
            q   = q3plus+(z-L2)                 # propagate q to z position
            aux = 1/q                           # auxilliary for diameter calculation
            w.append((-lam0/(np.pi*aux.imag))**0.5)  # beam diameter along z axis
    ymax=max(w)*1.1
    ymin=-0
    fig, axs=plt.subplots()
    fig.tight_layout()
    axs.plot(z_vect,w)
    axs.set_xlabel('$z$ [mm]')
    axs.set_ylabel('beam diameter [mm]')
    axs.grid(True,'both'); axs.set_ylim([ymin,ymax])
    #axs.vlines([d0, L1, L2],ymin,ymax,linestyles="dashdot",color="magenta")
    #plt.show()
    print(th0); print(w0);print(th1); print(w1)
    
if isNotebook(): # run widget only if in interactive mode
    get_ipython().run_line_magic('matplotlib', 'widget')

# %% second row of the table
BeamExpander(0.0006328,0.5,100,20,115.002,-10,10,100,500000)


