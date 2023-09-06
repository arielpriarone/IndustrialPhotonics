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
    return np.array(R)  # reflectivity [%]

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

# %% data of the problem
lam0    =   1070*10**(-9)               # design wavelength
nair    =   1                           # left medim refr index
nH      =   1.46                        # refr index layer H (silicon dioxide)
nL      =   1.38                        # refr index layer L (magnesium fluoride)
ng      =   1.5                         # right medim refr index (glass)
dH      =   lam0/4/nH                   # thickness layer H
dL      =   lam0/4/nL                   # thickness layer L

lam     =   np.linspace(lam0*0.80,lam0*1.2,1000)

# fig, axs=plt.subplots(3, 1, sharex=True, sharey=True)
# fig.tight_layout()
# ii = 0
# for ncouples in [15,30,80]:
#     for theta in np.linspace(0,45,5):
#         n=np.tile([nH, nL],ncouples)    # concatenate bilayers indeces
#         d=np.tile([dH, dL],ncouples)    # concatenate bilayers thicknesses
#         n=np.concatenate(([nair],n))    # add air medium
#         n=np.concatenate((n,[ng]))      # add glass medium

#         Rtm=mlayer(n,d,lam,theta,'TM')  # tm polarization reflectivity
#         Rte=mlayer(n,d,lam,theta,'TE')  # te polarization reflectivity

#         axs[ii].plot(lam/lam0,(Rte+Rtm)/2,label=f'$\\theta={round(theta,3)}^\\circ$')
#         print(f'$\\theta={round(theta,3)}^\\circ$')
#     axs[ii].grid(True, 'Both')
#     axs[ii].set_title(f'{ncouples} bilayers')
#     axs[ii].set_ylabel('$R$ [%]')
#     ii+=1
# axs[-1].set_xlabel('$\\lambda/\\lambda_0$')
# axs[-1].legend(loc='lower right')
# tikzplotlib_fix_ncols(fig)
# tikzplotlib.save('Assignment3/fig1.tex',axis_width='0.9\\textwidth',axis_height ='7cm')


# %% design of the dichroic mirror
# data of the problem
lam0_c  =   1070*10**(-9)/0.867         # design wavelength - compensated for the AOI
dH      =   lam0_c/4/nH                 # thickness layer H
dL      =   lam0_c/4/nL                 # thickness layer L
ncouples =  80                          # how many bilayers
theta   =   45                          # AOU - degrees
lam     =   np.linspace(lam0*0.9,lam0*1.1,1000) # around laser wavelength

fig, axs=plt.subplots()
fig.tight_layout()
n=np.tile([nH, nL],ncouples)    # concatenate bilayers indeces
d=np.tile([dH, dL],ncouples)    # concatenate bilayers thicknesses
n=np.concatenate(([nair],n))    # add air medium
n=np.concatenate((n,[ng]))      # add glass medium

Rtm=mlayer(n,d,lam,theta,'TM')  # tm polarization reflectivity
Rte=mlayer(n,d,lam,theta,'TE')  # te polarization reflectivity

# axs.plot(lam,(Rte),label=f'$TE$')
# axs.plot(lam,(Rtm),label=f'$TM$')
# axs.plot(lam,(Rte+Rtm)/2,label=f'$average$')
# axs.vlines([lam0],-5,105,colors='red',linestyles='dashdot')
# axs.grid(True, 'Both')
# axs.set_title(f'{ncouples} bilayers')
# axs.set_ylabel('$R$ [%]')
# axs.set_xlabel('$\\lambda$ [m]')
# axs.legend(loc='upper right')

# tikzplotlib_fix_ncols(fig)
# tikzplotlib.save('Assignment3/fig2.tex',axis_width='0.9\\textwidth',axis_height ='7cm')


# %% check reflectivity in visible range
# data of the problem
lam     =   np.linspace(380,700,1000)*10**(-9) # visible range

fig, axs=plt.subplots()
fig.tight_layout()

Rtm=mlayer(n,d,lam,theta,'TM')  # tm polarization reflectivity
Rte=mlayer(n,d,lam,theta,'TE')  # te polarization reflectivity

# axs.plot(lam,(Rte),label=f'$TE$')
# axs.plot(lam,(Rtm),label=f'$TM$')
# axs.plot(lam,(Rte+Rtm)/2,label=f'$average$')
# axs.grid(True, 'Both')
# axs.set_ylabel('$R$ [%]')
# axs.set_xlabel('$\\lambda$ [m]')
# axs.legend(loc='upper right')

# tikzplotlib_fix_ncols(fig)
# tikzplotlib.save('Assignment3/fig3.tex',axis_width='0.9\\textwidth',axis_height ='7cm')

#%% compute the irradiance for the focus spot
# first laser
M2  = 1.2                           # quality factor
f   = 0.2                           # focal length of lens  [m]
wl  = 0.012/2                       # collimated beam radius[m]
wf  = M2*lam0*f/np.pi/wl            # focused spot radius   [m]
P   = 1000                          # laser power           [w]
r   = np.linspace(-2*wf,2*wf,200)   # radial distance from the focus point [m]
I   = 2*P/np.pi/wf*np.exp(-2*r**2/(wf**2)) # irradiance distribution [w/m^2]

fig, axs=plt.subplots()
fig.tight_layout()
axs.plot(r*10**3,I/10**7,label='first laser') # converted in kw/cm^2

# second laser
BPP = 5/10**6                       # beam parameter product [mm*rad]
M2  = BPP/lam0*np.pi                # quality factor
wf  = M2*lam0*f/np.pi/wl            # focused spot radius   [m]
P   = 5000                          # laser power           [w]

r   = np.linspace(-2*wf,2*wf,200)   # radial distance from the focus point [m]
I   = 2*P/np.pi/wf*np.exp(-2*r**2/(wf**2)) # irradiance distribution [w/m^2]

axs.plot(r*10**3,I/10**7,label='second laser')           # converted in kw/cm^2
axs.grid(True, 'Both')
axs.set_ylabel('Irradiance [kW/cm]')
axs.set_xlabel('radial position $r$ [mm]')
axs.legend()

axs.grid(True, 'Both')
axs.set_ylabel('Irradiance [kW/cm]')
axs.set_xlabel('radial position $r$ [mm]')
axs.legend()


tikzplotlib_fix_ncols(fig)
tikzplotlib.save('Assignment3/fig4.tex',axis_width='0.9\\textwidth',axis_height ='7cm')


plt.show()
# %%
