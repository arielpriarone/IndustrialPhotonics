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

def mLayerReflectivity(n,L,lam0,theta=0,pol='TE'):
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

def BeamExpander(lam0,w0,d0,d1,d2,f1,f2,f3,npoint=1000,fig=None,axs=None,plot=True,MS=1):
    # this function aim to produce a plot of a gausiann beam that passes thru three thin lenses, the approach used is tho compute the complex beam parameter q and propagate that thru air and lenses, then compute the radius and show a plot
    # parameters:
    #   lam0        wavelength considered                           [mm]
    #   w0          initial beam waist                              [mm]
    #   d0          from initial waist to first thin lens           [mm]
    #   d1          between first and second thin lenses            [mm]
    #   d2          between second and third thin lenses            [mm]
    #   f1          focal length first lens                         [mm]
    #   f2          focal length first lens                         [mm]
    #   f3          focal length first lens                         [mm]
    #   npoint      number of points of the plot (resolution)       [--]
    #   fig=None    figure handle
    #   axs=None    axis handle
    #   plot=True   T=generate plot; F=generate only the data
    #   Ms          quality factor of the beam (ref slide 05/177)   [--]
    # returns:
    #   fig         figure handle of the plot
    #   axs         axis handle of the plot
    #   M           overall magnification of the system             [--]
    #   d3 (s3II)   location of the output waist w.r.t. last lens   [mm]
    #   th3*10**5   angle of output beam *10^5                      [mrad*100]
    #   w3          output waist (real beam)                        [mm]
    #   w_end       beam radius at the end of the system (real beam)[mm]
    
    L1      =   d0+d1                           # second length position
    L2      =   d0+d1+d2                        # third length position
    zr0     =   np.pi*w0**2/lam0                # Rayleigh range
    q0      =   1j*zr0                          # complex beam parameter
    th0     =   lam0/np.pi/w0                   # divergence at the left of the first lens
    M       =   MS**0.5                         # sqrt of quality factor
    
    M1      =   f1/((d0-f1)**2+zr0**2)**0.5     # magnification first lens
    M1      =   abs(M1)
    w1      =   M1*w0                           # weist of second beam
    zr1     =   zr0*M1**2                       # Rayleigh range right first lens
    th1     =   th0/M1                          # Divergence right first lens
    q1minus =   q0+d0                           # propagate left side first lens
    A,B,C,D =   (1,0,-1/f1,1)                   # matrix entries of first lens
    q1plus  =   (A*q1minus+B)/(C*q1minus+D)     # propagate right side first lens
    
    s1I     =   d0                              # distance from first lens and waist (on the left)
    s1II    =   f1+M1**2*(s1I-f1)               # distance from first lens and waist (on the right)
    S2I     =   (d1-s1II)                       # distance from second lens and waist (on the left)
    M2      =   f2/((S2I-f2)**2+zr1**2)**0.5    # magnification second lens
    M2      =   abs(M2)
    w2      =   M2*w1                           # weist of third beam
    zr2     =   zr1*M2**2                       # Rayleigh range right second lens
    th2     =   th1/M2                          # Divergence right second lens
    q2minus =   q1plus+d1                       # propagate left side second lens
    A,B,C,D =   (1,0,-1/f2,1)                   # matrix entries of second lens
    q2plus  =   (A*q2minus+B)/(C*q2minus+D)     # propagate right side second lens
    
    s2II    =   f2+M2**2*(S2I-f2)               # distance from second lens and waist (on the right)
    S3I     =   (d2-s2II)                       # distance from third lens and waist (on the left)
    M3      =   f3/((S3I-f3)**2+zr2**2)**0.5    # magnification third lens
    M3      =   abs(M3)
    w3      =   M3*w2                           # weist of third beam
    zr3     =   zr2*M3**2                       # Rayleigh range right second lens
    th3     =   th2/M3                          # Divergence right second lens
    q3minus =   q2plus+d2                       # propagate left side third lens
    A,B,C,D =   (1,0,-1/f3,1)                   # matrix entries of third lens
    q3plus  =   (A*q3minus+B)/(C*q3minus+D)     # propagate right side third lens
    z_vect  =   np.linspace(0,L2+2*f3,npoint)   # points of z axis
    w       =   []                              # initialize beam radius along z
    w_r     =   []                              # this will be the real beam (not gaussian)

    S3II    =   f3+M3**2*(S3I-f3)               # location of output waist w.r.t. last lens
    for z in z_vect:
        if      0<=z<d0:
            q   = q0+(z-0)                  # propagate q to z position
            aux = 1/q                       # auxilliary for radius calculation
            w.append((-lam0/(np.pi*aux.imag))**0.5)  # beam radius along z axis
            w_r.append(M*w0*(1+((lam0*z)/(np.pi*w0**2))**2)**0.5)           # real beam radius along z axis
        elif    d0<=z<L1:
            q   = q1plus+(z-d0)             # propagate q to z position
            aux = 1/q                       # auxilliary for radius calculation
            w.append((-lam0/(np.pi*aux.imag))**0.5)  # beam radius along z axis
            w_r.append(M*w1*(1+((lam0*(z-d0-s1II))/(np.pi*w1**2))**2)**0.5)  # real beam radius along z axis
        elif    L1<=z<L2:
            q   = q2plus+(z-L1)             # propagate q to z position
            aux = 1/q                       # auxilliary for radius calculation
            w.append((-lam0/(np.pi*aux.imag))**0.5)  # beam radius along z axis
            w_r.append(M*w2*(1+((lam0*(z-L1-s2II))/(np.pi*w2**2))**2)**0.5)  # real beam radius along z axis
        elif    L2<=z:
            q   = q3plus+(z-L2)             # propagate q to z position
            aux = 1/q                       # auxilliary for radius calculation
            w.append((-lam0/(np.pi*aux.imag))**0.5)  # beam radius along z axis
            w_r.append(M*w3*(1+((lam0*(z-L2-S3II))/(np.pi*w3**2))**2)**0.5)  # real beam radius along z axis (if negative, the beam is already diverging)
        ymax=max(w)*1.1;    ymin=-0
    xmin=0.75*d0; xmax=L2+2*f3
    if plot:                                    # plot if needed, skip if not
        if fig == None or axs == None:
            fig, axs=plt.subplots()
        fig.tight_layout()
        axs.plot(z_vect,w,label=f'$d_1={d1}$; $d_2={d2}$')
        if (MS > 1):
            axs.fill_between(z_vect, w_r, w, alpha=0.2) # if it's a gaussian beam, no need to plot the shade
        axs.set_xlabel('$z$ [mm]')
        axs.set_ylabel('beam radius [mm]')
        axs.set_ylim([ymin,ymax]); axs.set_xlim([xmin,xmax])
        axs.vlines([d0, L1, L2],ymin,ymax,linestyles="dashdot",color="magenta")
        axs.grid(True, 'major')
        axs.legend()
    return fig, axs, M1*M2*M3, S3II, th3*10**5, w3*M, w_r[-1]

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

Rtm=mLayerReflectivity(n,d,lam,theta,'TM')  # tm polarization reflectivity
Rte=mLayerReflectivity(n,d,lam,theta,'TE')  # te polarization reflectivity

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

Rtm=mLayerReflectivity(n,d,lam,theta,'TM')  # tm polarization reflectivity
Rte=mLayerReflectivity(n,d,lam,theta,'TE')  # te polarization reflectivity

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



# %% study the first laser with three heads
w0  =   14/2/10**6                      # approximate the waist with the core radius [um]
MS  =   1.2                             # quality factor
(d0,d1,d2)  = (100,10,10)               # spacing configuration
(f1,f2,f3)  = (100,float('inf'),125)    # lenses configuration
fig, axs, Mag, d3, div, w_out, w_end = BeamExpander(lam0,w0,d0,d1,d2,f1,f2,f3,npoint=1000,MS=MS)


plt.show()
