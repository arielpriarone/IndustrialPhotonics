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

#  Function for plotting the beam radius
def BeamExpander(lam0,w0,d0,d1,d2,f1,f2,f3,npoint=1000,fig=None,axs=None,plot=True):
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
    # returns:
    #   fig         figure handle of the plot
    #   axs         axis handle of the plot
    #   M           overall magnification of the system
    #   d3          location of the output waist w.r.t. last lens
    #   th3*10**5   angle of output beam *10^5
    #   w3          output waist
    #   w_end       beam radius at the end of the system
    
    L1      =   d0+d1                           # second length position
    L2      =   d0+d1+d2                        # third length position
    zr0     =   np.pi*w0**2/lam0                # Rayleigh range
    q0      =   1j*zr0                          # complex beam parameter
    th0     =   lam0/np.pi/w0                   # divergence at the left of the first lens
    
    M1      =   -f1/((d0-f1)**2+zr0**2)**0.5    # magnification first lens (- to keep it positive)
    w1      =   M1*w0                           # weist of second beam
    zr1     =   zr0*M1**2                       # Rayleigh range right first lens
    th1     =   th0/M1                          # Divergence right first lens
    q1minus =   q0+d0                           # propagate left side first lens
    A,B,C,D =   (1,0,-1/f1,1)                   # matrix entries of first lens
    q1plus  =   (A*q1minus+B)/(C*q1minus+D)     # propagate right side first lens
    
    dp      =   (d1-(f1+M1**2*(d0-f1)))         # from left beam waist to second lens (slide 125)
    M2      =   f2/((dp-f2)**2+zr1**2)**0.5     # magnification second lens
    w2      =   M2*w1                           # weist of third beam
    zr2     =   zr1*M2**2                       # Rayleigh range right second lens
    th2     =   th1/M2                          # Divergence right second lens
    q2minus =   q1plus+d1                       # propagate left side second lens
    A,B,C,D =   (1,0,-1/f2,1)                   # matrix entries of second lens
    q2plus  =   (A*q2minus+B)/(C*q2minus+D)     # propagate right side second lens
    
    dp      =   (d2-(f2+M2**2*(dp-f2)))         # from left beam waist to third lens
    M3      =   f3/((dp-f3)**2+zr2**2)**0.5     # magnification third lens
    w3      =   M3*w2                           # weist of third beam
    zr3     =   zr2*M3**2                       # Rayleigh range right second lens
    th3     =   th2/M3                          # Divergence right second lens
    q3minus =   q2plus+d2                       # propagate left side third lens
    A,B,C,D =   (1,0,-1/f3,1)                   # matrix entries of third lens
    q3plus  =   (A*q3minus+B)/(C*q3minus+D)     # propagate right side third lens
    z_vect  =   np.linspace(0,L2+2*f3,npoint)   # points of z axis
    w       =   []                              # initialize beam radius along z

    d3      =   f3+M3**2*(dp-f3)                # location of output waist w.r.t. last lens (if negative diverges already from lens position)
    for z in z_vect:
        if      0<=z<d0:
            q   = q0+(z-0)                  # propagate q to z position
            aux = 1/q                       # auxilliary for radius calculation
            w.append((-lam0/(np.pi*aux.imag))**0.5)  # beam radius along z axis
        elif    d0<=z<L1:
            q   = q1plus+(z-d0)             # propagate q to z position
            aux = 1/q                       # auxilliary for radius calculation
            w.append((-lam0/(np.pi*aux.imag))**0.5)  # beam radius along z axis
        elif    L1<=z<L2:
            q   = q2plus+(z-L1)             # propagate q to z position
            aux = 1/q                       # auxilliary for radius calculation
            w.append((-lam0/(np.pi*aux.imag))**0.5)  # beam radius along z axis
        elif    L2<=z:
            q   = q3plus+(z-L2)             # propagate q to z position
            aux = 1/q                       # auxilliary for radius calculation
            w.append((-lam0/(np.pi*aux.imag))**0.5)  # beam radius along z axis
        ymax=max(w)*1.1;    ymin=-0
    xmin=0.75*d0; xmax=L2+2*f3
    if plot:                                    # plot if needed, skip if not
        if fig == None or axs == None:
            fig, axs=plt.subplots()
        fig.tight_layout()
        axs.plot(z_vect,w,label=f'$d_1={d1}$; $d_2={d2}$')
        axs.set_xlabel('$z$ [mm]')
        axs.set_ylabel('beam radius [mm]')
        axs.set_ylim([ymin,ymax]); axs.set_xlim([xmin,xmax])
        axs.vlines([d0, L1, L2],ymin,ymax,linestyles="dashdot",color="magenta")
        axs.grid(True, 'major')
        axs.legend()
    return fig, axs, M1*M2*M3, d3, th3*10**5, w3, w[-1]
        
if isNotebook(): # run widget only if in interactive mode
    get_ipython().run_line_magic('matplotlib', 'widget')

# %% check result for all the row of the table
table=[(10,120.006),
       (20,115.002),
       (30,113.334),
       (40,112.500),
       (50,112.000)]
fig, ax = plt.subplots()
for (d1,d2) in table:
    fig, ax, Mg, dout, thout, wout, w_end = BeamExpander(lam0=0.0006328,w0=0.5,d0=100,d1=d1,d2=d2,f1=-10,f2=10,f3=100,npoint=1000,fig=fig,axs=ax)
    print(f'Mg={Mg}; dout={dout}; thout={thout}; wout={wout}')

tikzplotlib_fix_ncols(fig)
tikzplotlib.save('Assignment2/PLOT.tex',axis_width='0.9\\textwidth',axis_height ='7cm')

# %% check linearity
fig, axs=plt.subplots()
fig.tight_layout()
Mg_vect=[]
d2      =   112.5                       # note that this is optimized for 40x !!!
d1_vect =   np.linspace(38,42,500)      # try some d1
for d1 in d1_vect:
    Mg = BeamExpander(lam0=0.0006328,w0=0.5,d0=100,d1=d1,d2=d2,f1=-10,f2=10,f3=100,plot=False)[2]
    Mg_vect.append(Mg)
axs.plot(d1_vect,Mg_vect,label=f'$d_2={round(d2,3)}$')

axs.set_xlabel('$d_1$ [mm]')
axs.set_ylabel('Magnification [-]')
axs.grid(True, 'Both')
axs.legend()

tikzplotlib_fix_ncols(fig)
tikzplotlib.save('Assignment2/dvsm.tex',axis_width='0.9\\textwidth',axis_height ='7cm')


# %% check linearity - envelope
fig, axs=plt.subplots()
fig.tight_layout()
maximum=[]
d_max=[]
for d2 in np.linspace(120,112,11):
    LinRel = [[],[]] 
    for d1 in np.linspace(10,50,600):
        Mg = BeamExpander(lam0=0.0006328,w0=0.5,d0=100,d1=d1,d2=d2,f1=-10,f2=10,f3=100,plot=False)[2]
        LinRel[0].append(d1)
        LinRel[1].append(Mg)
    maximum.append(max(LinRel[1]))
    maxindex=LinRel[1].index(max(LinRel[1]))
    d_max.append(LinRel[0][maxindex])
    axs.scatter(LinRel[0],LinRel[1],marker='.',label=f'$d_2={round(d2,3)}$')
axs.plot(d_max,maximum,'k',label=f'peack values')
axs.set_xlabel('$d_1$ [mm]')
axs.set_ylabel('Magnification [-]')
axs.grid(True, 'Both')
axs.legend(ncol=4)

tikzplotlib_fix_ncols(fig)
tikzplotlib.save('Assignment2/LinApprox.tex',axis_width='0.9\\textwidth',axis_height ='7cm')

# %% check linearity - definition with useful beam radius
fig, axs=plt.subplots()
fig.tight_layout()
d1_vect =   np.linspace(10,50,10)      # try some d1
d2_vect =   np.linspace(120,112,5)     # try some d2

for d2 in d2_vect:
    Mg_vect =   []
    for d1 in d1_vect:
        W_out = BeamExpander(lam0=0.0006328,w0=0.5,d0=100,d1=d1,d2=d2,f1=-10,f2=10,f3=100,plot=False)[6]
        Mg_vect.append(W_out/0.5)
    axs.plot(d1_vect,Mg_vect,label=f'$d_2={round(d2,3)}$')
axs.set_xlabel('$d_1$ [mm]')
axs.set_ylabel('$\\frac{w(z=L_2+2\\cdot f_3)}{w_0}$ [-]')
axs.grid(True, 'Both')
axs.legend()
tikzplotlib_fix_ncols(fig)
tikzplotlib.save('Assignment2/Woutvsm.tex',axis_width='0.9\\textwidth',axis_height ='7cm')

# %% check result for all the row of the table
table=[(10,120.006),
       (20,115.002),
       (30,113.334),
       (40,112.500),
       (50,112.000)]
fig, ax = plt.subplots()
for (d1,d2) in table:
    fig, ax, Mg, dout, thout, wout, w_end = BeamExpander(lam0=0.0006328,w0=0.5,d0=100,d1=d1,d2=d2,f1=10,f2=-10,f3=100,npoint=1000,fig=fig,axs=ax)
    print(f'Mg={Mg}; dout={dout}; thout={thout}; wout={wout}')

tikzplotlib_fix_ncols(fig)
tikzplotlib.save('Assignment2/AssArrangment.tex',axis_width='0.9\\textwidth',axis_height ='7cm')


# %% try to optimize oyher expander
table=[(10,119.98),
       (20,115),
       (30,113.35),
       (40,112.5),
       (50,112)]
fig, ax = plt.subplots()
for (d1,d2) in table:
    fig, ax, Mg, dout, thout, wout, w_end = BeamExpander(lam0=0.0006328,w0=0.5,d0=100,d1=d1,d2=d2,f1=10,f2=-10,f3=120,npoint=1000,fig=fig,axs=ax)
    print(f'Mg={Mg}; dout={dout}; thout={thout}; wout={wout}')

tikzplotlib_fix_ncols(fig)
tikzplotlib.save('Assignment2/Mydesign.tex',axis_width='0.9\\textwidth',axis_height ='7cm')

# %%
plt.show()

