# import the libraries
import numpy as np 
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy import sqrt, cos, sin
from matplotlib.widgets import Slider, Button
import matplotlib
matplotlib.use('TkAgg')
## constants ##
a = 1.419      # lattice parameter
Minit = 0      # initial value of the mass term
t1 = 2.9         # NN hopping
phi = 0          # initial phase
t2=t1*0.02  # the hopping

### TB Equation ###
def Tight_binding(x,y,M,phi):
    h0 = np.tensordot(np.array([[1,0],[0,1]]), 2*t2*cos(phi)*(cos(sqrt(3)/2*x*a+3/2*y*a)+cos(-sqrt(3)/2*x*a+3/2*y*a)+np.cos(a*x*sqrt(3))),axes = 0).T
    hx = np.tensordot(np.array([[0,1],[1,0]]), -t1*(cos(1/2*y*a+x*a*sqrt(3)/2)+cos(1/2*y*a-x*a*sqrt(3)/2)+cos(a*y)),axes = 0).T
    hy = np.tensordot(np.array([[0,1j],[-1j,0]]).T,t1* (sin(1/2*y*a+x*a*sqrt(3)/2)+np.sin(1/2*y*a-x*a*sqrt(3)/2)-np.sin(a*y)),axes = 0).T
    hz= np.tensordot(np.array([[1,0],[0,-1]]),(np.zeros(len(x))+M)-2*t2*np.sin(phi)*(np.sin(-sqrt(3)/2*x*a-3/2*y*a)+np.sin(-sqrt(3)/2*x*a+3/2*y*a)+np.sin(a*x*sqrt(3))),axes=0).T
    H=h0+hx+hy+hz
    E=LA.eigvalsh(H)
    return E

## reciprocal lattice ##
cell = sqrt(3)*a*np.array([[-1/2,sqrt(3)/2],[1/2,sqrt(3)/2,]])
icell = 2*np.pi*np.linalg.inv(cell).T
b1=icell[0,:]
b2=icell[1,:]   
K = [b1[0]*1/3+b2[0]*2/3,b1[1]*1/3+b2[1]*2/3]
M = [b1[0]*1/2+b2[0]*1/2,b1[1]*1/2+b2[1]*1/2]

## Line Segments ##
P1 = np.linspace(0,sqrt(((K[0]-0)**2+(K[1]-0)**2)),101) # gm to K
P2 = np.linspace(P1[-1],P1[-1]+sqrt(((M[0]-K[0])**2+(M[1]-K[1])**2)),101) # K to M
P3 = np.linspace(P2[-1],P2[-1]+sqrt(((0-M[0])**2+(0-M[1])**2)),101) # M to Gamma

kx1 = np.linspace(0.00,K[0],len(P1))  # Gamma to Kx
ky1 = np.linspace(0.00,K[1],len(P1)) # Gamma to Ky

kx2 = np.linspace(kx1[-1],M[0],len(P2)) # Kx to Mx
ky2 = np.linspace(ky1[-1],M[1],len(P2)) # Ky to My


kx3 = np.linspace(kx2[-1],0.0,len(P3)) # Mx to Gamma
ky3 = np.linspace(ky2[-1],0.0,len(P3)) # My to Gamma

# getting the eigenvalues for the line segments

Eig1 = Tight_binding(kx1,ky1,Minit,phi)
E1=Eig1[:,0]
E2=Eig1[:,1]
Eig2 = Tight_binding(kx2,ky2,Minit,phi)
E3=Eig2[:,0]
E4=Eig2[:,1]
Eig3 = Tight_binding(kx3,ky3,Minit,phi)
E5=Eig3[:,0]
E6=Eig3[:,1]

# PLOTS
fig, ay = plt.subplots() # creat a subplot 
fig.subplots_adjust(left=0.15, bottom=0.25) # size of the window 
# add the slider
axmass = fig.add_axes([0.15, 0.1, 0.65, 0.03])  
axphase = fig.add_axes([0.15, 0.15, 0.65, 0.03])
mass_term = Slider(ax=axmass,label='$M$',valmin=0,valmax=1,valinit=Minit)
phase = Slider(ax=axphase,label='$\\phi$',valmin=-np.pi,valmax=np.pi,valinit=phi)
# plot settings
ay.set_ylim(-10,10)
ay.set_xlim(P1[0],P3[-1])
ay.set_xlabel('k-points')
ay.set_xticks([0.00, P1[-1],P2[-1],P3[-1]])
ay.set_xticklabels([u'$\\Gamma$',u'K',u'M',u'$\\Gamma$'])
ay.vlines(P1[-1],-10,10,colors='k',linestyles='dashed', lw=1)
ay.vlines(P2[-1],-10,10,colors='k',linestyles='dashed', lw=1)
ay.set_ylabel('$E-E_f$')
ay.set_title('Haldane model')

# plot the dispersion
seg11, =ay.plot(P1,E1,color='blue')
seg12, =ay.plot(P1,E2,color='blue')

seg21, =ay.plot(P2,E3,color='red')
seg22, =ay.plot(P2,E4,color='red')

seg31, =ay.plot(P3,E5,color='green')
seg32, =ay.plot(P3,E6,color='green')

#update the eigenvalues when using the sliders
def update(val):
    seg11.set_ydata(Tight_binding(kx1,ky1,mass_term.val,phase.val)[:,0])
    seg12.set_ydata(Tight_binding(kx1,ky1,mass_term.val,phase.val)[:,1])
    seg21.set_ydata(Tight_binding(kx2,ky2,mass_term.val,phase.val)[:,0])
    seg22.set_ydata(Tight_binding(kx2,ky2,mass_term.val,phase.val)[:,1])
    seg31.set_ydata(Tight_binding(kx3,ky3,mass_term.val,phase.val)[:,0])
    seg32.set_ydata(Tight_binding(kx3,ky3,mass_term.val,phase.val)[:,1])
    fig.canvas.draw_idle()

# register the update function with each slider
mass_term.on_changed(update)
phase.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')

def reset(event):
    mass_term.reset()
    phase.reset()
button.on_clicked(reset)

plt.show()