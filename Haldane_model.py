import numpy as np 
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy import sqrt, exp
from matplotlib.widgets import Slider, Button

## constants ##
a = 1.419      # lattice parameter
Minit = 0
### TB Equation ###
def Tight_binding(x,y,M,t1,phi):
    t2=t1*0.02
    h0 = np.tensordot(np.array([[1,0],[0,1]]), -2*t2*np.cos(phi)*(np.cos(sqrt(3)/2*x*a+3/2*y*a)+np.cos(sqrt(3)/2*x*a-3/2*y*a)+np.cos(-a*x*sqrt(3))),axes = 0).T
    hx = np.tensordot(np.array([[0,1],[1,0]]), -t1*(np.cos(1/2*y*a+x*a*sqrt(3)/2)+np.cos(1/2*y*a-x*a*sqrt(3)/2)+np.cos(a*y)),axes = 0).T
    hy = np.tensordot(np.array([[0,1j],[-1j,0]]).T,-t1* (np.sin(1/2*y*a+x*a*sqrt(3)/2)+np.sin(1/2*y*a-x*a*sqrt(3)/2)-np.sin(a*y)),axes = 0).T
    hz= np.tensordot(np.array([[1,0],[0,-1]]),(np.zeros(len(x))+M)-2*t2*np.sin(phi)*(np.sin(sqrt(3)/2*x*a+3/2*y*a)+np.sin(sqrt(3)/2*x*a-3/2*y*a)-np.sin(a*x*sqrt(3))),axes=0).T
    H=h0+hx+hy+hz
    E=LA.eigvalsh(H)
    return E
## Line Segments ##
p1 = np.linspace(0,1.704298205,101) # Gamma to K
p2 = np.linspace(p1[-1],2.556447307,101) # K to M
p3 = np.linspace(p2[-1],4.032412848,101) # M to Gamma

x1 = np.linspace(0,1.704298205,len(p1)) # Gamma to Kx
y1 = np.linspace(0,0,len(p1)) # Gamma to Ky

x2 = np.linspace(1.704298205,1.278223653,len(p2))  # Kx to Mx
y2 = np.linspace(0,0.7379827704,len(p2)) # Ky to My

x3 = np.linspace(1.278223653,0,len(p3))
y3 = np.linspace(0.7379827704,0,len(p3))
# getting the eigenvalues for the line segments
t1 = 2.9         # NN hopping
phi = 0          # initial phase

Eig1 = Tight_binding(x1,y1,Minit,t1,phi)
E1=Eig1[:,0]
E2=Eig1[:,1]
Eig2 = Tight_binding(x2,y2,Minit,t1,phi)
E3=Eig2[:,0]
E4=Eig2[:,1]
Eig3 = Tight_binding(x3,y3,Minit,t1,phi)
E5=Eig3[:,0]
E6=Eig3[:,1]
# PLOTS
fig, ay = plt.subplots()
fig.subplots_adjust(left=0.15, bottom=0.25)
axfreq = fig.add_axes([0.15, 0.1, 0.65, 0.03])
axfreq2 = fig.add_axes([0.15, 0.15, 0.65, 0.03])
mass_term = Slider(ax=axfreq,label='$M$',valmin=0,valmax=1,valinit=Minit)
phase = Slider(ax=axfreq2,label='$\\phi$',valmin=-np.pi,valmax=np.pi,valinit=phi)
ay.set_ylim(-10,10)
ay.set_xlim(p1[0],p3[-1])
ay.set_xlabel('k-points')
ay.set_xticks([0.00, 1.704298205,2.556447307,4.032412848])
ay.set_xticklabels([u'$\\Gamma$',u'K',u'M',u'$\\Gamma$'])
ay.vlines(1.704298205,-10,10,colors='k',linestyles='dashed', lw=1)
ay.vlines(2.556447307,-10,10,colors='k',linestyles='dashed', lw=1)
ay.set_ylabel('$E-E_f$')
ay.set_title('Haldane model')
seg11, =ay.plot(p1,E1,color='blue')
seg12, =ay.plot(p1,E2,color='blue')

seg21, =ay.plot(p2,E3,color='red')
seg22, =ay.plot(p2,E4,color='red')

seg31, =ay.plot(p3,E5,color='green')
seg32, =ay.plot(p3,E6,color='green')
def update(val):
    seg11.set_ydata(Tight_binding(x1,y1,mass_term.val,t1,phase.val)[:,0])
    seg12.set_ydata(Tight_binding(x1,y1,mass_term.val,t1,phase.val)[:,1])
    seg21.set_ydata(Tight_binding(x2,y2,mass_term.val,t1,phase.val)[:,0])
    seg22.set_ydata(Tight_binding(x2,y2,mass_term.val,t1,phase.val)[:,1])
    seg31.set_ydata(Tight_binding(x3,y3,mass_term.val,t1,phase.val)[:,0])
    seg32.set_ydata(Tight_binding(x3,y3,mass_term.val,t1,phase.val)[:,1])
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