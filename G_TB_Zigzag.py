import numpy as np 
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy import sqrt, exp
from matplotlib.widgets import Slider, Button

import matplotlib
matplotlib.use('TkAgg')

## constants ##
t1 = 2.9         # NN hopping
a = 1.419      # lattice parameter
Minit = 0.5
N=15
phi= 0
### TB Equation ###
def Tight_binding(x,y,M,phi,N):
    t2=t1*0.02
    h0 = np.tensordot(np.array([[1,0],[0,1]]), -2*t2*np.cos(phi)*(np.cos(sqrt(3)/2*x*a+3*np.pi*y/N)+np.cos(sqrt(3)/2*x*a-3*np.pi*y/N)+np.cos(-a*x*sqrt(3))),axes = 0).T
    hx = np.tensordot(np.array([[0,1],[1,0]]), -t1*(np.cos(np.pi*y/N+x*a*sqrt(3)/2)+np.cos(np.pi*y/N-x*a*sqrt(3)/2)+np.cos(-2*np.pi*y/N)),axes = 0).T
    hy = np.tensordot(np.array([[0,1j],[-1j,0]]).T,-t1* (np.sin(np.pi*y/N+x*a*sqrt(3)/2)+np.sin(np.pi*y/N-x*a*sqrt(3)/2)+np.sin(-2*np.pi*y/N)),axes = 0).T
    hz= np.tensordot(np.array([[1,0],[0,-1]]),(np.zeros(len(x))+M)-2*t2*np.sin(phi)*(np.sin(sqrt(3)/2*x*a+3*np.pi*y/N)+np.sin(sqrt(3)/2*x*a-3*np.pi*y/N)+np.sin(-a*x*sqrt(3))),axes=0).T
    H=h0+hx+hy+hz
    E=LA.eigvalsh(H)
    return E

## Line Segments ##
p1 = np.linspace(-np.pi/(sqrt(3)*a),np.pi/(sqrt(3)*a),101) 

# plot's labels
fig, ay = plt.subplots(figsize=(9, 6), dpi=100)

#sliders
fig.subplots_adjust(left=0.15, bottom=0.25)
axfreq = fig.add_axes([0.15, 0.1, 0.65, 0.03])
axfreq2 = fig.add_axes([0.15, 0.15, 0.65, 0.03])
mass_term = Slider(ax=axfreq,label='$M$',valmin=0,valmax=6,valinit=Minit)
phase = Slider(ax=axfreq2,label='$\\phi$',valmin=-np.pi,valmax=np.pi,valinit=phi)

# ay.set_ylim(-10,10)
ay.set_xlim(p1[0],p1[-1])
ay.set_xlabel('k-points')
ay.set_ylabel('$E-E_f$')

# plotting in for loop
Valence = []
Conduction = []
for i in range(N):
    line1, = ay.plot(p1,Tight_binding(p1,i,Minit,phi,N)[:,0])
    Valence.append(line1)
    line2, = ay.plot(p1,Tight_binding(p1,i,Minit,phi,N)[:,1])
    Conduction.append(line2)


# update function to update the value when sliding
def update(val):
    for i in range(N):
        Valence[i].set_ydata(Tight_binding(p1,i,mass_term.val,phase.val,N)[:,0])
        Conduction[i].set_ydata(Tight_binding(p1,i,mass_term.val,phase.val,N)[:,1])
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
