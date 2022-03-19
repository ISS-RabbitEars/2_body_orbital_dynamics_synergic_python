import numpy as np
import sympy as sp
from sympy.physics.vector import dynamicsymbols
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import animation
from sympy.physics.vector.printing import vpprint, vlatex

def integrate(ic, ti, p):
	m1, m2, Gc = p
	r1, v1, r2, v2, th1, omega1, th2, omega2 = ic

	print(ti)
	
	return [v1, a1.subs({M2:m2, R1:r1, R2:r2, THETA1:th1, THETA2:th2, theta1_dot:omega1, G:Gc}),\
		v2, a2.subs({M1:m1, R1:r1, R2:r2, THETA1:th1, THETA2:th2, theta2_dot:omega2, G:Gc}),\
		omega1, alpha1.subs({M2:m2, R1:r1, R2:r2, r1_dot:v1, THETA1:th1, THETA2:th2, theta1_dot:omega1, G:Gc}),\
		omega2, alpha2.subs({M1:m1, R1:r1, R2:r2, r2_dot:v2, THETA1:th1, THETA2:th2, theta2_dot:omega2, G:Gc})]


G, M1, M2 = sp.symbols('G M1 M2')
t = sp.Symbol('t')
R1, R2, THETA1, THETA2 = dynamicsymbols('R1 R2 THETA1 THETA2')

X1 = R1 * sp.cos(THETA1)
Y1 = R1 * sp.sin(THETA1)
X2 = R2 * sp.cos(THETA2)
Y2 = R2 * sp.sin(THETA2)

v1s = X1.diff(t, 1)**2 + Y1.diff(t, 1)**2
v2s = X2.diff(t, 1)**2 + Y2.diff(t, 1)**2
R21 = sp.sqrt((X2 - X1)**2 + (Y2 - Y1)**2)
R12 = sp.sqrt((X1 - X2)**2 + (Y1 - Y2)**2)

T = sp.simplify(sp.Rational(1,2) * (M1 * v1s + M2 * v2s))
V = sp.simplify(-((G * M1 * M2) / R21) - ((G * M1 * M2) / R12))

L = T - V

r1_dot = R1.diff(t, 1)
r2_dot = R2.diff(t, 1)
theta1_dot = THETA1.diff(t, 1)
theta2_dot = THETA2.diff(t, 1)

dLdr1 = L.diff(R1, 1)
dLdr2 = L.diff(R2, 1)
dLdtheta1 = L.diff(THETA1, 1)
dLdtheta2 = L.diff(THETA2, 1)
dLdr1dot = L.diff(r1_dot, 1)
dLdr2dot = L.diff(r2_dot, 1)
dLdtheta1dot = L.diff(theta1_dot, 1)
dLdtheta2dot = L.diff(theta2_dot, 1)
ddtdLdr1dot = dLdr1dot.diff(t, 1)
ddtdLdr2dot = dLdr2dot.diff(t, 1)
ddtdLdtheta1dot = dLdtheta1dot.diff(t, 1)
ddtdLdtheta2dot = dLdtheta2dot.diff(t, 1)

Lgr1 = ddtdLdr1dot - dLdr1
Lgr2 = ddtdLdr2dot - dLdr2
Lgtheta1 = ddtdLdtheta1dot - dLdtheta1
Lgtheta2 = ddtdLdtheta2dot - dLdtheta2

r1_ddot = R1.diff(t, 2)
r2_ddot = R2.diff(t, 2)
theta1_ddot = THETA1.diff(t, 2)
theta2_ddot = THETA2.diff(t, 2)

a1_sol = sp.solve(Lgr1,r1_ddot)
a2_sol = sp.solve(Lgr2,r2_ddot)
alpha1_sol = sp.solve(Lgtheta1,theta1_ddot)
alpha2_sol = sp.solve(Lgtheta2,theta2_ddot)

a1 = a1_sol[0]
a2 = a2_sol[0]
alpha1 = alpha1_sol[0]
alpha2 = alpha2_sol[0]

#-----------------------------------------------------------

Gc=1
m1, m2 = [1, 1]
r1o, r2o = [0.1, 1]
v1o, v2o = [0, 0]
theta1o, theta2o = [0, 180] 
omega1o, omega2o = [0, 15]

cnvrt = np.pi/180
theta1o *= cnvrt
theta2o *= cnvrt
omega1o *= cnvrt
omega2o *= cnvrt

p = [m1, m2, Gc]
ic = [r1o, v1o, r2o, v2o, theta1o, omega1o, theta2o, omega2o]

tf = 15 
nfps = 30
nframes = tf * nfps
ta = np.linspace(0, tf, nframes)

rth = odeint(integrate, ic, ta, args = (p,))

x1 = np.zeros(nframes)
y1 = np.zeros(nframes)
x2 = np.zeros(nframes)
y2 = np.zeros(nframes)
for i in range(nframes):
	x1[i] = X1.subs({R1:rth[i,0], THETA1:rth[i,4]})
	y1[i] = Y1.subs({R1:rth[i,0], THETA1:rth[i,4]})
	x2[i] = X2.subs({R2:rth[i,2], THETA2:rth[i,6]})
	y2[i] = Y2.subs({R2:rth[i,2], THETA2:rth[i,6]})

xmin = min(x1)
if min(x2) < xmin:
	xmin = min(x2)
xmax = max(x1)
if max(x2) > xmax:
	xmax = max(x2)
ymin = min(y1)
if min(y2) < ymin:
	ymin = min(y2)
ymax = max(y1)
if max(y2) > ymax:
	ymax = max(y2)

ke = np.zeros(nframes)
pe = np.zeros(nframes)
for i in range(nframes):
	ke[i] = T.subs({M1:m1, M2:m2, R1:rth[i,0], R2:rth[i,2], r1_dot:rth[i,1], r2_dot:rth[i,3], theta1_dot:rth[i,5], theta2_dot:rth[i,7]})
	pe[i] = V.subs({G:Gc, M1:m1, M2:m2, R1:rth[i,0], R2:rth[i,2], THETA1:rth[i,4], THETA2:rth[i,6]})

E = ke + pe

fig, a=plt.subplots()

rad=0.1
xmax+=2*rad
xmin-=2*rad
ymax+=2*rad
ymin-=2*rad


def run(frame):
	plt.clf()
	plt.subplot(211)
	circle=plt.Circle((x1[frame],y1[frame]),radius=rad,fc='xkcd:cerulean')
	plt.gca().add_patch(circle)
	circle=plt.Circle((x2[frame],y2[frame]),radius=rad,fc='xkcd:red')
	plt.gca().add_patch(circle)
	plt.title("Orbital Dynamics")
	ax=plt.gca()
	ax.set_aspect(1)
	plt.xlim([xmin,xmax])
	plt.ylim([ymin,ymax])
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.set_facecolor('xkcd:black')
	plt.subplot(212)
	plt.plot(ta[0:frame],ke[0:frame],'xkcd:red',lw=0.5)
	plt.plot(ta[0:frame],pe[0:frame],'xkcd:cerulean',lw=0.5)
	plt.plot(ta[0:frame],E[0:frame],'xkcd:bright green',lw=1.0)
	plt.xlim([0,tf])
	plt.title("Energy")
	ax=plt.gca()
	ax.legend(['T','V','E'],labelcolor='w',frameon=False)
	ax.set_facecolor('xkcd:black')

ani=animation.FuncAnimation(fig,run,frames=nframes)
#writervideo = animation.FFMpegWriter(fps=nfps)
#ani.save('gravitation.mp4', writer=writervideo)
plt.show()
 
