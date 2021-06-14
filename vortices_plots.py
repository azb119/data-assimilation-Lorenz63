from vortices import *
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
from scipy.interpolate import UnivariateSpline

# Figure 19
t = 30
res = 10000
tarr = np.linspace(0,t,res)
a = solve_ivp(rhs, (0,t), v, t_eval=tarr)
plt.figure()

for i in  range(0, 1, 2):
    plt.scatter(a.y[i], a.y[i+1], linewidth=0.4, c=tarr , cmap=plt.cm.viridis , marker=".", s=6)
plt.colorbar(plt.scatter(a.y[0], a.y[1], linewidth=0.4, c=tarr ,cmap=plt.cm.viridis , marker=".",s=6), shrink=0.8, label='t')

plt.axis('equal')
plt.xlabel("x")
plt.ylabel("y")

# init_conditions from gaussian_dist
t_course = np.linspace(0,t,int(res/100))
for i in range(6):
    init_dist = np.random.default_rng().normal(0,0.05,6)
    init_dist[2:] = 0
    v1 = v+init_dist
    b = solve_ivp(rhs, (0,t), v1, t_eval=tarr)
    for i in range(0,1, 2):
        plt.plot(b.y[i], b.y[i+1], linewidth=0.6)
plt.show()

# Figure 21 & 22
# compare reference and mechanistic models
t=20
t2=60
t3=250
dt_out=t3*dt
v1_ref = [[item[0][0] for item in vref(t)][0::t3],[item[0][1] for item in vref(t)][0::t3]]
v2_ref = [[item[1][0] for item in vref(t)][0::t3],[item[1][1] for item in vref(t)][0::t3]]
v3_ref = [[item[2][0] for item in vref(t)][0::t3],[item[2][1] for item in vref(t)][0::t3]]

v1_mod = [[item[0][0] for item in vmod(t)][0::t3],[item[0][1] for item in vmod(t)][0::t3]]
v2_mod = [[item[1][0] for item in vmod(t)][0::t3],[item[1][1] for item in vmod(t)][0::t3]]
v3_mod = [[item[2][0] for item in vmod(t)][0::t3],[item[2][1] for item in vmod(t)][0::t3]]

v1_err = [[m - r for m, r in zip(v1_mod[0], v1_ref[0])][0:t2],[m - r for m, r in zip(v1_mod[1], v1_ref[1])][0:t2]]
v2_err = [[m - r for m, r in zip(v2_mod[0], v2_ref[0])][0:t2],[m - r for m, r in zip(v2_mod[1], v2_ref[1])][0:t2]]
v3_err = [[m - r for m, r in zip(v3_mod[0], v3_ref[0])][0:t2],[m - r for m, r in zip(v3_mod[1], v3_ref[1])][0:t2]]

fig1 = plt.figure(figsize=(12,3))
x1ref, = plt.plot(np.arange(0,(len(v1_ref[0])-0.5)*dt_out,dt_out), v1_ref[0], marker="x",label ='reference')
x1mod, = plt.plot(np.arange(0,(len(v1_ref[0])-0.5)*dt_out,dt_out), v1_mod[0], marker="o",label='model')
plt.title('x-coordinates of 1st vortex')
plt.xlabel("time")
plt.ylabel("x")
plt.legend()

fig2 = plt.figure(figsize=(12,3))
y1ref, = plt.plot(np.arange(0,(len(v1_ref[1])-0.5)*dt_out,dt_out), v1_ref[1], marker="x",label='reference')
y1mod, = plt.plot(np.arange(0,(len(v1_ref[1])-0.5)*dt_out,dt_out), v1_mod[1], marker="o",label='model')
plt.xlabel("time")
plt.ylabel("y")
plt.title('y-coordinates of 1st vortex')
plt.legend()

fig3 = plt.figure(figsize=(12,3))
x2ref, = plt.plot(np.arange(0,(len(v2_ref[0])-0.5)*dt_out,dt_out), v2_ref[0], marker="x",label ='reference')
x2mod, = plt.plot(np.arange(0,(len(v2_ref[0])-0.5)*dt_out,dt_out), v2_mod[0], marker="o",label='model')
plt.title('x-coordinates of 2nd vortex')
plt.xlabel("time")
plt.ylabel("x")
plt.legend()

fig4 = plt.figure(figsize=(12,3))
y2ref, = plt.plot(np.arange(0,(len(v2_ref[1])-0.5)*dt_out,dt_out), v2_ref[1], marker="x",label='reference')
y2mod, = plt.plot(np.arange(0,(len(v2_ref[1])-0.5)*dt_out,dt_out), v2_mod[1], marker="o",label='model')
plt.xlabel("time")
plt.ylabel("y")
plt.title('y-coordinates of 2nd vortex')
plt.legend()

fig7 = plt.figure(figsize=(12,3))
x3ref, = plt.plot(np.arange(0,(len(v3_ref[0])-0.5)*dt_out,dt_out), v3_ref[0], marker="x",label='reference')
x3mod, = plt.plot(np.arange(0,(len(v3_ref[0])-0.5)*dt_out,dt_out), v3_mod[0], marker="o",label='model')
plt.xlabel("time")
plt.ylabel("x")
plt.title('x-coordinates of 3rd vortex')
plt.legend()

fig8 = plt.figure(figsize=(12,3))
y3ref, = plt.plot(np.arange(0,(len(v3_ref[1])-0.5)*dt_out,dt_out), v3_ref[1], marker="x",label='reference')
y3mod, = plt.plot(np.arange(0,(len(v3_ref[1])-0.5)*dt_out,dt_out), v3_mod[1], marker="o",label='model')
plt.xlabel("time")
plt.ylabel("y")
plt.title('y-coordinates of 3rd vortex')
plt.legend()

fig5 = plt.figure(figsize=(12,3))
plt.plot(np.linspace(0, t2*dt_out, t2), v1_err[0],marker='x',label='x')
plt.plot(np.linspace(0, t2*dt_out, t2), v1_err[1],marker='x',label='y')
plt.ylim([-1.3, 1.3])

plt.xlabel("time")
plt.ylabel("error")
plt.title('Vortex 1')
plt.legend()

fig6 = plt.figure(figsize=(12,3))
plt.plot(np.linspace(0, t2*dt_out, t2), v2_err[0],marker='x',label='x')
plt.plot(np.linspace(0, t2*dt_out, t2), v2_err[1],marker='x',label='y')
plt.ylim([-1.3, 1.3])
plt.xlabel("time")
plt.ylabel("error")
plt.title('Vortex 2')
plt.legend()

fig9 = plt.figure(figsize=(12,3))
plt.plot(np.linspace(0, t2*dt_out, t2), v3_err[0],marker='x',label='x')
plt.plot(np.linspace(0, t2*dt_out, t2), v3_err[1],marker='x',label='y')
plt.ylim([-1.3, 1.3])
plt.xlabel("time")
plt.ylabel("error")
plt.title('Vortex 3')
plt.legend()
plt.show()

# Figure 23
N_obs=200
v1=[item[0] for item in vref(dt_out*N_obs)]
v1s=v1[slice(250,len(v1),250)]
xs= [item[0] for item in v1s]
ys =[item[1] for item in v1s]
xobs = [item[0] for item in v1_obs(N_obs, dt_out, vref)]
yobs = [item[1] for item in v1_obs(N_obs, dt_out, vref)]
t = np.arange(dt_out,dt_out*(len(xs)+1),dt_out)


plt.figure(figsize=(11,4))
col = plt.get_cmap('tab10')
plt.plot(t, xobs, 'x', color=col(0), label='$x_{obs}$')
plt.plot(t,xs,marker='o',color=col(1),label='$v_{1,ref}$ x coord')
plt.xlabel("time")
plt.ylabel("x")
plt.legend()

plt.figure(figsize=(11,4))
plt.plot(t,yobs,'x', color = col(0), label = '$y_{\text{obs}}$') 
plt.plot(t,ys,marker='o',color=col(1),label='$v_{1,ref}$ y coord')
plt.xlabel("time")
plt.ylabel("y")
plt.legend()

plt.figure(figsize=(11,4))
plt.plot(t,np.array(xs)-np.array(xobs)) 
plt.xlabel("time")
plt.ylabel("error in x coord of the observations");

plt.figure(figsize=(11,4))
plt.plot(t,np.array(ys)-np.array(yobs)) 
plt.xlabel("time")
plt.ylabel("error in y coord of the observations");

plt.show()

# Figure 24
# comparing reference model to stochastic model 
t=20

v1_ref = [[item[0][0] for item in vref(t)][0::250],[item[0][1] for item in vref(t)][0::250]]
v2_ref = [[item[1][0] for item in vref(t)][0::250],[item[1][1] for item in vref(t)][0::250]]
v3_ref = [[item[2][0] for item in vref(t)][0::250],[item[2][1] for item in vref(t)][0::250]]

plt.figure(figsize=(12,3))
xs=v1_ref[0]
x2s=[item[0][0] for item in sim_v(int(t/dt),ICs)][0::250]
plt.xlabel('time')
plt.ylabel('x')
plt.plot(np.arange(0,(len(xs)-0.5)*dt_out,dt_out),xs,marker = 'x',label = 'reference')
plt.plot(np.arange(0,(len(xs)-0.5)*dt_out,dt_out),x2s,marker = 'o',label='stochastic')
plt.title('Vortex 1')
plt.legend()

plt.figure(figsize=(12,3))
ys=v1_ref[1]
y2s=[item[0][1] for item in sim_v(int(t/dt),ICs)][0::250]
plt.xlabel('time')
plt.ylabel('y')
plt.plot(np.arange(0,(len(xs)-0.5)*dt_out,dt_out),ys,marker = 'x',label = 'reference')
plt.plot(np.arange(0,(len(xs)-0.5)*dt_out,dt_out),y2s,marker = 'o',label='stochastic')
plt.title('Vortex 1')
plt.legend()

plt.figure(figsize=(12,3))
xs=v2_ref[0]
x2s=[item[1][0] for item in sim_v(int(t/dt),ICs)][0::250]
plt.xlabel('time')
plt.ylabel('x')
plt.plot(np.arange(0,(len(xs)-0.5)*dt_out,dt_out),xs,marker = 'x',label = 'reference')
plt.plot(np.arange(0,(len(xs)-0.5)*dt_out,dt_out),x2s,marker = 'o',label='stochastic')
plt.title('Vortex 2')
plt.legend()

plt.figure(figsize=(12,3))
ys=v2_ref[1]
y2s=[item[1][1] for item in sim_v(int(t/dt),ICs)][0::250]
plt.xlabel('time')
plt.ylabel('y')
plt.plot(np.arange(0,(len(xs)-0.5)*dt_out,dt_out),ys,marker = 'x',label = 'reference')
plt.plot(np.arange(0,(len(xs)-0.5)*dt_out,dt_out),y2s,marker = 'o',label='stochastic')
plt.title('Vortex 2')
plt.legend()

plt.figure(figsize=(12,3))
xs=v3_ref[0]
x2s=[item[2][0] for item in sim_v(int(t/dt),ICs)][0::250]
plt.xlabel('time')
plt.ylabel('x')
plt.plot(np.arange(0,(len(xs)-0.5)*dt_out,dt_out),xs,marker = 'x',label = 'reference')
plt.plot(np.arange(0,(len(xs)-0.5)*dt_out,dt_out),x2s,marker = 'o',label='stochastic')
plt.title('Vortex 3')
plt.legend()

plt.figure(figsize=(12,3))
ys=v3_ref[1]
y2s=[item[2][1] for item in sim_v(int(t/dt),ICs)][0::250]
plt.xlabel('time')
plt.ylabel('y')
plt.plot(np.arange(0,(len(xs)-0.5)*dt_out,dt_out),ys,marker = 'x',label = 'reference')
plt.plot(np.arange(0,(len(xs)-0.5)*dt_out,dt_out),y2s,marker = 'o',label='stochastic')
plt.title('Vortex 3')
plt.legend()

plt.show()

# Figure 25
# SIS mean trajectory of first vortex 
N_obs = 350
sis=SIS(15)
xobserve=[item[0] for item in v1_obs(N_obs, dt_out, vref)]
yobserve = [item[1] for item in v1_obs(N_obs, dt_out, vref)]
enpaths, weights = sis[1],sis[3]
t_list = np.arange(0, dt_out*(N_obs+0.5), dt_out)
x_sim = [mean_y(enpaths, k,0, 0,weights) for k in range(0, N_obs+1)]
y_sim = [mean_y(enpaths,k,0,1,weights) for k in range(0,N_obs+1)]

xfit = UnivariateSpline(t_list[1:], xobserve)
yfit = UnivariateSpline(t_list[1:], yobserve)
xfit.set_smoothing_factor(12.5)
yfit.set_smoothing_factor(12.5)

plt.figure(figsize=(12,6))
plt.plot(t_list[1:], xobserve, '.k', label = 'x_obs')
plt.plot(t_list[1:],xfit(t_list[1:]),'k',label='cubic spline')
plt.plot(t_list[1:], x_sim[1:], 'c',ls='--', label = 'sample size = 15')
plt.xlabel('time')
plt.ylabel('x coord of vortex 1')
plt.xlim((0,60))
plt.legend()
plt.figure(figsize=(12,6))
plt.plot(t_list[1:], yobserve, '.k', label = 'y_obs')
plt.plot(t_list[1:],yfit(t_list[1:]),'k',label='cubic spline')
plt.plot(t_list[1:], y_sim[1:], 'c',ls='--', label = 'sample size = 15')
plt.xlabel('time')
plt.ylabel('y coord of vortex 1')
plt.xlim((0,60))
plt.legend()

plt.show()

# Figure 26
# SIS Effective sample size
N_obs=90
trial2 = []
M_vals = [15,50,100,1000]
for value in M_vals:
    trial2.append(SIS(value)[0])

plt.figure(figsize=(8,6))
t_ls = np.arange(0, len(trial2[0])*dt_out,dt_out)
count=-1
for runs in trial2:
    count+=1
    plt.semilogy(t_ls, runs, label=M_vals[count])
plt.xlabel('time')
plt.ylabel('effective sample size')
plt.legend()
plt.show()

# Figure 27
# SIS RMSE
N_obs=60
M_vals = [50,100,1000]
trial4paths = []
trial4weights = []
for value in M_vals:
    sis2 = SIS(value)
    trial4paths.append(sis2[1])
    trial4weights.append(sis2[3])
    print(value)
trial4=np.matrix([trial4paths,trial4weights])

M_vals = [10,25,50,100,1000]
xrmses=[]
yrmses=[]
for i in range(len(M_vals)):
    xrmses.append(v1_xrmse(M_vals[i],N_obs,trial4[0,i],trial4[1,i],0))
    yrmses.append(v1_yrmse(M_vals[i],N_obs,trial4[0,i],trial4[1,i],0))

plt.figure(figsize=(8,4))
plt.semilogx(M_vals, xrmses,marker = 'x',label='x-coord')
plt.semilogx(M_vals, yrmses,marker = 'o',label='y- coord')
plt.legend()
plt.title('SIS RMSE against sample size of 1st Vortex')

plt.show()

# Figure 28
# SIR mean trajectory
N_obs = 350
sir=SIR(15)
xobserve2=[item[0] for item in v1_obs(N_obs, dt_out, vref)]
yobserve2 = [item[1] for item in v1_obs(N_obs, dt_out, vref)]
enpaths = sir[1]
t_list2 = np.arange(0, dt_out*(N_obs+0.5), dt_out)
x_sim2 = [mean_y2(enpaths, k,0, 0) for k in range(0, N_obs+1)]
y_sim2 = [mean_y2(enpaths,k,0,1) for k in range(0,N_obs+1)]

xfit2 = UnivariateSpline(t_list2[1:], xobserve2)
yfit2 = UnivariateSpline(t_list2[1:], yobserve2)
xfit2.set_smoothing_factor(10.5)
yfit2.set_smoothing_factor(10.5)

plt.figure(figsize=(12,6))
plt.plot(t_list2[1:], xobserve2, '.k', label = 'x_obs')
plt.plot(t_list2[1:],xfit2(t_list[1:]),'k',label='cubic spline')
plt.plot(t_list2[1:], x_sim2[1:], 'c',ls='--', label = 'sample size = 15')
plt.xlabel('time')
plt.ylabel('x coord of vortex 1')
plt.xlim((0,60))

plt.legend()
plt.figure(figsize=(12,6))
plt.plot(t_list2[1:], yobserve2, '.k', label = 'y_obs')
plt.plot(t_list2[1:],yfit2(t_list2[1:]),'k',label='cubic spline')
plt.plot(t_list2[1:], y_sim2[1:], 'c',ls='--', label = 'sample size = 15')
plt.xlabel('time')
plt.ylabel('y coord of vortex 1')
plt.xlim((0,60))

plt.legend()
plt.show()

# Figure 29
# Effective sample size SIR
N_obs=100
trial = []
M_vals = [10, 20, 60]
for value in M_vals:
    trial.append(SIR(value)[0])

plt.figure(figsize=(10,6))
t_ls = np.arange(0, len(trial[0])*dt_out,dt_out)
for runs in trial:
    plt.plot(t_ls, runs)
plt.xlabel('time')
plt.ylabel('effective sample size')
plt.show()

# Figure 30
# SIR RMSE
N_obs=600
xrmses=[]
yrmses=[]
M_vals = [5,25,50,100,150,200,250,300]
trial3 = []

for i in range(len(M_vals)):
    trial3.append(SIR(M_vals[i])[1])
    xrmses.append(v1_xrmse2(M_vals[i],N_obs,trial3[i],0))
    yrmses.append(v1_yrmse2(M_vals[i],N_obs,trial3[i],0))

plt.figure(figsize=(10,4))
plt.plot(M_vals, xrmses,marker = 'x',label='x-coord')
plt.plot(M_vals, yrmses,marker = 'o',label='y- coord')
plt.legend()
plt.show()
