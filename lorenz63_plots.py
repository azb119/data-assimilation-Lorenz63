import numpy as np
import matplotlib.pyplot as plt  # noqa F401
from numpy.core import multiarray  # noqa F401
from mpl_toolkits import mplot3d  # noqa F401
from mpl_toolkits.mplot3d import Axes3D
from lorenz63 import *

# Figure 1
WIDTH, HEIGHT, DPI = 1000, 750, 100
fig = plt.figure(facecolor='k', figsize=(WIDTH/DPI, HEIGHT/DPI))
ax = fig.gca(projection='3d')
ax.set_facecolor('k')
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
s = 10
cmap = plt.cm.winter
t = 9.05
z=zmod(t,(1.4,2.2,2.8),dt)
z1 = [i[0] for i in z]
z2 = [i[1] for i in z]
z3 = [i[2] for i in z]
p=[z1,z2,z3]
ax.plot(p[0], p[1], p[2], color='r')

cmap=plt.cm.summer
z=zmod(t,(1.1,1.9,3.2),dt)
z1 = [i[0] for i in z]
z2 = [i[1] for i in z]
z3 = [i[2] for i in z]
p2=[z1,z2,z3]
ax.plot(p2[0], p2[1], p2[2], color='g')

cmap=plt.cm.summer
z=zmod(t,(1,1.85,3.05),dt)
z1 = [i[0] for i in z]
z2 = [i[1] for i in z]
z3 = [i[2] for i in z]
p2=[z1,z2,z3]
ax.plot(p2[0], p2[1], p2[2], color='b')
ax.grid(False)
ax.w_xaxis.set_pane_color((0, 0, 0, 0))
ax.w_yaxis.set_pane_color((0, 0, 0, 0))
ax.w_zaxis.set_pane_color((0, 0, 0, 0))
ax.w_xaxis.line.set_color('white')
ax.w_yaxis.line.set_color('white')
ax.w_zaxis.line.set_color('white')
ax.w_zaxis.line.set_color('white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.zaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')  # only affects
ax.tick_params(axis='y', colors='white')  # tick labels
ax.tick_params(axis='z', colors='white')  # not tick marks
ax.plot([1.4,1.1,1],[2.2,1.9,1.85],[2.8,3.2,3.05],'x',color='w', label ='= Initial Conditions')
plt.legend()

plt.show()

# Figure 2
WIDTH, HEIGHT, DPI = 1000, 750, 100
fig = plt.figure(facecolor='k', figsize=(WIDTH/DPI, HEIGHT/DPI))
ax = fig.gca(projection='3d')
ax.set_facecolor('k')
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
s = 10
cmap = plt.cm.winter
t = 20
z = zref(t)
z1 = [i[0] for i in z]
z2 = [i[1] for i in z]
z3 = [i[2] for i in z]
p=[z1,z2,z3]
for i in range(0,len(p[0])-s,s):
    ax.plot(p[0][i:i+s+1], p[1][i:i+s+1], p[2][i:i+s+1], color=cmap(i/len(p[0])), alpha=0.4)
ax.grid(False)
ax.w_xaxis.set_pane_color((0, 0, 0, 0))
ax.w_yaxis.set_pane_color((0, 0, 0, 0))
ax.w_zaxis.set_pane_color((0, 0, 0, 0))
ax.w_xaxis.line.set_color('white')
ax.w_yaxis.line.set_color('white')
ax.w_zaxis.line.set_color('white')
ax.w_zaxis.line.set_color('white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.zaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')  # only affects
ax.tick_params(axis='y', colors='white')  # tick labels
ax.tick_params(axis='z', colors='white')  # not tick marks

plt.show()

# Figure 3 & 4
xs=[item[0] for item in zref(dt_out*200)]
x2s = xs[slice(50,len(xs),50)]
trial = x_obs(N_obs, dt_out, zref)
t = np.arange(0,dt_out*len(x2s),dt_out)
plt.figure(figsize=(11,4))
plt.plot(t, trial, 'x')
plt.xlabel("time")
plt.ylabel("x_obs")
plt.figure(figsize=(11,4))
plt.plot(t,x2s-trial) 
plt.xlabel("time")
plt.ylabel("error for $x_{obs}$")

# Figure 6
x_ref = [item[0] for item in zref(6)][0::50]
y_ref = [item[1] for item in zref(6)][0::50]
z_ref = [item[2] for item in zref(6)][0::50]

x_mod = [item[0] for item in zmod(6)][0::50]
y_mod = [item[1] for item in zmod(6)][0::50]
z_mod = [item[2] for item in zmod(6)][0::50]

x_err = [m - r for m, r in zip(x_mod, x_ref)][0:40]
y_err = [m - r for m, r in zip(y_mod, y_ref)][0:40]
z_err = [m - r for m, r in zip(z_mod, z_ref)][0:40]

fig1 = plt.figure()
x1, = plt.plot(np.arange(0,(len(x_ref)-0.5)*dt_out,dt_out), x_ref, marker="x")
x2, = plt.plot(np.arange(0,(len(x_ref)-0.5)*dt_out,dt_out), x_mod, marker="o")
plt.ylim([-20, 20])
plt.xlabel("time")
plt.ylabel("solution for x")
plt.legend([x1, x2], ['reference', 'model'], loc=2)

fig2 = plt.figure()
y1, = plt.plot(np.arange(0,(len(x_ref)-0.5)*dt_out,dt_out), y_ref, marker="x")
y2, = plt.plot(np.arange(0,(len(x_ref)-0.5)*dt_out,dt_out), y_mod, marker="o")
plt.ylim([-50, 50])
plt.xlabel("time")
plt.ylabel("solution for y")
plt.legend([y1, y2], ['reference', 'model'], loc=2)

fig3 = plt.figure()
z1, = plt.plot(np.arange(0,(len(x_ref)-0.5)*dt_out,dt_out), z_ref, marker="x")
z2, = plt.plot(np.arange(0,(len(x_ref)-0.5)*dt_out,dt_out), z_mod, marker="o")
plt.ylim([0, 50])
plt.xlabel("time")
plt.ylabel("solution for z")
plt.legend([z1, z2], ['reference', 'model'], loc=2)

fig4 = plt.figure()
plt.plot(np.linspace(0, 2, 40), x_err, marker="x")
plt.ylim([-2, 2])
plt.xlabel("time")
plt.ylabel("error for x")

fig5 = plt.figure()
plt.plot(np.linspace(0, 2, 40), y_err, marker="x")
plt.ylim([-5, 5])
plt.xlabel("time")
plt.ylabel("error for y")

fig6 = plt.figure()
plt.plot(np.linspace(0, 2, 40), z_err, marker="x")
plt.ylim([-5, 5])
plt.xlabel("time")
plt.ylabel("error for z")

plt.show()

# Figure 7
WIDTH, HEIGHT, DPI = 1000, 750, 100
fig = plt.figure(facecolor='k', figsize=(WIDTH/DPI, HEIGHT/DPI))
ax = fig.gca(projection='3d')
ax.set_facecolor('k')
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
s = 10
cmap = plt.cm.summer
t = int(20/0.001)
zs = sim_z(t, ICs)
x_sim = first_comp(zs)
t_list = np.arange(0, t+dt_sim, dt_sim)
z1 = [i[0] for i in zs]
z2 = [i[1] for i in zs]
z3 = [i[2] for i in zs]
p=[z1,z2,z3]
for i in range(0,len(p[0])-s,s):
    ax.plot(p[0][i:i+s+1], p[1][i:i+s+1], p[2][i:i+s+1], color=cmap(i/len(p[0])), alpha=0.4)
ax.grid(False)
ax.w_xaxis.set_pane_color((0, 0, 0, 0))
ax.w_yaxis.set_pane_color((0, 0, 0, 0))
ax.w_zaxis.set_pane_color((0, 0, 0, 0))
ax.w_xaxis.line.set_color('white')
ax.w_yaxis.line.set_color('white')
ax.w_zaxis.line.set_color('white')
ax.w_zaxis.line.set_color('white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.zaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')  # only affects
ax.tick_params(axis='y', colors='white')  # tick labels
ax.tick_params(axis='z', colors='white')  # not tick marks

plt.show()

# Figure 8
plt.figure(figsize=(12,3))
xs=[item[0] for item in zref(6)][0::50]
x2s=[item[0] for item in sim_z(int(6/dt),ICs)][0::50]
plt.xlabel('time')
plt.ylabel('x')
plt.plot(np.arange(0,(len(xs)-0.5)*dt_out,dt_out),xs,marker = 'x',label = 'reference')
plt.plot(np.arange(0,(len(xs)-0.5)*dt_out,dt_out),x2s,marker = 'o',label='stochastic')
plt.legend()

plt.figure(figsize=(12,3))
ys=[item[1] for item in zref(6)][0::50]
y2s=[item[1] for item in sim_z(int(6/dt),ICs)][0::50]
plt.plot(np.arange(0,(len(xs)-0.5)*dt_out,dt_out),ys,marker = 'x',label = 'reference')
plt.plot(np.arange(0,(len(xs)-0.5)*dt_out,dt_out),y2s,marker = 'o',label='stochastic')
plt.xlabel('time')
plt.ylabel('y')
plt.legend()

plt.figure(figsize=(12,3))
zs=[item[2] for item in zref(6)][0::50]
z2s=[item[2] for item in sim_z(int(6/dt),ICs)][0::50]
plt.plot(np.arange(0,(len(xs)-0.5)*dt_out,dt_out),zs,marker = 'x',label = 'reference')
plt.plot(np.arange(0,(len(xs)-0.5)*dt_out,dt_out),z2s,marker = 'o',label='stochastic')
plt.xlabel('time')
plt.ylabel('z')
plt.legend()

plt.show()

# Figure 10
# plotting  x trajectories for different SIS sample sizes
N_obs = 80
useless, enpath, observe, weights = SIS(50)
useless2, enpath2, observe2, weights2 = SIS(100)
t_list = np.linspace(0, dt_out * N_obs, N_obs + 1)
x_sim = [mean_y(enpath, k, 0,weights) for k in range(0, N_obs+1)]
x_sim2 = [mean_y(enpath2, k, 0, weights2) for k in range(0, N_obs+1)]

plt.figure(figsize=(8,6))
plt.plot(t_list[1:], observe, '-k', label = 'x_obs')
plt.plot(t_list, x_sim, '-c', label = 'particle mean50')
plt.plot(t_list, x_sim2, 'red', label = 'particle mean100')

plt.xlabel("time")
plt.ylabel("x coordinates")
plt.legend()
plt.show()

# Figure 11
# data for RMSE of SIS 
# N_obs=80
# trial3paths = []
# trial3weights= []

# M_vals = [50, 100,1000,10**4,10**5]
# for value in M_vals:
#     sis2 = SIS(value)
#     trial3paths.append(sis2[1])
#     trial3weights.append(sis2[3])
#     print(value)
# trial3=np.matrix([trial3paths,trial3weights])

# data then saved and loaded below
# np.save('trial3_data',trial3)
# trial3test = np.load('trial3_data.npy',allow_pickle=True)
# SIS RMSE plot
M_vals=[50,100,1000,10**4,10**5]
plt.figure(figsize=(8,4))
N_obs=80
count=-1
xrmses=[]
zrmses=[]
yrmses=[]
for i in range(len(M_vals)):
    xrmses.append(xrmse2(M_vals[i],N_obs,trial3[0,i],trial3[1,i]))
    zrmses.append(zrmse2(M_vals[i],N_obs,trial3[0,i],trial3[1,i]))

plt.semilogx(M_vals, xrmses,marker = 'x',label='x-coord')
plt.semilogx(M_vals, zrmses,marker = 'o',label='z- coord')
plt.xlim(10)
plt.legend()
plt.show()

# Figure 12
N_obs = 200
observe=x_obs(N_obs,dt_out,zref)
sis=SIS(100)
enpaths, weights = sis[1],sis[3]
t_list = np.arange(0, dt_out*(N_obs+0.5), dt_out)
x_sim = [mean_y(enpaths, k, 0,weights) for k in range(0, N_obs+1)]
plt.figure(figsize=(12,6))
plt.plot(t_list[1:], observe, '-k', label = 'x_obs')
plt.plot(t_list, x_sim, 'c',ls='--', label = 'sample size = 100')
plt.xlabel('x')
plt.ylabel('time')
plt.legend()

plt.show()

# Figure 13
# effective sample size plot for SIS

# N_obs=50
# trial2 = []
# M_vals = [50, 100,1000,10**4,10**5]
# for value in M_vals:
#     trial2.append(SIS(value)[0])
#     print(value)

# data is then saved and loaded below
# np.save('trial2_data',trial2)
trial2test=np.load('trial2_data.npy',allow_pickle=True)
plt.figure(figsize=(8,6))
t_ls = np.arange(0, len(trial2test[0])*dt_out,dt_out)
for runs in trial2test:
    plt.semilogy(t_ls, runs)
plt.xlim(0, 2)
plt.xlabel('time')
plt.ylabel('effective sample size')
plt.show()

# Figure 15
# effective sample size plot for SIR
N_obs=200
trial = []
M_vals = [30, 100, 300]
for value in M_vals:
    trial.append(SIR(value)[0])
    
plt.figure(figsize=(14,6))
t_ls = np.arange(0, len(trial[0])*dt_out,dt_out)
for runs in trial:
    plt.plot(t_ls, runs)
plt.show()

# Figure 16
N_obs = 400
useless, enpath, observe = SIR(100)
t_list = np.linspace(0, dt_out * N_obs, N_obs + 1)
x_sim = [mean_y(enpath, k, 0) for k in range(0, N_obs+1)]
plt.figure()
plt.plot(t_list[1:], observe, '-k', label = 'x_obs')
plt.plot(t_list, x_sim, '--c', label = 'particle mean')
# for paths in enpath:
#   plt.plot(t_list, [i[0] for i in paths], '-c')
plt.xlabel("time")
plt.ylabel("x coordinates")
plt.legend()
plt.show()

# Figure 17
# SIR RMSE plot
M_vals=[50,100,150,200,250,300,350]
plt.figure(figsize=(8,4))
N_obs=1000
count=-1
xrmses=[]
zrmses=[]
for i in range(len(M_vals)):
    xrmses.append(xrmse2(M_vals[i],N_obs,trial3[0,i],trial3[1,i]))
    zrmses.append(zrmse2(M_vals[i],N_obs,trial3[0,i],trial3[1,i]))

plt.plot(M_vals, xrmses,marker = 'x',label='x-coord')
plt.plot(M_vals, zrmses,marker = 'o',label='z- coord')
plt.xlim(10)
plt.legend()
plt.show()
