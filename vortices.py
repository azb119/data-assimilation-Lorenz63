import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np


# v = Initail points of vortices [x0,y0,x1,y1...]
# gamma = Strengths for points 0,1,2..
# t = time endpoint
# res = Number of equally spaced time points

ICs =  np.array([[1.4,0], [-0.5,0.8],[-1,-2.1]])
gamma = [12, 5,6]
dt = 0.001
a = 1/(10*np.sqrt(dt))



def f(v):
    """
    Calculate f for the next time step.
    -----------
    Parameters:
    v: 3d array of x,y coords [(x1, y1),(x2,y2),(x3,y3)]
        current position

    Return: 3d array
        next v
    """
    x = [v[i][0] for i in range(0,len(v))]
    y = [v[i][1] for i in range(0,len(v))]
    fn =  [[-1/(2*np.pi)*sum(gamma[k]*(y[i]-y[k])/((x[i]-x[k])**2+(y[i]-y[k])**2) for k in range(len(x)) if k!=i), 1/(2*np.pi)*sum(gamma[k]*(x[i]-x[k])/((x[i]-x[k])**2+(y[i]-y[k])**2) for k in range(len(x)) if k!=i)] for i in range(len(v))]
    return np.array([item for item in fn])

def vmod(t, ic=ICs, dt=dt):
    z_new = ic
    points = [ic]
    for i in range(int(t/dt)):
        z_new = z_new + dt*(f(z_new))
        points.append(z_new)
    return points


g0 = np.array([[a * ((2+(-0.2)*i) ** -0.5 - 0.5),a * ((3+(-0.2)*i) ** -0.5 - 0.5)] for i in range(3)])

def tent_fn(a, gi):
    """
    Calculate the tent function for g and observation error
    ----------
    Parameters:
    a: float
        scale for the tent function
    gi: float
        last tent function output

    Return: float
        next tent function output
    """
    if gi < 0:
        out = (1.9999*gi + a/2)
    else:
        out = (-1.9999*gi + a/2)
    return out

def g(a, g):
    """
    Calculate driver g for the next time step.
    ---------
    Parameters:
    a: float
        input for tent_fn
    g: 2d np.array
        the last g calculated

    Return: 2d np.array of floats
        next g
    """
    out1 = [tent_fn(a, g[0][i]) for i in range(2)]
    out2 = [tent_fn(a, g[1][i]) for i in range(2)]
    out3 = [tent_fn(a, g[2][i]) for i in range(2)]
    out = np.array([out1,out2,out3])
    return out


# df for lorenz 63 reference
df = lambda last_z, last_g: np.array(f(last_z) + g(a, last_g))


def forward_e(df, dt, *last_points):
    """
    Use forward euler to propagate the surrogate physical process.
    ----------
    Parameters
    df: function
        differential function for propagation
    dt: float
        time step
    last_points:
        last inputs calculated for df, with first one as the last postn z

    Return: float
        next point for ODE by forward euler
    """
    return np.array(last_points[0] + dt * df(*last_points))


# calculate zref
def vref(t):
    """
    Calculate reference physical process.
    ----------
    Parameters:
    t: endpoint time

    Returns:
    list of coordinates
    """
    out = [ICs]
    gs = [g0]
    for i in range(int(t/dt)):
        lastz = out[-1]
        lastg = gs[-1]
        nextz = forward_e(df, dt, lastz, lastg)
        # print(lastg)
        out.append(nextz)
        gs.append(g(a, lastg))
    return out


dt_out=350*dt



def first_comp(v):
    """
    Take the 1st vortex from list of coords
    ----------
    Parameter:
    z: list of 3d coords

    Return: 1d np.array
        list of x coords
    """
    return np.array([i[0] for i in v])


def xi(k):
    """
    Generate the 'semi random variables' for big xi
    -----------
    Parameters:
    k: int
        number of random vars to generate

    Return: list with len k
    """
    a = 4
    errs = [a * (2 ** -0.5 - 0.5)]
    for i in range(k):
        errs.append(tent_fn(a, errs[-1]))

    errs2 = [a * (2.1 ** -0.5 - 0.5)]
    for i in range(k):
        errs2.append(tent_fn(a, errs2[-1]))

    return errs[1:], errs2[1:]


def bigxi(k):
    """
    Generate more of xi and take every 10th
    ---------
    Parameter:
    k: int
        number of bigxi to make

    Return: list with len k
    """
    e = xi(10*k)
    er = [[e[0][10*i - 1] for i in range(1, k+1)],[e[1][10*i - 1] for i in range(1, k+1)]]
    return er


def errorsum(c, N_obs):
    """
    Calculate the sum terms for measurement error in observations.
    ---------
    Parameters:
    c: int
        Number of bigxi to be summed
    N_obs: int
        Number of observations

    Return: N_obs len np.array of floats
        Observation error terms in a list
    """
    k = c * N_obs
    E = bigxi(k)
    err_ls = []
    err_ls2=[]
    for i in range(N_obs):
        err_ls.append(sum(E[0][i*20:i*20 + 20]))
        err_ls2.append(sum(E[1][i*20:i*20 + 20]))

    err_ls = np.array(err_ls) / c
    err_ls2 = np.array(err_ls2) / c

    return err_ls,err_ls2


def v1_obs(N_obs, dt_out, zref):
    """
    Create list of x&y-coord observations
    ----------
    Parameters:
    N_obs: int
        number of observations
    dt_out: float
        time step for observations
    zref: fn
        fn for coordinates of the real surrogate process

    Return: N_obs len list
        list of observations
    """
    c = 20
    err_ls = errorsum(c, N_obs)
    
    vreal = vref(N_obs * dt_out)
    xreal = np.array(first_comp(vreal))
    xreal = xreal[350::350]

    return xreal + np.transpose(err_ls)


t = dt_out*len(xs)
v1_ref = [[item[0][0] for item in vref(t)][0::350],[item[0][1] for item in vref(t)][0::350]]

N_obs=200
v1=[item[0] for item in vref(dt_out*N_obs)]
v1s=v1[slice(350,len(v1),350)]
xs= [item[0] for item in v1s]
ys =[item[1] for item in v1s]
xobs = [item[0] for item in v1_obs(N_obs, dt_out, vref)]
yobs = [item[1] for item in v1_obs(N_obs, dt_out, vref)]

tlist = np.arange(0,dt_out*len(xs),dt_out)
plt.figure(figsize=(11,4))
plt.plot(tlist, xobs, 'x')
plt.plot(tlist, v1_ref[0], marker="x",label ='reference')
plt.xlabel("time")
plt.ylabel("x_obs")

plt.figure(figsize=(11,4))
plt.plot(tlist,yobs,'x') 
plt.xlabel("time")
plt.ylabel("y_obs")

plt.figure(figsize=(11,4))
plt.plot(tlist,np.array(xs)-np.array(xobs))
plt.plot(tlist, v1_ref[1], marker="x",label='reference')
plt.xlabel("time")
plt.ylabel("error in x coord of the observations")

plt.figure(figsize=(11,4))
plt.plot(tlist,np.array(ys)-np.array(yobs)) 
plt.xlabel("time")
plt.ylabel("error in y coord of the observations")

plt.show()