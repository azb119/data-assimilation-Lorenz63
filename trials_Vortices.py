import numpy as np
import matplotlib.pyplot as plt

ICs = np.array([[1.4,0], [-0.5,0.8],[-1,-2.1]])
gamma = [12, 5,6]
dt = 0.001
a = 1/(10*np.sqrt(dt))
g0 = np.array([[a * ((2+(-0.2)*i) ** -0.5 - 0.5),a * ((3+(-0.2)*i) ** -0.5 - 0.5)]
               for i in range(3)])


# surrogate physical process
def f(z):
    """
    Calculate f for the next time step.
    -----------
    Parameters:
    z: Nd array of 2d array of floats [[x0, y0], [x1, y1], ..]
        current positions of each vortex

    Return: array of 2d array of floats
        next positions of every vortices
    """
    x = [i[0] for i in z]
    y= [i[1] for i in z]
    x1 = np.array([-1 / (2 * np.pi) *sum([gamma[b] * (y[a] - y[b]) / ((y[a]-y[b])**2 + (x[a]-x[b])**2)
               for b in range(len(gamma)) if a!=b]) for a in range(len(gamma))])
    x2 = np.array([-1 / (-2 * np.pi) *sum([gamma[b] * (x[a] - x[b]) / ((y[a]-y[b])**2 + (x[a]-x[b])**2)
               for b in range(len(gamma)) if a!=b]) for a in range(len(gamma))]) 
    return np.array([[x1[i], x2[i]] for i in range(len(gamma))])


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
    return np.array(last_points[0] + dt * 1.1* df(*last_points))


# without g
def v_det(t, ICs, dt):
    z_new = ICs
    points = [ICs]
    for i in range(int(t/dt)):
        z_new = forward_e(f, dt, z_new)
        points.append(z_new)
    return points


"""
t=10
v=v_det(t,ICs,dt)
x1 = [i[0][0] for i in v]
y1 = [i[0][1] for i in v]
x2 = [i[1][0] for i in v]
y2 = [i[1][1] for i in v]
x3 = [i[2][0] for i in v]
y3 = [i[2][1] for i in v]

plt.figure()
plt.plot(x1,y1)
plt.plot(x2,y2)
plt.plot(x3,y3)
plt.xlabel('x')
plt.ylabel('y')
plt.plot([ics[0] for ics in ICs], [ics[1] for ics in ICs], 'ob')
plt.show()
"""


# with g
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
        out = (1.99999*gi + a/2)
    else:
        out = (-1.99999*gi + a/2)
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
    out = []
    for j in g:
        out.append([tent_fn(a, j[i]) for i in range(2)])
    return np.array(out)


df = lambda last_z, last_g: np.array(f(last_z) + g(a, last_g))


def vref(t):
    """
    Calculate reference physical process.
    ----------
    Parameters:
    t: endpoint time

    Returns:
    list of 3d coordinates
    """
    out = [ICs]
    gs = [g0]
    for i in range(int(t/dt)):
        lastz = out[-1]
        lastg = gs[-1]
        nextz = forward_e(df, dt, lastz, lastg)
        out.append(nextz)
        gs.append(g(a, lastg))
    return out

"""
t=10
v=vref(t)
x1 = [i[0][0] for i in v]
y1 = [i[0][1] for i in v]
x2 = [i[1][0] for i in v]
y2 = [i[1][1] for i in v]
x3 = [i[2][0] for i in v]
y3 = [i[2][1] for i in v]


plt.figure()
plt.plot(x1,y1)
plt.plot(x2,y2)
plt.plot(x3,y3)
plt.show()
"""

# partial obs
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
    xreal = xreal[250::250]

    return xreal + np.transpose(err_ls)


# stochastic difference equations
dt_sim = 0.001


def stochastic_g():
    """
    Calculate randomised g
    -------
    Return: 3d np.array of random samples
    """
    a = 1/(10*np.sqrt(dt_sim))
    sigma = np.sqrt(0.0838)
    vals = np.random.normal(0, sigma, (3,2))
    return a * vals
# print(stochastic_g())


def sde_df(last_v):
    """
    df for stochastic difference eqn
    --------
    Parameters:
    last_z: 3d array
        coordinate of the last position

    Return: 3d np.array
        calculated df for the next forward euler
    """
    return f(last_v) + stochastic_g()
# print(sde_df(ICs))


def sim_v(n, ICs):
    """
    Propagate v n steps
    ----------
    Parameters:
    n: int
        number of steps to propagate
    ICs: 3d array
        initial conditions to propagate from

    Return: n+1 array
        list of z predicted from initial conditions till nth
    """
    out = [ICs]
    for i in range(n):
        out.append(forward_e(sde_df, dt_sim, out[-1]))
    return out


N_obs = 200  # number of observations

from numpy import linalg as LA

def likelihoodfn(current_obs, current_forecast):
    """
    calculate pi_y for a specific particle given the observed and forecast
    ---------
    Parameters:
    current_obs: float
        (x,y) vals for the current obs
    current_forecast: 3d array
        predicted current coordinates

    Return: float
        likelihood
    """
    R = 1 / 15
    p = -0.5 / R * pow(LA.norm(current_obs - current_forecast[0]), 2)
    return np.exp(p) / np.sqrt(2*np.pi*R)


def init_ensemble(M, mu, sig):
    """
    Give an initial ensemble from normal dist N(mu, sig^2)
    ---------
    Parameters:
    M: int
        ensemble sample size
    mu: 3d array
        sampling mean for each axis
    sig: float
        standard dev for sampling

    Return: M len array of 3d coords
        list of coords of ensemble of initial particles
    """
    vals = []
    for mui in mu:
        vals.append(np.random.normal(mui, sig, (M,2)))

    coords = [[vals[0][i], vals[1][i], vals[2][i]] for i in range(M)]
    return coords
# print(init_ensemble(4, ICs, 0.1),np.shape(init_ensemble(4, ICs, 0.1)))

def M_eff(w):
    """
    calc the effective sample size
    ----------
    Parameters:
    w: M len array
        weights array

    Return: float
        effective sample size
    """
    return 1 / sum(np.array(w) ** 2)


def next_w(current_w, current_obs, current_fore):
    """
    calc the next set of weights
    ---------
    Parameters:
    current_w: M len array
        old weights
    current_obs: float (as we only see x coord)
        newest observed position
    current_fore: M length array of 3d coords, same len as current_w
        forecasts of each particle

    Return: M len array
        new weights
    """
    # M = len(current_w)
    likelies = []
    for forecast in current_fore:
        likelies.append(likelihoodfn(current_obs, forecast))
    probs = np.array(likelies) * np.array(current_w)
    denom = sum(probs)
    # print("denom", denom)
    return probs / denom


def resampling(M, current_w, old_ensemble):
    """
    sample new ensemble from the old ensemble using weights as probabilities
    ----------
    Parameters:
    M: int
        ensemble size
    current_w: M len array
        current weights
    old_ensemble: M len array of 3d coords
        ensemble to be sampled from

    Return: M len array of 3d coords
        new ensemble with repetitions
    """
    sampling = np.random.multinomial(M, current_w)
    new_ensemble = []
    # print("tot weights", sum(weights))
    # print("resampling..")
    for idx, amnt in enumerate(sampling):
        if amnt:
            new_ensemble += [old_ensemble[idx] for i in range(amnt)]
    # print(new_ensemble)
    return new_ensemble


def SIR(M):
    """
    Run the SIR filter, using x_obs with underlying zref.
    The ikelihood used will be likelihoodfn
    """
    observations = x_obs(N_obs, dt_out, zref)

    initial_samples = init_ensemble(M, ICs, 0.1)
    initial_w = [1/M for i in range(M)]

    weights = initial_w.copy()
    current_ensemble = initial_samples.copy()
    ensemble_paths = [[i] for i in initial_samples]
    # ensemble paths will look as below
    # ensemble_paths = [
    #   [[1st coord of 1st path], [2nd coord], ..],  # particle one
    #   ...
    #   [[1st coord of Mth path], [2nd coord], ..]  # particle M
    # ]
    eff_M_hist = [M_eff(weights)]  # history of effective sample size to graph

    for observing in range(N_obs):

        # propagate each point to the next observation
        new_ens = []
        for i in range(M):
            new_coord = sim_z(int(dt_out/dt), current_ensemble[i])[-1]
            new_ens.append(new_coord)
        current_ensemble = new_ens

        # add the new coords to ensemble_path
        for path in range(M):
            ensemble_paths[path].append(current_ensemble[path])

        # new weights
        weights = next_w(weights, observations[observing], current_ensemble)

        # effective sample size
        eff_M_hist.append(M_eff(weights))

        # resampling
        if eff_M_hist[-1] < M/2:
            current_ensemble = resampling(M, weights, current_ensemble)
            weights = [1/M for i in range(M)]

    return eff_M_hist, ensemble_paths, observations

dt_out=250*dt
N_obs=200
v1=[item[0] for item in vref(dt_out*N_obs)]
v1s=v1[slice(250,len(v1),250)]
xs= [item[0] for item in v1s]
ys =[item[1] for item in v1s]
t = dt_out*len(xs)
xobs = [item[0] for item in v1_obs(N_obs, dt_out, vref)]
yobs = [item[1] for item in v1_obs(N_obs, dt_out, vref)]
tlist = np.arange(0,dt_out*len(xs),dt_out)

v1_ref = [[item[0][0] for item in vref(t)][0::250],[item[0][1] for item in vref(t)][0::250]]

fig, axs = plt.subplots(4)
# plt.figure(figsize=(11,4))
axs[0].plot(tlist, xobs, 'x', label='x_obs')
axs[0].plot(np.arange(0,(len(v1_ref[0])-0.5)*dt_out,dt_out), v1_ref[0], marker="x",label ='reference')
axs[0].set(xlabel='time', ylabel='x')
axs[0].legend()

# plt.figure(figsize=(11,4))
axs[1].plot(tlist,yobs,'x', label='y_obs') 
axs[1].plot(np.arange(0,(len(v1_ref[1])-0.5)*dt_out,dt_out), v1_ref[1], marker="x",label='reference')
axs[1].set(xlabel='time', ylabel='y')
axs[1].legend()

# axs[2].figure(figsize=(11,4))
axs[2].plot(tlist,np.array(xs)-np.array(xobs)) 
axs[2].set(xlabel='time', ylabel='error in x coord of the observation')

# axs[3].figure(figsize=(11,4))
axs[3].plot(tlist,np.array(ys)-np.array(yobs)) 
axs[3].set(xlabel='time', ylabel='error in y coord of the observation')

plt.show()
