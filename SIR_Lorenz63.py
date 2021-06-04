import numpy as np
import matplotlib.pyplot as plt  # noqa F401
from numpy.core import multiarray  # noqa F401
from numpy.random import default_rng  # noqa F401
from mpl_toolkits import mplot3d  # noqa F401


# real underlying Lorenz 63 model

def f(z):
    """
    Calculate f for the next time step.
    -----------
    Parameters:
    z: 3d array of floats [x, y, z]
        current position

    Return: 3d array
        next z
    """
    x1 = 10 * (z[1] - z[0])
    x2 = z[0] * (28 - z[2]) - z[1]
    x3 = z[0] * z[1] - 8 * z[2] / 3
    return np.array([x1, x2, x3])


ICs = np.array([-0.587, -0.563, 16.870])
dt = 0.001
a = 1/np.sqrt(dt)
g0 = np.array([
    a * (2 ** -0.5 - 0.5),
    a * (3 ** -0.5 - 0.5),
    a * (5 ** -0.5 - 0.5)
])


def g(a, g):
    """
    Calculate driver g for the next time step.
    ---------
    Parameters:
    a: float
        input for tent_fn
    g: 3d np.array
        the last g calculated

    Return: 3d np.array of floats
        next g
    """
    out = np.array([tent_fn(a, gi) for gi in g])
    return out


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


# df for lorenz 63 reference
df = lambda last_z, last_g: np.array(f(last_z) + g(a, last_g))  # noqa E731


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
def zref(t):
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


"""
t = 10
z = zref(t)
fig = plt.figure()
ax = plt.axes(projection='3d')
z1 = [i[0] for i in z]
z2 = [i[1] for i in z]
z3 = [i[2] for i in z]
ax.plot(z1, z2, z3)
plt.show()
"""

# partial observations
# say we have the model and zref is represented by zreal as below
# t = 10
# zreal = zref(t)
# zreal is an array of 3d positions, eg. [[0,0,0], [1,2,3], [2,3,4]]


def first_comp(z):
    """
    Take the x coordinate from list of coords
    ----------
    Parameter:
    z: list of 3d coords

    Return: 1d np.array
        list of x coords
    """
    return np.array([i[0] for i in z])


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
    return errs[1:]


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
    er = [e[10*i - 1] for i in range(1, k+1)]
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
    for i in range(N_obs):
        err_ls.append(sum(E[i*20:i*20 + 20]))
    err_ls = np.array(err_ls) / c
    return err_ls


def x_obs(N_obs, dt_out, zref):
    """
    Create list of x-coord observations
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
    zreal = zref(N_obs * dt_out)
    xreal = np.array(first_comp(zreal))
    xreal = xreal[50::50]
    return xreal + err_ls


# intiliasing
dt_out = 0.05
N_obs = 200

"""
trial = x_obs(N_obs, dt_out, zref)
t = np.arange(0,dt_out*200,dt_out)
plt.figure()
plt.plot(t, trial, 'x')
plt.show()
"""


# stochastic difference equations
dt_sim = 0.001


def stochastic_g():
    """
    Calculate randomised g
    -------
    Return: 3d np.array of random samples
    """
    a = 1/np.sqrt(dt_sim)
    sigma = np.sqrt(0.0838)
    vals = np.random.normal(0, sigma, 3)
    return a * vals
# print(stochastic_g())


def sde_df(last_z):
    """
    df for stochastic difference eqn
    --------
    Parameters:
    last_z: 3d array
        coordinate of the last position

    Return: 3d np.array
        calculated df for the next forward euler
    """
    return f(last_z) + stochastic_g()
# print(sde_df(ICs))


def sim_z(n, ICs):
    """
    Propagate z n steps
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
# print(len(sim_z(5, ICs)))


"""
t = int(20/0.001)
zs = sim_z(t, ICs)
x_sim = first_comp(zs)
t_list = np.arange(0, t+dt_sim, dt_sim)
plt.figure()
#plt.plot(t_list, x_sim, 'x')
#plt.show()
ax = plt.axes(projection='3d')
z1 = [i[0] for i in zs]
z2 = [i[1] for i in zs]
z3 = [i[2] for i in zs]
ax.plot(z1, z2, z3)
plt.show()
"""

# SIR implementation
# Initialise values
# M = 100  # number of particles
N_obs = 200  # number of observations


def likelihoodfn(current_obs, current_forecast):
    """
    calculate pi_y for a specific particle given the observed and forecast
    ---------
    Parameters:
    current_obs: float
        x val for the current obs
    current_forecast: 3d array
        predicted current coordinates

    Return: float
        likelihood
    """
    R = 1 / 15
    p = -0.5 / R * pow(current_obs - current_forecast[0], 2)
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
        vals.append(np.random.normal(mui, sig, M))
    coords = [[vals[0][i], vals[1][i], vals[2][i]] for i in range(M)]
    return coords
# print(init_ensemble(4, ICs, 0.1))


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


# SIR run
def SIR(M):
    observations = x_obs(N_obs, dt_out, zref)

    initial_samples = init_ensemble(M, ICs, 0.1)
    initial_w = [1/M for i in range(M)]

    weights = initial_w.copy()
    current_ensemble = initial_samples.copy()
    ensemble_paths = [[i] for i in initial_samples]
    # ensemble paths will look as below
    # ensemble_paths = [
    #   [[1st coord of path], [2nd coord], ..],  # particle one
    #   ...
    #   [[1st coord of path], [2nd coord], ..]  # particle M
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
        # print("eff M = ", eff_M_hist[-1])

        # resampling
        if eff_M_hist[-1] < M/2:
            # print(observing)
            current_ensemble = resampling(M, weights, current_ensemble)
            weights = [1/M for i in range(M)]

    return eff_M_hist, ensemble_paths, observations


# effective sample size plot
"""
trial = []
M_vals = [30, 100, 300]
for value in M_vals:
    trial.append(SIR(value)[0])
print(len(trial[0]))
print(N_obs)
plt.figure()
t_ls = np.linspace(0, N_obs*dt_out, len(trial[0]))
for runs in trial:
    plt.plot(t_ls, runs)
plt.show()
"""

"""
coords = SIR(8)[1]
plt.figure()
ax = plt.axes(projection='3d')
for i in coords:
    x = [j[0] for j in i]
    y = [j[1] for j in i]
    z = [j[2] for j in i]
    ax.plot(x, y, z)
plt.show()
"""


# RMSE
def mean_y(enpaths, k, axis):
    yi_tk = [path[k][axis] for path in enpaths]
    # print(len(yi_tk))
    return sum(yi_tk) / len(yi_tk)


def RMSE(enpaths, obs, axis):
    """
    Parameters:
    enpaths: list of list of coords
        list of each ensemble paths
    obs: list of coords
        all observations
    axis: 0, 1, or 2
        x, y or z coordinate to be calculated
    """
    total = 0
    for k in range(len(obs)):
        ym = mean_y(enpaths, k, axis)
        total += (ym - obs[k]) ** 2
    return np.sqrt(total / len(obs))


"""
storex = []
# storey = []
# storez = []
M_vals = [i*50 for i in range(1,7)]
for i in M_vals:
    useless, ep, ob = SIR(i)
    storex.append(RMSE(ep, ob, 0))
    #storey.append(RMSE(ep, ob, 1))
    #storez.append(RMSE(ep, ob, 2))

plt.figure()
plt.plot(M_vals, storex)
#plt.plot(M_vals, storey, '-.')
#plt.plot(M_vals, storez, '-x')

plt.show()
"""
