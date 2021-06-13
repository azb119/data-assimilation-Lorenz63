import numpy as np
import matplotlib.pyplot as plt
from numpy.core import multiarray
from mpl_toolkits import mplot3d

# Section 2.1 & 2.2
# surrogate physical process: Lorenz 63 model

ICs = np.array([-0.587, -0.563, 16.870])  # initial conditions
dt = 0.001  # real process time step
a = 1/np.sqrt(dt)
g0 = np.array([  # initial g
    a * (2 ** -0.5 - 0.5),
    a * (3 ** -0.5 - 0.5),
    a * (5 ** -0.5 - 0.5)
])


def f(z):
    """
    Calculate f for the next time step.
    -----------
    Parameters:
    z: 3d array of floats [x, y, z]
        current position, z^n

    Return: 3d array
        f(z^n)
    """
    x1 = 10 * (z[1] - z[0])
    x2 = z[0] * (28 - z[2]) - z[1]
    x3 = z[0] * z[1] - 8 * z[2] / 3
    return np.array([x1, x2, x3])


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
        out.append(nextz)
        gs.append(g(a, lastg))
    return out


# Section 2.4
# mechanistic model
def zmod(t, ic=ICs, dt=dt):
    """
    Calculate the coordinates of the mechanistic model.
    ---------
    Parameters:
    t: int or float
        endpoint time
    ic: 3d array of floats
        initial conditions
    dt: float
        time step for the simulation

    Return: list of 3d arrays
        coordinates of each point of mechanistic model
    """
    z_new = ic
    points = [ic]
    for i in range(int(t/dt)):
        z_new = z_new + dt*(f(z_new))
        points.append(z_new)
    return points


# Section 2.3
# generating partial observations
dt_out = 0.05  # observation time step
N_obs = 200  # number of observations


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
    Calculate the summation terms for measurement error in observations.
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
    Create list of x-coord observations.
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


# Section 2.5
# stochastic difference equations
dt_sim = 0.001  # model time steps


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
print(stochastic_g())


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


# Section 3
# For use in SIR + SIS implementation
M = 100  # number of particles
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
    likelies = []
    for forecast in current_fore:
        likelies.append(likelihoodfn(current_obs, forecast))
    probs = np.array(likelies) * np.array(current_w)
    denom = sum(probs)
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
    for idx, amnt in enumerate(sampling):
        if amnt:
            new_ensemble += [old_ensemble[idx] for i in range(amnt)]
    return new_ensemble


# Section 3.2
# SIS Implementation
def SIS(M):
    """
    Run the SIS filter for a given sample size.
    ----------
    Parameters:
    M: int
        sample size
    
    Return: 3 tuple of lists
        effective sample size for every time step,
        paths of every ensemble member,
        weights of every ensemble for every time step
    """
    observations = x_obs(N_obs, dt_out, zref)
    ensemble_weights =np.zeros((M,N_obs+1))
    initial_samples = init_ensemble(M, ICs, 0.1)
    initial_w = [1/M for i in range(M)]
    weights = initial_w.copy()
    ensemble_weights[:,0]=weights.copy()
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
        ensemble_weights[:,observing+1]= np.transpose(weights.copy())

        # effective sample size
        eff_M_hist.append(M_eff(weights))

    return eff_M_hist, ensemble_paths, observations, ensemble_weights


# Section 3.3 & 3.4
# SIR implementation
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


# functions used to compute rmses and averages of ensemble members
# functions not ending in 2 used for SIS model; use weighted averages
# functions ending in 2 used for SIR model; use the mean
def mean_y(enpaths, k, axis, weights):
    yi_tk = [path[k][axis] for path in enpaths]
    ws = np.transpose(weights[:, k])
    return np.average(yi_tk, weights=ws)


def mean_y2(enpaths, k, axis):
    yi_tk = [path[k][axis] for path in enpaths]
    return np.mean(yi_tk)


def xrmse(m, N_obs, path, weights):
    x_sim = [mean_y(path, k, 0, weights) for k in range(0, N_obs+1)]
    observe = x_obs(N_obs, dt_out, zref)
    return np.sqrt(np.mean((np.asarray(x_sim[1:])
                   - np.asarray(observe[1:]))**2))


def yrmse(m, N_obs, path, weights):
    y_sim = [mean_y(path, k, 1, weights) for k in range(0, N_obs+1)]
    observe = [item[1] for item in zref(N_obs*dt_out)][0::50]
    return np.sqrt(np.mean((np.asarray(y_sim[1:])
                   - np.asarray(observe[1:]))**2))


def zrmse(m, N_obs, path, weights):
    z_sim = [mean_y(path, k, 2, weights) for k in range(0, N_obs+1)]
    observe = [item[2] for item in zref(N_obs*dt_out)][0::50]
    return np.sqrt(np.mean((np.asarray(z_sim[1:])
                   - np.asarray(observe[1:]))**2))


def xrmse2(m, N_obs, path, weights):
    x_sim = [mean_y2(path, k, 0) for k in range(0, N_obs+1)]
    observe = x_obs(N_obs, dt_out, zref)
    return np.sqrt(np.mean((np.asarray(x_sim[1:])
                   - np.asarray(observe))**2))


def yrmse2(m, N_obs, path, weights):
    y_sim = [mean_y2(path, k, 1) for k in range(0, N_obs+1)]
    observe = [item[1] for item in zref(N_obs*dt_out)][0::50]
    return np.sqrt(np.mean((np.asarray(y_sim[1:])
                   - np.asarray(observe[1:]))**2))


def zrmse2(m, N_obs, path, weights):
    z_sim = [mean_y2(path, k, 2) for k in range(0, N_obs+1)]
    observe = [item[2] for item in zref(N_obs*dt_out)][0::50]
    return np.sqrt(np.mean((np.asarray(z_sim[1:])
                   - np.asarray(observe[1:]))**2))
