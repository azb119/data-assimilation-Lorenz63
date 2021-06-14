from scipy.integrate import solve_ivp  # noqa F401
import numpy as np
from numpy import linalg as LA


# Section 4.1
# first implementation of ODEs
def rhs(s, v):
    pairs = [v[i:i+2] for i in range(0, len(v), 2)]
    x = v[::2]
    y = v[1::2]

    # implement ODE for each pair x_alpha, y_alpha
    fn = [[
        -1/(2*np.pi)*sum(gamma[k]*(y[i]-y[k])/((x[i]-x[k])**2+(y[i]-y[k])**2) for k in range(len(x)) if k!=i),  # noqa E501
        1/(2*np.pi)*sum(gamma[k]*(x[i]-x[k])/((x[i]-x[k])**2+(y[i]-y[k])**2) for k in range(len(x)) if k!=i)  # noqa E501
        ] for i in range(len(pairs))]
    return [item for sub in fn for item in sub]


# Section 4.2
# mechanistic model
ICs = np.array([[1.4, 0], [-0.5, 0.8], [-1, -2.1]])  # intial points
gamma = [12, 5, 6]  # vortex strengths
dt = 0.001  # time step
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
    x = [v[i][0] for i in range(0, len(v))]
    y = [v[i][1] for i in range(0, len(v))]
    fn = [[-1/(2*np.pi)*sum(gamma[k]*(y[i]-y[k])/((x[i]-x[k])**2+(y[i]-y[k])**2) for k in range(len(x)) if k != i),  # noqa E501
          1/(2*np.pi)*sum(gamma[k]*(x[i]-x[k])/((x[i]-x[k])**2+(y[i]-y[k])**2) for k in range(len(x)) if k != i)]  # noqa E501
          for i in range(len(v))]
    return np.array([item for item in fn])


def vmod(t, ic=ICs, dt=dt):
    """
    Calculate the coordinates of the mechanistic model.
    ----------
    Parameters:
    t: int or float
        endpoint time
    ic: 3d array of 2d coordinates
        intial conditions
    dt: float
        timestep for simulations

    Return: list of 2d coordinates
        list of points of the mechanistic model
    """
    z_new = ic
    points = [ic]
    for i in range(int(t/dt)):
        z_new = z_new + dt*(f(z_new))
        points.append(z_new)
    return points


# reference model
g0 = np.array([[a * ((2+(-0.2)*i) ** -0.5 - 0.5),  # initial g
                a * ((3+(-0.2)*i) ** -0.5 - 0.5)] for i in range(3)])


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
    out = np.array([out1, out2, out3])
    return out


# df for vortices ODEs with g
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


def vref(t):
    """
    Calculate reference physical process.
    ----------
    Parameters:
    t: int or float
        endpoint time

    Returns: array of 3d arrays containing x-y coordinates
        list of coordinates of reference model
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


# Section 4.3
# generating partial observations
def first_comp(v):
    """
    Take the 1st vortex from list of coords
    ----------
    Parameter:
    v: list of 3d coords

    Return: 1d np.array
        list of first item in v
    """
    return np.array([i[0] for i in v])


def xi(k):
    """
    Generate the 'semi random variables' for big xi
    -----------
    Parameters:
    k: int
        number of random vars to generate

    Return: 2 lists with len k
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

    Return: 2 lists with len k
    """
    e = xi(10*k)
    er = [[e[0][10*i - 1] for i in range(1, k+1)],
          [e[1][10*i - 1] for i in range(1, k+1)]]
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
    err_ls2 = []
    for i in range(N_obs):
        err_ls.append(sum(E[0][i*20:i*20 + 20]))
        err_ls2.append(sum(E[1][i*20:i*20 + 20]))

    err_ls = np.array(err_ls) / c
    err_ls2 = np.array(err_ls2) / c

    return err_ls, err_ls2


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


# Section 4.4
# stochastic difference equations
dt_sim = 0.001  # time step for simulations


def stochastic_g():
    """
    Calculate randomised g
    -------
    Return: np.array with shape (3,2) of random samples
    """
    a = 1/(10*np.sqrt(dt_sim))
    sigma = np.sqrt(0.0838)
    vals = np.random.normal(0, sigma, (3, 2))
    return a * vals


def sde_df(last_v):
    """
    df for stochastic difference eqn
    --------
    Parameters:
    last_z: 3d array with shape (3,2)
        coordinate of the last position

    Return: np.array with shape (3,2)
        calculated df for the next forward euler
    """
    return f(last_v) + stochastic_g()


def sim_v(n, ICs):
    """
    Propagate v n steps
    ----------
    Parameters:
    n: int
        number of steps to propagate
    ICs: 3d array with shape (3,2)
        initial conditions to propagate from

    Return: n+1 array of 3d arrays of xy coordinates
        list of z predicted from initial conditions till nth
    """
    out = [ICs]
    for i in range(n):
        out.append(forward_e(sde_df, dt_sim, out[-1]))
    return out


# Section 4.5 & 4.6
N_obs = 200  # number of observations
dt_out = 0.25


# functions to be used in SIs and SIR filters
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
        vals.append(np.random.normal(mui, sig, (M, 2)))

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


# SIS Implementation
def SIS(M):
    observations = v1_obs(N_obs, dt_out, vref)
    ensemble_weights = np.zeros((M, N_obs+1))
    initial_samples = init_ensemble(M, ICs, 0.1)
    initial_w = [1/M for i in range(M)]
    weights = initial_w.copy()
    ensemble_weights[:, 0] = weights.copy()
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
            new_coord = sim_v(int(dt_out/dt), current_ensemble[i])[-1]
            new_ens.append(new_coord)
        current_ensemble = new_ens

        # add the new coords to ensemble_path
        for path in range(M):
            ensemble_paths[path].append(current_ensemble[path])

        # new weights
        weights = next_w(weights, observations[observing], current_ensemble)
        ensemble_weights[:, observing + 1] = np.transpose(weights.copy())

        # effective sample size
        eff_M_hist.append(M_eff(weights))

    return eff_M_hist, ensemble_paths, observations, ensemble_weights


# Functions to be used in RMSE and weighted average calculations
def mean_y(enpaths, k, vortex, axis, weights):
    yi_tk = [path[k][vortex][axis] for path in enpaths]
    ws = np.transpose(weights[:, k])
    return np.average(yi_tk, weights=ws)


def v1_xrmse(m, N_obs, path, weights, vortex):
    x_sim = [mean_y(path, k, vortex, 0, weights) for k in range(0, N_obs+1)]
    observe = [item[0] for item in v1_obs(N_obs, dt_out, vref)]
    return np.sqrt(np.mean((np.asarray(x_sim[1:]) - np.asarray(observe))**2))


def v1_yrmse(m, N_obs, path, weights, vortex):
    y_sim = [mean_y(path, k, vortex, 1, weights) for k in range(0, N_obs+1)]
    observe = [item[1] for item in v1_obs(N_obs, dt_out, vref)]
    return np.sqrt(np.mean((np.asarray(y_sim[1:]) - np.asarray(observe))**2))


# SIR implementation
def SIR(M):
    observations = v1_obs(N_obs, dt_out, vref)

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
            new_coord = sim_v(int(dt_out/dt), current_ensemble[i])[-1]
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


# Functions to be used in RMSE and mean calculations
def mean_y2(enpaths, k, vortex, axis):
    yi_tk = [path[k][vortex][axis] for path in enpaths]
    return np.mean(yi_tk)


def v1_xrmse2(m, N_obs, path, vortex):
    x_sim = [mean_y2(path, k, vortex, 0) for k in range(0, N_obs+1)]
    observe = [item[0] for item in v1_obs(N_obs, dt_out, vref)]
    return np.sqrt(np.mean((np.asarray(x_sim[1:]) - np.asarray(observe))**2))


def v1_yrmse2(m, N_obs, path, vortex):
    y_sim = [mean_y2(path, k, vortex, 1) for k in range(0, N_obs+1)]
    observe = [item[1] for item in v1_obs(N_obs, dt_out, vref)]
    return np.sqrt(np.mean((np.asarray(y_sim[1:]) - np.asarray(observe))**2))
