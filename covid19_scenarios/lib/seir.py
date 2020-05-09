"""
Run SEIR model, retrospectively estimate R_e, and plot different possible
scenarios of social distancing relaxation
"""
import os
from collections import namedtuple
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numba import jit, njit
from matplotlib.ticker import FuncFormatter

S_COLORS= ["#176BA0", "#3B9B1C", "#9C19A3", "#E46511", "#Ae0868", "#008FE0", "#076E00", "#BC70C7"][::-1]

L_COLORS = ["#bcbfc0", "#858585", "#3b3b3b"]

DESC = {
    "S": "Susceptible",
    "E": "Exposed",
    "I": "Infected",
    "R": "Recovered",
    "C": "Daily Cases",
}

MODEL_DEFAULTS = {
    "population": 918702,
    "infected_days": 5,
    "exposed_days": 3,
    "initial_daily_cases": 50,
    "test_catch_ratio": 0.1,
    "cases": 0,
    "start_date": pd.Timestamp.now().normalize(),
}

STEP_DEFAULTS = {
    "N": 0,
    "beta": 0.03,
    "gamma": 0.2,
    "sigma": 0.3333333333333333,
    "test_catch_ratio": 0.1,
    "cases": 0,
}
Parameters = namedtuple(
    "Parameters", list(MODEL_DEFAULTS.keys()), defaults=list(MODEL_DEFAULTS.values()),
)

StepParameters = namedtuple(
    "StepParameters", list(STEP_DEFAULTS.keys()), defaults=list(STEP_DEFAULTS.values()),
)

State = namedtuple("State", list(DESC.keys()), defaults=[0] * len(DESC.keys()))


def seir_step(state, params):
    import warnings

    N, beta, gamma, sigma = params.N, params.beta, params.gamma, params.sigma
    S, E, I, R = state.S, state.E, state.I, state.R
    with np.errstate(over="raise"):
        try:
            return State(
                S=S - beta * S * I / N,
                E=E + beta * S * I / N - sigma * E,
                I=I + sigma * E - gamma * I,  # noqa: E741
                R=R + gamma * I,  # nopq: E741
                C=sigma * E * params.test_catch_ratio,
            )
        except FloatingPointError:
            print(state, params)
            raise ValueError


def get_beta(R_0, gamma, sigma):
    return max(0, qsolve(1.0 / gamma / sigma, 1 / sigma + 1 / gamma, 1 - R_0)) + gamma


def seir(params, R_0):
    gamma = 1.0 / params.infected_days
    sigma = 1.0 / params.exposed_days
    initial_daily_cases = max(1, params.initial_daily_cases)
    initial_infected = initial_daily_cases / gamma / params.test_catch_ratio
    initial_exposed = sigma / (sigma + gamma) * initial_infected
    E = initial_exposed
    I = initial_infected  # noqa: E741
    R = params.cases / params.test_catch_ratio - E - I
    S = params.population - E - I - R

    # herd immunity correction
    test_step = seir_step(
        State(S, E, I, R),
        StepParameters(
            N=params.population,
            beta=get_beta(R_0[0], gamma, sigma),
            gamma=gamma,
            sigma=sigma,
            test_catch_ratio=params.test_catch_ratio,
        ),
    )
    herd_correct = initial_daily_cases / test_step.C
    E *= herd_correct
    I *= herd_correct
    R = params.cases / params.test_catch_ratio - E - I
    S = params.population - E - I - R

    state = State(S, E, I, R)
    states = [state]
    for current_R_0 in R_0:
        states.append(
            seir_step(
                states[-1],
                StepParameters(
                    N=params.population,
                    beta=get_beta(current_R_0, gamma, sigma),
                    gamma=gamma,
                    sigma=sigma,
                    test_catch_ratio=params.test_catch_ratio,
                ),
            )
        )
    out = pd.DataFrame(states[1:])
    out.index = pd.date_range(start=params.start_date, periods=len(out.index))
    return out


def exp_relax(a, b, t, k=0.5):
    """Relax linear function exponentially"""
    # ce^kt0 + d = a * t0 + b
    # cke^kt0 = a
    k = -np.abs(k)
    t0 = t[0]
    c = a / k * np.exp(-k * t0)
    d = a * t0 + b - c * np.exp(k * t0)
    return c * np.exp(k * t) + d


def social_distancing_reduction_scenario(
    data,
    daily_cases,
    R_e,
    population,
    social_distancing_reduction=0.05,
    additional_days=200,
    relaxation_date="5/15/2020",
    relaxation_days=10,
    relaxation_ratio=0.1,
):
    R_0 = R_e.max()
    # sanity check on R_0 values
    R_0 = max(R_0, 3.0)
    R_0 = min(R_0, 7.0)

    R_e_current = R_e.iloc[-1]
    if pd.to_datetime(relaxation_date) <= R_e.index[-1]:
        relaxation_date = R_e.index[-1]
    pre_relax_days = (pd.to_datetime(relaxation_date) - R_e.index[-1]).days
    try:
        initial_fit = fit_poly(np.arange(-4, 1, dtype=float), R_e.iloc[-5:].values, 1)
    except np.linalg.LinAlgError:
        print(R_e)
        print(data)
        raise ValueError
    if pre_relax_days > 0:
        pre_relax_R_e = exp_relax(
            initial_fit[-2],
            R_e_current,
            np.arange(pre_relax_days + 1),
            k=max(1.5, pre_relax_days / 10.0),
        )[1:]
    else:
        pre_relax_R_e = np.array([])
    R_e_forecast = pre_relax_R_e[-1] if pre_relax_days > 0 else R_e_current
    R_e_eventual = (
        social_distancing_reduction * max(R_e_forecast, R_0)
        + (1 - social_distancing_reduction) * R_e_forecast
    )
    R_e_future = smooth_clamp(
        pd.Series(range(relaxation_days + 1 + additional_days)),
        R_e_forecast,
        R_e_eventual,
        cutoff=relaxation_days / 2,
        width=relaxation_days / 2,
    )[1:]
    R_e_future = np.concatenate([pre_relax_R_e, R_e_future])
    daily_fit = np.polyfit(range(-4, 0), daily_cases.iloc[-4:], 1)
    initial_daily_cases = daily_fit[-1]
    results = seir(
        Parameters(
            infected_days=5,
            exposed_days=3,
            initial_daily_cases=initial_daily_cases,
            population=population,
            test_catch_ratio=0.15,
            cases=data[-1],
        ),
        R_e_future,
    )
    R_e_future = pd.Series(R_e_future, index=results.index)
    return results, R_e_future


def plot_scenarios(
    data,
    daily_cases,
    R_e,
    population,
    reductions,
    relaxation_date="5/15/2020",
    suptitle=None,
    show=True,
    save_filename=None,
    debug=False,
):
    if save_filename:
        try:
            os.remove(save_filename)
        except FileNotFoundError:
            pass
    pre_relax_days = (pd.to_datetime(relaxation_date) - R_e.index[-1]).days
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=False, figsize=(13, 14))
    legend_ratio = 0.60
    plt.tight_layout(pad=3, h_pad=8, w_pad=3, rect=(-(1 - legend_ratio), 0, legend_ratio, 0.93))
    data.loc[daily_cases.index].plot(ax=axes[1], label="_nolegend_", color="black")
    for reduction, color in zip(reductions[::-1], S_COLORS):
        results, R_e_future = social_distancing_reduction_scenario(
            data, daily_cases, R_e, population, reduction, relaxation_date=relaxation_date
        )
        res = results.C.iloc[: 30 + pre_relax_days]
        res.loc[daily_cases.index[-1]] = daily_cases.iloc[-1]
        res.sort_index(inplace=True)
        res.plot(ax=axes[0], label=f"{reduction * 100:.0f}%", color=color)
        res = pd.Series(R_e_future, index=results.index).iloc[: 30 + pre_relax_days]
        res.loc[daily_cases.index[-1]] = R_e.iloc[-1]
        res.sort_index(inplace=True)
        res.plot(ax=axes[2], label=f"{reduction * 100:.0f}%", color=color)
        pd.concat([data.iloc[-1:], results.C]).sort_index().cumsum().iloc[
            0 : 31 + pre_relax_days
        ].plot(ax=axes[1], label=f"{reduction * 100:.0f}%", color=color)
    daily_cases.plot(ax=axes[0], label="_nolegend_", color="black")
    R_e.loc[daily_cases.index].plot(ax=axes[2], label="_nolegend_", color="black")
    axes[0].set_title("Projected Daily Case Counts")
    axes[1].set_title("Projected Cumulative Case Counts")
    axes[2].set_title("Projected $R_e$")
    axes[2].axhline(y=1, linestyle="-", color="black", alpha=0.4, linewidth=2, label="Zero Growth")
    axes[0].set_ylabel("Cases")
    axes[1].set_ylabel("Cases")
    axes[2].set_ylabel("$R_e$")

    for ax in axes:
        plt.subplots_adjust(left=0.2)
        ax.get_yaxis().get_major_formatter().set_scientific(False)
        if ax != axes[2]:
            ax.get_yaxis().set_major_formatter(
                FuncFormatter(lambda x, p: format(int(x), ","))
            )
        ax.axvline(
            x=pd.to_datetime(relaxation_date), linestyle="--", label="Relaxation Date", color=L_COLORS[2]
        )
        ax.axvline(x=data.index[-1], linestyle="dotted", label="Today", color=L_COLORS[2])
        ax.legend(
            title="Social Distancing \nRelaxation",
            bbox_to_anchor=(1 / legend_ratio - 0.1, 1.02),
            fontsize=8,
            title_fontsize=8,
        )
        ax.grid(color="black", alpha=0.2, linewidth=0.5)
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
    axes[1].set_xlim([daily_cases.index[0], results.index[29 + pre_relax_days]])
    if suptitle:
        plt.suptitle(suptitle)
    if save_filename:
        plt.gcf().savefig(save_filename, bbox_inches="tight", pad_inches=0.5)
    if show:
        plt.show()
    else:
        plt.close()


@njit
def _coeff_mat(x, deg):
    mat_ = np.zeros(shape=(x.shape[0], deg + 1))
    const = np.ones_like(x)
    mat_[:, 0] = const
    mat_[:, 1] = x
    if deg > 1:
        for n in range(2, deg + 1):
            mat_[:, n] = x ** n
    return mat_


@jit
def _fit_x(a, b):
    # linalg solves ax = b
    det_ = np.linalg.lstsq(a, b)[0]
    return det_


@jit
def fit_poly(x, y, deg):
    a = _coeff_mat(x, deg)
    p = _fit_x(a, y)
    # Reverse order so p[0] is coefficient of highest order
    return p[::-1]


@jit
def eval_polynomial(P, x):
    """
    Compute polynomial P(x) where P is a vector of coefficients, highest
    order coefficient at P[0].  Uses Horner's Method.
    """
    result = 0
    for coeff in P:
        result = x * result + coeff
    return result


@njit
def _rolling_regression(x, y, interval=7):
    interval_up = int(np.floor(interval / 2))
    interval_down = interval - interval_up - 1
    out = np.zeros_like(x)
    out.fill(np.nan)
    for idx in range(x.size):
        int_slice = slice(
            max(0, idx - interval_down), min(x.size, idx + interval_up + 1)
        )
        if idx + interval_up + 1 > x.size:
            fit = fit_poly(x[int_slice], y[int_slice], 1)
            x0 = x[idx]
            # calculate slope at x0
            out[idx] = fit[-2]
        else:
            fit = fit_poly(x[int_slice], y[int_slice], 2)
            x0 = x[idx]
            # calculate slope at x0
            out[idx] = fit[-2] + 2 * fit[-3] * x0  # + 3 * fit[-4] * x0 ** 2
    return out


def rolling_regression_log(x, y, interval=7):
    is_pd = isinstance(y, pd.Series)
    out = _rolling_regression(np.array(x), np.log(np.array(y)), interval=interval)
    if is_pd:
        out = pd.Series(out, index=y.index)
    return out


def rolling_regression(x, y, interval=7):
    is_pd = isinstance(y, pd.Series)
    try:
        out = _rolling_regression(np.array(x), np.array(y), interval=interval)
    except np.linalg.LinAlgError:
        print(x, y, interval)
        raise ValueError
    if is_pd:
        out = pd.Series(out, index=y.index)
    return out


@njit
def _noise_AR1(loc, errs, factor=0.5):
    noise = np.zeros_like(loc)
    noise[0] = errs[0]
    for idx in range(1, noise.size):
        noise[idx] = factor * noise[idx - 1] + (1 - factor) * errs[idx]
    return loc + noise


def noise_AR1(loc, scale, factor=0.5):
    theta = scale ** 2 / np.clip(loc, 1, None)
    k = loc ** 2 / np.clip(scale, 1, None) ** 2
    errs = np.random.gamma(k, theta)
    return _noise_AR1(loc, errs, factor)


def _fuzz_diff(s, noise=0.1, factor=0.5):
    diff = np.diff(s)
    diff[diff < 0] = 0
    diff_out = noise_AR1(
        diff,
        np.clip(noise * np.clip(diff,0.5 * np.median(diff), None), 1, None),
        factor if not np.isnan(factor) else 0.5,
    )
    diff_out[diff_out < 0.0] = 0.0
    diff_out = diff_out.round()
    return np.cumsum(np.concatenate([s[:1], diff_out]))


def fuzz_diff(s, noise=0.2, factor=0.5):
    return pd.Series(_fuzz_diff(s.values, noise, factor), index=s.index)


def qsolve(a, b, c):
    if b ** 2 < 4 * a * c:
        return -b / (2 * a)
    else:
        return (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)


def growth_data(
    data,
    cutoff_date=None,
    cutoff_threshold=10,
    t_e=5,
    t_i=3,
    interval=11,
    window=11,
    lpad=False,
):
    daily_cases = (
        data.rolling(int(window / 2 + 1), win_type="gaussian", center=True)
        .mean(std=2)
        .fillna(data)
        .pipe(
            (rolling_regression, "y"),
            x=np.array(range(len(data)), dtype=float),
            interval=window,
        )
        .clip(1, None)
    )
    growth_rate = (
        (
            daily_cases.clip(1, None)
            .pipe(
                (rolling_regression_log, "y"),
                x=np.array(range(len(data)), dtype=float),
                interval=window,
            )
            .fillna(method="bfill")
            .dropna()
            .loc[data.index]
        )
        .ewm(halflife=4)
        .mean()
    )

    cases = (
        data.rolling(window, center=True, win_type="gaussian").mean(std=2).fillna(data)
    )
    cases.iloc[-int(np.ceil(window / 2)) :] = data.iloc[-int(np.ceil(window / 2)) :]
    first_above_threshold = daily_cases[
        daily_cases > max(10, 0.01 * daily_cases.quantile(0.9))
    ].index[0]
    if cutoff_date is None:
        cutoff_date = first_above_threshold
    z = growth_rate
    Z_BOUND = 0.3
    R_e = (
        1
        + z * (t_e + t_i)
        + np.where(
            z <= Z_BOUND,
            z ** 2,
            Z_BOUND ** 2 + 2 * Z_BOUND * (np.abs(z) - Z_BOUND) * t_e * t_i,
        )
    )
    R_e = R_e.where(z >= -(t_e + t_i) / (2 * t_e * t_i), 0).clip(0, None)
    if lpad:
        cutoff_date_adj = max(
            cutoff_date - pd.Timedelta(days=np.floor(window / 2)), daily_cases.index[0]
        )
    else:
        cutoff_date_adj = cutoff_date

    daily_cases = daily_cases.loc[cutoff_date_adj:]
    cases = cases.loc[cutoff_date_adj:]
    z = z.loc[cutoff_date_adj:]
    R_e = R_e.loc[cutoff_date_adj:]

    growth_rate = growth_rate.loc[daily_cases.index]

    if len(daily_cases) < 3:
        raise ValueError

    return cases, daily_cases, growth_rate, z, R_e


def smooth_clamp(s, s_i, s_f, cutoff, width, decreasing=False):
    """
    Apply a smooth cutoff to a continuous series
    Values below cutoff - width are s_i, values above cutoff + width are s_f,
    and in between, we use a cubic spline with 0 derivative at the
    cutoff - width and cutoff + width
    If decreasing=True, switch the roles of 0 and 1
    """

    # map the interval [cutoff - width, cutoff + width]
    # to the interval [0, 1]
    t = (s - cutoff + width) / (2 * width)
    out = (
        # cubic interpolation from 0 to 1
        # that has 0 derivative at cutoff - width and cutoff + width
        (3 * t ** 2 - 2 * t ** 3)
        .where((t > 0) | t.isnull(), 0.0)
        .where((t < 1) | t.isnull(), 1.0)
        .fillna(0)
    )
    if decreasing:
        out = 1 - out
    return s_i + out * (s_f - s_i)


def arma_noise(s, order=(1, 0), window=5):
    from statsmodels.tsa.arima_model import ARMA
    from statsmodels.tsa.arima_process import arma_generate_sample

    t = s.diff()
    est = t.rolling(window, win_type="gaussian").mean(std=2)
    ratio = t.div(est.clip(1, None)).dropna()
    mod = ARMA(ratio.values, order=order).fit()
    ar = np.concatenate([np.array([1.0]), -mod.arparams])
    ma = np.concatenate([np.array([1.0]), -mod.maparams])
    return (
        pd.Series(
            2 / np.pi * np.arctan(0.25 * arma_generate_sample(ar, ma, nsample=s.size))
            + 1,
            index=s.index,
        )
        .mul(est.fillna(t))
        .round()
        .fillna(s)
        .cumsum()
        .astype(int)
    )


def lag1_autocorr(s, window=11):
    x = s.diff()
    x = x - x.rolling(window=window, center=True, min_periods=1).mean()
    return x.corr(x.shift(1))


def make_growth_plots(
    data,
    cutoff_date=None,
    cutoff_threshold=10,
    t_e=5,
    t_i=3,
    suptitle=None,
    show=True,
    save_filename=None,
    debug=False,
    error=True,
    window=11,
    sim_count=50,
    color_func=None,
    interp_interval="2H",
):
    if save_filename:
        try:
            os.remove(save_filename)
        except FileNotFoundError:
            pass
    cases, daily_cases, growth_rate, z, R_e = growth_data(
        data,
        cutoff_date=cutoff_date,
        cutoff_threshold=cutoff_threshold,
        t_e=t_e,
        t_i=t_i,
        window=window,
    )
    growth_data_partial = partial(
        growth_data,
        cutoff_date=cutoff_date,
        cutoff_threshold=cutoff_threshold,
        window=window,
        lpad=True,
    )
    daily_cut = data.diff().loc[
        pd.to_datetime(
            cutoff_date if cutoff_date else data.index[-1] - pd.Timedelta(days=20)
        ) :
    ]
    from statsmodels.tsa.stattools import acf
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    factor = acf(daily_cut, nlags=1, fft=True)[0]
    std = daily_cut.std() / daily_cut.mean()
    std = min(std, 0.3)
    sim = (
        growth_data_partial(fuzz_data, t_e=fuzz_t_e, t_i=fuzz_t_i)
        for fuzz_data, fuzz_t_e, fuzz_t_i in zip(
            (fuzz_diff(data, noise=std, factor=factor) for _ in range(sim_count)),
            (t_e for _ in range(sim_count)),
            (t_i for _ in range(sim_count)),
        )
    )
    R_e = R_e.ewm(halflife=2).mean()
    R_e_err = (
        pd.concat(
            (
                x[-1].loc[[y for y in daily_cases.index if y in x[-1].index]]
                for x in sim
            ),
            axis=1,
        )
        .sort_index()
        .fillna(method="ffill")
        .fillna(method="bfill")
        .loc[R_e.index]
        .copy(deep=False)
    )
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13, 9), sharex=False)
    plt.tight_layout(pad=4, h_pad=7, w_pad=4, rect=(0, 0, 1, 0.9))
    axes_flat = iter(axes.flatten())

    ax = next(axes_flat)
    daily_cases.plot(ax=ax, linewidth=3, label="Smoothed")
    data.diff().loc[daily_cases.index[0] :].plot(ax=ax, marker="x", linestyle="None", label="Raw", color="black")
    ax.legend(borderpad=0.2, framealpha=0.5)
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ",")))
    ax.grid(color="black", alpha=0.2, linewidth=0.5)
    ax.set_ylabel("Cases")
    ax.set_title("Daily Cases")

    ax = next(axes_flat)
    data.loc[cases.index].plot(ax=ax, linewidth=3)
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ",")))
    ax.grid(color="black", alpha=0.2, linewidth=0.5)
    ax.set_ylabel("Cases")
    ax.set_title("Cumulative Cases")

    ax = next(axes_flat)
    growth_rate.plot(label="Growth Rate", ax=ax, linewidth=3)
    ax.axhline(y=0, linestyle="-", color="black", alpha=0.4, linewidth=2)
    ax.grid(color="black", alpha=0.2, linewidth=0.5)
    ax.set_ylabel("Days${}^{-1}$")
    ax.set_title("Logarithmic Growth Rate")

    ax = next(axes_flat)
    ax.set_ylabel("$R_e$")
    # dummy plot to generate ticks
    R_e.plot(ax=ax, alpha=0, label="__nolegend__")
    # save ticks
    thresholds = color_func(0, return_thresholds=True)
    R_e_interp = R_e.resample(interp_interval).interpolate()
    R_e_thresh = pd.Series(color_func(R_e_interp), index=R_e_interp.index)
    R_e_thresh = (R_e_thresh != R_e_thresh.shift(1)).cumsum()
    R_e_df = pd.DataFrame({"R_e": R_e_interp, "thresh": R_e_thresh})
    R_e_df["group"] = R_e_df.thresh.diff().fillna(True).cumsum()
    for _, segment in R_e_df.groupby("group"):
        idx = segment.index.to_list()
        start = idx[0]
        end = idx[-1] + pd.Timedelta(interp_interval)
        color = color_func(segment.R_e.iloc[0])
        R_e_interp.loc[start:end].plot(ax=ax, color=color, linewidth=3, label="__nolegend__")

    #    R_e.clip(0, None).plot(ax=ax, linewidth=3)
    ax.axhline(y=1, linestyle="-", color="black", alpha=0.4, linewidth=2)
    R_e_err_5 = R_e_err.quantile(0.05, axis=1)
    R_e_err_34 = R_e_err.quantile(0.34, axis=1)
    R_e_err_68 = R_e_err.quantile(0.68, axis=1)
    R_e_err_95 = R_e_err.quantile(0.95, axis=1)

    err_ratio = (R_e - R_e_err_34).div(R_e_err_68 - R_e_err_34).clip(0.25, 0.75)
    err_ratio = err_ratio.where(R_e_err_68 != R_e_err_34, 0.5)
    R_e_err_l1 = err_ratio * (R_e_err_68 - R_e_err_34).clip(0.1, None)
    R_e_err_h1 = (1 - err_ratio) * (R_e_err_68 - R_e_err_34).clip(0.1, None)
    R_e_err_l2 = err_ratio * (R_e_err_95 - R_e_err_5).clip(0.2, None)
    R_e_err_h2 = (1 - err_ratio) * (R_e_err_95 - R_e_err_5).clip(0.2, None)
    R_e_err_l1 = R_e_err_l1.clip(
        0.5
        * pd.concat([R_e_err_l1.shift(x) for x in range(-1, 2)], axis=1).max(
            axis=1, skipna=True
        ),
        None,
    )
    R_e_err_h1 = R_e_err_h1.clip(
        0.5
        * pd.concat([R_e_err_h1.shift(x) for x in range(-1, 2)], axis=1).max(
            axis=1, skipna=True
        ),
        None,
    )
    R_e_err_l2 = R_e_err_l2.clip(
        0.5
        * pd.concat([R_e_err_l2.shift(x) for x in range(-2, 3)], axis=1).max(
            axis=1, skipna=True
        ),
        None,
    )
    R_e_err_h2 = R_e_err_h2.clip(
        0.5
        * pd.concat([R_e_err_h2.shift(x) for x in range(-2, 3)], axis=1).max(
            axis=1, skipna=True
        ),
        None,
    )

    R_e_err_l1 = R_e_err_l1.clip(0.05, None)
    R_e_err_h1 = R_e_err_h1.clip(0.05, None)
    R_e_err_l2 = R_e_err_l2.clip(0.10, None)
    R_e_err_h2 = R_e_err_h2.clip(0.10, None)
    ax.grid(color="black", alpha=0.2, linewidth=0.5)
    ax.fill_between(
        R_e.index,
        R_e - R_e_err_l1,
        R_e + R_e_err_h1,
        facecolor="#000000",
        alpha=0.2,
        label="\u00B11 Std. Dev.",
    )
    ax.fill_between(
        R_e.index,
        R_e - R_e_err_l2,
        R_e + R_e_err_h2,
        facecolor="#000000",
        alpha=0.1,
        label="\u00B12 Std. Dev.",
    )
    ylim = ax.get_ylim()
    legend_elements = ([
        Line2D([0], [0], color=color_func(thresholds[-1] + 0.001), lw=3, label=f"CRRI 4"),
    ] + [
        Line2D([0], [0], color=color_func(thresholds[idx] + 0.001), lw=3, label=f"CRRI {idx+2}")
        for idx in range(len(thresholds) - 2, -1, -1)
        ] + [
        Line2D([0], [0], color=color_func(thresholds[0] - 0.001), lw=3, label=f"CRRI 1")
        ])[::-1] + [
        Patch(facecolor="#000000", edgecolor=None, alpha=0.3, label="\u00B11 Std."),
        Patch(facecolor="#000000", edgecolor=None, alpha=0.1, label="\u00B12 Std."),
    ]

    legend = ax.legend(handles=legend_elements, framealpha=0.5, borderpad=0.2)
    ax.set_ylim([max(-0.1, ylim[0]), min(8.0, ylim[1])])
    ax.set_title("Cerner Reopening Risk Index and $R_e$")

    if suptitle is not None:
        plt.suptitle(suptitle)
    if save_filename:
        plt.gcf().savefig(save_filename)
    if show:
        plt.show()
    else:
        plt.close()
    return R_e, daily_cases
