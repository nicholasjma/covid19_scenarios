# -*- coding: utf-8 -*-
---
jupyter:
  jupytext:
    formats: ipynb,py:light,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.4
  kernelspec:
    display_name: Environment (conda_anaconda3)
    language: python
    name: conda_anaconda3
---

<!-- #region heading_collapsed=true -->
## Setup
<!-- #endregion -->

<!-- #region heading_collapsed=true hidden=true -->
### Packages and Functions
<!-- #endregion -->

```python hidden=true
import io
import logging
import os
import re
from collections.abc import Iterable
from datetime import datetime
from functools import partial
from textwrap import wrap

import markdown2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.core.display import HTML  # noqa: F401
from IPython.core.display import display
from PIL import ImageColor
from numba import njit

from cipy.file import pd_read

%load_ext autoreload
%autoreload 1
%aimport lib.seir
from lib.seir import (
    make_growth_plots,
    plot_scenarios,
)


def in_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    return True


if in_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
```

```python hidden=true
def name_fix(s, language="HTML"):
    if language == "HTML":
        emdash = "&mdash;"
        endash = "&ndash;"
        mu = "&mu;"
    elif language == "py":
        emdash = "\u2014"
        endash = "\u2013"
        mu = "\u03BC"
    elif language == "comma":
        emdash = ", "
        endash = ", "
        mu = "\u03BC"
    return (
        s.replace("--", emdash)
        .replace("-", endash)
        .replace("Metropolitan Statistical Area", "MSA")
        .replace("Micropolitan Statistical Area", f"{mu}SA")
        .replace(" town", " Town")
        .replace("Metropolitan NECTA", "MNECTA")
        .replace("Micropolitan NECTA", f"{mu}NECTA")
    )


name_fix_py = partial(name_fix, language="py")
name_fix_comma = partial(name_fix, language="comma")


@njit
def _get_class(
    value, thresholds, values,
):
    out = np.full(len(value), values[0])
    for idx in range(len(thresholds)):
        out[value >= thresholds[idx]] = values[idx + 1]
    return out


def get_class(value, thresholds, values, return_thresholds=False):
    if not isinstance(value, Iterable):
        if return_thresholds:
            return thresholds
        value = [value]
        out = _get_class(np.array(value), np.array(thresholds), np.arange(len(values)))
        return values[out[0]]
    else:
        out = _get_class(np.array(value), np.array(thresholds), np.arange(len(values)))
        return values[out]


# get colors from css
color_values = []
with open("../docs/modal.css", "r") as f:
    for line in f:
        if (
            "valueLow" in line
            or "valueMed" in line
            or "valueHigh" in line
            or "valueCritical" in line
        ):
            color_values.append(f.readline()[-9:-2])
R_e_class = partial(
    get_class,
    thresholds=np.array([0.9, 1.1, 1.3]),
    values=np.array(["valueLow", "valueMed", "valueHigh", "valueCritical"]),
)


def R_0_class(x):
    return ""


R_e_colors = partial(
    get_class, thresholds=np.array([0.9, 1.1, 1.3]), values=np.array(color_values),
)

CRRI = partial(
    get_class,
    thresholds=np.array([0.0, 0.9, 1.1, 1.3]),
    values=np.array([0, 1, 2, 3, 4]),
)
```

```python code_folding=[13] hidden=true
def make_grid_plot(
    R_e_dict,
    regions,
    n_graphs,
    colorfunc,
    dispfunc,
    ncols=4,
    linewidth=3,
    interp_interval="2H",
    start_date="3/17/2020",
    suptitle="",
    save_filename=None,
    h_pad=7,
):
    nrows = np.ceil(n_graphs / ncols)
    height = (3 + h_pad / 10) * nrows + 1.5
    fig, axes = plt.subplots(
        ncols=ncols,
        nrows=np.ceil(n_graphs / ncols).astype(int),
        sharex=False,
        sharey=True,
        figsize=(5 * ncols, height),
    )
    plt.tight_layout(pad=15, h_pad=h_pad, w_pad=2, rect=(0, 0, 1, 1 - 1.5 / height))
    g_iter = tqdm(regions)
    for region, ax in zip(g_iter, axes.flatten()):
        ax.set_ylim([0, 3])
        R_e = R_e_dict[region]
        disp = dispfunc(region)
        g_iter.set_description(disp)
        R_e_interp = (
            R_e.loc[pd.to_datetime(start_date) : pd.Timestamp.now().normalize()]
            .resample(interp_interval)
            .interpolate()
        )
        dummy_series = pd.Series(
            0,
            index=pd.date_range(
                start=pd.to_datetime(start_date), end=pd.Timestamp.now().normalize()
            ).values,
        )
        dummy_series.plot(ax=ax, alpha=0)
        R_e_thresh = pd.Series(colorfunc(R_e_interp), index=R_e_interp.index)
        R_e_thresh = (R_e_thresh != R_e_thresh.shift(1)).fillna(True).cumsum()
        R_e_df = pd.DataFrame({"R_e": R_e_interp, "thresh": R_e_thresh})
        R_e_df["group"] = R_e_df.thresh.diff().fillna(1).cumsum()
        for _, segment in R_e_df.groupby("group"):
            idx = segment.index.to_list()
            color = colorfunc(segment.R_e.iloc[0])
            new_idx = idx + [idx[-1] + pd.Timedelta(interp_interval)]
            new_idx = [x for x in new_idx if x in R_e_interp.index]
            R_e_interp.loc[new_idx].plot(ax=ax, color=color, linewidth=linewidth)
        ax.axhline(y=1.0, linewidth=1.5, color="black", alpha=0.4, label="Zero Growth")
        ax.grid(color="black", alpha=0.1, linewidth=0.5)
        ax.set_title("\n".join(wrap(disp, 25)))

    # cleanup unused axes
    for ax in axes.flatten()[len(regions) :]:
        ax.remove()
    plt.suptitle(suptitle)
    if save_filename:
        fig.savefig(save_filename)
```

```python code_folding=[0] hidden=true
# heattime plots

ImageColor.getcolor("#23a9dd", "RGBA")

def add_alpha(hex_color, alpha=0.6):
    rgb_tuple = ImageColor.getcolor(hex_color, "RGB")
    return "rgba" + repr(tuple(list(rgb_tuple) + [alpha]))


def discrete_colorscale(bvals, colors):
    """
    bvals - list of values bounding intervals/ranges of interest
    colors - list of rgb or hex colorcodes for values in [bvals[k], bvals[k+1]],0<=k < len(bvals)-1
    returns the plotly  discrete colorscale
    """
    if len(bvals) != len(colors) + 1:
        raise ValueError("len(boundary values) should be equal to  len(colors)+1")
    bvals = sorted(bvals)
    nvals = [
        (v - bvals[0]) / (bvals[-1] - bvals[0]) for v in bvals
    ]  # normalized values

    dcolorscale = []  # discrete colorscale
    for k in range(len(colors)):
        dcolorscale.extend([[nvals[k], colors[k]], [nvals[k + 1], colors[k]]])
    return dcolorscale


def draw_heattime(R_e_dict, colorfunc, min_date="3/3/2020", width=700):
    bvals = [-1, 0, 1, 2, 3, 4]
    colors = ["#f3f3f3"] + list(map(colorfunc, [0, 1, 1.2, 1.5]))
    colors = [add_alpha(x) for x in colors]

    dcolorsc = discrete_colorscale(bvals, colors)

    tickvals = np.arange(0.5, 6, 1) * 0.8
    ticktext = ["No Reported Cases", "CRRI 1", "CRRI 2", "CRRI 3", "CRRI 4"]
    z = pd.DataFrame(R_e_dict)
    z.index = pd.to_datetime(z.index).rename("Date")
    z = z.reset_index().fillna(-1)
    # z = z.groupby(pd.Grouper(key="Date", freq="W-MON")).max()

    for col in z.columns:
        if col == "Date":
            continue
        z[col] = CRRI(z[col])
    z.set_index("Date", inplace=True)
    if min_date is not None:
        z = z.loc[min_date:].copy(deep=False)
    z = z.T.sort_values(
        by=[
            (pd.Timestamp.now() - pd.Timedelta(days=x)).strftime("%Y-%m-%d")
            for x in range(1, 30)
        ],
        ascending=False,
    ).T
    heatmap = go.Heatmap(
        x=z.index,
        y=z.columns,
        z=z.T,
        colorscale=dcolorsc,
        colorbar=dict(
            thickness=25,
            tickvals=tickvals,
            ticktext=ticktext,
            y=0.9,
            len=0.18 * 51 / len(R_e_dict.keys()),
        ),
        xgap=0.5,
        ygap=1.5,
    )

    fig = go.Figure(data=[heatmap])
    fig.update_layout(
        width=width,
        height=20 * len(R_e_dict.keys()),
        yaxis=dict(autorange="reversed", side="left"),
        margin=dict(l=0, r=0, t=0, b=0, pad=0),  # noqa: E741
        yaxis_showgrid=False,
    )
    fig.update_xaxes(ticks="outside")
    return fig
```

<!-- #region hidden=true -->
### Params
<!-- #endregion -->

```python hidden=true
# raise error when setting on copy
pd.set_option("mode.chained_assignment", "raise")

# disable INFO output from penn_Chime
logging.disable(level=logging.INFO)

today = datetime.now()
```


<!-- #region heading_collapsed=true hidden=true -->
#### Align population data to COVID-19 Dataset
<!-- #endregion -->

```python code_folding=[] hidden=true
# get confirmed case data for each county
county_data = pd_read(
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
)
state_cases = (
    county_data.groupby("Province_State")
    .sum()
    .loc[:, [x for x in county_data if x.endswith("/20")]]
)
county_data["Admin2"] = county_data["Admin2"].replace("Unassigned", np.nan)
county_data.loc[
    ~county_data["Admin2"].isnull() & county_data["Admin2"].str.contains("Out of "),
    "Admin2",
] = np.nan
county_data = (
    county_data.drop(columns=["UID", "iso2", "iso3"])
    .dropna(subset=["Admin2"])
    .set_index(["Province_State", "Admin2"])
    .rename_axis(["State", "County"])
)

# getting the counties aligned with the other datasets is a challenge
county_pop = pd_read(
    "https://www2.census.gov/programs-surveys/popest/datasets/"
    "2010-2019/counties/totals/co-est2019-alldata.csv",
    columns=["STNAME", "CTYNAME", "POPESTIMATE2019"],
).drop_duplicates()
state_pop = county_pop.loc[
    county_pop.STNAME == county_pop.CTYNAME, ["STNAME", "POPESTIMATE2019"]
]
# fmt: off
state_pop["ABBREV"] = (
    ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA",
     "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
     "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
     "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
     "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
)
# fmt: on
state_pop.set_index("STNAME", inplace=True)
state_cases = state_cases.loc[state_pop.index].copy(deep=False)

county_pop = county_pop[county_pop.CTYNAME != county_pop.STNAME].copy(deep=False)
county_pop.CTYNAME = county_pop.CTYNAME.str.replace("Doña Ana", "Dona Ana")
county_pop.CTYNAME = (
    county_pop.CTYNAME.str.replace(" County", "")
    .str.replace(" Borough", "")
    .str.replace(" Municipality", "")
    .str.replace(" Census Area", "")
    .str.replace(" City and", "")
    .str.replace(" Parish", "")
    .str.replace(" city", " City")
)

virginia_city_removal = [
    "Alexandria",
    "Bristol",
    "Buena Vista",
    "Charlottesville",
    "Chesapeake",
    "Colonial Heights",
    "Covington",
    "Danville",
    "Emporia",
    "Falls Church",
    "Fredericksburg",
    "Galax",
    "Hampton",
    "Harrisonburg",
    "Hopewell",
    "Lexington",
    "Lynchburg",
    "Manassas",
    "Manassas Park",
    "Martinsville",
    "Newport News",
    "Norfolk",
    "Norton",
    "Petersburg",
    "Poquoson",
    "Portsmouth",
    "Radford",
    "Salem",
    "Staunton",
    "Suffolk",
    "Virginia Beach",
    "Waynesboro",
    "Williamsburg",
    "Winchester",
]
for city in virginia_city_removal:
    county_pop.loc[county_pop.STNAME == "Virginia", "CTYNAME"] = county_pop.loc[
        county_pop.STNAME == "Virginia", "CTYNAME"
    ].str.replace(f"{city} City", city)
county_pop = county_pop.set_index(["STNAME", "CTYNAME"]).rename_axis(
    ["State", "County"]
)

county_pop.loc[("Massachusetts", "Dukes and Nantucket"), "POPESTIMATE2019"] = (
    county_pop.loc[("Massachusetts", "Dukes"), "POPESTIMATE2019"].squeeze()
    + county_pop.loc[("Massachusetts", "Nantucket"), "POPESTIMATE2019"].squeeze()
)

mitigation = pd_read(
    "https://opendata.arcgis.com/datasets/97792521be744711a291d10ecef33a61_0.csv"
)
county_data = county_data.merge(
    mitigation.set_index("fips")["last_update"]
    .pipe(pd.to_datetime)
    .dt.normalize()
    .dt.tz_localize(None)
    .to_frame()
    .rename(columns={"last_update": "social_distancing_date"}),
    left_on="FIPS",
    right_index=True,
    how="left",
)
county_data = county_data.join(county_pop)
# these two populations are missing, and manually provided
county_data.loc[
    ("District of Columbia", "District of Columbia"), "POPESTIMATE2019"
] = 711571
county_data.loc[("Missouri", "Kansas City"), "POPESTIMATE2019"] = 459787
county_cases = county_data[
    [x for x in county_data.columns if x.endswith("/20")]
    + ["POPESTIMATE2019", "social_distancing_date", "FIPS"]
].copy(deep=False)
county_cases.dropna(subset=["POPESTIMATE2019", "FIPS"], inplace=True)
```

<!-- #region hidden=true -->
#### Read CBSA/CSA data and cleanup FIPS codes
<!-- #endregion -->

```python code_folding=[3] hidden=true
cbsa_df = pd.read_excel("https://www.bls.gov/lau/lmadir2015.xlsx", encoding="latin-1")
cbsa_df["FIPS"] = cbsa_df.st_fips * 1000 + cbsa_df.cnty_fips
cbsa_df.lma_fips = cbsa_df.lma_fips.astype(str).str.zfill(5)
lma_fips = [  # replace lma_fips with lma_code when appropriate
    13,
    27,
    29,
    35,
    39,
    41,
    59,
    75,
    99,
    105,
    131,
    1,
    17,
    37,
    47,
    61,
    95,
    117,
    121,
    137,
    51,
    7,
    79,
    115,
    123,
    205,
    139,
    151,
    157,
    159,
    171,
]
lma_fips = [str(x).zfill(5) for x in lma_fips]
cbsa_df.loc[cbsa_df.lma_fips.isin(lma_fips), "lma_fips"] = cbsa_df.loc[
    cbsa_df.lma_fips.isin(lma_fips), "lma_code"
]
cbsa_codes = cbsa_df.lma_fips.unique()
```

```python hidden=true
# get CSA-FIPS crosswalk
csa_df = pd.read_csv(
    "http://data.nber.org/cbsa-csa-fips-county-crosswalk/cbsa2fipsxw.csv"
)
csa_df = csa_df.dropna(subset=["csacode", "fipsstatecode", "fipscountycode"]).copy(
    deep=False
)
csa_df["FIPS"] = csa_df.fipsstatecode.mul(1000).add(csa_df.fipscountycode)
csa_df.csacode = csa_df.csacode.astype(int).astype(str)
csa_codes = csa_df.csacode.unique()
```

```python code_folding=[0] hidden=true
state_fips = (
    pd.read_csv(
        io.StringIO(
            """Name	Postal Code	FIPS
            Alabama	AL	01
            Alaska	AK	02
            Arizona	AZ	04
            Arkansas	AR	05
            California	CA	06
            Colorado	CO	08
            Connecticut	CT	09
            Delaware	DE	10
            District of Columbia	DC	11
            Florida	FL	12
            Georgia	GA	13
            Hawaii	HI	15
            Idaho	ID	16
            Illinois	IL	17
            Indiana	IN	18
            Iowa	IA	19
            Kansas	KS	20
            Kentucky	KY	21
            Louisiana	LA	22
            Maine	ME	23
            Maryland	MD	24
            Massachusetts	MA	25
            Michigan	MI	26
            Minnesota	MN	27
            Mississippi	MS	28
            Missouri	MO	29
            Montana	MT	30
            Nebraska	NE	31
            Nevada	NV	32
            New Hampshire	NH	33
            New Jersey	NJ	34
            New Mexico	NM	35
            New York	NY	36
            North Carolina	NC	37
            North Dakota	ND	38
            Ohio	OH	39
            Oklahoma	OK	40
            Oregon	OR	41
            Pennsylvania	PA	42
            Rhode Island	RI	44
            South Carolina	SC	45
            South Dakota	SD	46
            Tennessee	TN	47
            Texas	TX	48
            Utah	UT	49
            Vermont	VT	50
            Virginia	VA	51
            Washington	WA	53
            West Virginia	WV	54
            Wisconsin	WI	55
            Wyoming	WY	56
            American Samoa	AS	60
            Guam	GU	66
            Northern Mariana Islands	MP	69
            Puerto Rico	PR	72
            Virgin Islands	VI	78"""
        ),
        sep="\t",
        usecols=["Name", "FIPS"],
    )
    .set_index("FIPS")
    .squeeze()
    .str.replace(" +", " ")
    .str.replace("^ ", "")
)
```

<!-- #region hidden=true -->
#### Set Plotting rcParams
<!-- #endregion -->

```python hidden=true
params = {
    "legend.fontsize": 8,
    "figure.titlesize": 25,
    "figure.figsize": (15, 10),
    "axes.labelsize": 14,
    "axes.titlesize": 20,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.titlepad": 15,
    "axes.linewidth": 1,
    "legend.title_fontsize": 10,
    "figure.facecolor": (1, 1, 1, 1),
    "figure.dpi": 144,
    "savefig.pad_inches": 0,
}
plt.rcParams.update(params)
```

## Run scenarios for each CBSA, CSA, state, and country


### Run scenarios for each CBSA, CSA, and state

```python code_folding=[0, 2]
def run_scenarios(
    cbsa_data, cbsa_code, area_name, pop, debug=False, relaxation_date="5/15/2020",
):
    temp = cbsa_data.diff().fillna(cbsa_data.iloc[0]).clip(0, None)
    temp2 = temp.copy()
    temp2.iloc[-3:] = np.round(temp2.iloc[-6:-3].mean())
    # make sure daily counts aren't decreasing in last 3 days to account for lag
    cbsa_data = temp.clip(temp2, None).cumsum()

    cbsa_data.index = pd.to_datetime(cbsa_data.index)

    try:
        R_e, daily_cases = make_growth_plots(
            cbsa_data,
            suptitle=f"{area_name} COVID-19",
            show=debug,
            save_filename=f"../docs/summary/{cbsa_code}.svg",
            color_func=R_e_colors,
        )
    except (ValueError, IndexError):
        return
    if len(R_e) >= 5:
        plot_scenarios(
            cbsa_data,
            daily_cases,
            R_e,
            pop,
            [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1],
            relaxation_date=relaxation_date,
            suptitle=f"{name_fix_py(area_name)} COVID-19",
            show=debug,
            save_filename=f"../docs/scenarios/{cbsa_code}.svg",
        )
    return R_e, daily_cases
```


#### Demo Graphs

```python
# thresholds = R_e_colors(0, return_thresholds=True)
# threshold_colors = list(map(R_e_colors, [0, 1, 1.2, 1.5]))
# x = np.arange(30)
# k_values = [-0.06, 0, 0.03, 0.06]
# fig, ax = plt.subplots(figsize=(8, 5))
# for idx, k in enumerate(k_values):
#     y = 2 * np.exp(k * x) + 1 + idx * 0.5
#     plt.plot(
#         x, pd.Series(y), linewidth=5, label=f"{idx+1}", color=threshold_colors[idx]
#     )
# plt.ylim([0, 15])
# plt.legend(fontsize=20, title_fontsize=20, title="CRRI")
# plt.xlabel("Days", fontsize=20)
# plt.gca().set_yticks([])
# plt.title("Cerner Reopening Risk Index (CRRI)\nCategory Examples", fontsize=30)
# plt.ylabel("Daily Cases", fontsize=20)
# plt.xlim([0, 25])
# plt.savefig("../docs/examples.svg", bbox_inches="tight", pad_inches=0.3)
# plt.show()
```

<!-- #region heading_collapsed=true -->
#### Model options
<!-- #endregion -->

```python code_folding=[] hidden=true
# correct data errors
# key is the state, CSA, or CBSA
# value is a list of tuples, of which the first element is a slice
#     and the second element is the numerical correction to add to that slice
leavenworth_fixes = [
    (slice("4/30/2020", None), -157),
    (slice("5/3/2020", None), -100),
    (slice("5/6/2020", None), -166),
    (slice("5/7/2020", None), -87),
]
corrections = {
    # KC CBSA, Leavenworth prison mass test
    "28140": leavenworth_fixes,
    # KC CSA, Leavenworth prison mass test
    "312": leavenworth_fixes,
    # Kansas, Leavenworth prison mass test
    "Kansas": leavenworth_fixes,
}

# set to True to run on debug_list instead of all CBSAs
debug_list = [
    #     "71650",  # Boston
    "28140",  # KC CBSA
    "312",  # KC CSA
    "Kansas",
    "Missouri",
    #     "41180",  # St. Louis
    #     "14780",
    #     "82980",
    #     "C_United States of America",
    "C_Iceland",
    #     "C_Ecuatorial Guinea",
    #     "Alaska",
    "Florida",
    "New York",
    #     "Alabama",
    #     "25200",
    #     "CN3115100000000",
    #     "518",
    #     "00011",
    #     "25200",
    "CN0101300000000",
]
debug = False
```

### CBSAs

```python code_folding=[0]
# run scenarios and generate plots for CBSAs
if not debug:
    cbsa_R_e = {}
    cbsa_pop = {}
cbsa_iter = tqdm(cbsa_codes)
for cbsa_code in cbsa_iter:
    if debug and cbsa_code not in debug_list:
        continue
    area_name = name_fix_py(cbsa_df[cbsa_df.lma_fips == cbsa_code].lma_title.iloc[0])
    cbsa_iter.set_description(area_name)

    cbsa_mask = county_data.FIPS.isin(cbsa_df[cbsa_df.lma_fips == cbsa_code].FIPS)
    cbsa_data = county_data.loc[
        cbsa_mask, [x for x in county_data if x.endswith("/20")],
    ].sum()
    cbsa_data.index = pd.to_datetime(cbsa_data.index)

    if cbsa_code in corrections:
        for corr_slice, amount in corrections[cbsa_code]:
            if debug:
                print(cbsa_code, area_name, corr_slice, amount)
            cbsa_data.loc[corr_slice] += amount
    cbsa_data = cbsa_data[cbsa_data > 0].copy(deep=False)
    if len(cbsa_data.index) < 7 or cbsa_data.max() < 100:
        continue
    cbsa_pop[cbsa_code] = county_data.loc[cbsa_mask, "POPESTIMATE2019"].sum()
    out = run_scenarios(
        cbsa_data, cbsa_code, area_name, cbsa_pop[cbsa_code], debug=debug,
    )
    if out:
        cbsa_R_e[cbsa_code] = out[0]
```

```python code_folding=[0]
# Build R_e plot for each cbsa
CBSAS = 60
cbsa_pop_s = pd.Series(cbsa_pop).sort_values(ascending=False).head(CBSAS).sort_index()
cbsa_names = (
    cbsa_pop_s.to_frame()
    .merge(
        cbsa_df[["lma_fips", "lma_title"]].drop_duplicates(),
        left_index=True,
        right_on="lma_fips",
    )
    .sort_values(by=["lma_title"])
)

def cbsa_disp(cbsa_code, df):
    return name_fix_comma(df[df.lma_fips == cbsa_code].lma_title.iloc[0])


make_grid_plot(
    cbsa_R_e,
    cbsa_names.lma_fips,
    n_graphs=CBSAS,
    colorfunc=R_e_colors,
    dispfunc=partial(cbsa_disp, df=cbsa_df),
    suptitle="Cerner Reopening Risk Index and $R_e$ by CBSA",
    save_filename="../docs/all_cbsas.svg",
    h_pad=12,
)
if not in_notebook():
    plt.close()
```

```python code_folding=[0]
# Build bar graph showing highest R_e in last 2 weeks for each state
CBSAS = 50
cbsa_highest_R_e_2wks = (
    pd.Series(
        {
            cbsa_code: cbsa_R_e[cbsa_code].iloc[-14:].max()
            for cbsa_code in cbsa_pop_s.index
        }
    )
    .to_frame()
    .reset_index()
)
cbsa_highest_R_e_2wks.columns = ["CBSA", "R_e"]
cbsa_highest_R_e_2wks.sort_values(by=["R_e"], inplace=True)
cbsa_highest_R_e_2wks = cbsa_highest_R_e_2wks.merge(
    cbsa_df[["lma_title", "lma_fips"]].drop_duplicates(),
    left_on="CBSA",
    right_on="lma_fips",
)
cbsa_highest_R_e_2wks.lma_title = cbsa_highest_R_e_2wks.lma_title.apply(name_fix_comma)
cbsa_highest_R_e_2wks["pop"] = cbsa_highest_R_e_2wks["lma_fips"].map(cbsa_pop_s)
cbsa_highest_R_e_2wks = (
    cbsa_highest_R_e_2wks.sort_values(by=["pop"], ascending=False)
    .head(CBSAS)
    .sort_values(by=["R_e"], ascending=True)
)
color = cbsa_highest_R_e_2wks.R_e.apply(R_e_colors)
color_desc = dict(zip(color_values, [f"{x}" for x in range(1, 5)]))
fig = go.Figure(
    data=[
        go.Bar(
            y=cbsa_highest_R_e_2wks.loc[color == col, "lma_title"],
            x=cbsa_highest_R_e_2wks.loc[color == col, "R_e"].round(2),
            hovertext=cbsa_highest_R_e_2wks.loc[color == col, "CBSA"],
            marker=dict(color=col,),
            name=color_desc[col],
            orientation="h",
        )
        for col in color_values
    ],
)
fig.update_layout(
    annotations=[],
    width=832,
    height=900,
    plot_bgcolor="white",
    paper_bgcolor="white",
    showlegend=True,
    legend_title_text="CCRI",
    #     legend=dict(traceorder="reversed"),
    margin=dict(l=0, r=0, t=0, b=0, pad=0),  # noqa: E741
    shapes=[
        dict(
            type="line",
            yref="y",
            x0=1,
            x1=1,
            xref="x",
            y0=0,
            y1=CBSAS * 1.02,
            line=dict(color="#b8b9ba", width=3,),
        )
    ],
    bargap=0.01,
    yaxis=dict(title="R<sub>e</sub>", autorange="reversed"),
)
fig.update_yaxes(tickfont=dict(size=10))
fig.write_html("cbsa_summary_plot.html", include_plotlyjs=False)
with open("cbsa_summary_plot.html", "r") as f:
    lines = f.readlines()
os.remove("cbsa_summary_plot.html")
cbsa_summary_lines = lines[4:-3]
```

### CSAs

```python code_folding=[0]
# same analysis for CSAs
if not debug:
    csa_R_e = {}
    csa_pop = {}
csa_iter = tqdm(csa_df.csacode.unique())
for csa_code in csa_iter:
    if debug and csa_code not in debug_list:
        continue
    area_name = name_fix_py(csa_df[csa_df.csacode == csa_code].csatitle.iloc[0])
    csa_iter.set_description(area_name)
    csa_mask = county_data.FIPS.isin(csa_df[csa_df.csacode == csa_code].FIPS)
    csa_data = county_data.loc[
        csa_mask, [x for x in county_data if x.endswith("/20")],
    ].sum()
    csa_data.index = pd.to_datetime(csa_data.index)

    if csa_code in corrections:
        for corr_slice, amount in corrections[csa_code]:
            csa_data.loc[corr_slice] += amount
    csa_data = csa_data[csa_data > 0].copy(deep=False)
    if len(csa_data.index) < 7 or csa_data.max() < 100:
        continue
    csa_pop[csa_code] = county_data.loc[csa_mask, "POPESTIMATE2019"].sum()
    out = run_scenarios(csa_data, csa_code, area_name, csa_pop[csa_code], debug=debug,)
    if out:
        csa_R_e[csa_code] = out[0]
```

```python code_folding=[0]
# Build R_e plot for each csa
CSAS = 60
csa_pop_s = pd.Series(csa_pop).sort_values(ascending=False).head(CSAS).sort_index()
csa_pop_s = (
    csa_pop_s.to_frame()
    .merge(
        csa_df[["csacode", "csatitle"]].drop_duplicates(),
        left_index=True,
        right_on="csacode",
    )
    .sort_values(by=["csatitle"])
)

def csa_disp(csa_code, df):
    return name_fix_comma(df[df.csacode == csa_code].csatitle.iloc[0])


make_grid_plot(
    csa_R_e,
    csa_pop_s.csacode,
    n_graphs=CSAS,
    colorfunc=R_e_colors,
    dispfunc=partial(csa_disp, df=csa_df),
    suptitle="Cerner Reopening Risk Index and $R_e$ by CSA",
    save_filename="../docs/all_csas.svg",
    h_pad=12,
)
if not in_notebook():
    plt.close()
```

```python code_folding=[0]
# Build bar graph showing highest R_e in last 2 weeks for each state
CSAS = 50
csa_highest_R_e_2wks = (
    pd.Series(
        {csa_code: csa_R_e[csa_code].iloc[-14:].max() for csa_code in csa_pop_s.csacode}
    )
    .to_frame()
    .reset_index()
)
csa_highest_R_e_2wks.columns = ["CSA", "R_e"]
csa_highest_R_e_2wks.sort_values(by=["R_e"], inplace=True)
csa_highest_R_e_2wks = csa_highest_R_e_2wks.merge(
    csa_df[["csatitle", "csacode"]].drop_duplicates(), left_on="CSA", right_on="csacode"
)
csa_highest_R_e_2wks.csatitle = csa_highest_R_e_2wks.csatitle.apply(name_fix_comma)
csa_highest_R_e_2wks["pop"] = csa_highest_R_e_2wks.csacode.map(csa_pop)
csa_highest_R_e_2wks = (
    csa_highest_R_e_2wks.sort_values(by=["pop"], ascending=False)
    .head(CSAS)
    .sort_values(by=["R_e"], ascending=True)
)
color = csa_highest_R_e_2wks.R_e.apply(R_e_colors)
color_desc = dict(zip(color_values, [f"CRRI {x}" for x in range(1, 5)]))
fig = go.Figure(
    data=[
        go.Bar(
            y=csa_highest_R_e_2wks.loc[color == col, "csatitle"],
            x=csa_highest_R_e_2wks.loc[color == col, "R_e"].round(2),
            hovertext=csa_highest_R_e_2wks.loc[color == col, "CSA"],
            marker=dict(color=col,),
            name=color_desc[col],
            orientation="h",
        )
        for col in color_values
    ],
)
fig.update_layout(
    annotations=[],
    width=832,
    height=900,
    plot_bgcolor="white",
    paper_bgcolor="white",
    showlegend=True,
    #     legend=dict(traceorder="reversed"),
    margin=dict(l=0, r=0, t=0, b=0, pad=0),  # noqa: E741
    shapes=[
        dict(
            type="line",
            yref="y",
            x0=1,
            x1=1,
            xref="x",
            y0=0,
            y1=CSAS * 1.01,
            line=dict(color="#b8b9ba", width=3,),
        )
    ],
    bargap=0.01,
    yaxis=dict(title="R<sub>e</sub>", autorange="reversed"),
)
fig.update_xaxes(tickfont=dict(size=10))
fig.write_html("csa_summary_plot.html", include_plotlyjs=False)
with open("csa_summary_plot.html", "r") as f:
    lines = f.readlines()
os.remove("csa_summary_plot.html")
csa_summary_lines = lines[4:-3]
```

### States

```python code_folding=[0]
state_relaxation_date = (
    pd.read_excel("50 states reopening.xlsx")
    .fillna(pd.Timestamp.now().normalize() + pd.Timedelta(days=14))
    .set_index("States")
    .squeeze()
)
state_relaxation_date.index = state_relaxation_date.index.str.replace(
    "West Virgina", "West Virginia"
)
display(state_relaxation_date.sort_values().to_frame().head(5))
```

```python code_folding=[0]
# same analysis for states
if not debug:
    state_R_e = {}
states_iter = tqdm(state_cases.index)
for state in states_iter:
    if debug and state not in debug_list:
        continue
    area_name = state
    states_iter.set_description(state)
    state_data = state_cases.loc[state]
    state_data.index = pd.to_datetime(state_data.index)
    if state in corrections:
        for corr_slice, amount in corrections[state]:
            state_data.loc[corr_slice] += amount
    state_data = state_data[state_data > 0].copy(deep=False)
    if len(state_data.index) < 7 or state_data.max() < 100:
        continue
    pop_est = state_pop.loc[state, "POPESTIMATE2019"]
    out = run_scenarios(
        state_data,
        state,
        area_name,
        pop_est,
        debug=debug,
        relaxation_date=state_relaxation_date.loc[state],
    )
    if out:
        state_R_e[state] = out[0]
```

#### State $R_e$ plots

```python code_folding=[0]
# Build R_e plot for each state

make_grid_plot(
    state_R_e,
    state_R_e.keys(),
    n_graphs=51,
    colorfunc=R_e_colors,
    dispfunc=lambda x: x,
    suptitle="Cerner Reopening Risk Index and $R_e$ by State",
    save_filename="../docs/all_states.svg",
)
if not in_notebook():
    plt.close()
```

#### State Summary Plot

```python code_folding=[0]
# Build bar graph showing highest R_e in last 2 weeks for each state
s_highest_R_e_2wks = (
    pd.Series({state: x.iloc[-14:].max() for state, x in state_R_e.items()})
    .to_frame()
    .reset_index()
)
s_highest_R_e_2wks.columns = ["State", "R_e"]
s_highest_R_e_2wks.sort_values(by=["R_e"], inplace=True)
s_highest_R_e_2wks = s_highest_R_e_2wks.merge(state_pop, left_on="State", right_on="STNAME")
color = s_highest_R_e_2wks.R_e.apply(R_e_colors)
color_desc = dict(zip(color_values, [f"{x}" for x in range(1, 5)]))
fig = go.Figure(
    data=[
        go.Bar(
            x=s_highest_R_e_2wks.loc[color == col, "ABBREV"],
            y=s_highest_R_e_2wks.loc[color == col, "R_e"].round(2),
            hovertext=s_highest_R_e_2wks.loc[color == col, "State"],
            marker=dict(color=col,),
            name=color_desc[col],
        )
        for col in color_values
    ],
)
fig.update_layout(
    annotations=[],
    width=832,
    height=250,
    plot_bgcolor="white",
    paper_bgcolor="white",
    showlegend=True,
    legend_title_text="CCRI",
    #     legend=dict(traceorder="reversed"),
    margin=dict(l=0, r=0, t=0, b=0, pad=0),  # noqa: E741
    shapes=[
        dict(
            type="line",
            yref="y",
            y0=1,
            y1=1,
            xref="x",
            x0=0,
            x1=51,
            line=dict(color="#b8b9ba", width=3,),
        )
    ],
    yaxis=dict(title="R<sub>e</sub>"),
    bargap=0.01,
)
fig.update_xaxes(tickfont=dict(size=10))
fig.write_html("summary_plot.html", include_plotlyjs=False)
with open("summary_plot.html", "r") as f:
    lines = f.readlines()
os.remove("summary_plot.html")
summary_lines = lines[4:-3]
```

### Countries

<!-- #region heading_collapsed=true -->
#### Load Data
<!-- #endregion -->

```python hidden=true
country_data = (
    pd_read(
        "https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/"
        "csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    )
    .set_index(["Country/Region", "Province/State"])
    .groupby("Country/Region")
    .sum()
)
```

```python hidden=true
country_pop = pd_read(
    #     "https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/CSV_FILES/"
    "WPP2019_TotalPopulationBySex.csv"
)
```

```python hidden=true
country_pop = country_pop.loc[
    (country_pop.Time == 2020) & (country_pop.Variant == "Medium"),
    ["Location", "PopTotal"],
].copy(deep=False)
country_pop.PopTotal = country_pop.PopTotal.mul(1000).astype(int)
```

```python hidden=true
country_map = {
    "Bolivia": "Bolivia (Plurinational State of)",
    "Brunei": "Brunei Darussalam",
    "Burma": "Myanmar",
    "Congo (Brazzaville)": "Democratic Republic of the Congo",
    "Congo (Kinshasa)": "Congo",
    "Cote d'Ivoire": "Côte d'Ivoire",
    "Diamond Princess": np.nan,
    "Iran": "Iran (Islamic Republic of)",
    "Korea, South": "Republic of Korea",
    "Kosovo": np.nan,
    "Laos": "Lao People's Democratic Republic",
    "Mauritius": np.nan,
    "MS Zaandam": np.nan,
    "Moldova": "Republic of Moldova",
    "Russia": "Russian Federation",
    "Syria": "Syrian Arab Republic",
    "Taiwan*": "China, Taiwan Province of China",
    "Tanzania": "United Republic of Tanzania",
    "US": "United States of America",
    "Venezuela": "Venezuela (Bolivarian Republic of)",
    #     "Vietnam": "Viet Nam",
    "Vietnam": np.nan,  # not enough recent cases reported
    "Montenegro": np.nan,  # not enough recent cases reported
    "West Bank and Gaza": np.nan,
}
country_data.index = country_data.index.to_series().replace(country_map)
country_data = country_data[country_data.index.notnull()].copy(deep=False)
country_data = country_data.merge(
    country_pop, left_index=True, right_on="Location"
).set_index("Location")
```

#### Build plots

```python code_folding=[0]
# country level plots
country_R_e = {}
country_iter = tqdm(country_data.index)
for country in country_iter:
    if debug and f"C_{country}" not in debug_list:
        continue
    area_name = country
    country_iter.set_description(country)
    c_data = country_data.loc[
        country, [x for x in country_data if x.endswith("/20")],
    ]
    c_data.index = pd.to_datetime(c_data.index)
    if country in corrections:
        for corr_slice, amount in corrections[country]:
            c_data.loc[corr_slice] += amount
    c_data = c_data[c_data > 0].copy(deep=False)
    if len(c_data.index) < 7 or c_data.max() < 100:
        continue
    pop_est = country_data.loc[country, "PopTotal"]
    out = run_scenarios(c_data, f"C_{country}", area_name, pop_est, debug=debug,)
    if out:
        country_R_e[country] = out[0]
```

```python code_folding=[0, 36]
# Build bar graph showing highest R_e in last 2 weeks for each country
COUNTRIES = 59
c_highest_R_e_2wks = (
    pd.Series({country: x.iloc[-14:].max() for country, x in country_R_e.items()})
    .to_frame()
    .reset_index()
)
c_highest_R_e_2wks.columns = ["Country", "R_e"]
c_highest_R_e_2wks.sort_values(by=["R_e"], inplace=True)
c_highest_R_e_2wks = c_highest_R_e_2wks.merge(
    country_pop, left_on="Country", right_on="Location"
)
c_highest_R_e_2wks = (
    c_highest_R_e_2wks.sort_values(by="PopTotal", ascending=False)
    .head(COUNTRIES)
    .sort_values(by="R_e", ascending=True)
)
country_keys = c_highest_R_e_2wks.Country.values
c_highest_R_e_2wks["Country"] = (
    c_highest_R_e_2wks["Country"]
    .replace({y: x for x, y in country_map.items()})
    .replace({"Taiwan*": "Taiwan", "US": "United States"})
)
color = c_highest_R_e_2wks.R_e.apply(R_e_colors)

fig = go.Figure(
    data=[
        go.Bar(
            x=c_highest_R_e_2wks.loc[color == col, "Country"],
            y=c_highest_R_e_2wks.loc[color == col, "R_e"].round(2),
            marker=dict(color=col,),
            name=color_desc[col],
        )
        for col in color_values
    ],
)
fig.update_layout(
    annotations=[],
    width=832,
    height=280,
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend_title_text="CCRI",
    #     legend=dict(traceorder="reversed"),
    margin=dict(l=0, r=0, t=0, b=0, pad=0),  # noqa: E741
    shapes=[
        dict(
            type="line",
            yref="y",
            y0=1,
            y1=1,
            xref="x",
            x0=0,
            x1=COUNTRIES,
            line=dict(color="#b8b9ba", width=3,),
        )
    ],
    bargap=0.01,
    yaxis=dict(title="R<sub>e</sub>"),
)
fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
fig.write_html("summary_plot.html", include_plotlyjs=False)  # js loaded from states
with open("summary_plot.html", "r") as f:
    lines = f.readlines()
os.remove("summary_plot.html")
c_summary_lines = lines[4:-3]
if in_notebook():
    fig.show()
```

```python code_folding=[0]
# Build R_e plot for each country
COUNTRIES = 60
c_highest_R_e_2wks = (
    pd.Series({country: x.iloc[-14:].max() for country, x in country_R_e.items()})
    .to_frame()
    .reset_index()
)
c_highest_R_e_2wks.columns = ["Country", "R_e"]
c_highest_R_e_2wks.sort_values(by=["R_e"], inplace=True)
c_highest_R_e_2wks = c_highest_R_e_2wks.merge(
    country_pop, left_on="Country", right_on="Location"
)
c_highest_R_e_2wks = (
    c_highest_R_e_2wks.sort_values(by="PopTotal", ascending=False)
    .head(COUNTRIES)
    .sort_values(by="R_e", ascending=True)
)
c_highest_R_e_2wks["Country_disp"] = (
    c_highest_R_e_2wks["Country"]
    .replace({y: x for x, y in country_map.items()})
    .replace({"Taiwan*": "Taiwan", "US": "United States"})
)
c_highest_R_e_2wks = c_highest_R_e_2wks.set_index("Country").sort_values(
    by=["Country_disp"]
)
country_keys = c_highest_R_e_2wks.index.values


def country_disp(country, df):
    return df.loc[country, "Country_disp"]

make_grid_plot(
    country_R_e,
    c_highest_R_e_2wks.index,
    n_graphs=COUNTRIES,
    colorfunc=R_e_colors,
    dispfunc=partial(country_disp, df=c_highest_R_e_2wks),
    ncols=4,
    linewidth=3,
    interp_interval="2H",
    start_date="3/17/2020",
    suptitle="Cerner Reopening Risk Index and $R_e$ by Country",
    save_filename="../docs/all_countries.svg",
)
if not in_notebook():
    plt.close()
```

## Heat Plots


#### Countries

```python code_folding=[0]
# R_e heattime map

def _country_disp(country, df):
    return df.loc[country, "Country_disp"]


country_disp = partial(_country_disp, df=c_highest_R_e_2wks)
fig = draw_heattime(
    {country_disp(x): country_R_e[x] for x in c_highest_R_e_2wks.index},
    R_e_colors,
#     min_date=None,
)
fig.write_html("country_heattime.html", include_plotlyjs=False)  # js loaded from states
with open("country_heattime.html", "r") as f:
    lines = f.readlines()
os.remove("country_heattime.html")
c_heattime = lines[4:-3]
if in_notebook():
    fig.show()
```

#### States

```python code_folding=[0]
# R_e heattime map
fig = draw_heattime(state_R_e, R_e_colors)
fig.write_html("state_heattime.html", include_plotlyjs=False)  # js loaded from states
with open("state_heattime.html", "r") as f:
    lines = f.readlines()
os.remove("state_heattime.html")
s_heattime = lines[4:-3]
if in_notebook():
    fig.show()
```

<!-- #region code_folding=[] -->
#### CSAs
<!-- #endregion -->

```python code_folding=[0]
# R_e heattime map

fig = draw_heattime(
    {
        csa_highest_R_e_2wks.set_index("CSA").loc[x, "csatitle"]: csa_R_e[x]
        for x in csa_highest_R_e_2wks.CSA
    },
    R_e_colors,
    width=832,
    #     min_date=None,
)
fig.write_html("csa_heattime.html", include_plotlyjs=False)  # js loaded from states
with open("csa_heattime.html", "r") as f:
    lines = f.readlines()
os.remove("csa_heattime.html")
csa_heattime = lines[4:-3]
if in_notebook():
    fig.show()
```

#### CBSAs

```python code_folding=[0]
# R_e heattime map

fig = draw_heattime(
    {
        cbsa_highest_R_e_2wks.set_index("CBSA").loc[x, "lma_title"]: cbsa_R_e[x]
        for x in cbsa_highest_R_e_2wks.lma_fips
    },
    R_e_colors,
    width=832,
    #     min_date=None,
)
fig.write_html("cbsa_heattime.html", include_plotlyjs=False)  # js loaded from states
with open("cbsa_heattime.html", "r") as f:
    lines = f.readlines()
os.remove("cbsa_heattime.html")
cbsa_heattime = lines[4:-3]
if in_notebook():
    fig.show()
```

## Build index.html

<!-- #region heading_collapsed=true -->
#### Header
<!-- #endregion -->

```python code_folding=[] hidden=true
# Build index.html
THRES = 1.0


def heattime_header(lines, region):
    lines.append(
        f"<h4>Historical Cerner Reopening Risk Index by {region.capitalize()}</h4>"
    )
    lines.append(
        f'<p>This plot shows the historical <a href="#CRRIDesc">CRRI</a> in each'
        f" {region} since March, sorted by CRRI scores over the last 2 weeks.</p>"
    )


def get_value_str(s, R_0_max=8.0):
    return f"{s:.2f}" if s <= R_0_max else f"{R_0_max:.0f}+"


def get_days_below(R_e, R_e_threshold=THRES, recent=14):
    out = R_e.iloc[-recent:].lt(R_e_threshold).sum()
    return f"{out} / {recent}"


def add_buttons(lines, s, prefix="", filetype="svg"):
    lines.append(
        '<td class="mdl-data-table__cell--non-numeric center">'
        '<button class="mdl-button mdl-js-button mdl-button--icon"'
        f'onclick="open_image(&quot;./summary/{prefix}{s}.{filetype}&quot;)">'
        '  <i class="material-icons">history</i>'
        "</button>"
    )
    lines.append(
        '<button class="mdl-button mdl-js-button mdl-button--icon"'
        f'onclick="open_image(&quot;./scenarios/{prefix}{s}.{filetype}&quot;)" style="margin-left:1.6rem;margin-right:0.5rem">'
        '  <i class="material-icons">timeline</i>'
        "</button>"
        "</td>"
    )


THRES_STRING = f"Recent Days R<sub>e</sub> < {THRES}"

lines = []
lines.append("<!DOCTYPE html>")
lines.append('<html lang="en">')
lines.append("<head>")
lines.append('<meta charset="utf-8"/>')
lines.append(
    """
        <link rel="icon" href="covid_fav.png" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="theme-color" content="#000000" />
        <meta name="description" content="Cerner COVID-19 Reopening and Social Distancing Model" />
        <link rel="apple-touch-icon" href="covid_512.png" />
    """
)
lines.append('<meta http-equiv="content-language" content="en" />')
lines.append('<meta name="google" value="notranslate">')
lines.append("  <title>Cerner COVID-19 Reopening and Social Distancing Model</title>")
lines.append(
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css'
    '?family=Roboto:300,400,500,700" type="text/css">'
)
lines.append(
    '<link rel="stylesheet" href="https://fonts.googleapis.com/'
    'icon?family=Material+Icons">'
)
lines.append(
    '<link rel="stylesheet" href="https://code.getmdl.io/1.3.0/'
    'material.indigo-pink.min.css">'
)
lines.append('<link rel="stylesheet" href="modal.css">')

lines.append(
    '<script defer src="https://code.getmdl.io/1.3.0/material.min.js">' "</script>"
)
lines.append(
    '<script type="text/javascript">window.PlotlyConfig = {MathJaxConfig:"local"};'
    '</script><script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
)
lines.append(
    '<link rel="stylesheet" href="https://cdn.datatables.net/1.10.20/css/dataTables.material.min.css">'
)
lines.append("</head>")
lines.append("<body>")
lines.append('<div id="myModal" class="modal">')
lines.append('  <span class="modal_close">&times;</span>')
lines.append('  <img class="modal-content" id="img01">')
lines.append("</div>")
lines.append('<div class = "mdl-layout mdl-js-layout">')
lines.append('<main class = "mdl-layout__content">')
lines.append('<div class="docs-text-styling">')
lines.append('  <section id="include">')
lines.append('    <div class="styles__content">')
```

<!-- #region heading_collapsed=true -->
#### Top card and tab bar
<!-- #endregion -->

```python code_folding=[] hidden=true
lines.append(
    """
    <p>Cerner has compiled a Reopening Risk Index (<a href="#CRRIDetails">CRRI</a>)
    with four risk categories.
    The index is based on the trend of COVID-19 cases within the last 14 days and
    provides guidance to stakeholders evaluating the potential consequences of
    reducing social distancing.
<div class="risk-scale mdl-card mdl-shadow--2dp">
  <div class="mdl-card__title">
  <img src="risk_index.svg" class="risk_index_img"></img>
  </div>
  <div class="mdl-card__supporting-text">
"""
)

with open("scenario.md", "r") as f:
    lines.append(markdown2.markdown(f.read()))
lines.append(
    """
  </div>
  <div class="mdl-card__actions mdl-card--border">
    <a class="mdl-button mdl-button--colored mdl-js-button mdl-js-ripple-effect" href="#tab-bar">
      See the risk index in your region
    </a>
  </div>
</div>
"""
)
lines.append('<div class="topdesc">')
# lines.append('<p><img src="examples.svg" class="example_image"></img></p>')

lines.append("</div>")
lines.append(
    """<div class = "mdl-tabs mdl-js-tabs" id="tab-bar">
       <div class = "mdl-tabs__tab-bar">
           <a href = "#country-panel" class = "mdl-tabs__tab is-active">Countries</a>
           <a href = "#state-panel" class = "mdl-tabs__tab">US States</a>
           <a href = "#csa-panel" class = "mdl-tabs__tab">US CSA</a>
           <a href = "#cbsa-panel" class = "mdl-tabs__tab">US CBSA</a>
       </div>"""
)
```

<!-- #region heading_collapsed=true -->
#### Countries
<!-- #endregion -->

```python code_folding=[] hidden=true
# Countries
lines.append('<div class="mdl-tabs__panel is-active" id="country-panel">')
lines.append(
    """
<h4>Highest Cerner Reopening Risk Index (CRRI) and R<sub>e</sub> in Last 2 Weeks by Country</h4>
<small><a href="#ReDesc">R<sub>e</sub></a> is an epidemiological measure of the rate of spread of the
epidemic. Values less than one represent decreasing case counts, while values greater than one represent
increasing case counts.</small>
<div class="countries-graph mdl-card mdl-shadow--2dp">
  <div class="mdl-card__title">
"""
)
lines.extend(c_summary_lines)
lines.append(
    """  </div>
  <div class="mdl-card__supporting-text"><p>
According to the WHO:</p>
    <blockquote style="font-size:12pt;" cite="https://www.who.int/emergencies/diseases/novel-coronavirus-2019/strategies-plans-and-operations">
Without careful planning, and in the absence of scaled
up public health and clinical care capacities, the premature
lifting of physical distancing measures is likely to lead
to an uncontrolled resurgence in COVID-19 transmission
and an amplified second wave of cases.</blockquote><p>
<a href="#CRRIDetails">Model details</a></p>
  </div>
  <div class="mdl-card__actions mdl-card--border">
    <a class="mdl-button mdl-button--colored mdl-js-button mdl-js-ripple-effect"
      href="javascript:open_image(&quot;./all_countries.svg&quot;)">
      Risk index over time for 60 most populous countries
    </a>
  </div>
</div>
"""
)
lines.append("<h4>Reopening Data by Country</h4>")
lines.append(
    '<table style="width:100%;" id="countries" class="mdl-data-table mdl-js-data-table">'
)
lines.append("  <thead><tr>")
lines.append('    <th class="mdl-data-table__cell--non-numeric">Country</th>')
lines.append('    <th class="mdl-data-table__cell--non-numeric">R<sub>0</sub></th>')
lines.append('    <th class="mdl-data-table__cell--non-numeric">R<sub>e</sub></th>')
lines.append(
    '    <th class="mdl-data-table__cell--non-numeric"> R<sub>e</sub> 2 Wk High</th>'
)
lines.append(f'    <th class="mdl-data-table__cell--non-numeric">{THRES_STRING}</th>')
lines.append(
    '    <th class="mdl-data-table__cell--non-numeric center">Past Data | Forecast</span></th>'
)
lines.append("  </tr></thead>")
lines.append("  <tbody>")
for country in tqdm(country_R_e.keys(), desc="Country Table"):
    if not os.path.exists(f"../docs/summary/C_{country}.svg"):
        print(country)
        continue
    lines.append("  <tr>")
    lines.append(
        f'    <td class="mdl-data-table__cell--non-numeric">'
        f"{name_fix(country)}</td>"
    )
    value = country_R_e[country].max()
    value_str = get_value_str(value)
    lines.append(f'    <td class="{R_0_class(value)}">{value_str}</td>')
    value = country_R_e[country].iloc[-1]
    value_str = get_value_str(value)
    lines.append(
        f'    <td class="valueOuter"><div class="innerBorder {R_e_class(value)}"><span>{value_str}</span></div></td>'
    )
    value = country_R_e[country].iloc[-14:].max()
    value_str = get_value_str(value)
    lines.append(
        f'    <td class="valueOuter"><div class="innerBorder {R_e_class(value)}"><span>{value_str}</span></div></td>'
    )
    lines.append(f"    <td>{get_days_below(country_R_e[country])}</td>")
    add_buttons(lines, country.replace(" ", "%20"), prefix="C_")
    lines.append("  </tr>")
lines.append("  </tbody>")
lines.append("</table>")
heattime_header(lines, "country")
lines.extend(c_heattime)
lines.append("</div>")
```

<!-- #region heading_collapsed=true -->
#### States
<!-- #endregion -->

```python code_folding=[] hidden=true
# States
lines.append('<div class = "mdl-tabs__panel" id = "state-panel">')
lines.append(
    """
<h4>Highest Cerner Reopening Risk Index (CRRI) and R<sub>e</sub> in Last 2 Weeks by State</h4>
<small><a href="#ReDesc">R<sub>e</sub></a> is an epidemiological measure of the rate of spread of the
epidemic. Values less than one represent decreasing case counts, while values greater than one represent
increasing case counts.</small>
<div class="states-graph mdl-card mdl-shadow--2dp">
  <div class="mdl-card__title">
"""
)
lines.extend(summary_lines)
lines.append(
    """  </div>
  <div class="mdl-card__supporting-text"><p>
  The White House recommends that states have consistently declining
    case counts for 2 weeks before reopening. States meeting this criterion have
    a Cerner Reopening Risk Index (CRRI) of 1
    and shown in blue, and states that are close to meeting this criterion have a
    CRRI of 2 and are shown in purple. <a href="#CRRIDetails">Model details</a></p>
  </div>
  <div class="mdl-card__actions mdl-card--border">
    <a class="mdl-button mdl-button--colored mdl-js-button mdl-js-ripple-effect"
      href="javascript:open_image(&quot;./all_states.svg&quot;)">
      Risk index over time for all states
    </a>
  </div>
</div>
"""
)
lines.append("<h4>Reopening Data by State</h4>")
lines.append(
    '<table style="width:100%" id="states" ' 'class="mdl-data-table mdl-js-data-table">'
)
lines.append("  <thead><tr>")
lines.append('    <th class="mdl-data-table__cell--non-numeric">State</th>')
lines.append('    <th class="mdl-data-table__cell--non-numeric">R<sub>0</sub></th>')
lines.append('    <th class="mdl-data-table__cell--non-numeric">R<sub>e</sub></th>')
lines.append(
    '    <th class="mdl-data-table__cell--non-numeric"> R<sub>e</sub> 2 Wk High</th>'
)
lines.append(f'    <th class="mdl-data-table__cell--non-numeric">{THRES_STRING}</th>')
lines.append(
    '    <th class="mdl-data-table__cell--non-numeric center">Past Data | Forecast</span></th>'
)
lines.append(f'    <th class="mdl-data-table__cell--non-numeric">Abbrev</th>')
lines.append("  </tr></thead>")
lines.append("  <tbody>")
for state in tqdm(sorted(state_R_e.keys()), desc="State Table"):
    if not os.path.exists(f"../docs/summary/{state}.svg"):
        continue
    lines.append("  <tr>")
    lines.append(f"    <td>" f"{name_fix(state)}</td>")
    value = state_R_e[state].max()
    value_str = get_value_str(value)
    lines.append(f'    <td class="{R_0_class(value)}">{value_str}</td>')
    value = state_R_e[state].iloc[-1]
    value_str = get_value_str(value)
    lines.append(
        f'    <td class="valueOuter"><div class="innerBorder {R_e_class(value)}"><span>{value_str}</span></div></td>'
    )
    value = state_R_e[state].iloc[-14:].max()
    value_str = get_value_str(value)
    lines.append(
        f'    <td class="valueOuter"><div class="innerBorder {R_e_class(value)}"><span>{value_str}</span></div></td>'
    )
    lines.append(f"    <td>{get_days_below(state_R_e[state])}</td>")
    add_buttons(lines, state.replace(" ", "%20"))
    lines.append(f"    <td>{state_pop.loc[state, 'ABBREV']}</td>")
    lines.append("  </tr>")
lines.append("  </tbody>")
lines.append("</table>")
heattime_header(lines, "state")
lines.extend(s_heattime)
lines.append("</div>")
```

<!-- #region heading_collapsed=true -->
#### CSAs
<!-- #endregion -->

```python code_folding=[] hidden=true
# CSAs
lines.append('<div class = "mdl-tabs__panel" id = "csa-panel">')
lines.append(
    """
<h4>Highest Cerner Reopening Risk Index (CRRI) and R<sub>e</sub> in Last 2 Weeks by CSA</h4>
<small><a href="#ReDesc">R<sub>e</sub></a> is an epidemiological measure of the rate of spread of the
epidemic. Values less than one represent decreasing case counts, while values greater than one represent
increasing case counts.</small>
<div class="csa-graph mdl-card mdl-shadow--2dp">
  <div class="mdl-card__title">
"""
)
lines.extend(csa_summary_lines)
lines.append(
    """  </div>
  <div class="mdl-card__supporting-text"><p>
The White House recommends that states have consistently declining
    case counts for 2 weeks before reopening. States meeting this criterion have
    a Cerner Reopening Risk Index (CRRI) of 1
    and shown in blue, and states that are close to meeting this criterion havea
    CRRI of 2 and are shown in purple. <a href="#CRRIDetails">Model details</a></p><p>
    A combined statistical area (<a href="#Geographical-Areas">CSA</a>) is a
    United States Office of Management
    and Budget (OMB) term for a combination of adjacent CBSAs in the United States
    and Puerto Rico that can demonstrate economic or social linkage.</p><p>
    To see all CSAs in a single state, type the state name in
    the search box.</p>
  </div>
  <div class="mdl-card__actions mdl-card--border">
    <a class="mdl-button mdl-button--colored mdl-js-button mdl-js-ripple-effect"
      href="javascript:open_image(&quot;./all_csas.svg&quot;)">
      Risk index over time for 60 most populous CSA
    </a>
  </div>
</div>
"""
)
lines.append("<h4>Reopening Data by CSA</h4>")
lines.append(
    '<table style="width:100%" id="csa" class="mdl-data-table mdl-js-data-table">'
)
lines.append("  <thead><tr>")
lines.append('    <th class="mdl-data-table__cell--non-numeric">CSA Name</th>')
lines.append('    <th class="mdl-data-table__cell--non-numeric">R<sub>0</sub></th>')
lines.append('    <th class="mdl-data-table__cell--non-numeric">R<sub>e</sub></th>')
lines.append(
    '    <th class="mdl-data-table__cell--non-numeric"> R<sub>e</sub> 2 Wk High</th>'
)
lines.append('    <th class="mdl-data-table__cell--non-numeric">States</th>')
lines.append(f'    <th class="mdl-data-table__cell--non-numeric">{THRES_STRING}</th>')
lines.append(
    '    <th class="mdl-data-table__cell--non-numeric center">Past Data | Forecast</span></th>'
)
if debug:
    lines.append('    <th class="mdl-data-table__cell--non-numeric">CSA Code</th>')
lines.append("  </tr></thead>")
lines.append("  <tbody>")
for csa_code in tqdm(sorted(csa_R_e.keys()), desc="CSA Table"):
    csa_name = csa_df.loc[csa_df.csacode == csa_code, "csatitle"].head(1).squeeze()
    if not os.path.exists(f"../docs/summary/{csa_code}.svg"):
        continue
    lines.append("  <tr>")
    lines.append(
        f'    <td class="mdl-data-table__cell--non-numeric">'
        f"{name_fix(csa_name)}</td>"
    )
    value = csa_R_e[csa_code].max()
    value_str = get_value_str(value)
    lines.append(f'    <td class="{R_0_class(value)}">{value_str}</td>')
    value = csa_R_e[csa_code].iloc[-1]
    value_str = get_value_str(value)
    lines.append(
        f'    <td class="valueOuter"><div class="innerBorder {R_e_class(value)}"><span>{value_str}</span></div></td>'
    )
    value = csa_R_e[csa_code].iloc[-14:].max()
    value_str = get_value_str(value)
    lines.append(
        f'    <td class="valueOuter"><div class="innerBorder {R_e_class(value)}"><span>{value_str}</span></div></td>'
    )
    states_list = ",".join(
        csa_df.loc[csa_df.csacode == csa_code, "fipsstatecode"]
        .drop_duplicates()
        .map(state_fips)
        .to_list()
    )
    lines.append(
        f'    <td class="mdl-data-table__cell--non-numeric">{states_list}</td>'
    )
    lines.append(f"    <td>{get_days_below(csa_R_e[csa_code])}</td>")
    add_buttons(lines, csa_code)
    lines.append(f"    <td>{csa_code}</td>")
    lines.append("  </tr>")
lines.append("  </tbody>")
lines.append("</table>")
heattime_header(lines, "CSA")
lines.extend(csa_heattime)
lines.append("</div>")
```

<!-- #region heading_collapsed=true -->
#### CBSAs
<!-- #endregion -->

```python code_folding=[] hidden=true
# CBSAs
lines.append('<div class = "mdl-tabs__panel" id = "cbsa-panel">')
lines.append(
    """
<h4>Highest Cerner Reopening Risk Index (CRRI) and R<sub>e</sub> in Last 2 Weeks by CBSA</h4>
<small><a href="#ReDesc">R<sub>e</sub></a> is an epidemiological measure of the rate of spread of the
epidemic. Values less than one represent decreasing case counts, while values greater than one represent
increasing case counts.</small>
<div class="cbsa-graph mdl-card mdl-shadow--2dp">
  <div class="mdl-card__title">
"""
)
lines.extend(cbsa_summary_lines)
lines.append(
    """  </div>
  <div class="mdl-card__supporting-text">
<p>The White House recommends that states have consistently declining
    case counts for 2 weeks before reopening. States meeting this criterion have
    a Cerner Reopening Risk Index (CRRI) of 1
    and shown in blue, and states that are close to meeting this criterion have a
    CRRI of 2 and are shown in purple. <a href="#CRRIDetails">Model details</a></p><p>
    A core-based statistical area (<a href="#Geographical-Areas">CBSA</a>) is a
    U.S. geographic area defined by the
    Office of Management and Budget (OMB) that consists of one or more counties
    (or equivalents) anchored by an urban center of at least 10,000 people plus
    adjacent counties that are socioeconomically tied to the urban center by
    commuting.</p><p>To see all CBSAs in a single state, type the state name in
    the search box.</p>
  </div>
  <div class="mdl-card__actions mdl-card--border">
    <a class="mdl-button mdl-button--colored mdl-js-button mdl-js-ripple-effect"
      href="javascript:open_image(&quot;./all_cbsas.svg&quot;)">
      Risk index over time for 60 most populous CBSA
    </a>
  </div>
</div>
"""
)
lines.append("<h4>Reopening Data by CBSA</h4>")
lines.append(
    '<table style="width:100%;" id="cbsa" class="mdl-data-table mdl-js-data-table">'
)
lines.append("  <thead><tr>")
lines.append('    <th class="mdl-data-table__cell--non-numeric">CBSA Name</th>')
lines.append('    <th class="mdl-data-table__cell--non-numeric">R<sub>0</sub></th>')
lines.append('    <th class="mdl-data-table__cell--non-numeric">R<sub>e</sub></th>')
lines.append(
    '    <th class="mdl-data-table__cell--non-numeric"> R<sub>e</sub> 2 Wk High</th>'
)
lines.append('    <th class="mdl-data-table__cell--non-numeric">States</th>')
lines.append(f'    <th class="mdl-data-table__cell--non-numeric">{THRES_STRING}</th>')
lines.append(
    '    <th class="mdl-data-table__cell--non-numeric center">Past Data | Forecast</span></th>'
)
if debug:
    lines.append('    <th class="mdl-data-table__cell--non-numeric">CBSA Code</th>')
lines.append("  </tr></thead>")
lines.append("  <tbody>")
cbsa_codes.sort()
for cbsa_code in tqdm(cbsa_R_e.keys(), desc="CBSA Table"):
    cbsa_name = (
        cbsa_df.loc[cbsa_df.lma_fips == cbsa_code, "lma_title"].head(1).squeeze()
    )
    if not os.path.exists(f"../docs/summary/{cbsa_code}.svg"):
        continue
    lines.append("  <tr>")
    lines.append(
        f'    <td class="mdl-data-table__cell--non-numeric">'
        f"{name_fix(cbsa_name)}</td>"
    )
    value = cbsa_R_e[cbsa_code].max()
    value_str = get_value_str(value)
    lines.append(f'    <td class="{R_0_class(value)}">{value_str}</td>')
    value = cbsa_R_e[cbsa_code].iloc[-1]
    value_str = get_value_str(value)
    lines.append(
        f'    <td class="valueOuter"><div class="innerBorder {R_e_class(value)}"><span>{value_str}</span></div></td>'
    )
    value = cbsa_R_e[cbsa_code].iloc[-14:].max()
    value_str = get_value_str(value)
    lines.append(
        f'    <td class="valueOuter"><div class="innerBorder {R_e_class(value)}"><span>{value_str}</span></div></td>'
    )
    states_list = ",".join(
        cbsa_df.loc[cbsa_df.lma_fips == cbsa_code, "st_fips"]
        .drop_duplicates()
        .map(state_fips)
        .to_list()
    )
    lines.append(
        f'    <td class="mdl-data-table__cell--non-numeric">{states_list}</td>'
    )
    lines.append(f"    <td>{get_days_below(cbsa_R_e[cbsa_code])}</td>")
    add_buttons(lines, cbsa_code)
    lines.append(f"    <td>{cbsa_code}</td>")
    lines.append("  </tr>")
lines.append("  </tbody>")
lines.append("</table>")
heattime_header(lines, "CBSA")
lines.extend(cbsa_heattime)
lines.append("</div>")
lines.append("</div>")
```

<!-- #region heading_collapsed=true -->
#### Details and footer
<!-- #endregion -->

```python code_folding=[] hidden=true
with open("scenario_details.md", "r") as f:
    lines.append(markdown2.markdown(f.read()))
lines.append("    </div>")
lines.append("  </section>")
lines.append("</div>")
lines.append('<footer class="mdl-mini-footer" id="footer" style="padding:1rem;">')
lines.append('<div class = "mdl-mini-footer__left-section">')
lines.append('  <div class = "mdl-logo">')
lines.append("    &copy;Cerner Corporation 2020. All rights reserved.")
lines.append("  </div>")
lines.append(
    """
    <ul class="mdl-mini-footer__link-list">
      <li><a href="data/country_R_e.csv">Country R<sub>e</sub> CSV</a></li>
      <li><a href="data/state_R_e.csv">US State R<sub>e</sub> CSV</a></li>
      <li><a href="data/csa_R_e.csv">US CSA R<sub>e</sub> CSV</a></li>
      <li><a href="data/cbsa_R_e.csv">US CBSA R<sub>e</sub> CSV</a></li>
    </ul>
"""
)
lines.append("</div>")
lines.append('<div class = "mdl-mini-footer__right-section">')
lines.append(
    pd.Timestamp.now()
    .strftime('  <div class = "mdl-logo">Last Updated %b %d, %Y at %I:%M %p</div>')
    .replace(" 0", " ")
)
lines.append("</div></footer>")
lines.append("</main></div>")
lines.append('<script src="modal.js"></script>')
lines.append(
    """<!-- Bootstrap -->

    <!-- DATA TABLES SCRIPT -->
    <script src="https://code.jquery.com/jquery-3.3.1.js" type="text/javascript"></script>
    <script src="https://cdn.datatables.net/1.10.20/js/jquery.dataTables.min.js" type="text/javascript"></script>
    <script src="https://cdn.datatables.net/1.10.20/js/dataTables.material.min.js" type="text/javascript"></script>
    <script type="text/javascript">
       $(function() {
        $("#states").dataTable({
            "iDisplayLength": 10,
            "aLengthMenu": [[10, 25, 50, 100,  -1], [10, 25, 50, 100, "All"]],
            "order": [[ 3, "asc" ]],
            "columnDefs": [{"targets":[6],"visible":false}]
           });
       });
        $(function() {
        $("#csa").dataTable({
            "iDisplayLength": 10,
            "aLengthMenu": [[10, 25, 50, 100,  -1], [10, 25, 50, 100, "All"]],
            "order": [[ 3, "asc" ]],
            "columnDefs": [{"targets":[4, 7],"visible":false}]
           });
       });
        $(function() {
        $("#cbsa").dataTable({
            "iDisplayLength": 10,
            "aLengthMenu": [[10, 25, 50, 100,  -1], [10, 25, 50, 100, "All"]],
            "order": [[ 3, "asc" ]],
            "columnDefs": [{"targets":[4, 7],"visible":false}]
           });
       });
        $(function() {
        $("#countries").dataTable({
            "iDisplayLength": 10,
            "aLengthMenu": [[10, 25, 50, 100,  -1], [10, 25, 50, 100, "All"]],
            "order": [[ 3, "asc" ]]
           });
       });
      </script>"""
)
lines.append("</body></html>")
```

#### Output

```python code_folding=[]
with open("../docs/index.html", "w") as f:
    html_out = re.sub("\n", " ", "".join(lines))
    html_out = re.sub(" +", " ", html_out).replace("> <", "><").replace(": ", ":")
    f.write(html_out)
```

```python
df = pd.DataFrame(country_R_e).T
df.columns = df.columns.to_series().dt.strftime("%Y-%m-%d")
df["CRRI"] = df.iloc[:, -1].pipe(CRRI)
df.to_csv("../docs/data/country_R_e.csv")

df = pd.DataFrame(state_R_e).T
df.columns = df.columns.to_series().dt.strftime("%Y-%m-%d")
df["CRRI"] = df.iloc[:, -1].pipe(CRRI)
df.to_csv("../docs/data/state_R_e.csv")

df = pd.DataFrame(csa_R_e).T
df.columns = df.columns.to_series().dt.strftime("%Y-%m-%d")
df["CRRI"] = df.iloc[:, -1].pipe(CRRI)
df.to_csv("../docs/data/csa_R_e.csv")

df = pd.DataFrame(cbsa_R_e).T
df.columns = df.columns.to_series().dt.strftime("%Y-%m-%d")
df["CRRI"] = df.iloc[:, -1].pipe(CRRI)
df.to_csv("../docs/data/cbsa_R_e.csv")
```
