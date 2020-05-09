### Introduction <a name="Intro"></a>
To keep the number of COVID-19 cases manageable, R<sub>e</sub> should be kept below 1. This model aims to provide retrospective estimates of R<sub>e</sub>, as well as forecast different scenarios in which social distancing is reduced by various percentages. For the purposes of these scenarios, the reopening date is currently set to planned reopening dates at the State level if known, otherwise May 15.

In the U.S., based on [recommendations](https://www.whitehouse.gov/openingamerica/) from the White House, beginning a partial reopening should only be undertaken when the number of documented cases has a downward trajectory over a 14-day period. This is an equivalent to R<sub>e</sub> being under 1 for the same period.

An export of R<sub>e</sub> data over time can be found in the <a href="#footer">footer</a>.


### What are R<sub>0</sub> and R<sub>e</sub>?<a name="ReDesc"></a>

An R<sub>0</sub> or R<sub>e</sub> is defined as the average number people infected by each patient who contracts COVID-19. For example, a value of 1 means on average 1 person is infected by each person who contracts COVID-19.

R<sub>0</sub> is the [basic reproduction number](https://en.wikipedia.org/wiki/Basic_reproduction_number) at the beginning of the epidemic and prior to social distancing. R<sub>e</sub> is the effective reproduction number that takes into account social distancing measures and partial herd immunity from antibody immunity and is the best measure of the current rate of spread of the epidemic.

### How is the Cerner Reopening Risk Index (CRRI) defined?<a name="CRRIDetails"></a>

The CRRI is defined as

* **1** (low risk) when R<sub>e</sub> is less than 0.9

* **2** (medium risk) when R<sub>e</sub> is between 0.9 and 1.1

* **3** (high risk) when R<sub>e</sub> is between 1.1 and 1.3

* **4** (very high risk) when R<sub>e</sub> is greater than 1.3


### Geographical areas<a name="Geographical-Areas"></a>

A [**core-based statistical area**](https://en.wikipedia.org/wiki/Core-based_statistical_area) (CBSA) is a U.S. geographic area defined by the Office of Management and Budget (OMB) that consists of one or more counties (or equivalents) anchored by an urban center of at least 10,000 people plus adjacent counties that are socioeconomically tied to the urban center by commuting. A CBSA is either a metropolitan statistical area (MSA) or a micropolitan statistical area (&mu;SA), determined by whether the area has a population of greater than or less than 50,000, respectively.

A [**combined statistical area**](https://en.wikipedia.org/wiki/Combined_statistical_area) (CSA) is a United States Office of Management and Budget (OMB) term for a combination of adjacent CBSAs in the United States and Puerto Rico that can demonstrate economic or social linkage.

A [**labor market area**](https://en.wikipedia.org/wiki/Labor_market_area) is an economically integrated region in which residents can find jobs within a reasonable commuting distance or can change their employment without changing their place of residence.

A [**New England city and town area**](https://en.wikipedia.org/wiki/New_England_city_and_town_area) (NECTA) is a geographic and statistical entity defined by the U.S. federal government for use in the six-state New England region. We use MNECTA to refer to a metropolitan NECTA and &mu;NECTA to refer to a micropolitan NECTA.

### Data sources

Only publicly available data sources were used to build the Cerner model. The public data sources include:

  - [Definitive Healthcare hospital bed data](https://coronavirus-resources.esri.com/datasets/definitivehc::definitive-healthcare-usa-hospital-beds)

  - [Johns Hopkins CSSE COVID-19 case count data](https://github.com/CSSEGISandData/COVID-19)

  - [U.S. Census county population counts](https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv)

  - [Dataset from the American Red Cross: &nbsp;COVID19 Public Health Emergency Status by County](https://coronavirus-disasterresponse.hub.arcgis.com/datasets/arc-nhq-gis::covid19-public-health-emergency-status-by-county)



### Methodology

We first use [local regression](https://en.wikipedia.org/wiki/Local_regression) on the cumulative case counts to derive a smoothed version of the daily case counts. Then, we apply a local exponential regression on these smoothed daily case counts in order to infer the logarithmic growth rate. The logarithmic growth rate can be thought of as the approximate day-to-day percentage increase in daily case count.

We then use the [SEIR model](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology) to calculate an estimate for R<sub>e</sub> over time and apply a moving average to smooth out short time-scale variations. To parametrize this SEIR model, based on the literature, we assume that the [incubation time](https://annals.org/aim/fullarticle/2762808/incubation-period-coronavirus-disease-2019-covid-19-from-publicly-reported) is 3 days, and the [serial interval](https://wwwnc.cdc.gov/eid/article/26/7/20-1595_article) is 5 days.

We assume that R<sub>e</sub> remains what it is today until social distancing is relaxed. The planned date of any partial reopening is used, if known, and is otherwise assumed to be May 15. Currently we have collated this data for U.S. states but not for other geographical groupings. After that date, we assume that R<sub>e</sub> will eventually increase by various percentages toward the initial R<sub>0</sub>. The model assumes that R<sub>e</sub> will reach its eventual value within 10 days of the reopening date.

Currently, we assume that 80% of those exposed to COVID-19 do not have a documented positive test, having not received a test because they were asymptomatic or experiencing only mild symptoms. Incorporating information about testing positivity rates to inform estimates of true case counts is planned for a future release of this model.

The SEIR model is used to predict the case counts. We assume here that reported case counts are about 10% of total cases (including asymptomatic cases). Exact [estimates](https://medicalxpress.com/news/2020-04-covid-average-actual-infections-worldwide.html) &nbsp;&nbsp;[differ](https://www.npr.org/sections/coronavirus-live-updates/2020/04/23/842818125/coronavirus-has-infected-a-fifth-of-new-york-city-testing-suggests).

This analysis is built for each [CSA](#Geographical-Areas) or [CBSA](#Geographical-Areas) with at least 100 cumulative cases and a daily case count of 10 for at least three days.

### Discussion

All interventions, including but not limited to social distancing policies, wearing masks in public, contact tracing, and expanded testing can reduce R<sub>e</sub> and any strategy for reopening should make use of all available strategies. Monitoring the state of R<sub>e</sub> is of paramount importance in making continuing decisions about social distancing policies and allocation of resources.

Finally, it's worth mentioning effective contact tracing and mass testing are some of the most effective ways to deal with the COVID-19 epidemic. A good example is [South Korea's](javascript:open_image("summary/C_Republic%20of%20Korea.svg")) success in reducing R<sub>e</sub> with such policies.

### Caveats<a name="Caveats"></a>

- Estimations of R<sub>e</sub> can have high variance for regions with few total cases documented, or a relatively short history of having 10 or more cases.

- Case counts can be unreliable, and the amount of unreliability may vary by jurisdiction. Many cases of COVID-19 go undetected because the patient is either asymptomatic or mildly symptomatic and doesn't undergo testing. We chose to use case counts instead of deaths because of the inherent time lag of deaths data, which can be a hindrance for public health planning and for hospital surge planning.

- Forecasted future daily case counts are based on current reporting practices, 
and estimate the number of cases that will be reported, not the actual total.

- Because many of the factors going into R<sub>e</sub> are inherently behavioral and policy-based rather than epidemiological, we assume a constant R<sub>e</sub> between the time the model is run and the assumed relaxation date.

- Results are not guaranteed.
