## Model Description

### Introduction

This model aims to inform reopening decisions and to provide some possible trajetories after a reopening to provide public health officials and other stakeholders information about the likely future trajectory to best make continuing decisions.

To keep the number of COVID-19 cases manageable, R<sub>e</sub> should be kept below 1. This model aims to provide retrospective estimates of R<sub>e</sub>, as well as forecast different scenarios in which social distancing is reduced by various percentages. For the purposes of these scenarios, the reopening date is currently set to May 15.

Based on recommendations from the White House, beginning a parital reopening should only be undertaken when the number of documented cases has a downward trajectory over a 14-day period. This is an equivalent to R<sub>e</sub> being under 1 for the same period.

### What are R<sub>0</sub> and R<sub>e</sub>?

R<sub>0</sub> is the [basic reproduction number](https://en.wikipedia.org/wiki/Basic_reproduction_number), defined as the average number people infected by each patient who contracts COVID-19. It incorporates the probability that each contact results in a transmission and the average number of contacts per day.

R<sub>e</sub> is the effective reproduction number, taking into account social distancing measures and herd immunity.



### Geographical Areas
A [**core-based statistical area**](https://en.wikipedia.org/wiki/Core-based_statistical_area) (CBSA) is a U.S. geographic area defined by the Office of Management and Budget (OMB) that consists of one or more counties (or equivalents) anchored by an urban center of at least 10,000 people plus adjacent counties that are socioeconomically tied to the urban center by commuting. A CBSA is either a metropolitcan statistical area (MSA) or a micropolitan statistical area (&mu;SA), determined by whether the area has a population of greater than or less than 50,000, respectively.

A [**combined statistical area**](https://en.wikipedia.org/wiki/Combined_statistical_area) (CSA) is a United States Office of Management and Budget (OMB) term for a combination of adjacent CBSAs in the United States and Puerto Rico that can demonstrate economic or social linkage.

A [**labor market area**](https://en.wikipedia.org/wiki/Labor_market_area) is an economically integrated region within which residents can find jobs within a reasonable commuting distance or can change their employment without changing their place of residence.

A [**New England city and town area**](https://en.wikipedia.org/wiki/New_England_city_and_town_area) (NECTA) is a geographic and statistical entity defined by the U.S. federal government for use in the six-state New England region of the United States. We use MNECTA to refer to a metropolitan NECTA and &mu;NECTA to refer to a micropolitan NECTA.

### Data Sources

Only publicly available data sources were used to build the Cerner model. The public data sources include:

  - [Definitive Healthcare hospital bed data](https://coronavirus-resources.esri.com/datasets/definitivehc::definitive-healthcare-usa-hospital-beds)

  - [Johns Hopkins CSSE COVID-19 case count data](https://github.com/CSSEGISandData/COVID-19)

  - [US Census county population counts](https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv)

  - [Dataset from American Red Cross: COVID19 Public Health Emergency Status by County](https://coronavirus-disasterresponse.hub.arcgis.com/datasets/arc-nhq-gis::covid19-public-health-emergency-status-by-county)



### Methodology

We first use [local regression](https://en.wikipedia.org/wiki/Local_regression) on the cumulative case counts to derive a smoothed version of the daily case counts. Then, we apply a local exponential regression on these smoothed daily case counts in order to infer the logarithmic growth rate. The logarithmic growth rate can be thought of as the approximate day-to-day percentage increase in daily case count.

We then use the [SEIR model](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology) to calculate an estimate for R<sub>e</sub> over time, and apply a moving average to smooth out short time-scale variations.

We assume that R<sub>e</sub> remains what it is today until social distancing is relaxed (assumed here to be May 15). After that date, we assume that R<sub>e</sub> will eventually increase by various percentages towards the initial R<sub>0</sub>. We model the gradual increase in R<sub>e</sub> exponentially, assuming it will reach 90% of its final value over the course of 10 days.

The SEIR model is used to model the case counts. We assume here that reported case counts are about 20% of total cases (including asymptomatic cases).

This analysis is built for each CSA or CBSA with at least 100 cumulative cases, and a daily case count of 10 or more at least 3 days ago.
