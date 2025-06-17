
<!-- README.md is generated from README.Rmd. Please edit that file -->

# GLOSSA - Global Ocean Species Spatio-temporal Analysis <a href="https://iMARES-group.github.io/glossa/"><img src="inst/app/www/logo_glossa.png" align="right" height="138" /></a>

<!-- badges: start -->

[![CRAN
status](https://www.r-pkg.org/badges/version/glossa)](https://CRAN.R-project.org/package=glossa)
[![License: GPL
v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![R-CMD-check](https://github.com/iMARES-group/glossa/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/iMARES-group/glossa/actions/workflows/R-CMD-check.yaml)
<!-- badges: end -->

------------------------------------------------------------------------

**GLOSSA** (Global Ocean Species Spatio-temporal Analysis) is an
open-source, user-friendly R Shiny application designed for modeling
marine species distribution. Written in R, GLOSSA uses the Shiny and
bs4Dash libraries to provide an intuitive interface for fitting species
distribution models using presence/absence or presence-only
(pseudo-absences will be generated) data. The app uses flexible machine
learning techniques like Bayesian Additive Regression Trees (BART),
enabling users to model species distributions across different temporal
and spatial scales and forecast future scenarios based on environmental
data. With GLOSSA, users can explore current and future suitable
habitats and visualize results with minimal coding expertise.

![](https://github.com/iMARES-group/glossa/blob/main/inst/app/www/img/glossa_short_flowchart.png)

## Getting started

You can install and run GLOSSA using the following R code:

``` r
install.packages("glossa")

library(glossa)
run_glossa()
```

To install the development version of GLOSSA from GitHub, use:

``` r
if (!require("devtools")) 
  install.packages("devtools")

devtools::install_github("iMARES-group/glossa")
```

## Documentation and resources

For detailed documentation, tutorials, and examples on how to use
**GLOSSA**, visit the [official
website](https://iMARES-group.github.io/glossa/):

- [Overview](https://iMARES-group.github.io/glossa/)
- [Installation and getting started
  guide](https://iMARES-group.github.io/glossa/get_started.html)
- [Full
  documentation](https://iMARES-group.github.io/pages/documentation/)
- [Tutorials and
  examples](https://iMARES-group.github.io/glossa/pages/tutorials_examples/)

### Current version

- **Development version** in progress (17/06/2025)
- CRAN release: **v1.2.0** (17/06/2024)

See the full
[changelog](https://github.com/iMARES-group/glossa/blob/main/NEWS.md)

## How to cite GLOSSA

The GLOSSA manuscript is currently in progress. In the meantime, you can
cite the [preprint](https://doi.org/10.48550/arXiv.2505.05862) as
follows:

> Mestre-Tomás, J., Fuster-Alonso, A., Bellido, J. M., and Coll, M.
> (2025). GLOSSA: a user-friendly R Shiny application for Bayesian
> machine learning analysis of marine species distribution. arXiv
> preprint arXiv:2505.05862. DOI:
> <https://doi.org/10.48550/arXiv.2505.05862>
