## Release summary

This is the CRAN submission of `glossa` v1.2.0.

Key changes:

* Added support for tuning the number of trees in the sum-of-trees formulation of BART and the end-node shrinkage prior parameter to control overfitting (12/06/2025).
* Resolution harmonization is now implemented for environmental layers by aggregating to the coarsest resolution, complementing the existing extent and CRS harmonization (10/06/2025).
* Resolved a UI issue introduced in version 1.1.0 that prevented users from selecting thinning options (28/06/2025).

## Test environments

The package was tested on the following environments:

* Windows Server 2022 x64 (build 20348)
* Debian GNU/Linux trixie/sid

## R CMD check results

The package passed `R CMD check` on all tested platforms with one NOTE:

```
0 errors | 0 warnings | 1 note
```

```
New submission

Package was archived on CRAN

Possibly misspelled words in DESCRIPTION:
  Chipman (15:18)
  GLOSSA (13:56, 20:55)
  McCulloch (15:39)
  Spatiotemporal (14:13)
```

* This is a new submission after the package was archived.
* The flagged words in the DESCRIPTION file refer to proper nouns and standard terminology.
