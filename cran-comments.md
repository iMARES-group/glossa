## Release summary

This is the CRAN submission of `glossa` v1.1.0.

Key changes:
* Added support for spatial block and temporal block cross-validation.
* Users can now choose between different pseudo-absence generation strategies (random background, target-group, and buffer-restricted).
* Added new model evaluation metrics such as the Continuous Boyce Index (CBI) and Area Under the Curve (AUC).
* Improved reproducibility by exporting a configuration file with all user inputs and settings.
* Added support for a timestamp file that allows proper alignment between occurrence records and covariate layers.
* Raster layers with factor levels are now supported.
* Exported file names have been shortened and standardized to avoid Windows file path length errors when unzipping.

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

```
CRAN repository db overrides:
  X-CRAN-Comment: Archived on 2025-05-08 as issues were not corrected
    in time.
```

* All previously noted issues have now been corrected.
