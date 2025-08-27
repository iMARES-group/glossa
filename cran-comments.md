## Release summary

This is the CRAN submission of `glossa` v1.2.3.

#### Added

* New pseudo-absence generation method: `"env_space_flexsdm"`, which samples pseudo-absences from regions with lower suitability in the environmental space using the `sample_pseudoabs(method=c(method='env_const', env = somevar))` function from the *flexsdm* package. (26/08/2025, #6)

#### Fixed

* Large raster previews that exceed the 4 MB limit from *leaflet* are now automatically downsampled for visualization, while full-resolution rasters are kept for analysis and export. (26/08/2025, #11)

## Test environments

The package was tested on the following environments:

* Windows Server 2022 x64 (build 20348)
* Debian GNU/Linux trixie/sid

## R CMD check results

The package passed `R CMD check` on all tested platforms with one NOTE:

```
0 errors | 1 warnings | 1 note
```

```
New submission

Package was archived on CRAN

Possibly misspelled words in DESCRIPTION:
  Chipman (15:29)
  GLOSSA (13:56, 20:55)
  McCulloch (15:50)
  Spatio (14:19)
```

* This is a new submission after the package was archived.
* The flagged words in the DESCRIPTION file refer to proper nouns and standard terminology.

```
CRAN repository db overrides:
  X-CRAN-Comment: Archived on 2025-07-30 as required archived package
    'blockCV'.
```

* The package 'blockCV' is back to CRAN. No further issues.

```
Suggests or Enhances not in mainstream repositories:
  flexsdm
```

* `flexsdm` remains in Suggests and is used conditionally. We have added a statement in DESCRIPTION indicating where to obtain it (GitHub). The package passes checks and runs without `flexsdm`.
* The package `flexsdm` is an R package only available in GitHub (<https://github.com/sjevelazco/flexsdm>) and listed under Suggests because it is only needed for an optional pseudo-absence generation method ("env_space_flexsdm"). All core functions of GLOSSA work fully without `flexsdm`. 
* Installation instructions for using the environmental-space pseudo-absence method with `flexsdm` (from GitHub) are provided in the online documentation (<https://imares-group.github.io/glossa/pages/documentation/installation_setup.html>).

```
# flexsdm installation
# install.packages("remotes")

# For Windows and Mac OS operating systems
remotes::install_github("sjevelazco/flexsdm")

# For Linux operating system
remotes::install_github("sjevelazco/flexsdm@HEAD")
```
