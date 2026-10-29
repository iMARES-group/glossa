## Release summary

This is the CRAN submission of `glossa` v1.2.5.

### Added

* Updated the citation to the journal reference. (08/12/2025)

### Changed

* Removed the dependency on `mcp`. GLOSSA now use the inverse probit link from `stats` package preserving existing results. Thanks to @lindeloev for the contribution. (29/10/2026, #14).

### Fixed

* Fixed handling of study area inputs with multiple polygon features: now applies `sf::st_union()` and prints a warning. (09/01/2026, #13)
* Fixed issue where environmental layers files with different resolutions did not load. (14/04/2026)

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

Possibly misspelled words in DESCRIPTION:
  Chipman (15:29)
  GLOSSA (13:56, 20:55)
  McCulloch (15:50)
  Spatio (14:19)
```

* The flagged words in the DESCRIPTION file refer to proper nouns and standard terminology.

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
