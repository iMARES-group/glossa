## Release summary

This is the CRAN submission of `glossa` v1.2.1.

#### Changed

* Moved the `automap` package to `Imports` and `jsonlite` to `Suggests` (20/06/2025).

#### Fixed

* Fixed an UI issue in the validation table of the study area polygon. Although the upload worked, the validation table sometimes incorrectly flagged the format as invalid. This did not affect the analysis but was misleading (20/06/2025).

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
