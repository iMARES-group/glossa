## Release summary

This is the CRAN submission of `glossa` v1.2.2.

#### Added

* Results are now automatically assigned to `glossa_autosave` in the global environment after analysis completes, unless `clear_global_env = TRUE` in the `run_glossa()` function. This allows recovery even if the browser is accidentally closed before completion. (23/06/2025, #8)

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
  Chipman (15:18)
  GLOSSA (13:56, 20:55)
  McCulloch (15:39)
  Spatiotemporal (14:13)
```

* This is a new submission after the package was archived.
* The flagged words in the DESCRIPTION file refer to proper nouns and standard terminology.

```
CRAN repository db overrides:
  X-CRAN-Comment: Archived on 2025-06-30 as requires archived package
    'automap'.
```

* The package 'automap' is back to CRAN. No further issues.
