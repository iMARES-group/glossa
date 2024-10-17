## Resubmission

This is a resubmission to address the feedback from the previous CRAN submission.

* Updated the `Title` field in the DESCRIPTION file to place 'shiny' in single quotes.
* Added BART references in the `Description` field.
* Added the missing `\value{}` tags to `.Rd` files for the following exported functions: `getFprTpr`, `glossa_export`, `misClassError`, `optimalCutoff`, `run_glossa`, `sensitivity`, `specificity`, `variable_importance`, and `youdensIndex`.
* Replaced `print()` statements with `message()` in the `glossa_analysis` function.

## R CMD check results

The package passed `R CMD check` on all tested platforms with the following results.
There was one NOTE:

```
New submission

Possibly misspelled words in DESCRIPTION:
  Chipman (15:18)
  GLOSSA (13:56, 20:55)
  McCulloch (15:39)
  Spatiotemporal (14:13)
```

* This is a new submission to CRAN.
* The flagged words in the DESCRIPTION file refer to proper nouns and standard terminology.
