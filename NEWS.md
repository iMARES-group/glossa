# Changelog

All notable changes to this project will be documented in this file.

## glossa (development version)

### Added

* Results are now automatically assigned to `glossa_autosave` in the global environment after analysis completes, unless `clear_global_env = TRUE` in the `run_glossa()` function. This allows recovery even if the browser is accidentally closed before completion. (23/06/2025, #8)

## glossa 1.2.1 - 20/06/2025

### Changed

* Moved the `automap` package to `Imports` and `jsonlite` to `Suggests` (20/06/2025).

### Fixed

* Fixed an UI issue in the validation table of the study area polygon. Although the upload worked, the validation table sometimes incorrectly flagged the format as invalid. This did not affect the analysis but was misleading (20/06/2025).

## glossa 1.2.0 - 17/06/2025

### Added

* Added support for tuning the number of trees in the sum-of-trees formulation of BART and the end-node shrinkage prior parameter to control overfitting (12/06/2025).
* Resolution harmonization is now implemented for environmental layers by aggregating to the coarsest resolution, complementing the existing extent and CRS harmonization (10/06/2025).

### Fixed

* Resolved a UI issue introduced in version 1.1.0 that prevented users from selecting thinning options (28/05/2025).

## glossa 1.1.0 - 28/05/2025

### New features

* Added support for spatial block and temporal block cross-validation. Spatial block size can be determined manually or using residual or predictor autocorrelation (27/05/2025).
* Continuous Boyce Index (CBI) and Area Under the ROC Curve (AUC) are now computed for each model based on fitted values and in cross-validation (27/05/2025).
* Now three different pseudo-absence generation can be used: random background, target-group, and delimited by a buffer around presences (21/05/2025).
* A new configuration file including the input files and settings is now automatically created when exporting the results for better reproducibility (15/05/2025).
* Users can now upload a file of raster timestamps to correctly align irregular or custom time series with occurrence data. This prevents mismatches between occurrence timestamps and raster layers when temporal coverage is not sequential or has a different starting point (11/04/2025).
* Categorical variables can now be included as predictor variables using raster layers with factors (25/11/2024).

### Minor improvements and bug fixes

* Added confirmation dialog to close the app to avoid accidental clicks on the "Close App" button (27/05/2025).
* Improved error handling so analysis does not stop if any of the species drops an error (15/05/2025).
* Fixed reactivity and selection reset issues when updating plot inputs in the report tab (e.g. functional response plot) (02/04/2025).
* Fixed the broken link in the Documentation button on the GLOSSA home page (31/03/2025).
* Fixed an issue where users could upload `.tif` files but not `.tiff` files, which are used interchangeably (31/03/2025).
* Added support for handling zipped files from macOS by ignoring hidden system files (31/03/2025).
* Fixed an issue where accessing factor level labels assumed the column was named "label". Now uses the second column (31/03/2025).
* Warning messages appear when `NA` values are present in factor levels. Those levels are ignored in the analysis (31/03/2025).
* Allow a numerical tolerance of 1e-7 when evaluating resolution equality between provided raster layers (31/03/2025).
* Updated functions to support raster files with metadata (e.g., `.xml`) for declaring categorical variables or factors (25/11/2024).
* Enhanced handling of mixed continuous and categorical datasets during layer reading and processing (25/11/2024).
* Scaling now applies only to continuous variables, with categorical variables excluded and appended untransformed (25/11/2024).
* Improved functional response calculations to handle categorical predictors, filtering out levels with no observations (25/11/2024).
* Added functionality to compute the mode across time layers for categorical variables, preserving factor levels, when computing the average environmental scenario (25/11/2024).
* Changed exported file names to avoid Windows file path length errors when unzipping. Also, changed extension from `.csv` to `.tsv` (26/05/2025). The following naming adjustments were made: `suitable_habitat` -> `sh`, `native_range` -> `nr`, `cross_validation` -> `cross_val`, `functional_responses` -> `func_res`, `variable_importance` -> `var_imp`, `confusion_matrix` -> `mod_diag`, `fit_layers` -> `fit`, `projections` -> `proj`.

## glossa 1.0.0 - 15/10/2024

* Initial CRAN submission.
