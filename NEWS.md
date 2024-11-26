# glossa (development version)

### New Features

- Categorical variables can now be included as predictor variables using raster layers with factors (2024-11-25).

### Improvements

- Updated functions to support raster files with metadata (e.g., `.xml`) for declaring categorical variables or factors (2024-11-25).
- Enhanced handling of mixed continuous and categorical datasets during layer reading and processing (2024-11-25).
- Scaling now applies only to continuous variables, with categorical variables excluded and appended untransformed (2024-11-25).
- Improved functional response calculations to handle categorical predictors, filtering out levels with no observations (2024-11-25).
- Added functionality to compute the mode across time layers for categorical variables, preserving factor levels, when computing the average environmental scenario (2024-11-25).

# glossa 1.0.0

* Initial CRAN submission.
