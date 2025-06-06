#' Remove Duplicated Points from a Dataframe
#'
#' This function removes duplicated points from a dataframe based on specified coordinate columns.
#'
#' @param df A dataframe object with each row representing one point.
#' @param coords A character vector specifying the names of the coordinate columns used for identifying duplicate points. Default is c("decimalLongitude", "decimalLatitude").
#'
#' @return A dataframe without duplicated points.
#'
#' @export
remove_duplicate_points <- function(df, coords = c("decimalLongitude", "decimalLatitude")) {
  # Check if df is a dataframe
  if (!is.data.frame(df)) {
    stop("Input must be a data frame.")
  }

  # Check if coordinate columns are specified correctly
  if (!is.character(coords) || length(coords) == 0) {
    stop("Coordinate columns must be specified as a non-empty character vector.")
  }

  # Ensure specified coordinate columns exist in the dataframe
  missing_coords <- setdiff(coords, colnames(df))
  if (length(missing_coords) > 0) {
    stop(paste("Coordinate column(s) not found in dataframe:", paste(missing_coords, collapse = ", ")))
  }

  # Remove duplicated points
  unique_data <- df[!duplicated(df[, coords]), ]

  return(unique_data)
}


#' Remove Points Inside or Outside a Polygon
#'
#' This function removes points from a dataframe based on their location relative to a specified polygon.
#'
#' @param df A dataframe object with rows representing points.
#' @param polygon An sf polygon object defining the region for point removal.
#' @param overlapping Logical indicating whether points overlapping the polygon should be removed (TRUE) or kept (FALSE).
#' @param coords Character vector specifying the column names for longitude and latitude. Default is c("decimalLongitude", "decimalLatitude").
#'
#' @return A dataframe containing the filtered points.
#'
#' @export
remove_points_polygon <- function(df, polygon, overlapping = FALSE, coords = c("decimalLongitude", "decimalLatitude")) {
  # Check if df is a dataframe
  if (!is.data.frame(df)) {
    stop("Input must be a data frame.")
  }

  # Check if coords is a character vector with exactly two elements
  if (!is.character(coords) || length(coords) != 2) {
    stop("Coordinate columns must be specified as a character vector with exactly two elements.")
  }

  # Ensure specified coordinate columns exist in the dataframe
  missing_coords <- setdiff(coords, colnames(df))
  if (length(missing_coords) > 0) {
    stop(paste("Coordinate column(s) not found in dataframe:", paste(missing_coords, collapse = ", ")))
  }

  # Convert dataframe to sf object and set CRS
  points_sf <- sf::st_as_sf(df, coords = coords, crs = sf::st_crs(polygon))

  # Ensure consistent handling of spatial attributes
  sf::st_agr(points_sf) <- "constant"

  # Filter points based on overlapping parameter
  if (overlapping) {
    filtered_points <- suppressMessages(points_sf[!sf::st_intersects(points_sf, polygon, sparse = FALSE), ])
  } else {
    filtered_points <- suppressMessages(points_sf[sf::st_intersects(points_sf, polygon, sparse = FALSE), ])
  }

  # Convert back to dataframe
  filtered_points_df <- cbind(sf::st_coordinates(filtered_points), as.data.frame(filtered_points))
  colnames(filtered_points_df)[1:2] <- coords
  filtered_points_df <- filtered_points_df[, !colnames(filtered_points_df) %in% c("geometry")]

  return(filtered_points_df)
}

#' Clean Coordinates of Presence/Absence Data
#'
#' This function cleans coordinates of presence/absence data by removing NA coordinates, rounding coordinates if specified, removing duplicated points, and removing points outside specified spatial polygon boundaries.
#'
#' @param df A dataframe object with rows representing points. Coordinates are in WGS84 (EPSG:4326) coordinate system.
#' @param study_area A spatial polygon in WGS84 (EPSG:4326) representing the boundaries within which coordinates should be kept.
#' @param overlapping Logical indicating whether points overlapping the polygon should be removed (TRUE) or kept (FALSE).
#' @param thinning_method Character; spatial thinning method to apply to occurrence data. Options are `c("None", "Distance", "Grid", "Precision")`. See `GeoThinneR` package for details.
#' @param thinning_value Numeric; value used for thinning depending on the selected method: distance in meters (`Distance`), grid resolution in degrees (`Grid`), or decimal precision (`Precision`).
#' @param coords Character vector specifying the column names for longitude and latitude.
#' @param by_timestamp If TRUE, clean coordinates taking into account different time periods defined in the column `timestamp`.
#' @param seed Optional; an integer seed for reproducibility of results.
#'
#' @return A cleaned data frame containing presence/absence data with valid coordinates.
#'
#' @details This function takes a data frame containing presence/absence data with longitude and latitude coordinates, a spatial polygon representing boundaries within which to keep points, and parameters for rounding coordinates and handling duplicated points. It returns a cleaned data frame with valid coordinates within the specified boundaries.
#'
#' @export
clean_coordinates <- function(df, study_area, overlapping = FALSE, thinning_method = NULL, thinning_value = NULL, coords = c("decimalLongitude", "decimalLatitude"), by_timestamp = TRUE, seed = NULL) {
  # Remove NA coordinates
  if (by_timestamp) {
    df <- df[complete.cases(df[, c(coords, "timestamp")]), ]
  } else {
    df <- df[complete.cases(df[, coords]), ]
  }

  # Remove duplicated points
  if (by_timestamp) {
    df <- remove_duplicate_points(df, coords = c(coords, "timestamp"))
  } else {
    df <- remove_duplicate_points(df, coords = coords)
  }

  # Remove points outside the study area
  if (!is.null(study_area)) {
    df <- remove_points_polygon(df, study_area, overlapping, coords)
  }

  # Occurrence spatial thinning
  if (!is.null(thinning_method) && thinning_method != "None"){
    if (thinning_method %in% c("distance", "grid", "precision") && is.null(thinning_value)) {
      warning("Missing `thinning_value` for thinning method: ", thinning_method, ". Skipping thinning.")
    } else if (!thinning_method %in% c("distance", "grid", "precision")) {
      warning(paste("Unknown thinning method:", thinning_method, ". Skipping thinning."))
    } else {
      group_col <- NULL
      if (by_timestamp) group_col <- "timestamp"

      message("Running thinning with method '", thinning_method, "' and thinning value ", thinning_value)
      if (thinning_method == "distance") {
        df <- GeoThinneR::thin_points(
          df, lon_col = coords[1], lat_col = coords[2], group_col = group_col,
          method = "distance", search_type = "local_kd_tree", thin_dist = thinning_value,
          trials = 1, all_trials = FALSE, seed = seed
        )
      } else if (thinning_method == "grid") {
        df <- GeoThinneR::thin_points(
          df, lon_col = coords[1], lat_col = coords[2], group_col = group_col,
          method = "grid", resolution = thinning_value, n = 1,
          trials = 1, all_trials = FALSE, seed = seed
        )
      } else if (thinning_method == "precision") {
        df <- GeoThinneR::thin_points(
          df, lon_col = coords[1], lat_col = coords[2], group_col = group_col,
          method = "precision", precision = thinning_value,
          trials = 1, all_trials = FALSE, seed = seed
        )
      }
      df <- GeoThinneR::largest(df)
    }
  }

  return(df)
}
