#=========================================================#
# Read and Validate inputs  ----
#=========================================================#

#' Read and validate presences/absences CSV file
#'
#' This function reads and validates a CSV file containing presences and absences data for species occurrences.
#' It checks if the file has the expected columns and formats.
#'
#' @param file_path The file path to the CSV file.
#' @param file_name Optional. The name of the file. If not provided, the base name of the file path is used.
#' @param show_modal Optional. Logical. Whether to show a modal notification for warnings (use in Shiny). Default is FALSE.
#' @param timestamp_mapping Optional. A vector with the timestamp mapping of the environmental layers.
#' @param coords Optional. Character vector of length 2 specifying the names of the columns containing the longitude and latitude coordinates. Default is c("decimalLongitude", "decimalLatitude").
#' @param sep Optional. The field separator character. Default is tab-separated.
#' @param dec Optional. The decimal point character. Default is ".".
#'
#' @return A data frame with the validated data if the file has the expected columns and formats, NULL otherwise.
#' @keywords internal
#' @export
read_presences_absences_csv <- function(file_path, file_name = NULL, show_modal = FALSE, timestamp_mapping = NULL, coords = c("decimalLongitude", "decimalLatitude"), sep = "\t", dec = ".") {

  # Helper function to show notifications
  show_warning <- function(message) {
    warning(message)
    if (show_modal) showNotification(message, duration = 5, closeButton = TRUE, type = "error")
  }

  # Determine file name if not provided
  if (is.null(file_name)) {
    file_name <- sub("([^.]+)\\.[[:alnum:]]+$", "\\1", basename(file_path))
  }

  # Load the CSV file
  data <- tryCatch(
    read.csv2(file_path, sep = sep, dec = dec, header = TRUE, stringsAsFactors = FALSE),
    error = function(e) {
      show_warning(paste("Check", file_name, "file format, delimiters, or separators."))
      return(NULL)
    }
  )
  if (is.null(data)) return(NULL)

  # Check if the data has the required columns
  if (!all(coords %in% colnames(data))) {
    show_warning(paste("The", file_name, "file does not have the correct column names."))
    return(NULL)
  }

  # Process timestamp/year columns
  if ("timestamp" %in% colnames(data)) {
    data$timestamp_original <- data$timestamp
    if (!is.null(timestamp_mapping)) {
      data$timestamp <- match(data$timestamp, timestamp_mapping)
      if (any(is.na(data$timestamp))) {
        show_warning("Some timestamps in your occurrence file do not match any raster timestamps Please check your data.")
        return(NULL)
      }
    } else {
      data$timestamp <- data$timestamp - min(data$timestamp) + 1
    }
  } else if ("year" %in% colnames(data)) {
    data$timestamp_original <- data$year
    if (!is.null(timestamp_mapping)) {
      data$timestamp <- match(data$year, timestamp_mapping)
      if (any(is.na(data$timestamp))) {
        show_warning("Some timestamps in your occurrence file do not match any raster timestamps Please check your data.")
        return(NULL)
      }
    } else {
      data$timestamp <- data$year - min(data$year) + 1
    }
  } else {
    data$timestamp_original <- 1
    data$timestamp <- 1
    show_warning(paste0("'timestamp' column is not present in file ", file_name, ". Assuming all occurrences were observed at time 1."))
  }

  # Check if "pa" column is present
  if (!("pa" %in% colnames(data))) {
    data$pa <- 1
    show_warning(paste0("'pa' column is not present in file ", file_name, ". Assuming all rows are presences (1)."))
  }

  # Subset to required columns
  data <- data[, c(coords, "timestamp_original", "timestamp", "pa")]

  # Remove rows with NA values
  data <- data[complete.cases(data), ]
  if (nrow(data) == 0) {
    show_warning(paste("No valid records in", file_name, "."))
    return(NULL)
  }

  # Validate column formats
  if (!all(sapply(data[, coords], is.numeric))) {
    show_warning(paste("Coordinate columns are not numeric in", file_name, "file."))
    return(NULL)
  }
  if (!is.numeric(data$timestamp)) {
    show_warning(paste("Column 'timestamp' is not numeric in", file_name, "file."))
    return(NULL)
  }
  if (!all(data$pa %in% c(0, 1))) {
    show_warning(paste("Column 'pa' has values other than 0 and 1 in", file_name, "file."))
    return(NULL)
  }

  return(data)
}


#' Load Covariate Layers from ZIP Files
#'
#' This function loads covariate layers from a ZIP file, verifies their spatial characteristics, and returns them as a list of raster layers.
#'
#' @param file_path Path to the ZIP file containing covariate layers.
#' @param extend If TRUE it will take the largest extent, if FALSE the smallest.
#' @param first_layer If TRUE it will return only the layers from the first timestamp.
#' @param show_modal Optional. Logical. Whether to show a modal notification for warnings. Default is FALSE.
#'
#' @return A list containing raster layers for each covariate.
#' @keywords internal
#' @export
read_layers_zip <- function(file_path, extend = TRUE, first_layer = FALSE, show_modal = FALSE) {

  # Helper function to show warnings and notifications
  show_warning <- function(message) {
    warning(message)
    if (show_modal) showNotification(message, duration = 5, closeButton = TRUE, type = "error")
  }

  # Extract contents of the zip file
  tmpdir <- tempdir()
  zip_contents <- utils::unzip(file_path, exdir = tmpdir)
  # Clean out macOS hidden files/folders (.__, .DS_Store, and __MACOSX folder)
  zip_contents <- zip_contents[!grepl("(^|/)__MACOSX|/\\._|/\\.DS_Store$", zip_contents)]
  covariates <- unique(dirname(zip_contents))

  # Filter files to include only raster formats (e.g., .tif, .asc, .nc)
  raster_extensions <- c("tif", "tiff", "asc", "nc")

  # Verify if each covariate has the same number of files
  n_files <- sapply(covariates, function(x) length(list.files(x, pattern = paste0("\\.(", paste(raster_extensions, collapse = "|"), ")$"), full.names = TRUE)))
  if (length(unique(n_files)) != 1) {
    show_warning("Error: The environmental layers uploaded differ in number between the different variables")
    return(NULL)
  }

  # Load the first layer of each covariate to check CRS and resolution
  layers <- lapply(covariates, function(x) {
    files <- list.files(x, pattern = paste0("\\.(", paste(raster_extensions, collapse = "|"), ")$"), full.names = TRUE)
    tryCatch(terra::rast(files[n_files[1]]), error = function(e) NULL)
  })
  names(layers) <- sub("([^.]+)\\.[[:alnum:]]+$", "\\1", basename(covariates))
  if (any(sapply(layers, is.null))) {
    show_warning("Error: check format from files.")
    return(NULL)
  }

  # Check if all layers have the same CRS
  crs_list <- sapply(layers, function(x) terra::crs(x, proj = TRUE))
  if (any(sapply(crs_list, is.na))) {
    show_warning("There are rasters with undefined coordinate reference system (CRS).")
    return(NULL)
  }
  if (length(unique(crs_list)) != 1) {
    show_warning("There are layers with different coordinate reference system (CRS).")
  }

  # Project all layers to WGS84
  layers <- lapply(layers, function(x) terra::project(x, "epsg:4326"))

  # Check if all layers have the same resolution
  res_list <- lapply(layers, terra::res)
  is_consistent <- all(sapply(res_list[-1], function(x) isTRUE(all.equal(x, res_list[[1]], tolerance = 1e-7))))
  if (!is_consistent) {
    show_warning("The layers uploaded have different resolution. We will aggregate to coarsest resolution.")
    max_res <- apply(do.call(rbind, res_list), 2, max)
  }

  # Check if all layers have the same extent
  ext_list <- lapply(layers, function(x) as.vector(terra::ext(x)))
  is_ext_consistent <- all(sapply(ext_list[-1], function(x) isTRUE(all.equal(x, ext_list[[1]], tolerance = 1e-7))))
  if (!is_ext_consistent) {
    show_warning("There are layers with different extent. We will transform layers extent.")
  }

  # Load and process all layers
  layers <- lapply(covariates, function(cov_dir) {
    cov_name <- basename(cov_dir)
    raster_layers <- terra::rast(list.files(cov_dir, pattern = paste0("\\.(", paste(raster_extensions, collapse = "|"), ")$"), full.names = TRUE))
    # Project to WGS84 CRS
    raster_layers <- terra::project(raster_layers, "epsg:4326")

    # Aggregate layers to coarsest resolution using mean
    if (!is_consistent) {
      raster_layers <- tryCatch(
        terra::aggregate(raster_layers, fact = max_res / terra::res(raster_layers), fun = "mean"),
        error = function(e) NULL
      )
    }

    # Check and clean NA factor levels
    if (any(terra::is.factor(raster_layers))) {
      for (j in seq_len(terra::nlyr(raster_layers))) {
        if (terra::is.factor(raster_layers[[j]])) {
          levs <- terra::levels(raster_layers[[j]])[[1]]
          if (any(is.na(levs[, 2]))) {
            # Remove rows with NA label
            levs <- levs[!is.na(levs[, 2]), ]
            levels(raster_layers[[j]]) <- levs
          }
        }
      }
    }

    return(raster_layers)
  })
  names(layers) <- basename(covariates)

  # Check if any layer failed (returned NULL)
  if (any(sapply(layers, is.null))) {
    show_warning("Error: One or more covariates could not be processed. Check CRS, extent, or resolution issues or provide harmonized data.")
    return(NULL)
  }

  # Organize layers by timestamp
  layers_result <- list()
  # Get the number of years (assuming each element in layers has the same number of layers)
  num_years <- ifelse(first_layer, 1, terra::nlyr(layers[[1]]))
  for (i in seq_len(num_years)) {
    # Get the extents and origins of each raster for the current year
    ext_list <- lapply(layers, function(x) as.vector(terra::ext(x[[i]])))
    orig_list <- lapply(layers, function(x) as.vector(terra::origin(x[[i]])))

    # Check if extents are not identical
    is_ext_consistent <- all(sapply(ext_list[-1], function(x) isTRUE(all.equal(x, ext_list[[1]], tolerance = 1e-7))))
    if (!is_ext_consistent) {

      # Determine the modified extent
      # (smaller or larger depending if user has uploaded a polygon to the shiny app)
      ext_df <- do.call(rbind, ext_list)
      if (extend) {
        modified_ext <- c(floor(min(ext_df[, 1])), ceiling(max(ext_df[, 2])),
                          floor(min(ext_df[, 3])), ceiling(max(ext_df[, 4])))
      } else {
        modified_ext <- c(ceiling(max(ext_df[, 1])), floor(min(ext_df[, 2])),
                          ceiling(max(ext_df[, 3])), floor(min(ext_df[, 4])))
      }

      # Set the maximum extent to the geographical coordinates of the world.
      modified_ext <- c(max(modified_ext[1], -180), min(modified_ext[2], 180),
                        max(modified_ext[3], -90), min(modified_ext[4], 90))
      modified_ext <- terra::ext(modified_ext)

      # Check if origins are not identical
      if (length(unique(orig_list)) != 1) {
        # If different origin, adjust extent and resample rasters for the current year
        layers_i <- lapply(layers, function(x) {
          r <- terra::rast(modified_ext, res = terra::res(x[[i]]))
          terra::resample(x[[i]], r)
        })
      } else {
        # If same origin, crop and extend rasters for the current year to the modified extent
        layers_i <- lapply(layers, function(x) terra::crop(terra::extend(x[[i]], modified_ext), modified_ext))
      }
    } else {
      # If extents are identical, use the original rasters for the current year
      layers_i <- lapply(layers, function(x) x[[i]])
    }

    # Combine adjusted rasters into a single raster stack for the current year
    layers_result[[i]] <- terra::rast(layers_i)
  }

  return(layers_result)
}


#' Read and Validate Extent Polygon
#'
#' This function reads and validates a polygon file containing the extent. It checks if the file has the correct format and extracts the geometry.
#'
#' @param file_path Path to the polygon file containing the extent.
#' @param show_modal Optional. Logical. Whether to show a modal notification for warnings. Default is FALSE.
#' @return A spatial object representing the extent if the file is valid, NULL otherwise.
#'
#' @keywords internal
#' @export
read_extent_polygon <- function(file_path, show_modal = FALSE) {

  # Helper function to show warnings and notifications
  show_warning <- function(message) {
    warning(message)
    if (show_modal) showNotification(message, duration = 5, closeButton = TRUE, type = "error")
  }

  # Supported drivers for reading polygon files
  supported_drivers <- c("GPKG", "ESRI Shapefile", "KML", "GeoJSON")

  # Try to read the polygon file
  extent_polygon <- tryCatch(
    sf::st_read(file_path, drivers = supported_drivers, quiet = TRUE),
    error = function(e) NULL
  )

  if (is.null(extent_polygon)) {
    show_warning("Error: Unable to read the file. Please check the file format.")
    return(NULL)
  }

  # Check if the geometry is valid
  if (!all(sf::st_is_valid(extent_polygon))) {
    show_warning("Warning: The polygon geometry is not valid. We will try to fix it.")
    extent_polygon <- sf::st_make_valid(extent_polygon)
    #return(NULL)
  }

  # Check if the geometry type is a polygon or multipolygon
  if (!all(sf::st_geometry_type(extent_polygon) %in% c("POLYGON", "MULTIPOLYGON"))) {
    show_warning("Error: The file does not contain valid polygon geometries.")
    return(NULL)
  }

  return(sf::st_geometry(extent_polygon))
}


#' Validate Layers Zip
#'
#' This function validates a ZIP file containing environmental layers. It checks if the layers have the same number of files, CRS (Coordinate Reference System), and resolution.
#'
#' @param file_path Path to the ZIP file containing environmental layers.
#' @param timestamp_mapping Optional. A vector with the timestamp mapping of the environmental layers.
#' @param show_modal Optional. Logical. Whether to show a modal notification for warnings. Default is FALSE.
#'
#' @return TRUE if the layers pass validation criteria, FALSE otherwise.
#' @keywords internal
#' @export
validate_layers_zip <- function(file_path, timestamp_mapping = NULL, show_modal = FALSE) {
  # Helper function to show warnings and notifications
  show_warning <- function(message, type = "error") {
    warning(message)
    if (show_modal) showNotification(message, duration = 5, closeButton = TRUE, type = type)
  }

  # Extract contents of the ZIP file
  tmpdir <- tempdir()
  zip_contents <- utils::unzip(file_path, exdir = tmpdir)
  # Clean out macOS hidden files/folders (.__, .DS_Store, and __MACOSX folder)
  zip_contents <- zip_contents[!grepl("(^|/)__MACOSX|/\\._|/\\.DS_Store$", zip_contents)]
  # Get unique covariate directories
  covariate_dirs <- unique(dirname(zip_contents))

  # Filter files to include only raster formats (e.g., .tif, .asc, .nc)
  raster_extensions <- c("tif", "tiff", "asc", "nc")

  # Verify if each covariate has the same number of files
  n_files <- sapply(covariate_dirs, function(dir) {
    length(list.files(dir, pattern = paste0("\\.(", paste(raster_extensions, collapse = "|"), ")$"), full.names = TRUE))
  })
  if (length(unique(n_files)) != 1) {
    show_warning("Error: The environmental layers uploaded differ in number between the different variables.")
    return(FALSE)
  }

  if (!is.null(timestamp_mapping)){
    # Check if the number of files matches the length of the timestamp mapping
    if (any(length(timestamp_mapping) != n_files)) {
      show_warning("Error: The number of timestamps in the environmental layers does not match the length of the timestamp mapping.")
      return(FALSE)
    }
  }

  # Load one layer from each covariate to check CRS and resolution
  layers <- lapply(covariate_dirs, function(dir) {
    tryCatch({
      files <- list.files(dir, pattern = paste0("\\.(", paste(raster_extensions, collapse = "|"), ")$"), full.names = TRUE)
      terra::rast(files[1])
    }, error = function(e) NULL)
  })
  names(layers) <- basename(covariate_dirs)
  if (any(sapply(layers, is.null))) {
    show_warning("Error: Check the format of the files.")
    return(FALSE)
  }

  # Check if all layers have the same CRS
  crs_list <- sapply(layers, terra::crs)
  if (any(is.na(crs_list))) {
    show_warning("There are rasters with an undefined coordinate reference system (CRS).")
    return(FALSE)
  }
  if (length(unique(crs_list)) != 1) {
    show_warning("Warning: Layers have different coordinate reference systems (CRS). Projecting to WGS84.", type = "warning")
  }

  layers <- lapply(layers, function(layer) terra::project(layer, "epsg:4326"))

  # Check if all layers have the same resolution
  res_list <- lapply(layers, terra::res)
  is_consistent <- all(sapply(res_list[-1], function(x) isTRUE(all.equal(x, res_list[[1]], tolerance = 1e-7))))
  if (!is_consistent) {
    show_warning("The layers uploaded have different resolutions.")
    return(FALSE)
  }

  # Check if all layers have the same extent
  ext_list <- lapply(layers, terra::ext)
  is_ext_consistent <- all(sapply(ext_list[-1], function(x) isTRUE(all.equal(x, ext_list[[1]], tolerance = 1e-7))))
  if (!is_ext_consistent) {
    show_warning("Warning: There are layers with different extents. We will transform layers to the largest extent.", type = "warning")
  }

  # Check if factor levels for categorical variables are not NA
  categorical_vars <- sapply(layers, function(x) any(terra::is.factor(x)))
  if (any(categorical_vars)) {
    na_factor_levels <- sapply(layers[categorical_vars], function(x){
      any(is.na(terra::levels(x[[1]])[[1]][, 2]))
    })

    if (any(na_factor_levels)) {
      show_warning("Warning: There are layers with NA factor levels. We will remove those levels.", type = "warning")
    }
  }

  # If all checks passed, return TRUE
  return(TRUE)
}


#' Validate Fit and Projection Layers
#'
#' This function validates fit and projection layers by checking their covariates.
#'
#' @param fit_layers_path Path to the ZIP file containing fit layers.
#' @param proj_layers_path Path to the ZIP file containing projection layers.
#' @param show_modal Optional. Logical. Whether to show a modal notification for warnings. Default is FALSE.
#'
#' @return TRUE if the layers pass validation criteria, FALSE otherwise.
#' @keywords internal
#' @export
validate_fit_projection_layers <- function(fit_layers_path, proj_layers_path, show_modal = FALSE) {
  # Helper function to show warnings and notifications
  show_warning <- function(message) {
    warning(message)
    if (show_modal) showNotification(message, duration = 5, closeButton = TRUE, type = "error")
  }

  # Extract contents of the ZIP files
  tmpdir_fit <- tempdir()
  tmpdir_proj <- tempdir()
  fit_layers_content <- utils::unzip(fit_layers_path, exdir = tmpdir_fit)
  proj_contents <- utils::unzip(proj_layers_path, exdir = tmpdir_proj)
  # Clean out macOS hidden files/folders (.__, .DS_Store, and __MACOSX folder)
  fit_layers_content <- fit_layers_content[!grepl("(^|/)__MACOSX|/\\._|/\\.DS_Store$", fit_layers_content)]
  proj_contents <- proj_contents[!grepl("(^|/)__MACOSX|/\\._|/\\.DS_Store$", proj_contents)]


  # Filter files to include only raster formats (e.g., .tif, .asc, .nc)
  raster_extensions <- c("tif", "tiff", "asc", "nc")
  fit_covariates <- basename(unique(dirname(fit_layers_content[grepl(paste0("\\.(", paste(raster_extensions, collapse = "|"), ")$"), fit_layers_content)])))
  proj_covariates <- basename(unique(dirname(proj_contents[grepl(paste0("\\.(", paste(raster_extensions, collapse = "|"), ")$"), proj_contents)])))

  # Check if they have the same covariates
  if (!all(sort(fit_covariates) == sort(proj_covariates))) {
    show_warning("The projection layers uploaded have different covariates from the fit layers.")
    return(FALSE)
  }

  # If all checks passed, return TRUE
  return(TRUE)
}


#' Validate Match Between Presence/Absence Files and Fit Layers
#'
#' This function validates whether the time periods of the presence/absence data match the environmental layers.
#'
#' @param pa_data Data frame containing the presence/absence data with a 'timestamp' column.
#' @param fit_layers_path Path to the ZIP file containing fit layers.
#' @param show_modal Optional. Logical. Whether to show a modal notification for warnings. Default is FALSE.
#'
#' @return TRUE if the timestamps match the fit layers, FALSE otherwise.
#' @keywords internal
#' @export
validate_pa_fit_time <- function(pa_data, fit_layers_path, show_modal = FALSE) {
  # Helper function to show warnings and notifications
  show_warning <- function(message) {
    warning(message)
    if (show_modal) showNotification(message, duration = 5, closeButton = TRUE, type = "error")
  }

  # Calculate the length of the time period in the presence/absence data
  pa_time_period_length <- max(pa_data$timestamp) - min(pa_data$timestamp) + 1

  # Calculate the length of the time period in the fit layers
  zip_contents <- utils::unzip(fit_layers_path, list = TRUE)
  # Clean out macOS hidden files/folders (.__, .DS_Store, and __MACOSX folder)
  zip_contents <- zip_contents[!grepl("(^|/)__MACOSX|/\\._|/\\.DS_Store$", zip_contents$Name), ]

  raster_extensions <- c("tif", "tiff", "asc", "nc")
  files <- zip_contents$Name[grepl(paste0("\\.(", paste(raster_extensions, collapse = "|"), ")$"), zip_contents$Name)]
  covariates <- unique(dirname(files))
  fit_layers_time_period_length <- length(files) / length(covariates)

  # Validate the match between time periods
  if (pa_time_period_length > fit_layers_time_period_length) {
    show_warning("The time period in the P/A files exceeds that in the fit layers.")
    return(FALSE)
  }

  return(TRUE)
}




