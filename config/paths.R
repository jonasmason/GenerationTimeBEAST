# -------------------------------------------------------------------
# Machine-specific repo/input roots
# -------------------------------------------------------------------

node <- Sys.info()[["nodename"]]

if (node %in% c("sce-bio-c04553", "sce-bio-c06973", "SCE-BIO-C07467")) {
  repo_dir <-"~/Work/BEAST/GenerationTimeBEAST"
}

if (node == "DESKTOP-EBKAL1K") {
  repo_dir <- "C:/Users/jonas/OneDrive - ETH Zurich/Desktop/msc_thesis/GenerationTimeBEAST_repo"
}

if (!exists("repo_dir")) {
  repo_dir <- Sys.getenv("GT_BEAST_REPO")
}

if (!nzchar(repo_dir) || !dir.exists(repo_dir)) {
  stop("Could not find repo_dir. Set GT_BEAST_REPO or add this machine to config/paths.R.")
}

repo_dir <- normalizePath(repo_dir, winslash = "/", mustWork = TRUE)


# -------------------------------------------------------------------
# Machine-specific large-output roots
# -------------------------------------------------------------------

if (node %in% c("sce-bio-c04553", "sce-bio-c06973", "SCE-BIO-C07467")) {
  output_root <- "~/Work/BEAST/GenerationTimeBEAST"
}

if (node == "DESKTOP-EBKAL1K") {
  output_root <- "D:/BEAST/GenerationTimeBEAST"
}

if (!exists("output_root") || !nzchar(output_root)) {
  stop("Could not find output_root. Set GT_BEAST_OUTPUT_ROOT or add this machine to config/paths.R.")
}

output_root <- normalizePath(output_root, winslash = "/", mustWork = FALSE)
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)


# -------------------------------------------------------------------
# Standard repo paths: small/static/project files
# -------------------------------------------------------------------

script_dir <- file.path(repo_dir, "scripts", "BEAST_GT")
config_dir <- file.path(repo_dir, "config")
lib_dir <- file.path(repo_dir, "lib")
src_dir <- file.path(repo_dir, "src")
repo_data_dir <- file.path(repo_dir, "data")
input_dir <- file.path(repo_data_dir, "input")


# -------------------------------------------------------------------
# Standard output paths: large/generated files
# -------------------------------------------------------------------

make_run_paths <- function(run_name) {
  if (missing(run_name) || !nzchar(run_name)) {
    stop("run_name must be a non-empty string.")
  }

  run_root <- file.path(output_root, run_name)

  paths <- list(
    run_name = run_name,
    run_root = run_root,

    sim_data_dir = file.path(run_root, "simulated_data"),
    xml_dir = file.path(run_root, "xml"),
    beast_output_dir = file.path(run_root, "beast_output"),
    debug_dir = file.path(run_root, "debug"),
    validation_dir = file.path(run_root, "validation"),
    tree_export_dir = file.path(run_root, "tree_exports"),
    plots_dir = file.path(run_root, "plots"),
    tables_dir = file.path(run_root, "tables")
  )

  dirs <- unlist(paths[names(paths) != "run_name"])
  invisible(lapply(dirs, dir.create, recursive = TRUE, showWarnings = FALSE))

  paths
}