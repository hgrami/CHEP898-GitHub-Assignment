# activity_rf.R
# Activity classification on accelerometer data.
# Fits a baseline Random Forest, a tuned Random Forest, and a multinomial
# logistic regression for comparison. Times each stage and saves artifacts.
# Dataset: /project/chep_fuller/accel_data_no_id.csv (~22.5M rows, 2 GB).

options(scipen = 999)
set.seed(10)

t_start <- Sys.time()
cat("Job started at:", format(t_start), "\n\n")

# ---- Packages -----------------------------------------------------------
needed <- c("tidyverse", "tidymodels", "ranger", "yardstick", "nnet", "glmnet")
missing <- setdiff(needed, rownames(installed.packages()))
if (length(missing)) {
  install.packages(missing, repos = "https://muug.ca/mirror/cran/")
}

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidymodels)
  library(ranger)
  library(nnet)
})

cores_env <- Sys.getenv("SLURM_CPUS_PER_TASK")
n_cores <- if (nzchar(cores_env)) as.integer(cores_env) else parallel::detectCores()
cat("Using", n_cores, "cores\n")

# ---- Load ---------------------------------------------------------------
data_path <- "/project/chep_fuller/accel_data_no_id.csv"

t_read <- Sys.time()
raw <- read_csv(data_path, show_col_types = FALSE)
read_min <- round(as.numeric(Sys.time() - t_read, units = "mins"), 2)
cat("Read", nrow(raw), "rows in", read_min, "min\n")

# ---- Clean --------------------------------------------------------------
keep_vars <- c("x_axis", "y_axis", "wear_location",
               "VO2", "VO2kg", "VCO2", "BF", "MET", "EE",
               "activity")

d <- raw %>%
  select(any_of(keep_vars)) %>%
  mutate(
    wear_location = as.factor(wear_location),
    activity      = as.factor(activity)
  ) %>%
  drop_na()

cat("Complete-case rows:", nrow(d), "\n")
cat("Activity class counts:\n")
print(table(d$activity))

rm(raw); gc()

# ---- Stratified subsample ----------------------------------------------
subsample_n <- 500000L

if (nrow(d) > subsample_n) {
  set.seed(10)
  d <- d %>%
    group_by(activity) %>%
    slice_sample(prop = subsample_n / nrow(d), replace = FALSE) %>%
    ungroup()
  cat("After stratified subsample:", nrow(d), "rows\n")
  print(table(d$activity))
}

# ---- Train / test split -------------------------------------------------
set.seed(10)
split     <- initial_split(d, prop = 0.7, strata = activity)
train_dat <- training(split)
test_dat  <- testing(split)

cat("Train:", nrow(train_dat), "| Test:", nrow(test_dat), "\n")

p <- ncol(train_dat) - 1L
classes <- levels(train_dat$activity)

# ---- Helper: multi-class metrics ---------------------------------------
eval_classifier <- function(truth, pred_class, prob_matrix = NULL, label) {
  pred_class <- factor(pred_class, levels = classes)
  df <- tibble(truth = truth, pred = pred_class)

  base <- bind_rows(
    df %>% accuracy(truth, pred),
    df %>% sens(truth, pred, estimator = "macro"),
    df %>% spec(truth, pred, estimator = "macro"),
    df %>% f_meas(truth, pred, estimator = "macro")
  ) %>% mutate(Model = label)

  if (!is.null(prob_matrix)) {
    prob_df <- as_tibble(prob_matrix)
    names(prob_df) <- paste0(".pred_", colnames(prob_matrix))
    prob_cols <- names(prob_df)
    auc_df <- tibble(truth = truth) %>%
      bind_cols(prob_df) %>%
      roc_auc(truth, !!!syms(prob_cols), estimator = "macro_weighted") %>%
      mutate(Model = label)
    base <- bind_rows(base, auc_df)
  }
  base
}

# ---- Baseline Random Forest --------------------------------------------
cat("\n==== Baseline Random Forest ====\n")
t_fit <- Sys.time()

rf_baseline <- ranger(
  activity ~ .,
  data        = train_dat,
  num.trees   = 500,
  mtry        = floor(sqrt(p)),
  num.threads = n_cores,
  importance  = "impurity",
  probability = FALSE,
  classification = TRUE,
  verbose     = FALSE
)

rf_baseline_prob <- ranger(
  activity ~ .,
  data        = train_dat,
  num.trees   = 500,
  mtry        = floor(sqrt(p)),
  num.threads = n_cores,
  importance  = "none",
  probability = TRUE,
  verbose     = FALSE
)

rf_baseline_min <- round(as.numeric(Sys.time() - t_fit, units = "mins"), 2)
cat("Baseline RF fit (class + prob) in", rf_baseline_min, "min\n")

rf_baseline_pred  <- predict(rf_baseline,      data = test_dat)$predictions
rf_baseline_probm <- predict(rf_baseline_prob, data = test_dat)$predictions

rf_baseline_metrics <- eval_classifier(
  test_dat$activity, rf_baseline_pred, rf_baseline_probm, "RF Baseline"
)
print(rf_baseline_metrics)

# ---- Tuned Random Forest -----------------------------------------------
cat("\n==== Tuned Random Forest ====\n")

mtry_grid   <- c(floor(sqrt(p)), max(2, floor(p / 3)), max(2, floor(p / 2)))
min_n_grid  <- c(1, 5)
tune_grid   <- expand.grid(mtry = mtry_grid, min_node_size = min_n_grid)

cat("Tuning grid:\n"); print(tune_grid)

set.seed(10)
holdout_idx <- sample(seq_len(nrow(train_dat)), size = floor(0.2 * nrow(train_dat)))
tune_train  <- train_dat[-holdout_idx, ]
tune_val    <- train_dat[ holdout_idx, ]

t_tune <- Sys.time()
tune_results <- map_dfr(seq_len(nrow(tune_grid)), function(i) {
  params <- tune_grid[i, ]
  cat("  trying mtry =", params$mtry,
      ", min_node_size =", params$min_node_size, "... ")
  t0 <- Sys.time()
  m <- ranger(
    activity ~ .,
    data           = tune_train,
    num.trees      = 300,
    mtry           = params$mtry,
    min.node.size  = params$min_node_size,
    num.threads    = n_cores,
    probability    = FALSE,
    classification = TRUE,
    verbose        = FALSE
  )
  pred <- predict(m, data = tune_val)$predictions
  acc <- mean(pred == tune_val$activity)
  elapsed <- round(as.numeric(Sys.time() - t0, units = "mins"), 2)
  cat("val_acc =", round(acc, 4), "(", elapsed, "min )\n")
  tibble(mtry = params$mtry,
         min_node_size = params$min_node_size,
         val_accuracy = acc,
         minutes = elapsed)
})

tune_min <- round(as.numeric(Sys.time() - t_tune, units = "mins"), 2)
cat("Tuning total:", tune_min, "min\n")

best <- tune_results %>% arrange(desc(val_accuracy)) %>% slice(1)
cat("Best: mtry =", best$mtry, ", min_node_size =", best$min_node_size, "\n")

cat("Refitting tuned RF on full training set...\n")
t_refit <- Sys.time()

rf_tuned <- ranger(
  activity ~ .,
  data           = train_dat,
  num.trees      = 500,
  mtry           = best$mtry,
  min.node.size  = best$min_node_size,
  num.threads    = n_cores,
  importance     = "impurity",
  probability    = FALSE,
  classification = TRUE,
  verbose        = FALSE
)

rf_tuned_prob <- ranger(
  activity ~ .,
  data           = train_dat,
  num.trees      = 500,
  mtry           = best$mtry,
  min.node.size  = best$min_node_size,
  num.threads    = n_cores,
  importance     = "none",
  probability    = TRUE,
  verbose        = FALSE
)

rf_tuned_min <- round(as.numeric(Sys.time() - t_refit, units = "mins"), 2)
cat("Tuned RF refit in", rf_tuned_min, "min\n")

rf_tuned_pred  <- predict(rf_tuned,      data = test_dat)$predictions
rf_tuned_probm <- predict(rf_tuned_prob, data = test_dat)$predictions

rf_tuned_metrics <- eval_classifier(
  test_dat$activity, rf_tuned_pred, rf_tuned_probm, "RF Tuned"
)
print(rf_tuned_metrics)

# ---- Multinomial Logistic Regression -----------------------------------
cat("\n==== Multinomial Logistic Regression (comparison) ====\n")
t_mlr <- Sys.time()

mlr_fit <- multinom(activity ~ ., data = train_dat, MaxNWts = 5000, trace = FALSE)

mlr_min <- round(as.numeric(Sys.time() - t_mlr, units = "mins"), 2)
cat("Multinomial regression fit in", mlr_min, "min\n")

mlr_pred  <- predict(mlr_fit, newdata = test_dat)
mlr_probm <- predict(mlr_fit, newdata = test_dat, type = "probs")
if (is.null(dim(mlr_probm))) {
  mlr_probm <- cbind(1 - mlr_probm, mlr_probm)
  colnames(mlr_probm) <- classes
}

mlr_metrics <- eval_classifier(
  test_dat$activity, mlr_pred, mlr_probm, "Multinomial Logistic"
)
print(mlr_metrics)

# ---- Comparison --------------------------------------------------------
cat("\n==== Model Comparison ====\n")
all_metrics <- bind_rows(rf_baseline_metrics, rf_tuned_metrics, mlr_metrics)
print(all_metrics)

# ---- Confusion matrix for tuned RF -------------------------------------
cat("\n=== Confusion matrix: Tuned RF ===\n")
rf_tuned_cm <- conf_mat(
  tibble(truth = test_dat$activity,
         pred  = factor(rf_tuned_pred, levels = classes)),
  truth = truth, estimate = pred
)
print(rf_tuned_cm)

# ---- Feature importance (tuned RF) -------------------------------------
cat("\n=== Variable importance (tuned RF, impurity) ===\n")
imp <- importance(rf_tuned)
imp_df <- tibble(variable = names(imp), importance = as.numeric(imp)) %>%
  arrange(desc(importance))
print(imp_df)

# ---- Save artifacts ----------------------------------------------------
saveRDS(list(
  rf_baseline_metrics = rf_baseline_metrics,
  rf_tuned_metrics    = rf_tuned_metrics,
  mlr_metrics         = mlr_metrics,
  all_metrics         = all_metrics,
  rf_tuned_cm         = rf_tuned_cm,
  importance          = imp_df,
  tune_results        = tune_results,
  best_params         = best,
  timings = list(
    read       = read_min,
    rf_base    = rf_baseline_min,
    rf_tune    = tune_min,
    rf_tuned   = rf_tuned_min,
    mlr        = mlr_min,
    cores      = n_cores,
    subsample  = subsample_n
  )
), "activity_rf_results.rds")

t_end <- Sys.time()
cat("\nJob finished at:", format(t_end), "\n")
cat("Total wall time:",
    round(as.numeric(t_end - t_start, units = "mins"), 2), "min\n")
