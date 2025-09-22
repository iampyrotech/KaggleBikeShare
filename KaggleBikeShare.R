# ======================
# Libraries
# ======================
library(tidyverse)
library(vroom)
library(readr)    
library(GGally)
library(patchwork)
library(tidymodels)
library(recipes)
library(lubridate)

# ======================
# Read in data (parse datetime correctly at import)
# ======================
train_raw <- vroom(
  "data/train.csv",
  col_types = cols(
    datetime = col_datetime(format = "%Y-%m-%d %H:%M:%S"),
    .default = col_guess()
  )
)

test_raw <- vroom(
  "data/test.csv",
  col_types = cols(
    datetime = col_datetime(format = "%Y-%m-%d %H:%M:%S"),
    .default = col_guess()
  )
)

head(train_raw)
summary(train_raw)

# ======================
# Preliminary EDA: Homework 1
# ======================
train_raw %>%
  select_if(is.numeric) %>%
  gather() %>%
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = 'free') +
  geom_histogram(bins = 30, fill = 'steelblue', color = 'white')

train_raw %>%
  select_if(is.numeric) %>%
  gather() %>%
  ggplot(aes(x = key, y = value)) +
  geom_boxplot(fill = 'tomato', alpha = 0.6) +
  facet_wrap(~ key, scales = 'free')

train_raw %>% 
  select_if(is.numeric) %>% 
  cor(use = "pairwise.complete.obs") %>%
  round(2)

train_raw %>% 
  select_if(is.numeric) %>% 
  ggcorr(label = TRUE)

# Bar plot of average count by weather
p1 <- ggplot(train_raw, aes(x = factor(weather), y = count)) +
  stat_summary(fun = "mean", geom = "bar", fill = "steelblue") +
  labs(x = "Weather", y = "Avg Count", title = "Average Rentals by Weather")

# Count by season
p2 <- ggplot(train_raw, aes(x = factor(season), y = count)) +
  stat_summary(fun = "mean", geom = "bar", fill = "darkseagreen") +
  labs(x = "Season", y = "Avg Count", title = "Average Rentals by Season")

# Distribution of temp
p3 <- ggplot(train_raw, aes(x = temp)) +
  geom_histogram(bins = 30, fill = "orange", color = "white") +
  labs(x = "Temperature", y = "Frequency", title = "Temperature Distribution")

# Scatter plot of humidity vs count
p4 <- ggplot(train_raw, aes(x = humidity, y = count)) +
  geom_point(alpha = 0.4, color = "tomato") +
  geom_smooth(method = "lm", se = FALSE, color = "black") +
  labs(x = "Humidity", y = "Count", title = "Humidity vs Rentals")

# Arrange in grid
plot <- (p1 | p2) / (p3 | p4)
ggsave("data/bike_share_eda.pdf", plot = plot, width = 10, height = 8)

# ======================
# Linear Regression: Homework 2
# ======================
train_lm <- train_raw %>%
  mutate(log_count = log1p(count))

x <- intersect(
  setdiff(names(train_lm), c("log_count", "count", "datetime", "casual", "registered")),
  names(test_raw)
)

lm_model <- lm(reformulate(x, response = "log_count"), data = train_lm)

log_preds   <- predict(lm_model, newdata = test_raw[, x, drop = FALSE])
count_preds <- pmax(0, round(expm1(log_preds)))

submission_lm <- tibble(
  datetime = format(as.POSIXct(test_raw$datetime), "%Y-%m-%d %H:%M:%S"),
  count    = as.integer(count_preds)
)

vroom::vroom_write(submission_lm, "submission_lm.csv", delim = ",")

# ======================
# Workflows/Recipe: Homework 4
# ======================
train <- train_raw %>%
  mutate(
    log_count = log1p(count)
  ) %>%
  select(-casual, -registered, -count)   # drop helper cols for modeling

test <- test_raw  # already parsed

mybike_recipe <- recipe(log_count ~ ., data = train) %>%
  step_mutate(
    weather = factor(ifelse(weather == 4, 3, weather)),
    season  = factor(season)
  ) %>%
  step_time(datetime, features = "hour", keep_original_cols = FALSE) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

mybike_prep <- prep(mybike_recipe, training = train)
baked_data  <- bake(mybike_prep, new_data = NULL)
head(baked_data, 5)


# ======================
# Penalized Regression: Homework 5 and 6
# ======================

library(tidymodels)

## Penalized regression model
preg_model <- linear_reg(penalty = tune(), mixture = tune()) %>% 
  set_engine("glmnet")

## Set Workflow
preg_wf <- workflow() %>% 
  add_recipe(mybike_recipe) %>% 
  add_model(preg_model)

## Grid of values to tune over
L <- 12
grid_of_tuning_params <- grid_regular(penalty(), mixture(), levels = L)

## Split data for CV
K <- 5
folds <- vfold_cv(train, v = K)

## Run the CV
CV_results <- preg_wf %>%
  tune_grid(resamples = folds,
            grid = grid_of_tuning_params,
            metrics = metric_set(rmse, mae))

## Find Best Tuning Parameters
bestTune <- CV_results %>% select_best(metric = "rmse")

## Finalize the Workflow & fit it
final_wf <- preg_wf %>% 
  finalize_workflow(bestTune) %>% 
  fit(data = train)

## Predict on test (model was on log_count)
preds_log <- predict(final_wf, new_data = test) %>% pull(.pred)
counts    <- pmax(0, round(expm1(preds_log)))

## Save Kaggle submission
dir.create("submissions", showWarnings = FALSE)
submission <- tibble(
  datetime = format(as.POSIXct(test$datetime), "%Y-%m-%d %H:%M:%S"),
  count    = as.integer(counts)
)

pen_tag <- gsub("\\.", "p", sprintf("%.6f", bestTune$penalty))
mix_tag <- gsub("\\.", "p", sprintf("%.2f",  bestTune$mixture))
vroom::vroom_write(submission, "submissions/submission_glmnet.csv", delim = ",")

message("Best params -> penalty: ", signif(bestTune$penalty, 4),
        " | mixture: ", signif(bestTune$mixture, 4))


# ======================
# Regression Trees: Homework 7
# ======================
# Do it on the log of count

library(rpart)

my_mod <- decision_tree(
  tree_depth      = tune(),
  cost_complexity = tune(),
  min_n           = tune()
) %>%
  set_engine("rpart") %>%
  set_mode("regression")

# ---- Create a workflow with model & recipe (log_count outcome) ----
tree_recipe <- recipe(log_count ~ ., data = train) %>%
  step_mutate(
    weather = factor(ifelse(weather == 4, 3, weather)),
    season  = factor(season)
  ) %>%
  step_time(datetime, features = "hour", keep_original_cols = FALSE)

tree_wf <- workflow() %>%
  add_recipe(tree_recipe) %>%
  add_model(my_mod)

# ---- Set up grid of tuning values ----
tree_grid <- grid_regular(
  cost_complexity(),          # default log10 range
  tree_depth(range = c(2L, 12L)),
  min_n(range = c(2L, 30L)),
  levels = c(5, 6, 5)         # small, simple grid
)

# ---- Set up K-fold CV ----
set.seed(123)
folds <- vfold_cv(train, v = 5, strata = log_count)

# ---- Find best tuning parameters ----
tree_res  <- tune_grid(tree_wf, resamples = folds, grid = tree_grid,
                       metrics = metric_set(rmse))
best_tree <- select_best(tree_res, metric = "rmse")

# ---- Finalize workflow and predict ----
final_tree_wf  <- finalize_workflow(tree_wf, best_tree)
final_tree_fit <- fit(final_tree_wf, data = train)

preds_log <- predict(final_tree_fit, new_data = test) %>% pull(.pred)
counts    <- pmax(0, round(expm1(preds_log)))

submission_tree <- tibble(
  datetime = format(as.POSIXct(test$datetime), "%Y-%m-%d %H:%M:%S"),
  count    = as.integer(counts)
)

dir.create("submissions", showWarnings = FALSE)
vroom::vroom_write(submission_tree, "submissions/submission_tree.csv", delim = ",")

best_tree

