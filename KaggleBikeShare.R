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
# Penalized Regression: Homework 5
# ======================

# grid of penalty/mixture pairs
param_grid <- tibble::tribble(
  ~penalty, ~mixture,
  0.001,    0.00,   # ridge
  0.010,    0.25,
  0.100,    0.50,   # elastic net
  0.300,    0.75,
  1.000,    1.00    # lasso
)

wf_base <- workflow() %>% add_recipe(mybike_recipe)
dir.create("submissions", showWarnings = FALSE)

for (i in seq_len(nrow(param_grid))) {
  pen <- param_grid$penalty[i]
  mix <- param_grid$mixture[i]
  
  preg_model <- linear_reg(penalty = pen, mixture = mix) %>%
    set_engine("glmnet")
  preg_wf <- wf_base %>% add_model(preg_model)
  
  fit_obj   <- fit(preg_wf, data = train)
  preds_log <- predict(fit_obj, new_data = test)$.pred
  counts    <- pmax(0, round(expm1(preds_log)))
  
  submission <- tibble(
    datetime = format(as.POSIXct(test$datetime), "%Y-%m-%d %H:%M:%S"),
    count    = as.integer(counts)
  )
  
  # build filename tags
  pen_tag <- gsub("\\.", "p", sprintf("%.3f", pen))
  mix_tag <- gsub("\\.", "p", sprintf("%.2f", mix))
  fn <- sprintf("submissions/submission_pen%s_mix%s.csv", pen_tag, mix_tag)
  
  vroom::vroom_write(submission, fn, delim = ",")
  message("Wrote: ", fn)
}

