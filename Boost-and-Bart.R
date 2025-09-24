library(tidyverse)
library(tidymodels)
library(vroom)
library(lubridate)
library(bonsai)
library(lightgbm)


bikeshare <- vroom("KaggleBikeShare/bike-sharing-demand/train.csv")
trainData <- bikeshare |> 
  mutate(log_count = log(count)) |>
  select(-casual, -registered, -count)
testData <- vroom("KaggleBikeShare/bike-sharing-demand/test.csv")

# -------------------------------------------------------------------------
## Recipe
bike_recipe <- recipe(log_count ~ ., data = trainData) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>% 
  step_mutate(weather = factor(weather),
              season = factor(season),
              holiday = factor(holiday),
              workingday = factor(workingday)) %>%
  step_date(datetime, features="dow") %>%
  step_time(datetime, features="hour") %>% 
  step_rm(datetime) %>%
  step_mutate(
    hour_sin = sin(2 * pi * datetime_hour / 24),
    hour_cos = cos(2 * pi * datetime_hour / 24)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())
prepped_recipe <- prep(bike_recipe)
bake(prepped_recipe, new_data=testData)

# -------------------------------------------------------------------------
# BOOST
# Boosting updates your predictions sequentially with each successive update 
# being a better prediction than the previous. 
# Gradient is a general form of boosting for any response type (the previous 
# algorithm was for quantitative response only but the intuition stays the 
# same in gradient boosting).
# The “gradient” term signifies that, in non-quantitative settings, predictions 
# are updated sequentially by following the gradient to minimize the loss 
# function (gradient descent optimization).


# model
boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
  set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("regression")

# workflow
boost_bike_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(boost_model)

# tuning grid
boost_grid <- grid_regular(
  tree_depth(),
  trees(),
  learn_rate(),
  levels = 5)

# Set up K-fold CV
cv_folds <- vfold_cv(trainData, v = 10, repeats = 1)

boost_tune <- boost_bike_wf %>%
  tune_grid(resamples = cv_folds,
            grid = boost_grid,
            metrics = metric_set(rmse, mae, rsq))

# 6. Select best model
best_boost <- boost_tune %>%
  select_best(metric = "rmse")

# Finalize workflow and predict
final_boost_wf <- boost_bike_wf %>%
  finalize_workflow(best_boost) %>%
  fit(data = trainData)

boost_preds <- predict(final_boost_wf, new_data = testData) %>%
  mutate(.pred = exp(.pred))

# -------------------------------------------------------------------------
# Format the Predictions for Submission to Kaggle
kaggle_submission <- boost_preds |>
  bind_cols(testData) |>
  select(datetime, .pred) |>
  rename(count=.pred) |>
  mutate(count=pmax(0, count)) |>
  mutate(datetime=as.character(format(datetime)))

## Write out the file
vroom_write(x=kaggle_submission, 
            file="./KaggleBikeShare/k-subs/boost.csv", 
            delim=",")


# -------------------------------------------------------------------------
# BART
# BART (Bayesian Additive Regression Trees) is similar to boosting but fits and 
# “refits” all trees relative to each other (rather a single fit based on the 
# previous trees). An oversimplification is that BART is a looped boosting algorithm.

# model
bart_model <- bart(trees=tune()) %>% # BART figures out depth and learn_rate
  set_engine("dbarts") %>% # might need to install
  set_mode("regression")

# workflow
bart_bike_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(bart_model)

# tuning grid
bart_grid <- grid_regular(
  trees(),
  levels = 5)

# Set up K-fold CV
cv_folds <- vfold_cv(trainData, v = 10, repeats = 1)

bart_tune <- bart_bike_wf %>%
  tune_grid(resamples = cv_folds,
            grid = bart_grid,
            metrics = metric_set(rmse, mae, rsq))

# Select best model
best_bart <- bart_tune %>%
  select_best(metric = "rmse")

# Finalize workflow and predict
final_bart_wf <- bart_bike_wf %>%
  finalize_workflow(best_bart) %>%
  fit(data = trainData)

bart_preds <- predict(final_bart_wf, new_data = testData) %>%
  mutate(.pred = exp(.pred))


# -------------------------------------------------------------------------
# Format the Predictions for Submission to Kaggle
kaggle_submission <- bart_preds |>
  bind_cols(testData) |>
  select(datetime, .pred) |>
  rename(count=.pred) |>
  mutate(count=pmax(0, count)) |>
  mutate(datetime=as.character(format(datetime)))

## Write out the file
vroom_write(x=kaggle_submission, 
            file="./KaggleBikeShare/k-subs/bart.csv", 
            delim=",")
