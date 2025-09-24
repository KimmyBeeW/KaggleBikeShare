library(tidyverse)
library(tidymodels)
library(vroom)
library(lubridate)


bikeshare <- vroom("KaggleBikeShare/bike-sharing-demand/train.csv")
trainData <- bikeshare |> 
  mutate(log_count = log(count)) |>
  select(-casual, -registered, -count)
testData <- vroom("KaggleBikeShare/bike-sharing-demand/test.csv")

# -------------------------------------------------------------------------
## Recipe
bike_recipe <- recipe(log_count ~ ., data = trainData) %>% # Set model formula and dataset
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
  step_dummy(all_nominal_predictors()) %>% #create dummy variables
  step_normalize(all_numeric_predictors()) # all predictors are numeric and on the same scale.
prepped_recipe <- prep(bike_recipe) # Sets up the preprocessing using myDataSet
bake(prepped_recipe, new_data=testData)

# -------------------------------------------------------------------------
# Random Forest

# model
forest_mod <- rand_forest(mtry = tune(),
                          min_n = tune(),
                          trees = 1000) %>%
  set_engine("ranger") %>%
  set_mode("regression")

# workflow
bike_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(forest_mod)

# tuning grid
maxNumXs <- prepped_recipe %>%
  bake(new_data = NULL) %>%
  select(-log_count) %>%
  ncol()
maxNumXs # 20

forest_grid <- grid_regular(
  mtry(range = c(1,maxNumXs)),
  min_n(),
  levels = 5)

# Set up K-fold CV
cv_folds <- vfold_cv(trainData, v = 10, repeats = 1)

forest_tune <- bike_wf %>%
  tune_grid(resamples = cv_folds,
            grid = forest_grid,
            metrics = metric_set(rmse, mae, rsq))

# 6. Select best model
best_forest <- forest_tune %>%
  select_best(metric = "rmse")

# Finalize workflow and predict
final_forest_wf <- bike_wf %>%
  finalize_workflow(best_forest) %>%
  fit(data = trainData)

forest_preds <- predict(final_forest_wf, new_data = testData) %>%
  mutate(.pred = exp(.pred))


# -------------------------------------------------------------------------
# Format the Predictions for Submission to Kaggle
kaggle_submission <- forest_preds |>
  bind_cols(testData) |>
  select(datetime, .pred) |>
  rename(count=.pred) |>
  mutate(count=pmax(0, count)) |>
  mutate(datetime=as.character(format(datetime)))

## Write out the file
vroom_write(x=kaggle_submission, 
            file="./KaggleBikeShare/k-subs/forest1.csv", 
            delim=",")
