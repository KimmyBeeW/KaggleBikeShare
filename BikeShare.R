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
## Feature Engineering
bike_recipe <- recipe(log_count ~ ., data = trainData) %>% # Set model formula and dataset
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>% 
  step_mutate(weather = factor(weather),
              season = factor(season),
              holiday = factor(holiday),
              workingday = factor(workingday)) %>%
  step_date(datetime, features="dow") %>%
  step_time(datetime, features="hour") %>% 
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>% #create dummy variables
  # step_zv(all_predictors()) %>% #removes zero-variance predictors
  # step_corr(all_predictors(), threshold=0.5) %>% # removes > than .5 corr
  step_normalize(all_numeric_predictors()) # all predictors are numeric and on the same scale.
prepped_recipe <- prep(bike_recipe) # Sets up the preprocessing using myDataSet
bake(prepped_recipe, new_data=testData)

# -------------------------------------------------------------------------
# REGRESSION TREE

# 1. Define the tree model
tree_mod <- decision_tree(cost_complexity = tune(),
                          tree_depth = tune(),
                          min_n = tune()) %>%
  set_engine("rpart") %>%
  set_mode("regression")

# 2. Combine model with recipe into workflow
bike_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(tree_mod)

# 3. Set up tuning grid
tree_grid <- grid_regular(
  tree_depth(),
  cost_complexity(),
  min_n(),
  levels = 5)

# 4. Set up cross-validation
cv_folds <- vfold_cv(trainData, v = 10, repeats = 1)

# 5. Tune parameters
tree_tune <- bike_wf %>%
  tune_grid(resamples = cv_folds,
            grid = tree_grid,
            # control = control_grid(save_pred = TRUE)
            metrics = metric_set(rmse, mae, rsq)
            )

# 6. Select best model
best_tree <- tree_tune %>%
  select_best(metric = "rmse")

# 7. Finalize workflow and fit
final_tree_wf <- bike_wf %>%
  finalize_workflow(best_tree) %>%
  fit(data = trainData)

# 8. Predict on test data
tree_preds <- predict(final_tree_wf, new_data = testData) %>%
  mutate(.pred = exp(.pred))


# -------------------------------------------------------------------------
# Format the Predictions for Submission to Kaggle
kaggle_submission <- tree_preds |>
  bind_cols(testData) |>
  select(datetime, .pred) |>
  rename(count=.pred) |>
  mutate(count=pmax(0, count)) |>
  mutate(datetime=as.character(format(datetime)))

## Write out the file
vroom_write(x=kaggle_submission, 
            file="./KaggleBikeShare/tree2.csv", 
            delim=",")
