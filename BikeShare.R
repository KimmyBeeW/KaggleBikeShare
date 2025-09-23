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
tree_mod <- decision_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()) %>%
  set_engine("rpart") %>%
  set_mode("regression")

# 2. Combine model with recipe into workflow
tree_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(tree_mod)

# 3. Set up tuning grid
tree_grid <- grid_regular(
  tree_depth(range = c(1, 10)),
  cost_complexity(range = c(-3, -1)),
  min_n(range = c(2, 10)),
  levels = c(5, 5, 5)
)

# 4. Set up cross-validation
set.seed(123)
cv_folds <- vfold_cv(trainData, v = 10)

# 5. Tune parameters
tree_tune <- tune_grid(
  tree_wf,
  resamples = cv_folds,
  grid = tree_grid,
  control = control_grid(save_pred = TRUE)
)

# 6. Select best model
best_tree <- select_best(tree_tune, metric = "rmse")

# 7. Finalize workflow and fit
final_tree_wf <- finalize_workflow(tree_wf, best_tree)
final_tree_fit <- fit(final_tree_wf, data = trainData)

# 8. Predict on test data
tree_preds <- predict(final_tree_fit, new_data = testData) %>%
  mutate(.pred = exp(.pred))  # if using log(count) target


# -------------------------------------------------------------------------
# Format the Predictions for Submission to Kaggle
kaggle_submission <- tree_preds |>
  bind_cols(testData) |> #Bind predictions with test data
  select(datetime, .pred) |> #Just keep datetime and prediction variables
  rename(count=.pred) |> #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) |> #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file
vroom_write(x=kaggle_submission, 
            file="./KaggleBikeShare/tree.csv", 
            delim=",")
