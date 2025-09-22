library(tidyverse)
library(tidymodels)
library(vroom)
library(lubridate)


bikeshare <- vroom("KaggleBikeShare/bike-sharing-demand/train.csv")
# -------------------------------------------------------------------------
## Clean
# remove the casual and registered variables and change count to log(count) 
# in the training data only.
CleanData <- bikeshare |> 
  mutate(log_count = log(count)) |>
  select(-casual, -registered, -count)

trainData <- CleanData
testData <- vroom("KaggleBikeShare/bike-sharing-demand/test.csv")

# -------------------------------------------------------------------------
## Feature Engineering
# recipe that:
#      1. recodes weather “4” to a “3” then makes it a factor
#      2. extracts the hour variable from the timestamp
#      3. makes season a factor
#      4. does 1 other step of your choice
bike_recipe <- recipe(log_count ~ ., data = trainData) %>% # Set model formula and dataset
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>% # 1
  step_mutate(weather = factor(weather), # 1
              season = factor(season), # 3
              holiday = factor(holiday),
              workingday = factor(workingday)) %>%
  step_date(datetime, features="dow") %>% # gets day of week
  step_time(datetime, features="hour") %>% # 2
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>% #create dummy variables
  step_zv(all_predictors()) %>% #removes zero-variance predictors
  step_corr(all_predictors(), threshold=0.5) %>% # removes > than .5 corr
  step_normalize(all_numeric_predictors()) # all predictors are numeric and on the same scale.
prepped_recipe <- prep(bike_recipe) # Sets up the preprocessing using myDataSet
bake(prepped_recipe, new_data=NULL)


# -------------------------------------------------------------------------
# PENALIZED REGRESSION
preg_model <- linear_reg( #Set model and tuning
  penalty=tune(), # regularization strength (>0)
  mixture=tune()) %>% # elastic net mixing parameter (0 = ridge, 1 = lasso, in between = elastic net)
  set_engine("glmnet") # Function to fit in R

preg_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(preg_model)

penalty_grid <- grid_regular(
  penalty(range = c(-4, 1)), # log10 scale ~ 0.0001 to 10
  mixture(range = c(0, 1)),  # ridge → lasso
  levels = c(10, 5)           # gives 50 combos
)

set.seed(123)
cv_folds <- vfold_cv(trainData, v = 5)

# Tune parameters
preg_tune <- tune_grid(
  preg_wf,
  resamples = cv_folds,
  grid = penalty_grid,
  control = control_grid(save_pred = TRUE)
)

# Collect results
collect_metrics(preg_tune)

best_model <- select_best(preg_tune, metric = "rmse")

final_wf <- finalize_workflow(preg_wf, best_model)

final_fit <- fit(final_wf, data = trainData)

bike_penalized_preds <- predict(final_fit, new_data = testData) %>%
  mutate(.pred = exp(.pred))


# -------------------------------------------------------------------------
# Format the Predictions for Submission to Kaggle
kaggle_submission <- bike_penalized_preds |>
  bind_cols(testData) |> #Bind predictions with test data
  select(datetime, .pred) |> #Just keep datetime and prediction variables
  rename(count=.pred) |> #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) |> #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file9
vroom_write(x=kaggle_submission, 
            file="./KaggleBikeShare/LinearPreds_penalized_regression.csv", 
            delim=",")
