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
  step_corr(all_predictors(), threshold=0.5) # removes > than .5 corr
prepped_recipe <- prep(bike_recipe) # Sets up the preprocessing using myDataSet
bake(prepped_recipe, new_data=NULL)


# -------------------------------------------------------------------------
## Model
# Combine your linear regression model with your recipe into a linear regression workflow.

  
## Define a Model
lin_model <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

## Combine into a Workflow and fit
bike_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(lin_model) %>%
  fit(data=trainData)

# -------------------------------------------------------------------------
## Predictions
# Use your workflow to predict the test data (don’t forget to backtransform the log(count) prediction).

## Run all the steps on test data
bike_lin_preds <- predict(bike_workflow, new_data = testData) %>%
  mutate(.pred = exp(.pred))  # back-transform
head(bike_lin_preds, 5)


# -------------------------------------------------------------------------
# Print out and show the first 5 rows of your baked dataset and report your Kaggle score to LearningSuite.

prepped_recipe <- prep(bike_recipe) # Sets up the preprocessing using myDataSet
baked_train <- bake(prepped_recipe, new_data=NULL)
head(baked_train, 5)

# Format the Predictions for Submission to Kaggle
kaggle_submission <- bike_lin_preds |>
  bind_cols(testData) |> #Bind predictions with test data
  select(datetime, .pred) |> #Just keep datetime and prediction variables
  rename(count=.pred) |> #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) |> #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file9
vroom_write(x=kaggle_submission, file="./KaggleBikeShare/LinearPreds_logcount2.csv", delim=",")
