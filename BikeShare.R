library(tidyverse)
library(tidymodels)
library(vroom)
library(lubridate)

# read in the bike share training data using vroom().
bikeshare <- vroom("KaggleBikeShare/bike-sharing-demand/train.csv") #|>
#  mutate(weather = as.factor(weather), season = as.factor(season), 
#         holiday = as.factor(holiday), workingday = as.factor(workingday),
#         hour = as.factor(lubridate::hour(datetime)))

# In a well-documented cleaning section of your BikeShare.R file that comes 
# BEFORE the modeling, remove the casual and registered variables and change 
# count to log(count) in the training data only.
CleanData <- bikeshare |> 
  mutate(log_count = log(count)) |>
  select(-casual, -registered, -count)

trainData <- CleanData
testData <- vroom("KaggleBikeShare/bike-sharing-demand/test.csv")

# In a feature engineering section BEFORE the modeling, define a recipe that:
# 1. recodes weather “4” to a “3” then makes it a factor
# 2. extracts the hour variable from the timestamp
# 3. makes season a factor
# 4. does 1 other step of your choice
bike_recipe <- recipe(log_count ~ ., data = CleanData) %>% # Set model formula and dataset
  step_mutate(weather = ifelse(weather == 4, 3, weather))
  step_mutate(weather = factor(weather, levels=, labels=)) %>% #Make something a factor
  step_date(timestamp, features="dow") %>% # gets day of week
  step_time(timestamp, features=c("hour", "minute")) %>% #create time variable
  step_dummy(all_nominal_predictors()) %>% #create dummy variables
  step_zv(all_predictors()) %>% #removes zero-variance predictors
  step_corr(all_predictors(), threshold=0.5) %>% # removes > than .5 corr
prepped_recipe <- prep(bike_recipe) # Sets up the preprocessing using myDataSet
bake(prepped_recipe, new_data=testData)

# Combine your linear regression model with your recipe into a linear regression workflow.
# Use your workflow to predict the test data (don’t forget to backtransform the log(count) prediction).
# Print out and show the first 5 rows of your baked dataset and report your Kaggle score to LearningSuite.

  
## Define a Model
lin_model <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

## Combine into a Workflow and fit
bike_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(lin_model) %>%
  fit(data=bikeTrain)

## Run all the steps on test data
lin_preds <- predict(bike_workflow, new_data = bikeTest)

