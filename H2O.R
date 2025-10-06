library(tidyverse)
library(tidymodels)
library(vroom)
library(lubridate)
library(bonsai)
library(lightgbm)
library(agua)

# Load In Data

bikeshare <- vroom("KaggleBikeShare/bike-sharing-demand/train.csv")

bike_data <- bikeshare %>% 
  select(-casual, -registered) %>%
  mutate(count_log = log1p(count)) %>%
  select(-count)

test_data <- vroom("KaggleBikeShare/bike-sharing-demand/test.csv")

# Recipe
my_recipe <- recipe(count_log ~ ., data = bike_data) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather),
              weekend = ifelse(wday(datetime) %in% c(1,7), 1, 0),
              rush_hour = ifelse(hour(datetime) %in% c(7:9,16:19), 1, 0)) %>%
  step_date(datetime, features = c("month", "year", "dow")) %>%
  step_time(datetime, features = "hour") %>%
  step_rm(datetime) %>%
  step_interact(terms = ~ temp:humidity + temp:windspeed + season:humidity) %>%
  step_poly(temp, humidity, degree = 2) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

## Initialize an h2o session
h2o::h2o.init()

## Define the model
## max_runtime_secs = how long to let h2o.ai run
## max_models = how many models to stack
auto_model <- auto_ml() %>%
  set_engine("h2o", max_runtime_secs=180, max_models=6) %>%
  set_mode("regression")

## Combine into Workflow
automl_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(auto_model) %>%
  fit(data=bike_data)

# Predict on test set
bike_predictions <- predict(automl_wf, new_data = test_data) %>%
  mutate(.pred = exp(.pred) - 1,
         .pred = pmax(0, .pred))

# Kaggle submission
kaggle_submission <- test_data %>%
  select(datetime) %>%
  bind_cols(bike_predictions) %>%
  rename(count = .pred) %>%
  mutate(datetime = as.character(format(datetime)))

vroom_write(kaggle_submission, "./KaggleBikeShare/k-subs/H2OPreds.csv", delim = ",")
