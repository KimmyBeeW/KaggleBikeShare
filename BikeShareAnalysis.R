library(tidymodels)
library(tidyverse)
library(vroom)

## Linear Regression
## Read in the Data
trainData <- vroom("KaggleBikeShare/bike-sharing-demand/train.csv") |>
  mutate(weather = as.factor(weather), season = as.factor(season), 
         holiday = as.factor(holiday), workingday = as.factor(workingday)) |>
  select(-casual, -registered)

testData <- vroom("KaggleBikeShare/bike-sharing-demand/test.csv") |>
  mutate(weather = as.factor(weather), season = as.factor(season), 
         holiday = as.factor(holiday), workingday = as.factor(workingday))

## Setup and Fit the Linear Regression Model
my_linear_model <- linear_reg() |> #Type of model
  set_engine("lm") |> # Engine = What R function to use
  set_mode("regression") |> # Regression just means quantitative response
  fit(formula=count ~ .-datetime, data=trainData)

## Generate Predictions Using Linear Model
bike_predictions <- predict(my_linear_model,
                            new_data=testData) # Use fit to predict
bike_predictions ## Look at the output, 1 column .pred



# Format the Predictions for Submission to Kaggle
kaggle_submission <- bike_predictions |>
  bind_cols(testData) |> #Bind predictions with test data
  select(datetime, .pred) |> #Just keep datetime and prediction variables
  rename(count=.pred) |> #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) |> #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file9
vroom_write(x=kaggle_submission, file="./KaggleBikeShare/LinearPreds1.csv", delim=",")
