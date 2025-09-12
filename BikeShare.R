library(tidyverse)
library(tidymodels)
library(vroom)
library(lubridate)

# read in the bike share training data using vroom().
bikeshare <- vroom("KaggleBikeShare/bike-sharing-demand/train.csv") |>
  mutate(weather = as.factor(weather), season = as.factor(season), 
         holiday = as.factor(holiday), workingday = as.factor(workingday))

CleanData <- bikeshare |> 
  select(-casual, -registered) |>
  mutate(hour = as.factor(lubridate::hour(datetime)), log_count = log(count))
  


