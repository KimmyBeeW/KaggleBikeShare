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
  penalty(), # log10 scale
  mixture(), # default range 0 to 1, ridge → lasso
  levels = 5  # gives L^2 combos
)

set.seed(123)
cv_folds <- vfold_cv(trainData, v = 5, repeats = 1)

# Tune parameters
preg_tune <- tune_grid(
  preg_wf,
  resamples = cv_folds,
  grid = penalty_grid,
  metrics=metric_set(rmse, mae),
  control = control_grid(save_pred = TRUE)
)

# Collect results
## Plot Results
rmse_plot <- collect_metrics(preg_tune) %>% # Gathers metrics into DF8
  filter(.metric=="rmse") %>%
  ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
  geom_line() +
  labs(x = "Penalty", y = "Mean RMSE", color = "Mixture")
rmse_plot

ggsave(
  filename = "./KaggleBikeShare/rmse_plot_tune.png", # file path and name
  plot = rmse_plot,                              # the ggplot object
  width = 8, height = 5,                         # size in inches
  dpi = 300                                      # resolution
)


# Model
best_model <- select_best(preg_tune, metric = "rmse")
best_model

final_wf <- finalize_workflow(preg_wf, best_model)

final_fit <- fit(final_wf, data = trainData)

bike_penalized_preds <- predict(final_fit, new_data = testData) %>%
  mutate(.pred = exp(.pred))

# -------------------------------------------------------------------------
# REGRESSION TREE

# 1. Define the model
tree_mod <- decision_tree(
  tree_depth = tune(),
  cost_complexity = tune(),
  min_n = tune()
) %>%
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
cv_folds <- vfold_cv(trainData, v = 5)

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
            file="./KaggleBikeShare/LinearPreds_tree.csv", 
            delim=",")
