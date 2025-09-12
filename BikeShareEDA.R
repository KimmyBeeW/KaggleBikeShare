library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)
library(lubridate)
library(GGally)

# read in the bike share training data using vroom().
bikeshare <- vroom("KaggleBikeShare/bike-sharing-demand/train.csv") |>
  mutate(weather = as.factor(weather), season = as.factor(season), 
         holiday = as.factor(holiday), workingday = as.factor(workingday)) |>
  mutate(hour = as.factor(lubridate::hour(datetime)))

# Perform an EDA and identify key features of the dataset.
# EDA
# Strong vs. weak relationships (scatterplots etc.)
# Variables with near-zero variance
#Distribution of categorical data
# Missing data
# Outliers
# Collinearity
# Quantitative vs. categorical
rental_counts <- ggplot(bikeshare, aes(x = count)) +
  geom_histogram(binwidth = 50, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Bike Rentals", x = "Total rentals", y = "Frequency")
rental_counts

seasonal_trends <- ggplot(bikeshare, aes(x = season, y = count, fill = season)) +
  geom_boxplot() +
  labs(title = "Rental Counts by Season", x = "Season", y = "Total Rentals")
seasonal_trends

work_vs_holiday <- ggplot(bikeshare, aes(x = workingday, y = count, fill = holiday)) +
  geom_boxplot() +
  labs(title = "Rentals by Working Day and Holiday", x = "Working Day", y = "Total Rentals")
work_vs_holiday

weather_cond <- ggplot(bikeshare, aes(x = weather, y = count, fill = weather)) +
  geom_boxplot() +
  labs(title = "Rentals by Weather Condition", x = "Weather", y = "Total Rentals")
weather_cond

temp_vs_rentals <- ggplot(bikeshare, aes(x = temp, y = count)) +
  geom_point(alpha = 0.3, color = "steelblue") +
  geom_smooth(method = "loess", se = FALSE, color = "red") +
  labs(title = "Effect of Temperature on Rentals", x = "Temperature (C)", 
       y = "Total Rentals") +
  theme_minimal()
temp_vs_rentals

bikeshare_long <- bikeshare |>
  pivot_longer(cols = c(casual, registered), names_to = "user_type", values_to = "rentals")
registered_vs_casual <- ggplot(bikeshare_long, aes(x = user_type, y = rentals, fill = user_type)) +
  geom_boxplot() +
  labs(title = "Casual vs Registered Rentals", x = "User Type", y = "Rentals")
registered_vs_casual

cvr_time <- ggplot(bikeshare_long, aes(x = datetime, y = rentals, color = user_type)) +
  geom_line(alpha = 0.7) +
  labs(title = "Casual vs Registered Rentals Over Time",
       x = "Date and Time",
       y = "Number of Rentals",
       color = "User Type") +
  theme_minimal()
cvr_time

cvr_time_smooth <- ggplot(bikeshare_long, aes(x = datetime, y = rentals, color = user_type)) +
  geom_smooth(se = FALSE) +
  labs(title = "Casual vs Registered Rentals (Smoothed Trend)",
       x = "Date and Time",
       y = "Rentals",
       color = "User Type") +
  theme_minimal()
cvr_time_smooth

bikeshare_long_daily <- bikeshare_long |>
  mutate(date = as.Date(datetime)) |>
  group_by(date, user_type) |>
  summarise(total_rentals = sum(rentals), .groups = "drop")

cvr_date <- ggplot(bikeshare_long_daily, aes(x = date, y = total_rentals, color = user_type)) +
  geom_line() +
  labs(title = "Casual vs Registered Rentals (Daily Totals)",
       x = "Date",
       y = "Total Rentals",
       color = "User Type") +
  theme_minimal()
cvr_date


hourly_trends <- ggplot(bikeshare, aes(x = hour, y = count)) +
  geom_boxplot(fill = "lightblue") +
  labs(title = "Hourly Rental Trends", x = "Hour of Day", y = "Total Rentals") +
  theme_minimal()
hourly_trends

bikeshare_num <- bikeshare |>
  select(temp, atemp, humidity, windspeed, casual, registered, count)
nums <- ggpairs(bikeshare_num)
nums

weather_bar <- ggplot(bikeshare, aes(x = weather, fill = weather)) +
  geom_bar() +
  geom_text(stat = "count", 
            aes(label = ..count..,
                vjust = ifelse(weather == 4, -0.5, 1.4)), 
            color = "black") +
  labs(title = "Distribution of Weather Conditions",
       x = "Weather Condition",
       y = "Number of Hours Observed") +
  theme_minimal()
weather_bar

# Create a 4 panel ggplot that shows 4 different key features of the dataset.
# One of these panels must be a barplot of weather.
hw <- (weather_bar + hourly_trends) / (temp_vs_rentals + cvr_date)  # 4 panel plot
hw

