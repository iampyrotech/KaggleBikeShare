# Libraries
library(tidyverse)
library(vroom)
library(GGally)
library(patchwork)
library(tidymodels)

# ---- Read in data ----
train <- vroom("data/train.csv")
head(train)
summary(train)


# ---- Preliminary EDA ----

train %>%
  select_if(is.numeric) %>%
  gather() %>%
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = 'free') +
  geom_histogram(bins = 30, fill = 'steelblue', color = 'white')

train %>%
  select_if(is.numeric) %>%
  gather() %>%
  ggplot(aes(x = key, y = value)) +
  geom_boxplot(fill = 'tomato', alpha = 0.6) +
  facet_wrap(~ key, scales = 'free')


train %>% 
  select_if(is.numeric) %>% 
  cor(use = "pairwise.complete.obs") %>%
  round(2)

train %>% 
  select_if(is.numeric) %>% 
  ggcorr(label = TRUE)

# Bar plot of average count by weather
p1 <- ggplot(train, aes(x = factor(weather), y = count)) +
  stat_summary(fun = "mean", geom = "bar", fill = "steelblue") +
  labs(x = "Weather", y = "Avg Count", title = "Average Rentals by Weather")

# Count by season
p2 <- ggplot(train, aes(x = factor(season), y = count)) +
  stat_summary(fun = "mean", geom = "bar", fill = "darkseagreen") +
  labs(x = "Season", y = "Avg Count", title = "Average Rentals by Season")

# Distribution of temp
p3 <- ggplot(train, aes(x = temp)) +
  geom_histogram(bins = 30, fill = "orange", color = "white") +
  labs(x = "Temperature", y = "Frequency", title = "Temperature Distribution")

# Scatter plot of humidity vs count
p4 <- ggplot(train, aes(x = humidity, y = count)) +
  geom_point(alpha = 0.4, color = "tomato") +
  geom_smooth(method = "lm", se = FALSE, color = "black") +
  labs(x = "Humidity", y = "Count", title = "Humidity vs Rentals")

# Arrange in grid
plot <- (p1 | p2) / (p3 | p4)

ggsave("data/bike_share_eda.pdf", plot = plot, width = 10, height = 8)



