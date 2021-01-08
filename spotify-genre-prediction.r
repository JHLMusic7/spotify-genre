##########################################################
# setup necessary library
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)

options(dplyr.summarise.inform = FALSE)
##########################################################
# Import dataset and data wrangling
##########################################################

#import dataset
#save the excel sheet in the same working directory as this script

getwd() #check current working directory
spotify <- read.csv("genres_v2.csv")

head(spotify)

summary(spotify)

#keep features to predict genre
spotify <- spotify %>%
  select(danceability, energy, key, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms, time_signature, genre)

head(spotify)
##########################################################
# Data visualization and exploration
##########################################################

#investigate genre distribution
spotify %>%
  group_by(genre) %>%
  ggplot(aes(genre)) +
  geom_bar(fill = "red", color = "black") +
  coord_flip()

#investigate tempo distribution by genre
spotify %>%
  group_by(genre) %>%
  ggplot(aes(genre, tempo)) +
  geom_boxplot()

#investigate the average tempo by genre
spotify %>%
  group_by(genre) %>%
  summarise(mean_tempo = mean(tempo)) %>%
  arrange(desc(mean_tempo))

#investigate energy distribution by genre
spotify %>%
  group_by(genre) %>%
  ggplot(aes(genre, energy)) +
  geom_boxplot()

#investigate the average energy level by genre
spotify %>%
  group_by(genre) %>%
  summarise(mean_energy = mean(energy)) %>%
  arrange(desc(mean_energy))

#investigate danceability distribution by genre
spotify %>%
  group_by(genre) %>%
  ggplot(aes(genre, danceability)) +
  geom_boxplot()

#investigate valence distribution by genre
spotify %>%
  group_by(genre) %>%
  ggplot(aes(genre, valence)) +
  geom_boxplot()

#investigate loudness distribution by genre
spotify %>%
  group_by(genre) %>%
  ggplot(aes(genre, loudness)) +
  geom_boxplot()

#investigate duration distribution by genre
spotify %>%
  group_by(genre) %>%
  ggplot(aes(genre, duration_ms)) +
  geom_boxplot()



##########################################################
# Analysis - genre prediction
#
#training, test, validation dataset preparation
##########################################################

set.seed(7, sample.kind = "Rounding")

verify_index <- createDataPartition(spotify$genre, times = 1, p = 0.2, list = FALSE)
analysis <- spotify[-verify_index,]
temp <- spotify[verify_index,]

validation <- temp %>%
  semi_join(analysis, by = "genre")

removed <- anti_join(temp, validation)
analysis <- rbind(analysis, removed)

set.seed(7, sample.kind = "Rounding")

test_index <- createDataPartition(analysis$genre, times = 1, p = 0.2, list = FALSE)
training_set <- analysis[-test_index,]
test_set <- analysis[test_index,]

rm(removed, temp, verify_index, analysis, test_index)

##########################################################
# Analysis - genre prediction
#
#training and testing different Machine learning algorithms
##########################################################

#lda model

set.seed(7, sample.kind = "Rounding")

train_lda <- train(genre ~ ., data = training_set, method = "lda")
s_lda <- predict(train_lda, test_set)
mean(test_set$genre == s_lda)

#qda model

set.seed(7, sample.kind = "Rounding")

train_qda <- train(genre ~ ., data = training_set, method = "qda")
s_qda <- predict(train_qda, test_set)
mean(test_set$genre == s_qda)

#random forest model 1

set.seed(7, sample.kind = "Rounding")

train_rf <- train(genre ~ ., data = training_set, method = "rf",
                  tuneGrid = data.frame(mtry = seq(1:7)),
                  ntree = 100)

train_rf$bestTune

s_rf <- predict(train_rf, test_set)

mean(s_rf == test_set$genre)

varImp(train_rf)

#random forest model 2

set.seed(7, sample.kind = "Rounding")

train_rf2 <- train(genre ~ ., data = training_set, method = "rf",
                  tuneGrid = data.frame(mtry = seq(1:7)),
                  ntree = 500)

train_rf2$bestTune

s_rf2 <- predict(train_rf2, test_set)

mean(s_rf2 == test_set$genre)

varImp(train_rf2)

#knn model

set.seed(7, sample.kind = "Rounding")

train_knn <- train(genre ~ ., data = training_set, method = "knn",
                   tuneGrid = data.frame(k = seq(1, 51, 2)),
                   trControl = trainControl(method = "cv", number=10, p=0.9))

train_knn$bestTune

ggplot(train_knn)

s_knn <- predict(train_knn, test_set)

mean(s_knn == test_set$genre)

#knn model with improtant variables (determined from varImp from rf model)

set.seed(7, sample.kind = "Rounding")

train_knn2 <- train(genre ~ tempo + danceability + instrumentalness + loudness, data = training_set, method = "knn",
                   tuneGrid = data.frame(k = seq(1, 51, 2)),
                   trControl = trainControl(method = "cv", number=10, p=0.9))

train_knn2$bestTune

ggplot(train_knn2)

s_knn2 <- predict(train_knn2, test_set)

mean(s_knn2 == test_set$genre)

#results table

results <- tibble(Method = "LDA model", Accuracy = mean(test_set$genre == s_lda))
results <- bind_rows(results, tibble(Method = "QDA model", Accuracy = mean(test_set$genre == s_qda)))
results <- bind_rows(results, tibble(Method = "Random forest model (100 nodes)", Accuracy = mean(test_set$genre == s_rf)))
results <- bind_rows(results, tibble(Method = "Random forest model (500 nodes)", Accuracy = mean(test_set$genre == s_rf2)))
results <- bind_rows(results, tibble(Method = "kNN model - all predictors", Accuracy = mean(test_set$genre == s_knn)))
results <- bind_rows(results, tibble(Method = "kNN model - important variable", Accuracy = mean(test_set$genre == s_knn2)))

results %>%
  arrange(desc(Accuracy))

