
library(caret) # For creating training and testing splits

library(boot) # For leave-one-out cross-validation

library(tidyverse) # For general data manipulation

library(glmnet) # For ridge and lasso regression

library(randomForest) # For random forest models

library(gbm) # For gradient boosted models

boston <- read.csv("~/Downloads/boston.csv") %>% # Look this data set up on Kaggle under 'Boston Housing'
  na.omit()

penguins <- penguins %>% # Data set should be in base R :)
  na.omit() %>%
  mutate(female = ifelse(sex == "female", 1, 0))

## Validation-set approach

set.seed(1989) # For reproducability

train_index <- createDataPartition(y = boston$MEDV, p = 0.8, list = F)

nrow(boston[train_index,])/nrow(boston) # Percentage in training set

nrow(boston[-train_index,])/nrow(boston) # Percentage in testing set

## Leave-one-out cross-validation

set.seed(1989) # For reproducability

mod <- glm(female ~ bill_len + bill_dep + flipper_len + body_mass, 
           data = penguins, 
           family = "binomial")

loocv <- cv.glm(data = penguins, glmfit = mod)

loocv$delta[2] # Bias-adjusted mean squared error

## k-fold cross-validation

set.seed(1989) # For reproducability
folds <- createFolds(penguins$female, k = 5, list = T, returnTrain = T)
fold_MSE <- numeric(5) # Dummy vector to store mean squared error

for (i in 1:5) {
  train_indices <- folds[[i]] # Each fold is a new training-testing run
  
  train_data <- penguins[train_indices, ]
  test_data <- penguins[-train_indices, ]
  
  mod <- glm(female ~ bill_len + bill_dep + flipper_len + body_mass, 
             data = train_data, 
             family = "binomial")
  
  pred <- data.frame(test_data, 
                     female_hat = predict(mod, test_data, type = "response"))
  
  fold_MSE[i] <- mean((pred$female - pred$female_hat)^2)
}

mean(fold_MSE) # Very similar to LOOCV!

## Creating and testing a generalized linear model with Boston data

set.seed(1989)

folds <- createFolds(boston$MEDV, k = 5, list = T, returnTrain = T) # Creating k = 5 folds

glm_results <- tibble( # Dummy data
  fold = 1:5,
  coef = vector("list", 5),
  mse = NA_real_
)

for (i in 1:5) {
  
  train_indices <- folds[[i]]
  
  train <- boston[train_indices, ]
  test <- boston[-train_indices, ]
  
  mod <- lm(
    MEDV ~ ., 
    train
  )
  
  dfpred_lm <- data.frame( # Making predictions from training and testing data
    test, 
    pred = predict(mod, newdata = test)
  )
  
  glm_results$coef[[i]] <- as_tibble(coef(mod), rownames = "var") # Inputting coefficients to dummy data
  glm_results$mse[[i]] <- mean((dfpred_lm$MEDV - dfpred_lm$pred)^2) # Inputting MSE to dummy data
  
}

glm_results$coef %>% # Averaging beta coefficients in each fold
  bind_rows(.id = "fold") %>%
  group_by(var) %>%
  summarize(value = round(mean(value), 1)) %>%
  pull(var, value)

cat("Generalized linear model MSE:", mean(glm_results$mse)) # MSE

## Creating and testing a ridge regression model with Boston data

set.seed(1989)

ridge_results <- tibble(
  fold = 1:5,
  coef = vector("list", 5),
  mse = NA_real_
)

for (i in 1:5) {
  
  train_indices <- folds[[i]]
  
  train <- boston[train_indices, ]
  test <- boston[-train_indices, ]
  
  x_train <- as.matrix(train[, -14])
  y_train <- train[, 14]
  
  x_test <- as.matrix(test[, -14])
  y_test <- test[, 14]
  
  cv_ridge <- cv.glmnet(
    x = x_train,
    y = y_train,
    family = "gaussian",
    alpha = 0
  )
  
  final_ridge <- glmnet(
    x = x_train,
    y = y_train,
    family = "gaussian",
    alpha = 0,
    lambda = cv_ridge$lambda.1se
  )
  
  dfpred_ridge <- data.frame(
    x_test, 
    pred = predict(
      final_ridge, 
      newx = x_test
    )
  )
  
  ridge_results$coef[[i]] <- as.matrix(coef(final_ridge), rownames = "var")
  ridge_results$mse[[i]] <- mean((y_test - dfpred_ridge$s0)^2)
  
}

round(rowMeans(do.call(cbind, ridge_results$coef)), 1) # Standardized model coefficients

cat("Ridge Regression MSE:", round(mean(ridge_results$mse), 1))

## Creating and testing a lasso regression model using Boston data

set.seed(1989)

lasso_results <- tibble(
  fold = 1:5,
  coef = vector("list", 5),
  mse = NA_real_
)

for (i in 1:5) {
  
  train_indices <- folds[[i]]
  
  train <- boston[train_indices, ]
  test <- boston[-train_indices, ]
  
  x_train <- as.matrix(train[, -14])
  y_train <- train[, 14]
  
  x_test <- as.matrix(test[, -14])
  y_test <- test[, 14]
  
  cv_lasso <- cv.glmnet(
    x = x_train,
    y = y_train,
    family = "gaussian",
    alpha = 1
  )
  
  final_lasso <- glmnet(
    x = x_train,
    y = y_train,
    family = "gaussian",
    alpha = 1,
    lambda = cv_ridge$lambda.1se
  )
  
  dfpred_lasso <- data.frame(
    x_test, 
    pred = predict(
      final_ridge, 
      newx = x_test
    )
  )
  
  lasso_results$coef[[i]] <- as.matrix(coef(final_lasso), rownames = "var")
  lasso_results$mse[[i]] <- mean((y_test - dfpred_lasso$s0)^2)
  
}

round(rowMeans(do.call(cbind, lasso_results$coef)), 1) # Standardized odel coefficients

cat("Lasso Regression MSE:", round(mean(lasso_results$mse), 1))

## Creating and testing a random forest model using the Boston data

set.seed(1989)

rf_results <- tibble(
  fold = 1:5,
  influence = vector("list", 5),
  mse = NA_real_
)

for (i in 1:5) {
  
  train_indices <- folds[[i]]
  
  train <- boston[train_indices, ]
  test <- boston[-train_indices, ]
  
  mod_rf <- randomForest(
    MEDV ~ .,
    train,
    mtry = 3,
    ntree = 500
  )
  
  dfpred_rf <- data.frame(
    test,
    pred = predict(mod_rf, test)
  )
  
  rf_results$influence[[i]] <- as_tibble(importance(mod_rf, type = 2), rownames = "var")
  rf_results$mse[[i]] <- mean((dfpred_rf$MEDV - dfpred_rf$pred)^2)
  
}

rf_results$influence %>% # Average RSS reduction across each node split for each variable
  bind_rows(.id = "fold") %>%
  group_by(var) %>%
  summarize(value = round(mean(IncNodePurity), 1)) %>%
  pull(var, value)

cat("Random Forest MSE:", round(mean(rf_results$mse), 1))

## Creating and testing a gradient boosted model using Boston data

gbm_results <- tibble(
  fold = 1:5,
  influence = vector("list", 5),
  mse = NA_real_
)

for (i in 1:5) {
  
  train_indices <- folds[[i]]
  
  train <- boston[train_indices, ]
  test <- boston[-train_indices, ]
  
  mod_gbm <- gbm(
    MEDV ~ .,
    train,
    distribution = "gaussian",
    n.trees = 500,
    shrinkage = 0.01,
  )
  
  dfpred_gbm <- data.frame(
    test,
    pred = predict(mod_gbm, test)
  )
  
  gbm_results$influence[[i]] <- as_tibble(summary(mod_gbm, plotit = F))
  gbm_results$mse[[i]] <- mean((dfpred_rf$MEDV - dfpred_rf$pred)^2)
  
}

gbm_results$influence %>% # Average error each variable reduces across each subsequent model fit
  bind_rows(.id = "fold") %>%
  group_by(var) %>%
  summarize(value = round(mean(rel.inf), 1)) %>%
  pull(var, value)

cat("Gradient Boosted MSE:", round(mean(gbm_results$mse), 1))

