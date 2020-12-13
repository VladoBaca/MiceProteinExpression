library(corrplot)
library(VIM)
library(keras)

set.seed(9)

data <- read.csv(file="dataset.csv", header=TRUE, sep=",")

# Correlation matrix

imputedData <- kNN(data[, 2:78], k=5, imp_var = FALSE)
corMat <- cor(imputedData)
corrplot(corMat, order = "hclust", tl.pos = FALSE)

# Divide randomly into 80% training data, 20% test data

data <- data[sample(nrow(data)),]
data_train <- data[1:round(nrow(data)*0.8),]
data_test <- data[(round(nrow(data)*0.8)+1):nrow(data),]

# Imputation of missing data

missing_values_count <- length(data[is.na(data)])
missing_values_ratio <- missing_values_count / (77*nrow(data))
missing_values_count
missing_values_ratio

data_train <- kNN(data_train, k=5, imp_var = FALSE)

data_test[is.na(data_test)] <- 0

# Data preparation for learning

train_x <- as.matrix(data_train[, 2:78])

# Class composed of three binary attributes

genotype <- ifelse(data_train$Genotype == "Control", 0, 1)
treatment <- ifelse(data_train$Treatment == "Memantine", 1, 0)
behaviour <- ifelse(data_train$Behavior == "C/S", 1, 0)

train_y <- matrix(c(genotype, treatment, behaviour), ncol = 3, byrow = FALSE)

# Same preparation for test data

test_x <- as.matrix(data_test[, 2:78])

genotype <- ifelse(data_test$Genotype == "Control", 0, 1)
treatment <- ifelse(data_test$Treatment == "Memantine", 1, 0)
behaviour <- ifelse(data_test$Behavior == "C/S", 1, 0)

test_y <- matrix(c(genotype, treatment, behaviour), ncol = 3, byrow = FALSE)

# Preparing the neural network

model <- keras_model_sequential()
model %>% 
  layer_dense(units = 20, activation = "sigmoid", input_shape = c(77)) %>% 
  layer_dense(units = 10, activation = "sigmoid") %>% 
  layer_dense(units = 3, activation = "hard_sigmoid")
  # 96%
  
  #layer_dense(units = 3, activation = "sigmoid", input_shape = c(77))
  # 78%
  
  #layer_dense(units = 10, activation = "sigmoid", input_shape = c(77)) %>% 
  #layer_dense(units = 3, activation = "sigmoid")
  # 94%

model %>% compile(
  loss = 'mse',
  metrics = 'accuracy',
  optimizer = 'rmsprop'
)

summary(model)

# Training

model %>% fit(
  epochs = 1000, batch_size = 10,
  x = train_x, y = train_y, validation_split = 0.1
)

# Testing & evaluation

prediction <- predict(model, x = train_x)
prediction <- round(prediction)

diffs <- rowSums(abs(train_y - prediction))

trainAccuracy <- length(diffs[diffs == 0])/length(diffs)
trainAccuracy


prediction <- predict(model, x = test_x)
prediction <- round(prediction)

diffs <- rowSums(abs(test_y - prediction))

testAccuracy <- length(diffs[diffs == 0])/length(diffs)
testAccuracy

