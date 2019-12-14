lapply(c("dplyr", "ranger", "iml", "gridExtra"), require,character.only=T)
set.seed(1234)


#########################
### Helpful Functions ###
#########################
# Function useed to access performance
accuracy <- function(data_in){
  # calculate what percentage of the searches have one of the top 5 recommended hotels booked
  data_in2 <- data_in %>%
    group_by(search_uuid) %>%
    mutate(rank=dense_rank(y_hat))
  top_5 <- data_in2 %>%
    group_by(search_uuid) %>%
    arrange(rank) %>% 
    top_n(5)
  top_5$selected <- ifelse(top_5$selected == "yes", 1, 0)
  
  return(sum(top_5$selected) / length(unique(data_in2$search_uuid)))
}

the_booked_one <- function(data_in){
  # extract the booked hotel for one search 
  return(data_in[data_in$selected == "yes", ])
}

the_recommended_one <- function(data_in){
  # extract the top recommended hotel for one search
  data_in_rank <- data_in %>% mutate(rank=dense_rank(y_hat))
  recommended <- data_in_rank %>% arrange(rank) %>% top_n(1)
  return(recommended)
}

mispredicted_hotels <- function(data_in, correctly_predicted){
  # extract list of searches for which the top receommended hotel was mispredicted / correctly predicted 
  # data_in: training data that contrains the predicted probability in variable y_hat
  # correctly_predicted: do you want to extract correctly or incorrectly predicted searches
  data_in_rank <- data_in %>%
    group_by(search_uuid) %>%
    mutate(rank=dense_rank(y_hat))
  top_1 <- data_in_rank %>%
    group_by(search_uuid) %>%
    arrange(rank) %>% 
    top_n(1)
  data_sub <- top_1[top_1$selected == correctly_predicted & top_1$comp_bookings > 1, c("search_uuid", "y_hat")]
  return(data_sub)
}


#################
### Data Load ###
#################
# all data prep is done beforehand
search_data_train <- read.csv("~/Desktop/MLI/train_sample.csv")
search_data_test <- read.csv("~/Desktop/MLI/test_sample.csv")


#################################
### Exploratory Data Analysis ###
#################################
# quick look at the data
sprintf("List of columns: %s", paste(colnames(search_data_train), collapse=", "))
sprintf("Shape of training data: %s", paste(dim(search_data_train), collapse=", "))
sprintf("Number of unique searches in the training data: %s:", length(unique(search_data_train$search_uuid)))
sprintf("Shape of test data: %s", paste(dim(search_data_test), collapse=", "))
sprintf("Number of unique searches in the test data: %s:", length(unique(search_data_test$search_uuid)))

head(search_data_test)

# median number of hotels per search
search_data_train %>% group_by(search_uuid) %>% summarise(count=n()) %>% summarise(median(count))

# distribution of the DV
table(search_data_train$selected)
table(search_data_test$selected)


#############################
### Train a Random Forest ###
#############################
# list of independent variables
IV <- c("distance_km", 
       "user_contract", 
       "with_loyalty", 
       "comp_bookings", 
       "base_price_per_night", 
       "hotel_bookings",
       "reviews_score",
       "hotel_stars_rating",
       "is_out_of_policy",
       "reviews_count")

# keep only IVs and the DV
train_sub <- search_data_train[ ,c(IV, "selected")]
test_sub <- search_data_test[ ,c(IV, "selected")]

# fit a Random Forest
fit_rfTree <- ranger(
  selected~., 
  train_sub, 
  num.trees=100,
  importance="impurity", 
  probability=TRUE,
  max.depth=10,
  mtry=4,
  seed=1234)

# make a prediction
search_data_train["y_hat"] <- predict(fit_rfTree, data=train_sub)$predictions[, 2]
search_data_test["y_hat"] <- predict(fit_rfTree, data=test_sub)$predictions[, 2]

# look at the findal model
fit_rfTree

# check performance
train_acc <- accuracy(search_data_train)
test_acc <- accuracy(search_data_test)
sprintf("Train accuracy %f, Test accuracy %f", train_acc, test_acc)


#########################################
### Model Specific Feature Importance ###
#########################################
par(las=2)
par(mar=c(10,10,2,2))
barplot(sort(fit_rfTree$variable.importance), horiz=TRUE, main="Feature Importance")


#####################################################
### Exercise 1: Model Specific Feature Importance ###
#####################################################
# 1. create a copy of the train and the test data
train_sub_exercise <- train_sub
test_sub_exercise <- test_sub

# 2. create a random uniformly distributed variable with minimum 0 and maximum 100
train_sub_exercise$random_variable <- runif(dim(train_sub_exercise)[1], min=0, max=100)
test_sub_exercise$random_variable <- runif(dim(test_sub_exercise)[1], min=0, max=100)

# train_sub_exercise$random_variable <- rbinom(dim(train_sub_exercise)[1], size=1, prob=0.5)
# test_sub_exercise$random_variable <- rbinom(dim(test_sub_exercise)[1], size=1, prob=0.5)

# 3. retrain the same model
fit_rfTree_exercise <- ranger(
  selected~., 
  train_sub_exercise, 
  num.trees=100, 
  importance="impurity", 
  probability=TRUE,
  max.depth=10,
  mtry=4)

# 4. extract the variable importance and compare with the initial results
par(las=2)
par(mar=c(10,10,4,2))
barplot(sort(fit_rfTree_exercise$variable.importance), horiz=TRUE, main="Feature Importance with a Random Variable")


############################
### Local Shapley Values ###
############################
# model type ranger is still not 100% supported by iml and we neeed to define our own custom prediction function
pred <- function(model, newdata)  {
  results <- as.data.frame(predict(model, data=newdata)$predictions)
  return(as.vector(results))
}

# define predictor objeect
predictor <- Predictor$new(
  model=fit_rfTree, 
  data=subset(train_sub, select=IV), 
  y=train_sub$selected,
  predict.fun=pred,
  type="prob",
  class="yes"
)

# look at one specific search and extract its local explanations
# chose a search for which the booked hotel was not recommended at the top
mispredicted_hotels(search_data_train, correctly_predicted="yes")

# subset the data for one search
selected_search <- "3d11ba8b-7547-4dbd-802a-6e98ddaca1cf"
one_search <- search_data_train[search_data_train$search_uuid == selected_search, ]

# extract the top recommended hotel for the search
one_pred <- the_recommended_one(one_search)

# look at the Shapley Values for the top recommended hotel
shapley_pred <- Shapley$new(predictor, x.interest=subset(one_pred, select=IV))
options(repr.plot.width=4.5, repr.plot.height=3)
plot(shapley_pred)

########################################
### Exercise 2: Local Shapley Values ###
########################################
# 1. chose a search for which the top predicted hotel was not booked
mispredicted_hotels(search_data_train, correctly_predicted="no")

# 2. create a subset dataframe that contains only this search
selected_search_exercise <- "90851c82-fb74-4f91-9542-8757e2ff68aa" # "c1657667-466b-4b1c-a627-aec3fe1fb7d8"
one_search_exercise <- search_data_train[search_data_train$search_uuid == selected_search_exercise, ]

# 3. extract the top recommended hotel for this search
one_pred_exercise <- the_recommended_one(one_search_exercise)

# 4. look at the Shapley Values for the top recommended hotel
shapley_pred_exercise <- Shapley$new(predictor, x.interest=subset(one_pred_exercise, select=IV))
options(repr.plot.width=4.5, repr.plot.height=3)
plot(shapley_pred_exercise)

# 5. extract the booked hotel for this search
one_book_exercise <- the_booked_one(one_search_exercise)

# 6. look at the Shapley Values for the booked hotel
shapley_book_exercise <- Shapley$new(predictor, x.interest=subset(one_book_exercise, select=IV))
options(repr.plot.width=4.5, repr.plot.height=3)
plot(shapley_book_exercise)


#############################
### Global Shaplye Values ###
#############################
# NB: DONT RUN
shap_values <- vector("list", nrow(search_data_train))
for (i in seq_along(shap_values)) {
  shap_values[[i]] <- Shapley$new(predictor, x.interest=search_data_train[i, IV])$results
  shap_values[[i]]$sample_num <- i  # identifier to track our instances.
  if(i %% 100 == 0){
    cat(sprintf("Number of iterations: %s: ", i))
  }
}
data_shap_values <- dplyr::bind_rows(shap_values)
data_shap_values$abs_phi <- abs(data_shap_values$phi)

shap_values_global <- data_shap_values %>% 
  group_by(feature) %>%
  summarise(mean=mean(abs_phi))
shap_values_global <- shap_values_global[order(shap_values_global$mean),]
par(las=2) 
par(mar=c(5,8,4,2))
barplot(shap_values_global$mean, main="Global Shapley Values Importance", names.arg=shap_values_global$feature, horiz=TRUE, cex.names=0.8)
# END DONT RUN


#########################
### Global Surrogate ###
########################
# train a decision tree model to interpret our random forest model
tree <- TreeSurrogate$new(predictor, maxdepth=2)
options(repr.plot.width=5, repr.plot.height=5)
plot(tree) 

# how good is the tree in approximating the black box model
tree$r.squared


#######################
### Local Surrogate ###
#######################
lime.explain <- LocalModel$new(predictor, x.interest=subset(one_pred, select=IV))
plot(lime.explain)


###################################
### Exercise 3: Local Surrogate ###
###################################
# 1. extract the LIME explanation for one of the mispredicted hotels for which you looked at the local Shapley Values
options(repr.plot.width=15, repr.plot.height=5)
par(mfrow=c(1,2))
lime_pred_exercise <- LocalModel$new(predictor, x.interest=subset(one_pred_exercise, select=IV))
grid.arrange(plot(lime_pred_exercise), plot(shapley_pred_exercise), ncol=2) 

# 2. extract the LIME explanation for the actually booked hotels for which you looked at the local Shapley Values
par(mfrow=c(1,2))
lime_book_exercise <- LocalModel$new(predictor, x.interest=subset(one_book_exercise, select=IV))
grid.arrange(plot(lime_book_exercise), plot(shapley_book_exercise), ncol=2) 



####################################################################################################################
##################################################### Appendix #####################################################
####################################################################################################################

######################################
### Permutation feature importance ###
######################################
# we should not specify the specific class when defining the prediction prermutation object
predictor_perm_train <- Predictor$new(
  model=fit_rfTree, 
  data=subset(train_sub, select=IV), 
  y=train_sub$selected,
  predict.fun=pred,
  type="prob"
)
# NB: the below takes time! 
perm_imp_train = FeatureImp$new(predictor_perm_train, loss="ce", compare="difference")
options(repr.plot.width=4, repr.plot.height=3)
plot(perm_imp_train)

# define predictior on test data
predictor_perm_test <- Predictor$new(
  model=fit_rfTree, 
  data=subset(test_sub, select=IV), 
  y=test_sub$selected,
  predict.fun=pred,
  type="prob"
)
# NB: the below takes time! 
perm_imp_test = FeatureImp$new(predictor_perm_test, loss="ce", compare="difference")
options(repr.plot.width=4, repr.plot.height=3)
plot(perm_imp_test)