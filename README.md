# DS-PHASE4-PROJECT

# Problem Statement
The objective of this project is to build a movie recommendation system that help the user to find the right item by minimizing the options since all entertainment websites or online stores have alot of items. It becomes challenging for the customer to select the right one. 

# Business Objectives
The business objectives:

* To create a Collaborative Filtering based Movie Recommendation System.It provides top 5 recommendations to a user, based on their ratings of other movie.
* Predict the rating that a user would give to a movie that he has not yet rated.
* Minimize the difference between the estimated and actual rating (RMSE and MAE).

# Data
The data i worked with was accessed throught the link:(https://grouplens.org/datasets/movielens/latest/), the focus for this project being the  movies and the ratings csv's

# The source of the data

* MovieLens Dataset to an external site (Grouplens)

# Description of data
* Movies data - the datatypes include: int64(1), object(2)
* Ratings data - the datatypes include: int64(2), float64(1)
  
* Data Manipulation, the module used include: pandas and numpy
* Data Visuaalization, the module used include: seaborn, matplotlib
* Modelling, the modules were accessed from the surprise they include: Reader,Dataset,accuracy,train_test_split,cross_validate,GridSearchCV,KNNBasic, KNNBaseline, SVD, mean_squared_error.
* Algorithms used: The modules were accessed from surprise they include KNNBasic,KNNBaseline and the Matrix Factorization-based algorithm: SVD

# Loading of the Data
* First, i had to connect to my google drive, the load the data using the file path, which refers to the google drive as well as the folder where the data is located.

file_path = "/content/drive/MyDrive/Data"
movies = pd.read_csv(file_path + "/movies.csv")
ratings = pd.read_csv(file_path + "/ratings.csv")

* Feature Engineering, by drooping the timestamp column
* Merging the two datasets onto one usibg the common column the "movieid": data = pd.merge(ratings, movies, on="movieId")

# Data Preparation

## Cleaning of Data 
The data did not have any missing values or duplicates, so no cleaning was carried out for this specific data.

## Exploratory Data Analysis(EDA)

### Univariate Analysis 
* Checking the "movieID Feature

* Checking the "userID" Feature
  
* Checking the "title" Feature

* Visualization of the "Rating" feature

![image](https://github.com/user-attachments/assets/9041fdb8-0719-4d0d-a1a2-c1b567833574)

* Visualization of the "Genre" Feature

![image](https://github.com/user-attachments/assets/a66e6a88-938f-4e72-8810-4cad9c6b9c78)

* Visualization of  the relationship between the number of ratings a movie got and the average rating.

![image](https://github.com/user-attachments/assets/6d2c007a-b5aa-47fd-aeb3-bb622f9a1a66)


### Bivariate Analysis

* Visualization of the top 5 movies with the best combined score

![image](https://github.com/user-attachments/assets/bf62c434-bb62-4243-a57e-d8fe47dde0b1)

# Modelling

* The surprise library to be used in the modelling process.
* The algorithms used include:  (KNN basic algorithm), (KNNBaseline algorithm), the Matrix Factorization_based algorithm(SVD).
* During implementation the SVD model to be used in the recommendation system.

* Convertion of the dataframe to a surprise dataset.
* Splitting the dataset: train_set(size 80) and test_set(size=20)

## A baseline memory-based model (KNN basic)

* User-Based (user_based=True): Finds users with similar rating patterns and Predicts a movie's rating for a user by averaging ratings from similar users.
* Similarity metrics used, cosine similarity to measure the similarity between users

## KNNBaseline

* It improves upon KNNBasic by incorporating baseline estimates to address biases in user and item ratings.
* Item-Based (user_based=False): Finds similar movies based on their bias-adjusted rating patterns and Predicts a rating based on how a user rated similar movies.
* When working with moderate-sized datasets and when interpretability is important

## SVD 

* It is one of the most popular and effective algorithms for handling large and sparse recommendation datasets.
* Used when better accuracy is required and when capturing hidden patterns in user behavior

# Model Evaluation 

* The key focus being the accuracy which is the RMSE and MAE. They both  vary for the three models, KNNBasic(RMSE - 1.0100 and MAE - 0.7773), KNNBaseline (RMSE - 0.9758 and MAE - 0.7397) , SVD (RMSE - 0.6448 and MAE - 0.4996)
* KNNBasic only finds similar users/items but doesn’t adjust for rating biases.
* KNNBaseline improves accuracy by accounting for user/item biases.
* SVD learns latent features, capturing deeper patterns, leading to lower RMSE & MAE.
  
* A lower RMSE indicates better predictive performance.
* Similar to RMSE, a lower MAE indicates better accuracy.
* The collaborative filtering algorithm (SVD) achieved relatively low RMSE and MAE, indicating good predictive accuracy


# Visualization of all the models

![image](https://github.com/user-attachments/assets/e863ca3d-bd29-4f56-ab7e-93481f3aa1b3)

# Conclusions

* The focus is on building a movie recommendation system using user-user similarity and matrix factorization. These concepts can be applied to any user-item interaction system.
* Explored generating recommendations based on a similarity matrix and collaborative filtering techniques. Additionally, I attempted to predict movie ratings based on a user's past rating behavior and evaluated the accuracy using RMSE and MAE error metrics.
* There is significant scope for improvement, including experimenting with different techniques and exploring advanced ML/DL algorithms.

