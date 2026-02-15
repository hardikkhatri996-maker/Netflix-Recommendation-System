#https://drive.google.com/drive/folders/1IqaTG6_2K2g8DOQ6fDus6dwXrglZuGXL?usp=sharing

#https://colab.research.google.com/drive/1lW0SFZDyrceZVwr29OFXOakX0tC1MiKH?usp=sharing


'''
Docstring for Netflix_project_by_intellipaat
'''

'''
SVD (Singular Value Decomposition) in a recommendation system works by finding patterns in user preferences and item similarities.
Here's a basic idea without going deep into the topic

1) What the System Has: A big table (matrix) with users on one side and items (like movies) on the other. Users give ratings to items, but not everyone has rated everything

2) What SVD Does: SVD looks at the ratings that are available and tries to figure out the hidden connections between users and items. It learns what kind of movies users like based on their previous ratings

3) How It Helps: Once SVD understands these patterns, it can predict how a user might rate a movie they haven’t seen yet. Based on these predictions, the system recommends movies that the user is most likely to enjoy'''
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
netflix_dataset =  pd.read_csv('combined_data_1.csv', header=None, names=['UserID', 'Rating'], usecols=[0, 1], low_memory=False)

netflix_dataset

netflix_dataset.tail()

netflix_dataset.dtypes

netflix_dataset.isnull().sum()

#Finding the movie count by assumiung that if there is NaN in Rating column so in front of it , there is a movie id

#get the movie count ( as the NaN values in Rating column will show how many movie are there )
movie_count = netflix_dataset.isnull().sum()
movie_count = movie_count["Rating"]
print("Total number of movies in the dataset: ", movie_count)

#Finding the total unqiue customer id by substracting the movie count from uniuque values in cust id column

# to calculate how many customers we are having in the dataset
customer_count = netflix_dataset['UserID'].nunique() 
print("Total number of customers in the dataset: ", customer_count)

customer_count - movie_count

#get the total number of ratings goven by the customers tp all movies combined
ratings_count = netflix_dataset['UserID'].count() - movie_count
ratings_count


#To find out how many people have rated the movies as 1,2,3,4,5 stars ratings to the movies
stars = netflix_dataset['Rating'].value_counts().sort_index()
stars

ax = stars.plot(kind = 'barh', legend = False, figsize = (15, 10))

plt.title(f'Total pool : {movie_count} movies, {customer_count} customers, {ratings_count} ratings given', fontsize = 20)
plt.grid(True)

netflix_dataset

#lets just make a clear dataframe to find how many MovieId are there

movie_id = None
movie_np = []

# iterate over the dataset
for cust_id in netflix_dataset['UserID']:
    if ':' in str(cust_id):
        # Update the current movie ID
        movie_id = cust_id.replace(':', '')
    movie_np.append(movie_id)

movie_np

# Add the new column to the DataFrame
netflix_dataset['Movie_Id'] = movie_np

# to keep only the rows where the 'Rating' column is not null (i.e., to keep only the rows where there is a rating given by the customer)
netflix_dataset = netflix_dataset[netflix_dataset['Rating'].notna()]

netflix_dataset

netflix_dataset.info()

# Now we dont have movie if in cust id column removed all 1: Nan , 2:Nan so lets fix the data type of cust id column to int
netflix_dataset['UserID'] = netflix_dataset['UserID'].astype(int)
netflix_dataset.info()

#pre - filtering
#now we will remove all the users that have rated less movies and 
#also all those movies that has been rated less in numbers.

dataset_movie_summary = netflix_dataset.groupby('Movie_Id')['Rating'].agg(['count'])
dataset_movie_summary

#now we will create a bencmark
movie_benchmark = round(dataset_movie_summary['count'].quantile(0.60),0)
movie_benchmark


# in this 1st all movies there those are having less ratinig then benchmark will be removed and then all those users will be removed who have rated less movies than the benchmark
drop_movie_list = dataset_movie_summary[dataset_movie_summary['count'] < movie_benchmark].index

drop_movie_list


len(drop_movie_list)

# now we will remove all the users are in-active

dataset_cust_summary = netflix_dataset.groupby('UserID')['Rating'].agg(['count'])
dataset_cust_summary

customer_benchmark = round(dataset_cust_summary['count'].quantile(0.60),0)  
customer_benchmark

#Lets remove all movies that has been rated less than 908 times
#Lets remove all customers that have rated less than 36 movies

# Create drop_cust_list similar to drop_movie_list
drop_cust_list = dataset_cust_summary[dataset_cust_summary['count'] < customer_benchmark].index

netflix_dataset = netflix_dataset[~netflix_dataset['Movie_Id'].isin(drop_movie_list)]
netflix_dataset = netflix_dataset[~netflix_dataset['UserID'].isin(drop_cust_list)]

print('After the triming, the shape is : {}'. format(netflix_dataset.shape))

netflix_dataset # clean netflix data !!!!

netflix_dataset.info()

#Model Building

df_title = pd.read_csv('movie_titles.csv', encoding = 'latin' , header=None, usecols=[0, 1, 2], names=['Movie_Id', 'Year', 'Name'])

from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

reader = Reader()


##we only work with top 100K rows for quick runtime
data = Dataset.load_from_df(netflix_dataset[['UserID', 'Movie_Id', 'Rating']][:100000],reader)

Model  = SVD()

cross_validate(Model, data, measures=['RMSE', ], cv=3) # Trained the model



#Use model for making Recommendation for a specific user

netflix_dataset
df_title

# Filter the dataset for the specific user you have selected to make recokmnedation to that user
user_ratings = netflix_dataset[netflix_dataset['Cust_Id'] == 1331154]  # niharika
user_ratings

# Find the number of unique movies rated by the user
movies_rated_by_user = user_ratings['Movie_Id'].nunique()
print(f"User 1331154 has rated {movies_rated_by_user} unique movies.")

copy_data_for_user = df_title.copy()
copy_data_for_user

CLEAN_copy_data_for_user = copy_data_for_user[~copy_data_for_user['Movie_Id'].isin(drop_movie_list)]
CLEAN_copy_data_for_user   # Removing the movie les then benchmark for this user

CLEAN_copy_data_for_user['Estimate_Score'] = CLEAN_copy_data_for_user['Movie_Id'].apply(lambda x: Model.predict(1331154, x).est)

CLEAN_copy_data_for_user

CLEAN_copy_data_for_user.sort_values('Estimate_Score', ascending=False) # Descending order of the estimated score for the user 1331154

top5_movies = CLEAN_copy_data_for_user.sort_values('Estimate_Score', ascending=False).head(5)

top5_movies