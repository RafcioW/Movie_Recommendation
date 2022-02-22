import pandas as pd
import numpy as np

# load the data

movies = pd.read_csv("/movie_dataset/movies.csv")
ratings = pd.read_csv("/movie_dataset/ratings.csv")

# minimum number of ratings to consider a movie
minimum_film_ratings = 10

# minimum number of ratings user had to make (all users had rated at least 20 movies)
minimum_user_ratings = 50

# pivot table
ratings_cp = ratings.pivot(index='movieId', columns='userId', values='rating')
print(ratings_cp.head())

# drop all the columns with less than minimum user ratings
ratings_cp = ratings_cp.dropna(axis=1, thresh=minimum_user_ratings)

# drop all the rows with less than minimum film ratings
ratings_cp = ratings_cp.dropna(thresh=minimum_film_ratings)

# size of the dataset after cutting
column_number = len(ratings_cp.columns)
row_number = len(ratings_cp)

# replace NaN with zeros
ratings_cp.fillna(0, inplace=True)
print(ratings_cp)
#copy for getting id
ratings_cp_name = ratings_cp.copy(deep=True)
ratings_cp_name.reset_index(inplace=True)

ratings_cp_values = ratings_cp.values
print(ratings_cp_values)

# counting cosine similarity
def cosine_similarity(first_point, second_point):
    result = np.dot(first_point, second_point)
    norm_first = np.linalg.norm(first_point)
    norm_second = np.linalg.norm(second_point)
    # cosine distance: 1 - cosine similarity
    # return 1 - (result / (norm_first * norm_second))
    return result / (norm_first * norm_second)


# size = (row_number, column_number)
distance = np.zeros((row_number, row_number))


def distance_value(ratings_cp_values):
    for index, i in enumerate(ratings_cp_values):
        for index2, j in enumerate(ratings_cp_values):
            if index != index2:
                distance[index, index2] = cosine_similarity(i, j)
            # since we search for maximum values, similarity equal to one will be zero
            else:
                distance[index, index2] = 0


distance_value(ratings_cp_values)

dataset = pd.DataFrame()
ratings_cp_index = np.array(ratings_cp.index)
print(ratings_cp_index)


def name_to_id(movie_name):
    movie = movies[movies['title'].str.contains(movie_name)]
    if len(movie):
        movie_id = movie.iloc[0]['movieId']
        movie_id = ratings_cp_name[ratings_cp_name['movieId']==movie_id].index[0]
        return movie_id


name_of_movie = 'Battlefield Earth'

# getting id based on name of movie
movieid = name_to_id(name_of_movie)
print(movieid)


def nearest_neighbor(movieid, k):
    neighbors_index = np.argpartition(distance[movieid], -k)[-k:]
    distance_final = distance[movieid][neighbors_index]
    dataset['similarity'] = distance_final.tolist()
    return neighbors_index


neighbors_final = nearest_neighbor(movieid, 10)
print(neighbors_final)
# getting ids of neighbors
final_answer = ratings_cp_index[neighbors_final]
print(final_answer)
# output
dataset['movieId'] = final_answer.tolist()
final_answer = pd.merge(dataset, movies, on='movieId')
final_answer.sort_values('similarity', ascending=False, inplace=True)
final_answer = final_answer.reset_index(drop=True)

pd.set_option("display.max_rows", None, "display.max_columns", None)
print(final_answer)
