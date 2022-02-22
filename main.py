import pandas as pd
import numpy as np

#load the data
movies = pd.read_csv("C:/Users/rafal/Desktop/moviedata/movies.csv")
ratings = pd.read_csv("C:/Users/rafal/Desktop/moviedata/ratings.csv")

# minimum number of ratings to consider a movie
minimum_film_ratings = 10

# minimum number of ratings user had to make (all users had rated at least 20 movies)
minimum_user_ratings = 50

#pivot table
ratings_cp = ratings.pivot(index='movieId', columns='userId', values='rating')
#print(ratings_cp)

# drop all the columns with less than minimum user ratings
ratings_cp = ratings_cp.dropna(axis=1, thresh=minimum_user_ratings)

# drop all the rows with less than minimum film ratings
ratings_cp = ratings_cp.dropna(thresh=minimum_film_ratings)

#size of the dataset after cutting
column_number = len(ratings_cp.columns)
row_number = len(ratings_cp)

#replace NaN with zeros
ratings_cp.fillna(0, inplace=True)

ratings_cp_values = ratings_cp.values


# print(ratings_cp)
# ratings_cp = csr_matrix(ratings_cp)

# counting cosine distance
def cosine_distance(first_point, second_point):
    result = np.dot(first_point, second_point)
    norm_first = np.linalg.norm(first_point)
    norm_second = np.linalg.norm(second_point)
    # 1 - cosine similarity
    #return 1 - (result / (norm_first * norm_second))
    return result / (norm_first * norm_second)


size = (row_number, column_number)
vector = np.zeros(size)
distance = np.zeros((row_number, row_number))


def distance_value(ratings_cp_values):
    for index, i in enumerate(ratings_cp_values):
        for index2, j in enumerate(ratings_cp_values):
            if i is not j:
                distance[index, index2] = cosine_distance(i, j)
            else:
                distance[index, index2] = 0


distance_value(ratings_cp_values)
print(distance)


def nearest_neighbor(movieid, k):
    neighbors_index = np.argpartition(distance[0], -k)[-k:]
    distance[0][neighbors_index]
    return neighbors_index
    pass


neighbors_final = nearest_neighbor(0, 10)
ratings_cp_index = np.array(ratings_cp.index)

print(ratings_cp_index[neighbors_final])

final_answer = ratings_cp_index[neighbors_final]
dataset = pd.DataFrame()
dataset['movieId'] = final_answer.tolist()

final_answer = pd.merge(dataset, movies, on='movieId')
pd.set_option("display.max_rows", None, "display.max_columns", None)
print(final_answer)









